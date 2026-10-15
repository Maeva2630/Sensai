import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from rag.ingestion import (
    DocumentDecodingError,
    DocumentIngestor,
    DocumentNotFoundError,
    DOCXTextExtractionError,
    EmptyDocumentError,
    PDFTextExtractionError,
    UnsafeArchiveError,
    UnsupportedDocumentTypeError,
)


class TestDocumentIngestor(unittest.TestCase):
    def setUp(self):
        self.ingestor = DocumentIngestor()
        self.temp_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_directory.name)

    def tearDown(self):
        self.temp_directory.cleanup()

    def _write_file(self, name: str, content: str, encoding: str = "utf-8") -> Path:
        path = self.directory / name
        path.write_text(content, encoding=encoding)
        return path

    def _write_archive(self, name: str, entries: dict[str, str]) -> Path:
        path = self.directory / name
        with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
            for entry_name, content in entries.items():
                archive.writestr(entry_name, content)
        return path

    def test_ingests_text_file_with_normalized_content_and_metadata(self):
        document_path = self._write_file("guide.txt", "First line\r\nSecond line\r\n")

        document = self.ingestor.ingest(document_path)

        self.assertEqual(document.source_filename, "guide.txt")
        self.assertEqual(document.source_path, str(document_path.resolve()))
        self.assertEqual(document.text, "First line\nSecond line")
        self.assertEqual(document.character_count, len("First line\nSecond line"))
        self.assertEqual(len(document.document_id), 64)
        self.assertEqual(document.source_extension, ".txt")
        self.assertEqual(document.source_type, "document")
        self.assertIsNone(document.language)

    def test_ingests_markdown_file(self):
        document_path = self._write_file("notes.MD", "# Project notes\n\nUseful content.")

        document = self.ingestor.ingest(document_path)

        self.assertEqual(document.source_filename, "notes.MD")
        self.assertIn("Useful content.", document.text)
        self.assertEqual(document.source_type, "markup")
        self.assertEqual(document.language, "markdown")

    def test_ingests_python_code_without_executing_it(self):
        document_path = self._write_file(
            "example.py",
            "def add(left, right):\n    return left + right\n",
        )

        document = self.ingestor.ingest(document_path)

        self.assertEqual(document.text, "def add(left, right):\n    return left + right")
        self.assertEqual(document.source_extension, ".py")
        self.assertEqual(document.source_type, "code")
        self.assertEqual(document.language, "python")

    def test_ingests_svg_and_configuration_files_with_metadata(self):
        svg_path = self._write_file("logo.svg", "<svg><circle /></svg>")
        yaml_path = self._write_file("config.yaml", "database:\n  host: localhost\n")

        svg_document = self.ingestor.ingest(svg_path)
        yaml_document = self.ingestor.ingest(yaml_path)

        self.assertEqual((svg_document.source_type, svg_document.language), ("markup", "svg"))
        self.assertEqual((yaml_document.source_type, yaml_document.language), ("configuration", "yaml"))

    def test_reingesting_unchanged_content_produces_same_identifier(self):
        document_path = self._write_file("knowledge.md", "Reusable knowledge")

        first_document = self.ingestor.ingest(document_path)
        second_document = self.ingestor.ingest(document_path)

        self.assertEqual(first_document.document_id, second_document.document_id)

    def test_missing_document_raises_clear_error(self):
        with self.assertRaises(DocumentNotFoundError):
            self.ingestor.ingest(self.directory / "missing.md")

    def test_unsupported_document_type_raises_clear_error(self):
        document_path = self._write_file("report.doc", "Not a real Word document")

        with self.assertRaises(UnsupportedDocumentTypeError) as context:
            self.ingestor.ingest(document_path)

        self.assertIn(".doc", str(context.exception))
        self.assertIn(".txt", str(context.exception))

    @patch.object(DocumentIngestor, "_load_pdf_reader")
    def test_ingests_text_based_pdf_with_document_metadata(self, mock_reader_loader):
        document_path = self.directory / "guide.pdf"
        document_path.write_bytes(b"%PDF-not-used-by-mock")

        class FakePage:
            def extract_text(self):
                return "PDF native text"

        class FakeReader:
            def __init__(self, path):
                self.pages = [FakePage()]

        mock_reader_loader.return_value = FakeReader

        document = self.ingestor.ingest(document_path)

        self.assertEqual(document.text, "PDF native text")
        self.assertEqual(document.source_extension, ".pdf")
        self.assertEqual(document.source_type, "document")
        self.assertEqual(document.language, "pdf")

    @patch.object(DocumentIngestor, "_load_pdf_reader")
    def test_image_only_pdf_reports_that_ocr_is_not_supported(self, mock_reader_loader):
        document_path = self.directory / "scan.pdf"
        document_path.write_bytes(b"%PDF-not-used-by-mock")

        class ImageOnlyPage:
            def extract_text(self):
                return None

        class FakeReader:
            def __init__(self, path):
                self.pages = [ImageOnlyPage()]

        mock_reader_loader.return_value = FakeReader

        with self.assertRaises(PDFTextExtractionError) as context:
            self.ingestor.ingest(document_path)

        self.assertIn("OCR is not yet supported", str(context.exception))

    @patch.object(DocumentIngestor, "_load_pdf_reader")
    def test_corrupted_pdf_reports_a_clear_error(self, mock_reader_loader):
        document_path = self.directory / "broken.pdf"
        document_path.write_bytes(b"not-a-pdf")

        class BrokenReader:
            def __init__(self, path):
                raise ValueError("invalid PDF")

        mock_reader_loader.return_value = BrokenReader

        with self.assertRaises(PDFTextExtractionError) as context:
            self.ingestor.ingest(document_path)

        self.assertIn("corrupted or invalid", str(context.exception))

    @patch.object(DocumentIngestor, "_load_docx_document")
    def test_ingests_docx_paragraphs_and_headings_in_order(self, mock_document_loader):
        document_path = self.directory / "guide.docx"
        document_path.write_bytes(b"not-used-by-mock")

        class FakeParagraph:
            def __init__(self, text):
                self.text = text

        class FakeDocument:
            paragraphs = [
                FakeParagraph("Getting started"),
                FakeParagraph("Install SensAI locally."),
                FakeParagraph("Ask a question."),
            ]

        mock_document_loader.return_value = lambda path: FakeDocument()

        document = self.ingestor.ingest(document_path)

        self.assertEqual(
            document.text,
            "Getting started\nInstall SensAI locally.\nAsk a question.",
        )
        self.assertEqual(document.source_extension, ".docx")
        self.assertEqual(document.source_type, "document")
        self.assertEqual(document.language, "docx")

    @patch.object(DocumentIngestor, "_load_docx_document")
    def test_empty_docx_reports_a_clear_error(self, mock_document_loader):
        document_path = self.directory / "empty.docx"
        document_path.write_bytes(b"not-used-by-mock")

        class EmptyDocument:
            paragraphs = []

        mock_document_loader.return_value = lambda path: EmptyDocument()

        with self.assertRaises(DOCXTextExtractionError) as context:
            self.ingestor.ingest(document_path)

        self.assertIn("no extractable paragraph text", str(context.exception))

    @patch.object(DocumentIngestor, "_load_docx_document")
    def test_corrupted_or_protected_docx_reports_a_clear_error(self, mock_document_loader):
        document_path = self.directory / "protected.docx"
        document_path.write_bytes(b"not-used-by-mock")
        mock_document_loader.return_value = lambda path: (_ for _ in ()).throw(
            ValueError("unreadable package")
        )

        with self.assertRaises(DOCXTextExtractionError) as context:
            self.ingestor.ingest(document_path)

        self.assertIn("corrupted, encrypted, or protected", str(context.exception))

    def test_ingests_supported_archive_entries_with_archive_metadata(self):
        archive_path = self._write_archive(
            "project.zip",
            {
                "docs/guide.md": "# Guide\nUseful documentation.",
                "src/main.py": "print('hello')\n",
            },
        )

        result = self.ingestor.ingest_archive(archive_path)

        self.assertEqual(len(result.documents), 2)
        self.assertEqual(result.skipped_entries, [])
        self.assertEqual(result.failed_entries, [])
        first_document = result.documents[0]
        self.assertEqual(first_document.source_filename, "project.zip::docs/guide.md")
        self.assertEqual(first_document.archive_filename, "project.zip")
        self.assertEqual(first_document.archive_entry_path, "docs/guide.md")
        self.assertTrue(first_document.source_path.endswith("project.zip::docs/guide.md"))

    def test_archive_skips_unsupported_entries_and_continues_after_failed_extraction(self):
        archive_path = self._write_archive(
            "mixed.zip",
            {
                "useful.txt": "Useful text",
                "unsupported.exe": "binary",
                "empty.md": " \n\t",
            },
        )

        result = self.ingestor.ingest_archive(archive_path)

        self.assertEqual([document.archive_entry_path for document in result.documents], ["useful.txt"])
        self.assertEqual(result.skipped_entries, ["unsupported.exe"])
        self.assertEqual(len(result.failed_entries), 1)
        self.assertEqual(result.failed_entries[0].entry_path, "empty.md")
        self.assertIn("empty", result.failed_entries[0].reason)

    def test_archive_rejects_path_traversal_before_processing_entries(self):
        archive_path = self._write_archive("unsafe.zip", {"../../sensitive.txt": "Do not read"})

        with self.assertRaises(UnsafeArchiveError) as context:
            self.ingestor.ingest_archive(archive_path)

        self.assertIn("unsafe path", str(context.exception))

    def test_archive_rejects_too_many_files(self):
        archive_path = self._write_archive(
            "many-files.zip",
            {"one.txt": "one", "two.txt": "two"},
        )

        with patch.object(self.ingestor, "MAX_ARCHIVE_FILE_COUNT", 1):
            with self.assertRaises(UnsafeArchiveError) as context:
                self.ingestor.ingest_archive(archive_path)

        self.assertIn("more than 1 files", str(context.exception))

    def test_empty_document_raises_clear_error(self):
        document_path = self._write_file("empty.txt", " \n\t ")

        with self.assertRaises(EmptyDocumentError):
            self.ingestor.ingest(document_path)

    def test_non_utf8_document_raises_clear_error(self):
        document_path = self.directory / "legacy.txt"
        document_path.write_bytes(b"\xff\xfe\x00")

        with self.assertRaises(DocumentDecodingError):
            self.ingestor.ingest(document_path)

    def test_new_file_types_can_be_added_without_changing_ingest(self):
        document_path = self._write_file("future.pdf", "Extracted PDF text")
        self.ingestor.register_extractor(
            ".pdf",
            lambda path: path.read_text(encoding="utf-8"),
            source_type="document",
            language="pdf",
        )

        document = self.ingestor.ingest(document_path)

        self.assertEqual(document.text, "Extracted PDF text")
        self.assertEqual(document.language, "pdf")

    def test_rejects_unknown_source_types_for_custom_extractors(self):
        with self.assertRaises(ValueError):
            self.ingestor.register_extractor(".custom", lambda path: "text", source_type="binary")


if __name__ == "__main__":
    unittest.main()
