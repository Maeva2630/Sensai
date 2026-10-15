import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from rag.ingestion import (
    DocumentDecodingError,
    DocumentIngestor,
    DocumentNotFoundError,
    EmptyDocumentError,
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
        document_path = self._write_file("report.pdf", "Not a real PDF")

        with self.assertRaises(UnsupportedDocumentTypeError) as context:
            self.ingestor.ingest(document_path)

        self.assertIn(".pdf", str(context.exception))
        self.assertIn(".txt", str(context.exception))

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
