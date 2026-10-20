from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from pathlib import PurePosixPath
from tempfile import TemporaryDirectory
from typing import Callable, Dict, Sequence, Union
from zipfile import BadZipFile, ZipFile, ZipInfo


class DocumentIngestionError(Exception):
    """Base exception for document ingestion failures."""


class DocumentNotFoundError(DocumentIngestionError):
    """Raised when the requested path does not point to a readable file."""


class UnsupportedDocumentTypeError(DocumentIngestionError):
    """Raised when no text extractor is registered for a file extension."""


class DocumentDecodingError(DocumentIngestionError):
    """Raised when a supported text document cannot be decoded as UTF-8."""


class EmptyDocumentError(DocumentIngestionError):
    """Raised when a document contains no useful text after normalization."""


class PDFTextExtractionError(DocumentIngestionError):
    """Raised when a PDF cannot provide extractable, native text."""


class DOCXTextExtractionError(DocumentIngestionError):
    """Raised when a DOCX cannot provide readable paragraph text."""


class ArchiveIngestionError(DocumentIngestionError):
    """Raised when a ZIP archive cannot be inspected or read safely."""


class UnsafeArchiveError(ArchiveIngestionError):
    """Raised when a ZIP archive exceeds a security limit."""


@dataclass(frozen=True)
class IngestedDocument:
    """Normalized document and its source metadata, ready for chunking."""

    document_id: str
    source_filename: str
    source_path: str
    text: str
    character_count: int
    source_extension: str = ""
    source_type: str = "document"
    language: str | None = None
    archive_filename: str | None = None
    archive_entry_path: str | None = None


@dataclass(frozen=True)
class ArchiveEntryFailure:
    """One supported archive entry that could not be extracted."""

    entry_path: str
    reason: str


@dataclass(frozen=True)
class ArchiveIngestionResult:
    """Documents and non-fatal outcomes collected from one safe ZIP archive."""

    archive_filename: str
    archive_path: str
    documents: Sequence[IngestedDocument]
    skipped_entries: Sequence[str]
    failed_entries: Sequence[ArchiveEntryFailure]


TextExtractor = Callable[[Path], str]
PathLike = Union[str, Path]


@dataclass(frozen=True)
class ExtractorRegistration:
    """A text extractor and the metadata associated with its file extension."""

    extractor: TextExtractor
    source_type: str
    language: str | None


class DocumentIngestor:
    """Extract supported local documents into a stable, normalized format.

    New formats can be introduced by registering an extractor for an extension,
    for example ``ingestor.register_extractor('.pdf', extract_pdf_text)``.
    The callers and the rest of the RAG pipeline do not need to change.
    """

    _SOURCE_TYPES = frozenset({"document", "code", "configuration", "markup"})
    MAX_ARCHIVE_SIZE_BYTES = 50 * 1024 * 1024
    MAX_ARCHIVE_UNCOMPRESSED_SIZE_BYTES = 250 * 1024 * 1024
    MAX_ARCHIVE_FILE_COUNT = 1_000
    MAX_ARCHIVE_NESTING_DEPTH = 10
    MAX_ARCHIVE_COMPRESSION_RATIO = 100
    _DEFAULT_FORMATS = (
        (".txt", "document", None),
        (".md", "markup", "markdown"),
        (".pdf", "document", "pdf"),
        (".docx", "document", "docx"),
        (".py", "code", "python"),
        (".c", "code", "c"),
        (".h", "code", "c/c++ header"),
        (".cpp", "code", "c++"),
        (".hpp", "code", "c++"),
        (".js", "code", "javascript"),
        (".ts", "code", "typescript"),
        (".java", "code", "java"),
        (".go", "code", "go"),
        (".rs", "code", "rust"),
        (".html", "markup", "html"),
        (".css", "code", "css"),
        (".xml", "markup", "xml"),
        (".svg", "markup", "svg"),
        (".json", "configuration", "json"),
        (".yaml", "configuration", "yaml"),
        (".yml", "configuration", "yaml"),
        (".toml", "configuration", "toml"),
        (".ini", "configuration", "ini"),
        (".sql", "code", "sql"),
        (".sh", "code", "shell"),
    )

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding
        self._extractors: Dict[str, ExtractorRegistration] = {}
        for extension, source_type, language in self._DEFAULT_FORMATS:
            extractor = self._extract_text_file
            if extension == ".pdf":
                extractor = self._extract_pdf_file
            elif extension == ".docx":
                extractor = self._extract_docx_file
            self.register_extractor(
                extension,
                extractor,
                source_type=source_type,
                language=language,
            )

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """Return supported extensions in a predictable order for CLI messages."""
        return tuple(sorted(self._extractors))

    def register_extractor(
        self,
        extension: str,
        extractor: TextExtractor,
        *,
        source_type: str = "document",
        language: str | None = None,
    ) -> None:
        """Register an extractor for one file extension.

        Keeping extraction behind this small registry makes later PDF support a
        local change: only a PDF extractor needs to be added.
        """
        normalized_extension = extension.lower()
        if not normalized_extension.startswith("."):
            raise ValueError("A document extension must start with a dot, for example '.pdf'.")
        normalized_source_type = source_type.lower()
        if normalized_source_type not in self._SOURCE_TYPES:
            allowed_types = ", ".join(sorted(self._SOURCE_TYPES))
            raise ValueError(f"source_type must be one of: {allowed_types}.")
        normalized_language = language.lower() if language else None
        self._extractors[normalized_extension] = ExtractorRegistration(
            extractor=extractor,
            source_type=normalized_source_type,
            language=normalized_language,
        )

    def ingest(self, document_path: PathLike) -> IngestedDocument:
        """Validate and extract a document into text and source metadata."""
        path = Path(document_path).expanduser()
        if not path.exists() or not path.is_file():
            raise DocumentNotFoundError(f"Document not found: {path}")

        extension = path.suffix.lower()
        registration = self._extractors.get(extension)
        if registration is None:
            supported = ", ".join(self.supported_extensions)
            raise UnsupportedDocumentTypeError(
                f"Unsupported document type '{extension or '[no extension]'}'. "
                f"Supported types: {supported}."
            )

        try:
            extracted_text = registration.extractor(path)
        except DocumentIngestionError:
            raise
        except UnicodeDecodeError as error:
            raise DocumentDecodingError(
                f"Could not decode '{path.name}' as {self.encoding}. "
                "Save the file as UTF-8 and try again."
            ) from error
        except OSError as error:
            raise DocumentIngestionError(f"Could not read '{path}': {error}") from error

        normalized_text = self._normalize_text(extracted_text)
        if not normalized_text:
            raise EmptyDocumentError(f"Document '{path.name}' is empty.")

        document_id = sha256(normalized_text.encode("utf-8")).hexdigest()
        return IngestedDocument(
            document_id=document_id,
            source_filename=path.name,
            source_path=str(path.resolve()),
            text=normalized_text,
            character_count=len(normalized_text),
            source_extension=extension,
            source_type=registration.source_type,
            language=registration.language,
        )

    def ingest_archive(self, archive_path: PathLike) -> ArchiveIngestionResult:
        """Extract supported entries from a ZIP without writing archive paths to disk.

        Individual unsupported or malformed documents are recorded and do not
        prevent other safe entries from being ingested. Archive-level safety
        violations reject the entire archive before any entry is processed.
        """
        path = Path(archive_path).expanduser()
        if not path.exists() or not path.is_file():
            raise DocumentNotFoundError(f"Document not found: {path}")
        if path.suffix.lower() != ".zip":
            raise UnsupportedDocumentTypeError(f"Archive '{path.name}' is not a ZIP file.")
        if path.stat().st_size > self.MAX_ARCHIVE_SIZE_BYTES:
            raise UnsafeArchiveError(
                f"Archive '{path.name}' exceeds the maximum compressed size "
                f"of {self.MAX_ARCHIVE_SIZE_BYTES} bytes."
            )

        try:
            with ZipFile(path) as archive:
                entries = self._validate_archive_entries(archive, path.name)
                return self._ingest_archive_entries(path, archive, entries)
        except BadZipFile as error:
            raise ArchiveIngestionError(
                f"Could not read ZIP archive '{path.name}'. The file may be corrupted."
            ) from error
        except OSError as error:
            raise ArchiveIngestionError(f"Could not read archive '{path}': {error}") from error

    def _validate_archive_entries(self, archive: ZipFile, archive_name: str) -> list[ZipInfo]:
        entries = [entry for entry in archive.infolist() if not entry.is_dir()]
        if len(entries) > self.MAX_ARCHIVE_FILE_COUNT:
            raise UnsafeArchiveError(
                f"Archive '{archive_name}' contains more than {self.MAX_ARCHIVE_FILE_COUNT} files."
            )

        total_uncompressed_size = 0
        for entry in entries:
            entry_path = self._safe_archive_entry_path(entry.filename, archive_name)
            if len(entry_path.parts) - 1 > self.MAX_ARCHIVE_NESTING_DEPTH:
                raise UnsafeArchiveError(
                    f"Archive entry '{entry.filename}' exceeds the maximum nesting depth "
                    f"of {self.MAX_ARCHIVE_NESTING_DEPTH}."
                )
            if entry.flag_bits & 0x1:
                raise UnsafeArchiveError(
                    f"Archive entry '{entry.filename}' is password-protected, which is not supported."
                )

            total_uncompressed_size += entry.file_size
            if total_uncompressed_size > self.MAX_ARCHIVE_UNCOMPRESSED_SIZE_BYTES:
                raise UnsafeArchiveError(
                    f"Archive '{archive_name}' exceeds the maximum uncompressed size "
                    f"of {self.MAX_ARCHIVE_UNCOMPRESSED_SIZE_BYTES} bytes."
                )
            if entry.file_size and not entry.compress_size:
                raise UnsafeArchiveError(
                    f"Archive entry '{entry.filename}' has a suspicious compression ratio."
                )
            if entry.compress_size and entry.file_size / entry.compress_size > self.MAX_ARCHIVE_COMPRESSION_RATIO:
                raise UnsafeArchiveError(
                    f"Archive entry '{entry.filename}' exceeds the maximum compression ratio "
                    f"of {self.MAX_ARCHIVE_COMPRESSION_RATIO}:1."
                )
        return entries

    @staticmethod
    def _safe_archive_entry_path(entry_name: str, archive_name: str) -> PurePosixPath:
        normalized_name = entry_name.replace("\\", "/")
        entry_path = PurePosixPath(normalized_name)
        if (
            entry_path.is_absolute()
            or ".." in entry_path.parts
            or not entry_path.parts
            or entry_path.parts[0].endswith(":")
        ):
            raise UnsafeArchiveError(
                f"Archive '{archive_name}' contains an unsafe path: '{entry_name}'."
            )
        return entry_path

    def _ingest_archive_entries(
        self,
        archive_path: Path,
        archive: ZipFile,
        entries: Sequence[ZipInfo],
    ) -> ArchiveIngestionResult:
        documents: list[IngestedDocument] = []
        skipped_entries: list[str] = []
        failed_entries: list[ArchiveEntryFailure] = []

        with TemporaryDirectory(prefix="sensai-rag-") as temporary_directory:
            temporary_root = Path(temporary_directory)
            for index, entry in enumerate(entries):
                entry_path = self._safe_archive_entry_path(entry.filename, archive_path.name)
                extension = entry_path.suffix.lower()
                if extension not in self._extractors:
                    skipped_entries.append(entry_path.as_posix())
                    continue

                try:
                    temporary_path = temporary_root / f"entry-{index}{extension}"
                    temporary_path.write_bytes(archive.read(entry))
                    document = self.ingest(temporary_path)
                except (DocumentIngestionError, OSError, RuntimeError) as error:
                    failed_entries.append(ArchiveEntryFailure(entry_path.as_posix(), str(error)))
                    continue

                documents.append(
                    replace(
                        document,
                        source_filename=f"{archive_path.name}::{entry_path.as_posix()}",
                        source_path=f"{archive_path.resolve()}::{entry_path.as_posix()}",
                        archive_filename=archive_path.name,
                        archive_entry_path=entry_path.as_posix(),
                    )
                )

        return ArchiveIngestionResult(
            archive_filename=archive_path.name,
            archive_path=str(archive_path.resolve()),
            documents=documents,
            skipped_entries=skipped_entries,
            failed_entries=failed_entries,
        )

    def _extract_text_file(self, path: Path) -> str:
        return path.read_text(encoding=self.encoding)

    @staticmethod
    def _load_pdf_reader():
        """Load pypdf only when a PDF is requested.

        This keeps plain-text and source-code ingestion independent from the
        optional PDF implementation at import time.
        """
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise PDFTextExtractionError(
                "PDF support requires the 'pypdf' package. "
                "Install the project dependencies and try again."
            ) from error
        return PdfReader

    def _extract_pdf_file(self, path: Path) -> str:
        """Return native text from a PDF without OCR or layout reconstruction."""
        pdf_reader = self._load_pdf_reader()
        try:
            reader = pdf_reader(path)
            page_texts = [page.extract_text() or "" for page in reader.pages]
        except Exception as error:
            raise PDFTextExtractionError(
                f"Could not read PDF '{path.name}'. The file may be corrupted or invalid."
            ) from error

        extracted_text = "\n".join(page_texts)
        if not extracted_text.strip():
            raise PDFTextExtractionError(
                f"PDF '{path.name}' contains no extractable text. "
                "It may be scanned or image-only; OCR is not yet supported."
            )
        return extracted_text

    @staticmethod
    def _load_docx_document():
        """Load python-docx only when a Word document is requested."""
        try:
            from docx import Document
        except ImportError as error:
            raise DOCXTextExtractionError(
                "DOCX support requires the 'python-docx' package. "
                "Install the project dependencies and try again."
            ) from error
        return Document

    def _extract_docx_file(self, path: Path) -> str:
        """Return paragraphs and headings in their original document order."""
        document_factory = self._load_docx_document()
        try:
            document = document_factory(path)
            paragraphs = [paragraph.text for paragraph in document.paragraphs]
        except Exception as error:
            raise DOCXTextExtractionError(
                f"Could not read DOCX '{path.name}'. "
                "The file may be corrupted, encrypted, or protected."
            ) from error

        extracted_text = "\n".join(paragraphs)
        if not extracted_text.strip():
            raise DOCXTextExtractionError(
                f"DOCX '{path.name}' contains no extractable paragraph text."
            )
        return extracted_text

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize line endings while preserving the document's wording."""
        return text.replace("\r\n", "\n").replace("\r", "\n").strip()
