from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Dict, Union


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
    _DEFAULT_FORMATS = (
        (".txt", "document", None),
        (".md", "markup", "markdown"),
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
            self.register_extractor(
                extension,
                self._extract_text_file,
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

    def _extract_text_file(self, path: Path) -> str:
        return path.read_text(encoding=self.encoding)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize line endings while preserving the document's wording."""
        return text.replace("\r\n", "\n").replace("\r", "\n").strip()
