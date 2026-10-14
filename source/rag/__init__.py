"""Components used by the Retrieval-Augmented Generation (RAG) pipeline."""

from .ingestion import (
    DocumentDecodingError,
    DocumentIngestionError,
    DocumentIngestor,
    DocumentNotFoundError,
    EmptyDocumentError,
    IngestedDocument,
    UnsupportedDocumentTypeError,
)
from .chunking import ChunkingConfig, DocumentChunk, DocumentChunker
from .embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddedChunk,
    EmbeddingConfig,
    EmbeddingConnectionError,
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingGenerator,
    EmbeddingInputError,
    EmbeddingModelError,
    EmbeddingResponseError,
    QueryEmbedding,
)
from .storage import DEFAULT_TOP_K, InvalidVectorError, LocalVectorStore, SearchResult, VectorStoreError
from .context import (
    DEFAULT_CONTEXT_CHARACTER_BUDGET,
    IndexingResult,
    PreparedContext,
    RAGContextConfig,
    RAGContextProvider,
    RAGIndexer,
)

__all__ = [
    "DocumentDecodingError",
    "DocumentIngestionError",
    "DocumentIngestor",
    "DocumentNotFoundError",
    "EmptyDocumentError",
    "IngestedDocument",
    "UnsupportedDocumentTypeError",
    "ChunkingConfig",
    "DocumentChunk",
    "DocumentChunker",
    "DEFAULT_EMBEDDING_MODEL",
    "EmbeddedChunk",
    "EmbeddingConfig",
    "EmbeddingConnectionError",
    "EmbeddingDimensionError",
    "EmbeddingError",
    "EmbeddingGenerator",
    "EmbeddingInputError",
    "EmbeddingModelError",
    "EmbeddingResponseError",
    "QueryEmbedding",
    "DEFAULT_TOP_K",
    "InvalidVectorError",
    "LocalVectorStore",
    "SearchResult",
    "VectorStoreError",
    "DEFAULT_CONTEXT_CHARACTER_BUDGET",
    "IndexingResult",
    "PreparedContext",
    "RAGContextConfig",
    "RAGContextProvider",
    "RAGIndexer",
]
