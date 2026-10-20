"""Embedding generation for RAG document chunks and user queries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence, Tuple

from rag.chunking import DocumentChunk
from utilities.ollama_client import (
    OllamaConnectionError,
    OllamaError,
    OllamaHTTPClient,
    OllamaModelNotFoundError,
)


DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


class EmbeddingError(Exception):
    """Base exception for embedding generation failures."""


class EmbeddingInputError(EmbeddingError):
    """Raised when the text submitted for embedding is empty."""


class EmbeddingConnectionError(EmbeddingError):
    """Raised when the local Ollama server cannot be reached."""


class EmbeddingModelError(EmbeddingError):
    """Raised when the configured Ollama embedding model is unavailable."""


class EmbeddingResponseError(EmbeddingError):
    """Raised when Ollama returns an invalid embedding payload."""


class EmbeddingDimensionError(EmbeddingError):
    """Raised when vectors do not share the expected number of dimensions."""


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for the model and optional vector dimension validation."""

    model: str = DEFAULT_EMBEDDING_MODEL
    expected_dimension: int | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("An embedding model name is required.")
        if self.expected_dimension is not None and self.expected_dimension <= 0:
            raise ValueError("expected_dimension must be greater than zero when provided.")


@dataclass(frozen=True)
class EmbeddedChunk:
    """A document chunk paired with the vector that represents its meaning."""

    chunk: DocumentChunk
    vector: Tuple[float, ...]
    model: str


@dataclass(frozen=True)
class QueryEmbedding:
    """A user query paired with the vector used for semantic search."""

    query: str
    vector: Tuple[float, ...]
    model: str


class EmbeddingGenerator:
    """Create validated embeddings without performing any persistent writes."""

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        client: OllamaHTTPClient | None = None,
    ) -> None:
        self.config = config or EmbeddingConfig()
        self.client = client or OllamaHTTPClient(model=self.config.model)
        self._dimension = self.config.expected_dimension

    def embed_chunks(self, chunks: Sequence[DocumentChunk]) -> list[EmbeddedChunk]:
        """Generate one validated vector for every non-empty document chunk."""
        embedded_chunks: list[EmbeddedChunk] = []
        for chunk in chunks:
            if not chunk.text.strip():
                continue
            vector = self._embed_text(chunk.text)
            embedded_chunks.append(EmbeddedChunk(chunk=chunk, vector=vector, model=self.config.model))
        return embedded_chunks

    def embed_query(self, query: str) -> QueryEmbedding:
        """Generate a validated vector for one user query."""
        if not query.strip():
            raise EmbeddingInputError("Cannot generate an embedding for an empty query.")
        return QueryEmbedding(query=query, vector=self._embed_text(query), model=self.config.model)

    def _embed_text(self, text: str) -> Tuple[float, ...]:
        try:
            response = self.client.get_embedding(text)
        except OllamaConnectionError as error:
            raise EmbeddingConnectionError(
                "Unable to generate embeddings because Ollama is unavailable. "
                "Start Ollama with 'ollama serve' and try again."
            ) from error
        except OllamaModelNotFoundError as error:
            raise EmbeddingModelError(
                f"Embedding model '{self.config.model}' is unavailable. "
                f"Download it with 'ollama pull {self.config.model}'."
            ) from error
        except OllamaError as error:
            raise EmbeddingError(f"Ollama could not generate an embedding: {error}") from error

        vector = self._extract_vector(response)
        self._validate_dimension(vector)
        return vector

    @staticmethod
    def _extract_vector(response: Any) -> Tuple[float, ...]:
        """Accept current Ollama responses and the legacy single-vector shape."""
        if not isinstance(response, dict):
            raise EmbeddingResponseError("Ollama returned an invalid embedding response.")

        payload = response.get("embeddings", response.get("embedding"))
        if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], list):
            payload = payload[0]

        if not isinstance(payload, list) or not payload:
            raise EmbeddingResponseError("Ollama returned no embedding vector.")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in payload):
            raise EmbeddingResponseError("Ollama returned an embedding containing non-numeric values.")

        vector = tuple(float(value) for value in payload)
        if not all(math.isfinite(value) for value in vector):
            raise EmbeddingResponseError("Ollama returned an embedding containing non-finite values.")
        return vector

    def _validate_dimension(self, vector: Tuple[float, ...]) -> None:
        if self._dimension is None:
            self._dimension = len(vector)
            return
        if len(vector) != self._dimension:
            raise EmbeddingDimensionError(
                f"Expected embeddings with {self._dimension} dimensions, received {len(vector)}."
            )
