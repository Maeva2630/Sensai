from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .chunking import DocumentChunker
from .embeddings import EmbeddingGenerator
from .ingestion import DocumentIngestor, IngestedDocument
from .storage import DEFAULT_TOP_K, LocalVectorStore, SearchResult


DEFAULT_CONTEXT_CHARACTER_BUDGET = 4000


@dataclass(frozen=True)
class RAGContextConfig:
    """Controls how much retrieved evidence is passed to the chat model."""

    top_k: int = DEFAULT_TOP_K
    max_context_characters: int = DEFAULT_CONTEXT_CHARACTER_BUDGET

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if self.max_context_characters < 300:
            raise ValueError("max_context_characters must be at least 300.")


@dataclass(frozen=True)
class IndexingResult:
    """The outcome of indexing one source document."""

    document: IngestedDocument
    chunk_count: int
    inserted_chunk_count: int


@dataclass(frozen=True)
class PreparedContext:
    """The temporary system message and its retrieved source chunks."""

    system_message: str
    results: Sequence[SearchResult]


class RAGIndexer:
    """Run ingestion, chunking, embedding, and persistent storage in order."""

    def __init__(
        self,
        ingestor: DocumentIngestor | None = None,
        chunker: DocumentChunker | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        vector_store: LocalVectorStore | None = None,
    ) -> None:
        self.ingestor = ingestor or DocumentIngestor()
        self.chunker = chunker or DocumentChunker()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.vector_store = vector_store or LocalVectorStore()

    def index_document(self, document_path: str | Path) -> IndexingResult:
        document = self.ingestor.ingest(document_path)
        chunks = self.chunker.chunk(document)
        embedded_chunks = self.embedding_generator.embed_chunks(chunks)
        inserted_chunk_count = self.vector_store.store_chunks(embedded_chunks)
        return IndexingResult(document, len(chunks), inserted_chunk_count)


class RAGContextProvider:
    """Retrieve evidence and turn it into a safe, bounded system message."""

    _INSTRUCTIONS = (
        "You are a document-grounded assistant. Answer only with information supported by the "
        "retrieved excerpts below. Cite the source filename when you use an excerpt. If the "
        "excerpts do not contain the answer, say exactly: 'I cannot find this information in the "
        "indexed documents.' Do not use outside knowledge.\n\nRetrieved context:\n"
    )

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_store: LocalVectorStore,
        config: RAGContextConfig | None = None,
    ) -> None:
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.config = config or RAGContextConfig()

    def prepare(self, query: str) -> PreparedContext:
        # Avoid an unnecessary Ollama embedding call when there is no knowledge
        # base. The model still receives the missing-context instruction.
        if self.vector_store.count_chunks() == 0:
            return PreparedContext(self._build_message([]), [])

        query_embedding = self.embedding_generator.embed_query(query)
        results = self.vector_store.search(query_embedding, top_k=self.config.top_k)
        return PreparedContext(self._build_message(results), results)

    def _build_message(self, results: Sequence[SearchResult]) -> str:
        parts = [self._INSTRUCTIONS]
        if not results:
            return self._fit_to_budget("".join(parts) + "No relevant indexed excerpts were found.")

        for result in results:
            source_header = (
                f"[Source: {result.source_filename} | Chunk: {result.chunk_id} | "
                f"Score: {result.similarity_score:.3f}]\n"
            )
            remaining = self.config.max_context_characters - len("".join(parts))
            if remaining <= len(source_header):
                break
            excerpt_budget = remaining - len(source_header) - 2
            excerpt = result.text[:excerpt_budget].rstrip()
            if not excerpt:
                break
            parts.append(f"{source_header}{excerpt}\n\n")

        return self._fit_to_budget("".join(parts))

    def _fit_to_budget(self, message: str) -> str:
        return message[: self.config.max_context_characters]
