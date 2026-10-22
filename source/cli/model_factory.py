"""Factories that assemble model-backed CLI services from application configuration."""

from dataclasses import dataclass
from collections.abc import Callable

from config import ModelConfig, RagConfig
from rag.context import RAGContextConfig, RAGContextProvider, RAGIndexer
from rag.embeddings import EmbeddingConfig, EmbeddingGenerator
from rag.ingestion import DocumentIngestor
from utilities.ollama_client import OllamaHTTPClient


@dataclass(frozen=True)
class ModelRuntime:
    """Model-backed services required by one interactive CLI session."""

    chat_client: OllamaHTTPClient
    rag_indexer: RAGIndexer
    rag_context_provider: RAGContextProvider


def create_rag_context_config(config: RagConfig) -> RAGContextConfig:
    """Translate application settings into RAG context-provider settings."""
    return RAGContextConfig(
        top_k=config.top_k,
        max_context_characters=config.max_context_characters,
        relevance_threshold=config.relevance_threshold,
    )


def create_model_runtime(
    model_config: ModelConfig,
    rag_config: RagConfig,
    *,
    rag_indexer: RAGIndexer | None = None,
    rag_context_provider: RAGContextProvider | None = None,
    client_factory: Callable[..., OllamaHTTPClient] = OllamaHTTPClient,
) -> ModelRuntime:
    """Build chat and RAG services, preserving injected services for tests."""
    chat_client = client_factory(
        base_url=model_config.base_url,
        model=model_config.chat_model,
        options={"temperature": model_config.temperature, "num_ctx": model_config.context_window},
    )
    if rag_indexer is None:
        embedding_client = client_factory(
            base_url=model_config.base_url,
            model=model_config.embedding_model,
        )
        rag_indexer = RAGIndexer(
            ingestor=DocumentIngestor(),
            embedding_generator=EmbeddingGenerator(
                EmbeddingConfig(model=model_config.embedding_model),
                client=embedding_client,
            ),
        )
    if rag_context_provider is None:
        rag_context_provider = RAGContextProvider(
            embedding_generator=rag_indexer.embedding_generator,
            vector_store=rag_indexer.vector_store,
            config=create_rag_context_config(rag_config),
        )
    return ModelRuntime(chat_client, rag_indexer, rag_context_provider)
