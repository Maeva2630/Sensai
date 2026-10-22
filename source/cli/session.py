"""Shared CLI state and types for feature command handlers."""

from collections.abc import Callable
from dataclasses import dataclass

from config import FeatureConfig
from rag.context import RAGIndexer
from rag.storage import MetadataFilter


@dataclass
class CliSession:
    """Mutable state that belongs to one interactive CLI session."""

    rag_enabled: bool = False
    metadata_filter: MetadataFilter | None = None
    show_sources: bool = True


@dataclass(frozen=True)
class CliCommandContext:
    """Services and feature settings available to command handlers."""

    feature_config: FeatureConfig
    rag_indexer: RAGIndexer


CommandHandler = Callable[[str, CliSession, CliCommandContext], None]
