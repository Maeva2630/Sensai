"""Shared CLI state and types for feature command handlers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from config import FeatureConfig
from rag.context import RAGIndexer
from rag.storage import MetadataFilter


CommandHandler = Callable[[str, "CliSession", "CliCommandContext"], None]


@dataclass(frozen=True)
class CommandDefinition:
    """A documented CLI command registered by one SensAI feature."""

    name: str
    usage: str
    description: str
    feature: str
    handler: CommandHandler

    def matches(self, user_input: str) -> bool:
        """Return whether input starts with this command at a command boundary."""
        if not user_input.startswith(self.name):
            return False
        suffix = user_input[len(self.name) :]
        return not suffix or suffix[0].isspace()


@dataclass
class CliSession:
    """Mutable state that belongs to one interactive CLI session."""

    rag_enabled: bool = False
    metadata_filter: MetadataFilter | None = None
    show_sources: bool = True


@dataclass(frozen=True)
class CliCommandContext:
    """Services, feature settings, and command registry available to handlers."""

    feature_config: FeatureConfig
    rag_indexer: RAGIndexer
    command_definitions: tuple[CommandDefinition, ...] = ()
