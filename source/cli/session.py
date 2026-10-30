"""Shared CLI state and types for feature command handlers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from config import FeatureConfig, PersonasConfig
from memory.artifact_manager import ArtifactManager
from memory.profile_manager import ProfileManager
from memory.session_manager import SessionManager
from observability.monitoring import DisabledMonitoringStore, MonitoringStore
from rag.context import RAGIndexer
from rag.storage import MetadataFilter
from tools.registry import ToolRegistry
from utilities.ollama_client import OllamaHTTPClient


DEFAULT_BUDGET_ALERT_THRESHOLD = 0.8
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
    exit_requested: bool = False
    last_request_messages: list[dict[str, Any]] | None = None
    last_rag_context: str | None = None
    active_persona_id: str = ""
    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    context_window: int = 4096

    @property
    def last_turn_tokens(self) -> int:
        """Return total tokens consumed in the most recent interaction."""
        return self.last_prompt_tokens + self.last_completion_tokens

    @property
    def total_session_tokens(self) -> int:
        """Return cumulative tokens consumed across the active session."""
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def context_usage_ratio(self) -> float:
        """Return the ratio of active context token usage to the context window."""
        return self.last_turn_tokens / max(1, self.context_window)

    def get_budget_warning(self, threshold: float = DEFAULT_BUDGET_ALERT_THRESHOLD) -> str | None:
        """Return a warning message when active context usage reaches or exceeds threshold."""
        if self.context_usage_ratio >= threshold:
            percentage = self.context_usage_ratio * 100
            return (
                f"Warning: Context token usage is at {percentage:.1f}% "
                f"({self.last_turn_tokens}/{self.context_window} tokens)."
            )
        return None

    def record_turn_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Update per-turn and cumulative session token counters."""
        self.last_prompt_tokens = max(0, prompt_tokens)
        self.last_completion_tokens = max(0, completion_tokens)
        self.total_prompt_tokens += self.last_prompt_tokens
        self.total_completion_tokens += self.last_completion_tokens


@dataclass(frozen=True)
class CliCommandContext:
    """Services, feature settings, and command registry available to handlers."""

    feature_config: FeatureConfig
    rag_indexer: RAGIndexer
    conversation_session: SessionManager
    command_definitions: tuple[CommandDefinition, ...] = ()
    registry: ToolRegistry | None = None
    monitoring_store: MonitoringStore | DisabledMonitoringStore | None = None
    personas: PersonasConfig = PersonasConfig()
    profile_manager: ProfileManager | None = None
    artifact_manager: ArtifactManager | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    chat_client: OllamaHTTPClient | None = None
    rag_context_provider: Any = None
    permission_checker: Any = None
    quiz_store: Any = None
