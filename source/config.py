"""Validated application configuration loaded from the repository config file."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from personas.teacher_assistant import TEACHER_ASSISTANT_BASE_PROMPT

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"


class ConfigurationError(ValueError):
    """Raised when the application configuration is missing or invalid."""


@dataclass(frozen=True)
class ModelConfig:
    chat_model: str = "qwen2.5:3b"
    fallback_chat_models: tuple[str, ...] = ("llama3.2:latest",)
    embedding_model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    context_window: int = 4096


@dataclass(frozen=True)
class FeatureConfig:
    rag_enabled: bool = True
    tools_enabled: bool = True
    memory_enabled: bool = True
    source_display_enabled: bool = True
    # Web search and MCP servers marked `network` stay off unless this is true.
    network_enabled: bool = False


@dataclass(frozen=True)
class RagConfig:
    top_k: int = 3
    max_context_characters: int = 4000
    relevance_threshold: float = 0.35


@dataclass(frozen=True)
class LoggingConfig:
    """Controls file-only diagnostic logs without exposing them in the CLI."""

    level: str = "INFO"
    file_path: str = "logs/sensai.log"


@dataclass(frozen=True)
class MonitoringConfig:
    """Controls local, privacy-preserving interaction metrics."""

    enabled: bool = False
    database_path: str = "data/sensai_monitoring.sqlite3"
    retention_days: int = 30
    prompt_version: str = "default-v1"


@dataclass(frozen=True)
class PersonaConfig:
    """One configurable assistant persona and its system-prompt instructions."""

    identifier: str
    display_name: str
    role: str
    tone: str
    system_prompt: str
    response_scope: str

    @property
    def system_instructions(self) -> str:
        """Combine the persona metadata and authored prompt for the chat model."""
        return f"""PERSONA:
- Name: {self.display_name}
- Role: {self.role}
- Tone: {self.tone}
- Response scope: {self.response_scope}

PERSONA INSTRUCTIONS:
{self.system_prompt}"""


DEFAULT_TEACHER_ASSISTANT_PERSONA = PersonaConfig(
    identifier="teacher-assistant",
    display_name="Teacher assistant",
    role="Educational assistant for teachers and educators",
    tone="Clear, practical, supportive, and pedagogical",
    system_prompt=TEACHER_ASSISTANT_BASE_PROMPT,
    response_scope="Support educators with lesson preparation, teaching resources, and assessment.",
)


@dataclass(frozen=True)
class PersonasConfig:
    """Configured personas and the one selected for new conversations."""

    active_persona: str = DEFAULT_TEACHER_ASSISTANT_PERSONA.identifier
    profiles: tuple[PersonaConfig, ...] = (DEFAULT_TEACHER_ASSISTANT_PERSONA,)

    @property
    def active(self) -> PersonaConfig:
        """Return the validated persona selected for the current application."""
        return next(profile for profile in self.profiles if profile.identifier == self.active_persona)

    def find(self, identifier: str) -> PersonaConfig | None:
        """Return a configured persona by identifier, if it exists."""
        return next((profile for profile in self.profiles if profile.identifier == identifier), None)


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig = ModelConfig()
    features: FeatureConfig = FeatureConfig()
    rag: RagConfig = RagConfig()
    logging: LoggingConfig = LoggingConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    personas: PersonasConfig = PersonasConfig()


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load and validate the JSON configuration without silently ignoring errors."""
    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigurationError(f"Configuration file not found: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ConfigurationError(f"Invalid JSON in configuration file '{config_path}': {error.msg}") from error

    if not isinstance(data, Mapping):
        raise ConfigurationError("The configuration root must be a JSON object.")
    model = _read_model(_section(data, "model"))
    features = _read_features(_section(data, "features"))
    rag = _read_rag(_section(data, "rag"))
    logging = _read_logging(_optional_section(data, "logging"))
    monitoring = _read_monitoring(_optional_section(data, "monitoring"))
    personas = _read_personas(_optional_section(data, "personas"), config_path.parent)
    return AppConfig(
        model=model,
        features=features,
        rag=rag,
        logging=logging,
        monitoring=monitoring,
        personas=personas,
    )


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = data.get(name)
    if not isinstance(section, Mapping):
        raise ConfigurationError(f"Configuration section '{name}' must be an object.")
    return section


def _optional_section(data: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    section = data.get(name)
    if section is None:
        return None
    if not isinstance(section, Mapping):
        raise ConfigurationError(f"Configuration section '{name}' must be an object.")
    return section


def _string(section: Mapping[str, Any], name: str) -> str:
    value = section.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"'{name}' must be a non-empty string.")
    return value.strip()


def _string_list(
    section: Mapping[str, Any],
    name: str,
    default: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Read an optional ordered list of distinct, non-empty model names."""
    value = section.get(name, list(default))
    if not isinstance(value, list):
        raise ConfigurationError(f"'{name}' must be an array of non-empty strings.")
    values = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    if len(values) != len(value):
        raise ConfigurationError(f"'{name}' must be an array of non-empty strings.")
    if len(set(values)) != len(values):
        raise ConfigurationError(f"'{name}' must not contain duplicate model names.")
    return values


def _boolean(section: Mapping[str, Any], name: str) -> bool:
    value = section.get(name)
    if not isinstance(value, bool):
        raise ConfigurationError(f"'{name}' must be true or false.")
    return value


def _number(section: Mapping[str, Any], name: str) -> float:
    value = section.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigurationError(f"'{name}' must be a finite number.")
    return float(value)


def _integer(section: Mapping[str, Any], name: str, minimum: int) -> int:
    value = section.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"'{name}' must be an integer greater than or equal to {minimum}.")
    return value


def _read_model(section: Mapping[str, Any]) -> ModelConfig:
    temperature = _number(section, "temperature")
    if not 0.0 <= temperature <= 2.0:
        raise ConfigurationError("'temperature' must be between 0 and 2.")
    chat_model = _string(section, "chat_model")
    fallback_chat_models = _string_list(
        section,
        "fallback_chat_models",
        ModelConfig().fallback_chat_models,
    )
    if chat_model in fallback_chat_models:
        raise ConfigurationError("'fallback_chat_models' must not contain 'chat_model'.")
    return ModelConfig(
        chat_model=chat_model,
        fallback_chat_models=fallback_chat_models,
        embedding_model=_string(section, "embedding_model"),
        base_url=_string(section, "base_url"),
        temperature=temperature,
        context_window=_integer(section, "context_window", 512),
    )


def _read_features(section: Mapping[str, Any]) -> FeatureConfig:
    return FeatureConfig(
        rag_enabled=_boolean(section, "rag_enabled"),
        tools_enabled=_boolean(section, "tools_enabled"),
        memory_enabled=_boolean(section, "memory_enabled"),
        source_display_enabled=_boolean(section, "source_display_enabled"),
        network_enabled=_boolean(section, "network_enabled") if "network_enabled" in section else False,
    )


def _read_rag(section: Mapping[str, Any]) -> RagConfig:
    threshold = _number(section, "relevance_threshold")
    if not -1.0 <= threshold <= 1.0:
        raise ConfigurationError("'relevance_threshold' must be between -1 and 1.")
    top_k = _integer(section, "top_k", 1)
    return RagConfig(
        top_k=top_k,
        max_context_characters=_integer(section, "max_context_characters", 300),
        relevance_threshold=threshold,
    )


def _read_logging(section: Mapping[str, Any] | None) -> LoggingConfig:
    """Read safe logging options while keeping older configuration files compatible."""
    if section is None:
        return LoggingConfig()
    level = _string(section, "level").upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigurationError("'level' must be DEBUG, INFO, WARNING, or ERROR.")
    return LoggingConfig(level=level, file_path=_string(section, "file_path"))


def _read_monitoring(section: Mapping[str, Any] | None) -> MonitoringConfig:
    """Read optional local monitoring settings without breaking older configs."""
    if section is None:
        return MonitoringConfig()
    return MonitoringConfig(
        enabled=_boolean(section, "enabled"),
        database_path=_string(section, "database_path"),
        retention_days=_integer(section, "retention_days", 1),
        prompt_version=_string(section, "prompt_version"),
    )


def _read_personas(
    section: Mapping[str, Any] | None,
    config_directory: Path,
) -> PersonasConfig:
    """Read configurable personas while keeping earlier configuration files compatible."""
    if section is None:
        return PersonasConfig()
    active_persona = _string(section, "active_persona")
    raw_profiles = section.get("profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ConfigurationError("'personas.profiles' must be a non-empty array.")

    profiles = tuple(_read_persona(profile, config_directory) for profile in raw_profiles)
    identifiers = [profile.identifier for profile in profiles]
    if len(set(identifiers)) != len(identifiers):
        raise ConfigurationError("'personas.profiles' identifiers must be unique.")
    if active_persona not in identifiers:
        raise ConfigurationError("'personas.active_persona' must identify a configured persona.")
    return PersonasConfig(active_persona=active_persona, profiles=profiles)


def _read_persona(raw_persona: Any, config_directory: Path) -> PersonaConfig:
    """Validate one persona object from the configuration file."""
    if not isinstance(raw_persona, Mapping):
        raise ConfigurationError("Each item in 'personas.profiles' must be an object.")
    inline_prompt = raw_persona.get("system_prompt")
    prompt_file = raw_persona.get("prompt_file")
    if (inline_prompt is None) == (prompt_file is None):
        raise ConfigurationError(
            "Each persona must define exactly one of 'system_prompt' or 'prompt_file'."
        )
    return PersonaConfig(
        identifier=_string(raw_persona, "identifier"),
        display_name=_string(raw_persona, "display_name"),
        role=_string(raw_persona, "role"),
        tone=_string(raw_persona, "tone"),
        system_prompt=(
            _string(raw_persona, "system_prompt")
            if inline_prompt is not None
            else _read_prompt_file(_string(raw_persona, "prompt_file"), config_directory)
        ),
        response_scope=_string(raw_persona, "response_scope"),
    )


def _read_prompt_file(prompt_file: str, config_directory: Path) -> str:
    """Load a UTF-8 prompt file relative to the configuration file."""
    path = (config_directory / prompt_file).resolve()
    try:
        path.relative_to(config_directory.resolve())
    except ValueError as error:
        raise ConfigurationError("'prompt_file' must stay inside the configuration directory.") from error
    try:
        content = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as error:
        raise ConfigurationError(f"Persona prompt file not found: {path}") from error
    except UnicodeDecodeError as error:
        raise ConfigurationError(f"Persona prompt file must be UTF-8: {path}") from error
    except OSError as error:
        raise ConfigurationError(f"Unable to read persona prompt file '{path}': {error}") from error
    if not content:
        raise ConfigurationError(f"Persona prompt file must not be empty: {path}")
    return content
