"""Validated application configuration loaded from the repository config file."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"


class ConfigurationError(ValueError):
    """Raised when the application configuration is missing or invalid."""


@dataclass(frozen=True)
class ModelConfig:
    chat_model: str = "llama3.2"
    embedding_model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    context_window: int = 4096


@dataclass(frozen=True)
class FeatureConfig:
    rag_enabled: bool = True
    tools_enabled: bool = True
    memory_enabled: bool = False
    source_display_enabled: bool = True


@dataclass(frozen=True)
class RagConfig:
    top_k: int = 3
    max_context_characters: int = 4000
    relevance_threshold: float = 0.35


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig = ModelConfig()
    features: FeatureConfig = FeatureConfig()
    rag: RagConfig = RagConfig()


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
    return AppConfig(model=model, features=features, rag=rag)


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = data.get(name)
    if not isinstance(section, Mapping):
        raise ConfigurationError(f"Configuration section '{name}' must be an object.")
    return section


def _string(section: Mapping[str, Any], name: str) -> str:
    value = section.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"'{name}' must be a non-empty string.")
    return value.strip()


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
    return ModelConfig(
        chat_model=_string(section, "chat_model"),
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
