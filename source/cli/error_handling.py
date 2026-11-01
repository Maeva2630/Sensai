"""User-safe error reporting at the interactive CLI boundary."""

from __future__ import annotations

import logging

from config import ConfigurationError
from memory.session_manager import InvalidSessionIdError
from rag.embeddings import EmbeddingError
from rag.ingestion import DocumentIngestionError
from rag.storage import VectorStoreError
from utilities.ollama_client import (
    OllamaConnectionError,
    OllamaError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)

ErrorRule = tuple[type[Exception], str, str]

_ERROR_RULES: tuple[ErrorRule, ...] = (
    (
        OllamaConnectionError,
        "ollama_connection",
        "Connection error: unable to reach Ollama. Make sure Ollama is running, then try again.",
    ),
    (
        OllamaModelNotFoundError,
        "ollama_model",
        "Model error: the configured Ollama model was not found. Check config.json and install the model.",
    ),
    (OllamaError, "ollama_response", "Ollama could not complete this request. Please try again."),
    (
        DocumentIngestionError,
        "document_ingestion",
        "Document error: the file could not be ingested. Check that it exists, is supported, and readable.",
    ),
    (EmbeddingError, "rag_embedding", "RAG error: unable to generate document embeddings. Please try again."),
    (VectorStoreError, "rag_storage", "RAG error: the local knowledge base is unavailable. Please try again."),
    (ConfigurationError, "configuration", "Configuration error: check config.json and restart SensAI."),
    (
        InvalidSessionIdError,
        "session_id",
        "Session error: a session id may only contain letters, digits, '-' and '_' (64 characters at most).",
    ),
    (FileNotFoundError, "file_not_found", "File error: the requested file or session could not be found."),
    (OSError, "file_system", "File error: SensAI could not access a required local file."),
)
_UNEXPECTED_ERROR = (
    "unexpected",
    "An unexpected error occurred and was handled safely. You can continue using SensAI.",
)


def format_cli_error(error: Exception, logger: logging.Logger) -> str:
    """Log diagnostic details and return a concise, safe message for the user."""
    category, message = _user_message(error)
    logger.exception("cli.error category=%s error_type=%s", category, type(error).__name__)
    return message


def format_fatal_error(error: Exception, logger: logging.Logger) -> str:
    """Message for an error that stops SensAI: it must not promise that the user can continue."""
    category, message = _user_message(error)
    logger.exception("cli.fatal_error category=%s error_type=%s", category, type(error).__name__)
    if category == _UNEXPECTED_ERROR[0]:
        return "An unexpected error stopped SensAI."
    return message


def report_cli_error(error: Exception, logger: logging.Logger) -> None:
    """Log diagnostic details and display a concise, safe terminal message."""
    print(f"\n{format_cli_error(error, logger)}")


def _user_message(error: Exception) -> tuple[str, str]:
    """Map known failures to actionable messages without exposing internals."""
    specific = _specific_message(error)
    if specific is not None:
        return specific
    for error_type, category, message in _ERROR_RULES:
        if isinstance(error, error_type):
            return category, message
    return _UNEXPECTED_ERROR


def _specific_message(error: Exception) -> tuple[str, str] | None:
    """Name the exact cause when SensAI knows it, including behind a wrapping error.

    A missing embedding model surfaces as a RAG error; telling the teacher to "try
    again" would not help, while the model name and the install command do.
    """
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, OllamaModelNotFoundError) and current.model:
            return (
                "ollama_model",
                f"Model error: the Ollama model '{current.model}' is not installed. "
                f"Install it with: ollama pull {current.model} (or change the model in config.json).",
            )
        if isinstance(current, OllamaTimeoutError):
            return (
                "ollama_timeout",
                f"Ollama is running but did not answer within {current.timeout:g} s. The model may still "
                "be loading or the request is long: try again, or raise model.request_timeout_seconds "
                "in config.json.",
            )
        if isinstance(current, OllamaConnectionError) and current is not error:
            return "ollama_connection", _ERROR_RULES[0][2]
        current = current.__cause__
    if isinstance(error, DocumentIngestionError):
        # Ingestion messages are written by SensAI for the user (unsupported type,
        # missing PDF library, unsafe archive...), so the precise reason can be shown.
        return "document_ingestion", f"Document error: {error}"
    return None
