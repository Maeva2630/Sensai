"""Safe, file-only logging configuration for SensAI."""

from __future__ import annotations

import logging
from pathlib import Path

from config import LoggingConfig


LOGGER_NAME = "sensai"
_COMPONENT_LOGGERS = (LOGGER_NAME, "rag.context")
_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(config: LoggingConfig) -> logging.Logger:
    """Configure SensAI logs without adding output to the interactive terminal."""
    log_path = Path(config.file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_FORMAT)
    for logger_name in _COMPONENT_LOGGERS:
        logger = logging.getLogger(logger_name)
        logger.setLevel(config.level)
        logger.propagate = False
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(config.level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logging.getLogger(LOGGER_NAME)


def get_logger(component: str) -> logging.Logger:
    """Return a SensAI component logger configured by the application entry point."""
    root_logger = logging.getLogger(LOGGER_NAME)
    if not root_logger.handlers:
        root_logger.addHandler(logging.NullHandler())
        root_logger.propagate = False
    return logging.getLogger(f"{LOGGER_NAME}.{component}")
