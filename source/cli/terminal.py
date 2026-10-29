"""Small terminal presentation helpers with safe non-interactive fallbacks."""

from __future__ import annotations

import os
import sys
from typing import TextIO


_RESET = "\033[0m"
_BOLD_GREEN = "\033[1;32m"
_BOLD_RED = "\033[1;31m"
_READLINE_START_IGNORE = "\001"
_READLINE_END_IGNORE = "\002"


def configure_line_editing() -> bool:
    """Enable local line editing and history when the platform provides readline."""
    try:
        import readline
    except ImportError:
        return False
    readline.set_history_length(100)
    return True


def read_user_input(line_editing_enabled: bool) -> str:
    """Read one user entry and retain it in the in-memory terminal history."""
    user_input = input(user_prompt(line_editing_enabled))
    if line_editing_enabled and user_input.strip():
        import readline

        readline.add_history(user_input)
    return user_input


def ask_confirmation(question: str) -> bool:
    """Ask a yes/no question on the terminal; anything but yes (y, yes, o, oui) means no."""
    return input(f"{question} [y/N] ").strip().lower() in {"y", "yes", "o", "oui"}


def user_prompt(line_editing_enabled: bool, stream: TextIO | None = None) -> str:
    """Return the green bold user prompt, preserving readline cursor positioning."""
    return _styled_label("You:", _BOLD_GREEN, stream, line_editing_enabled)


def assistant_prefix(stream: TextIO | None = None) -> str:
    """Return the red bold assistant label used for streamed responses."""
    return _styled_label("Assistant:", _BOLD_RED, stream, False)


def _styled_label(
    label: str,
    color: str,
    stream: TextIO | None,
    readline_enabled: bool,
) -> str:
    output = stream if stream is not None else sys.stdout
    if not _supports_color(output):
        return f"{label} "
    if readline_enabled:
        return (
            f"{_READLINE_START_IGNORE}{color}{_READLINE_END_IGNORE}{label}"
            f"{_READLINE_START_IGNORE}{_RESET}{_READLINE_END_IGNORE} "
        )
    return f"{color}{label}{_RESET} "


def _supports_color(stream: TextIO) -> bool:
    return bool(getattr(stream, "isatty", lambda: False)() and not os.environ.get("NO_COLOR"))
