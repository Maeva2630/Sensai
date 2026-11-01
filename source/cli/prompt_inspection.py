"""Formatting of the reconstructed model request for explicit developer inspection."""

from __future__ import annotations

import json
from typing import Any

from memory.compression import is_summary_message
from tools.safety import EXCERPTS_START


def format_model_request(
    messages: list[dict[str, Any]],
    rag_context: str | None,
    *,
    include_system: bool = False,
    tools: list[str] | None = None,
    options: dict[str, Any] | None = None,
) -> str:
    """Return a readable, role-labelled representation of one Ollama request.

    System messages can contain profile information, so their content is hidden
    unless a developer explicitly requests it. The generation settings and the
    tools offered are listed too: they are part of what the model receives.
    """
    if not messages:
        return "No model request has been sent in this session yet."

    sections = ["Reconstructed model request:"]
    if options:
        settings = ", ".join(f"{name}={value}" for name, value in options.items())
        sections.append(f"Settings: {settings}")
    if tools is not None:
        sections.append(f"Tools offered ({len(tools)}): {', '.join(tools) if tools else 'none'}")
    for index, message in enumerate(messages, start=1):
        label = _message_label(message, rag_context)
        content = message.get("content", "")
        sections.append(f"\n--- {index}. {label} ---")
        if label == "System message" and not include_system:
            sections.append("<hidden; use /debug prompt --include-system to display it>")
        else:
            sections.append(content if isinstance(content, str) and content else "<empty>")
        if message.get("tool_calls"):
            sections.append("Tool calls:")
            sections.append(json.dumps(message["tool_calls"], ensure_ascii=False, indent=2))
    return "\n".join(sections)


def _message_label(message: dict[str, Any], rag_context: str | None) -> str:
    content = message.get("content")
    if message.get("role") == "system":
        if rag_context is not None and content == rag_context:
            return "RAG rules (SensAI's own instructions)"
        if is_summary_message(message):
            return "Conversation summary"
        return "System message"
    if message.get("role") == "user":
        if isinstance(content, str) and content.startswith(EXCERPTS_START):
            return "User message with document excerpts (framed as data)"
        return "User message"
    if message.get("role") == "assistant":
        return "Assistant history"
    if message.get("role") == "tool":
        return "Tool message"
    return f"Unknown message role ({message.get('role', 'missing')})"
