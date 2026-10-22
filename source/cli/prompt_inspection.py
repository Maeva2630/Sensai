"""Formatting of the reconstructed model request for explicit developer inspection."""

from __future__ import annotations

import json
from typing import Any


def format_model_request(
    messages: list[dict[str, Any]],
    rag_context: str | None,
    *,
    include_system: bool = False,
) -> str:
    """Return a readable, role-labelled representation of one Ollama request.

    System messages can contain profile information, so their content is hidden
    unless a developer explicitly requests it.
    """
    if not messages:
        return "No model request has been sent in this session yet."

    sections = ["Reconstructed model request:"]
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
    if message.get("role") == "system":
        return "RAG context" if rag_context is not None and message.get("content") == rag_context else "System message"
    if message.get("role") == "user":
        return "User message"
    if message.get("role") == "assistant":
        return "Assistant history"
    if message.get("role") == "tool":
        return "Tool message"
    return f"Unknown message role ({message.get('role', 'missing')})"
