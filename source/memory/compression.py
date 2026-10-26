"""Semantic context compression via sliding-window LLM summarization."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from observability.app_logging import get_logger
from utilities.ollama_client import OllamaError, OllamaHTTPClient


logger = get_logger("memory.compression")

SUMMARY_PREFIX = "[Previous Conversation Summary:"
SUMMARY_SUFFIX = "]"
DEFAULT_AUTO_COMPRESSION_THRESHOLD = 0.85

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are an expert conversation summarizer. "
    "Your role is to produce a concise, factual summary of the following dialogue history between a user and an assistant. "
    "Focus on key facts, user preferences, decisions, teaching context, rules, and outcomes. "
    "Do NOT include conversational filler, meta-announcements, or greeting phrases. "
    "Output ONLY the factual summary text."
)


@dataclass(frozen=True)
class CompressionResult:
    """Outcome of a conversation context compression attempt."""

    compressed: bool
    messages_before: int
    messages_after: int
    summary: str = ""
    reason: str = ""


def is_summary_message(message: dict[str, Any]) -> bool:
    """Return whether a message is an existing compressed conversation summary."""
    if message.get("role") != "system":
        return False
    content = message.get("content", "")
    return isinstance(content, str) and content.startswith(SUMMARY_PREFIX)


def format_dialogue_for_summary(messages: list[dict[str, Any]]) -> str:
    """Format dialogue messages into clean text lines for the summarization prompt."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue
        if is_summary_message(msg):
            raw = content[len(SUMMARY_PREFIX) :].rstrip(SUMMARY_SUFFIX).strip()
            lines.append(f"Previous Summary: {raw}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    return "\n".join(lines)


def split_history_for_compression(
    messages: list[dict[str, Any]],
    keep_recent_turns: int = 2,
    min_turns_to_compress: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Split history into (base_system, older_turns, recent_turns).

    Returns None if history does not contain enough dialogue messages.
    """
    if not messages:
        return None

    has_base_system = messages[0].get("role") == "system" and not is_summary_message(messages[0])
    base_system = [messages[0]] if has_base_system else []
    dialogue_pool = messages[1:] if has_base_system else list(messages)

    keep_count = max(2, keep_recent_turns * 2)
    min_dialogue_needed = max(min_turns_to_compress * 2, keep_count + 2)

    if len(dialogue_pool) < min_dialogue_needed:
        return None

    older_turns = dialogue_pool[:-keep_count]
    recent_turns = dialogue_pool[-keep_count:]
    return base_system, older_turns, recent_turns


def generate_summary(
    dialogue_text: str,
    chat_client: OllamaHTTPClient,
) -> str | None:
    """Call Ollama to generate a factual summary of past dialogue."""
    prompt_messages = [
        {"role": "system", "content": SUMMARIZATION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Dialogue history to condense:\n\n{dialogue_text}"},
    ]
    try:
        response = chat_client.complete_message(prompt_messages)
        data = response.json() if hasattr(response, "json") else json.loads(response.text)
        summary = (
            data.get("message", {}).get("content", "").strip()
            if isinstance(data, dict)
            else ""
        )
        return summary if summary else None
    except (OllamaError, Exception) as error:
        logger.warning("compression.summary_failed error_type=%s", type(error).__name__)
        return None


def create_summary_message(summary_text: str) -> dict[str, Any]:
    """Wrap summary text into a system message dictionary."""
    return {
        "role": "system",
        "content": f"{SUMMARY_PREFIX} {summary_text}{SUMMARY_SUFFIX}",
    }


def assemble_compressed_history(
    base_system: list[dict[str, Any]],
    summary_text: str,
    recent_turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assemble the final compressed messages list."""
    return [*base_system, create_summary_message(summary_text), *recent_turns]


def compress_history(
    messages: list[dict[str, Any]],
    chat_client: OllamaHTTPClient,
    keep_recent_turns: int = 2,
    min_turns_to_compress: int = 3,
) -> tuple[list[dict[str, Any]], CompressionResult]:
    """Compress older conversation turns into a compact summary system message.

    Preserves:
    - Base system prompt (messages[0], e.g. Persona + Profile).
    - The most recent `keep_recent_turns` turns for dialogue continuity.

    Returns the updated messages list and a CompressionResult.
    """
    if not messages:
        return messages, CompressionResult(
            compressed=False,
            messages_before=0,
            messages_after=0,
            reason="No messages in history.",
        )

    split = split_history_for_compression(messages, keep_recent_turns, min_turns_to_compress)
    if split is None:
        keep_count = max(2, keep_recent_turns * 2)
        min_dialogue_needed = max(min_turns_to_compress * 2, keep_count + 2)
        return messages, CompressionResult(
            compressed=False,
            messages_before=len(messages),
            messages_after=len(messages),
            reason=f"Not enough dialogue turns to compress (need at least {min_dialogue_needed} messages).",
        )

    base_system, older_turns, recent_turns = split
    dialogue_text = format_dialogue_for_summary(older_turns)
    if not dialogue_text.strip():
        return messages, CompressionResult(
            compressed=False,
            messages_before=len(messages),
            messages_after=len(messages),
            reason="No text found in older dialogue turns.",
        )

    summary_text = generate_summary(dialogue_text, chat_client)
    if not summary_text:
        return messages, CompressionResult(
            compressed=False,
            messages_before=len(messages),
            messages_after=len(messages),
            reason="Summarization call failed or returned an empty summary.",
        )

    new_messages = assemble_compressed_history(base_system, summary_text, recent_turns)
    logger.info(
        "compression.successful messages_before=%d messages_after=%d",
        len(messages),
        len(new_messages),
    )

    return new_messages, CompressionResult(
        compressed=True,
        messages_before=len(messages),
        messages_after=len(new_messages),
        summary=summary_text,
    )
