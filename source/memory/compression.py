"""Semantic context compression via sliding-window LLM summarization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from observability.app_logging import get_logger
from utilities.ollama_client import OllamaError, OllamaHTTPClient


logger = get_logger("memory.compression")

SUMMARY_PREFIX = "[Previous Conversation Summary:"
SUMMARY_SUFFIX = "]"
DEFAULT_AUTO_COMPRESSION_THRESHOLD = 0.85

# The summary replaces the older messages: a fact it leaves out is lost for the rest of
# the conversation, so the prompt lists the facts a teacher expects the assistant to remember.
SUMMARIZATION_SYSTEM_PROMPT = (
    "You summarize the beginning of a conversation between a teacher (User) and an assistant. "
    "Your summary replaces those messages: anything you leave out is forgotten.\n"
    "Keep every fact the teacher stated, with the exact names, numbers and dates, including what "
    "the teacher said when introducing themselves. Use these lines, and drop a line with nothing to keep:\n"
    "- Teacher: name, subject, class or grade level, school.\n"
    "- Pupils and classes: names, numbers (pupils, marks), needs.\n"
    "- Dates and deadlines: exams, lessons, meetings.\n"
    "- Decisions and preferences.\n"
    "- Tasks still open or promised for later.\n"
    "- Produced by the assistant: one line naming it (lesson plan, quiz, document), without copying it.\n"
    "Keep the facts of any 'Previous Summary' line as well. "
    "Write in the language of the conversation, translating the line names. "
    "Output only the summary, without adding a greeting or a comment of your own."
)
# Small local models follow the instruction closest to the end: it is repeated after the dialogue.
SUMMARIZATION_REQUEST = (
    "Conversation to summarize:\n<<<\n{dialogue}\n>>>\n"
    "Write the summary now. Keep the teacher's name, class, numbers, dates, decisions and open tasks."
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

    split_index = len(dialogue_pool) - keep_count
    # A tool result must follow the assistant message that requested it: never start the
    # kept turns in the middle of a tool exchange, keep the whole exchange instead.
    while split_index > 0 and dialogue_pool[split_index].get("role") == "tool":
        split_index -= 1
    if split_index == 0:
        return None

    older_turns = dialogue_pool[:split_index]
    recent_turns = dialogue_pool[split_index:]
    return base_system, older_turns, recent_turns


def generate_summary(
    dialogue_text: str,
    chat_client: OllamaHTTPClient,
) -> str | None:
    """Call Ollama to generate a factual summary of past dialogue."""
    prompt_messages = [
        {"role": "system", "content": SUMMARIZATION_SYSTEM_PROMPT},
        {"role": "user", "content": SUMMARIZATION_REQUEST.format(dialogue=dialogue_text)},
    ]
    try:
        data = chat_client.complete_message(prompt_messages).json()
    except (OllamaError, ValueError) as error:  # ValueError: the reply is not JSON
        logger.warning("compression.summary_failed error_type=%s", type(error).__name__)
        return None
    message = data.get("message") if isinstance(data, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    summary = content.strip() if isinstance(content, str) else ""
    if not summary:
        logger.warning("compression.summary_empty")
    return summary or None


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
