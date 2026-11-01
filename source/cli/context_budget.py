"""Token budget checked before a request is sent to the model.

Ollama never rejects a prompt longer than `num_ctx`; it silently cuts it. Measured
with qwen2.5:3b (num_ctx 2048): when the history is too long, the oldest turns are
dropped (the system prompt is kept, but facts given early are forgotten); when one
message alone is too long, the prompt is cut token by token and the system prompt is
lost with the beginning of the message. SensAI therefore estimates the size of each
request before sending it and keeps room for the answer.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

# Measured with qwen2.5:3b: 3.2 characters per token for French course text with
# fractions, 5.2 for plain English. Starting from the French value keeps the first
# estimate on the safe side; each answer from Ollama then corrects the ratio.
DEFAULT_CHARACTERS_PER_TOKEN = 3.2
MINIMUM_CHARACTERS_PER_TOKEN = 2.5
MAXIMUM_CHARACTERS_PER_TOKEN = 6.0
# Role markers added by the chat template around every message.
MESSAGE_OVERHEAD_TOKENS = 4
# Instructions the chat template adds once when tools are offered (measured: 117
# tokens for one 200-character schema, so about 80 tokens plus the schema itself).
TOOLS_OVERHEAD_TOKENS = 80
MINIMUM_ANSWER_RESERVE_TOKENS = 256


def answer_reserve(context_window: int) -> int:
    """Tokens kept free for the model's answer: a quarter of the window, at least 256."""
    return max(MINIMUM_ANSWER_RESERVE_TOKENS, context_window // 4)


def prompt_budget(context_window: int) -> int:
    """Largest prompt that still leaves room for the answer."""
    return max(0, context_window - answer_reserve(context_window))


@dataclass
class TokenEstimator:
    """Estimate prompt tokens from characters, calibrated by Ollama's real counts."""

    characters_per_token: float = DEFAULT_CHARACTERS_PER_TOKEN

    def estimate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
        """Estimated prompt tokens for one request, tool definitions included."""
        tokens = math.ceil(_message_characters(messages) / self.characters_per_token)
        tokens += MESSAGE_OVERHEAD_TOKENS * len(messages)
        if tools:
            tokens += TOOLS_OVERHEAD_TOKENS + math.ceil(_tool_characters(tools) / self.characters_per_token)
        return tokens

    def learn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        measured_prompt_tokens: int,
    ) -> None:
        """Adjust the ratio with the prompt size Ollama reported for a request that fitted."""
        fixed = MESSAGE_OVERHEAD_TOKENS * len(messages) + (TOOLS_OVERHEAD_TOKENS if tools else 0)
        text_tokens = measured_prompt_tokens - fixed
        characters = _message_characters(messages) + (_tool_characters(tools) if tools else 0)
        if text_tokens <= 0 or characters <= 0:
            return
        ratio = characters / text_tokens
        self.characters_per_token = min(MAXIMUM_CHARACTERS_PER_TOKEN, max(MINIMUM_CHARACTERS_PER_TOKEN, ratio))


def _message_characters(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        total += len(str(message.get("content", "")))
        if message.get("tool_calls"):
            total += len(json.dumps(message["tool_calls"], ensure_ascii=False))
    return total


def _tool_characters(tools: list[dict[str, Any]]) -> int:
    return len(json.dumps(tools, ensure_ascii=False))


def answer_limit_notice(context_window: int) -> str:
    """Shown when an answer reached the reserved length and was stopped by Ollama."""
    return (
        f"(The answer was stopped at {answer_reserve(context_window)} tokens, the room kept for it in the "
        "context window. Ask for a shorter answer or a part at a time, or use /set context 8192.)"
    )


def over_budget_message(estimated_tokens: int, context_window: int) -> str:
    """Explain a refused request and what the teacher can do about it."""
    return (
        f"This request needs about {estimated_tokens} tokens, but the context window is "
        f"{context_window} tokens and {answer_reserve(context_window)} are kept for the answer. "
        "Sent as is, Ollama would silently cut it: the oldest turns, or, for a message this "
        "long, the beginning of the prompt with the assistant's instructions. Shorten the "
        "message, index long documents with "
        "/ingest instead of pasting them, use /compress, or raise model.context_window in config.json."
    )
