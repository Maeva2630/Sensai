"""Execution of tool calls requested by the chat model."""

import json
from typing import Any

from tools.registry import ToolRegistry
from utilities.ollama_client import build_tool_message, parse_tool_calls
from observability.app_logging import get_logger


logger = get_logger("tools")


def run_tool_calls(
    registry: ToolRegistry,
    raw_tool_calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> None:
    """Execute model-requested tools and append their results to chat history."""
    for call in parse_tool_calls({"tool_calls": raw_tool_calls}):
        logger.info("tool.requested name=%s", call["name"])
        print(f"[tool] {call['name']}({json.dumps(call['arguments'])})")
        result = registry.execute(call["name"], call["arguments"])
        label = "error" if result.is_error else "result"
        if result.is_error:
            logger.warning("tool.failed name=%s", call["name"])
        else:
            logger.info("tool.completed name=%s", call["name"])
        print(f"[tool] {label}: {result.content}")
        messages.append(build_tool_message(call["name"], result.content))
