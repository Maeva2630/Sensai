"""Execution of tool calls requested by the chat model."""

import json
from typing import Any

from tools.registry import ToolRegistry
from tools.safety import mark_untrusted
from utilities.ollama_client import build_tool_message, parse_tool_calls


def run_tool_calls(
    registry: ToolRegistry,
    raw_tool_calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> None:
    """Execute model-requested tools and append their results to chat history."""
    for call in parse_tool_calls({"tool_calls": raw_tool_calls}):
        print(f"[tool] {call['name']}({json.dumps(call['arguments'])})")
        result = registry.execute(call["name"], call["arguments"])
        label = "error" if result.is_error else "result"
        print(f"[tool] {label}: {result.content}")
        tool = registry.get(call["name"])
        content = mark_untrusted(call["name"], result.content) if tool is not None and tool.untrusted else result.content
        messages.append(build_tool_message(call["name"], content))
