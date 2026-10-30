"""Execution of tool calls requested by the chat model."""

import json
import sqlite3
from time import perf_counter
from typing import Any, Callable

from observability.monitoring import DisabledMonitoringStore, MonitoringStore
from tools.registry import ToolRegistry, ToolResult
from tools.safety import mark_untrusted
from utilities.ollama_client import build_tool_message, parse_tool_calls
from observability.app_logging import get_logger


logger = get_logger("tools")

DECLINED_MESSAGE = "The user declined this request, so nothing was sent. Answer without it and say so."


def run_tool_calls(
    registry: ToolRegistry,
    raw_tool_calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    monitoring_store: MonitoringStore | DisabledMonitoringStore | None = None,
    model_interaction_id: int = 0,
    confirm: Callable[[str], bool] | None = None,
) -> None:
    """Execute model-requested tools and append their results to chat history.

    A tool that asks for confirmation (it sends data outside the machine) runs only
    if `confirm` approves; without a `confirm` function it is refused.
    """
    for call in parse_tool_calls({"tool_calls": raw_tool_calls}):
        logger.info("tool.requested name=%s", call["name"])
        print(f"[tool] {call['name']}({json.dumps(call['arguments'])})")
        started_at = perf_counter()
        tool = registry.get(call["name"])
        question = _confirmation_question(tool, call["arguments"])
        if question is not None and not (confirm is not None and confirm(question)):
            logger.info("tool.declined name=%s", call["name"])
            result = ToolResult(DECLINED_MESSAGE, is_error=True)
        else:
            result = registry.execute(call["name"], call["arguments"])
        latency_ms = (perf_counter() - started_at) * 1000
        label = "error" if result.is_error else "result"
        if result.is_error:
            logger.warning("tool.failed name=%s", call["name"])
        else:
            logger.info("tool.completed name=%s", call["name"])
        if monitoring_store is not None:
            try:
                monitoring_store.record_tool_interaction(
                    model_interaction_id=model_interaction_id,
                    tool_name=call["name"],
                    status="error" if result.is_error else "success",
                    latency_ms=latency_ms,
                )
            except (OSError, sqlite3.Error) as error:
                # Observability is best effort and must never block a tool result.
                logger.warning("monitoring.tool_write_failed error_type=%s", type(error).__name__)
        print(f"[tool] {label}: {result.content}")
        content = mark_untrusted(call["name"], result.content) if tool is not None and tool.untrusted else result.content
        messages.append(build_tool_message(call["name"], content))


def _confirmation_question(tool: Any, arguments: Any) -> str | None:
    """The question a tool wants answered before it runs, if any."""
    if tool is None:
        return None
    return tool.confirmation_prompt(arguments if isinstance(arguments, dict) else {})
