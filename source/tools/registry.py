import json
import logging
from typing import Any, Dict, List, Optional

from tools.base import Tool, ToolError, ToolResult
from tools.safety import MAX_RESULT_CHARS, truncate_result

logger = logging.getLogger(__name__)

_JSON_TYPES = {
    "null": type(None),
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _check_type(name: str, value: Any, declared: Any) -> Optional[str]:
    """Check a value against a JSON schema `type` (a name or a list of names)."""
    names = declared if isinstance(declared, list) else [declared]
    expected = tuple(_JSON_TYPES[n] for n in names if isinstance(n, str) and n in _JSON_TYPES)
    if not expected:
        return None  # no usable type constraint (e.g. anyOf): accept
    # bool is a subclass of int, but is not a valid number
    is_bool_as_number = isinstance(value, bool) and "boolean" not in names
    if is_bool_as_number or not isinstance(value, expected):
        return f"argument '{name}' must be of type {' or '.join(n for n in names if isinstance(n, str))}"
    return None


def _coerce_value(value: str, declared: Any) -> Any:
    """Turn a string produced by the model into the JSON type the schema declares."""
    names = declared if isinstance(declared, list) else [declared]
    if "string" in names:
        return value
    if "boolean" in names and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    try:
        parsed = json.loads(value)
    except ValueError:
        return value
    wanted = tuple(_JSON_TYPES[n] for n in names if isinstance(n, str) and n in _JSON_TYPES and n != "boolean")
    if wanted and isinstance(parsed, wanted) and not isinstance(parsed, bool):
        return parsed
    return value


def coerce_arguments(schema: Dict[str, Any], arguments: Any) -> Any:
    """Fix arguments that small models send as strings ("10", "[1, 2]") instead of numbers or arrays.

    Only string values for properties declared with a non-string type are converted, and
    only when the string is valid JSON of that type. Anything else is left for validation.
    """
    if not isinstance(arguments, dict):
        return arguments
    properties = schema.get("properties", {})
    coerced = {}
    for name, value in arguments.items():
        spec = properties.get(name)
        if isinstance(value, str) and isinstance(spec, dict) and "type" in spec:
            value = _coerce_value(value, spec["type"])
        coerced[name] = value
    return coerced


def validate_arguments(schema: Dict[str, Any], arguments: Any) -> Optional[str]:
    """Check arguments against a tool schema. Return an error message, or None if valid."""
    if not isinstance(arguments, dict):
        return "arguments must be an object"

    properties = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in arguments:
            return f"missing required argument '{name}'"

    for name, value in arguments.items():
        if name not in properties:
            if schema.get("additionalProperties"):
                continue
            return f"unknown argument '{name}'"
        spec = properties[name]
        if not isinstance(spec, dict):
            continue  # e.g. the JSON schema shorthand `true`
        error = _check_type(name, value, spec.get("type"))
        if error:
            return error
        if "enum" in spec and value not in spec["enum"]:
            return f"argument '{name}' must be one of {spec['enum']}"
    return None


class ToolRegistry:
    """Holds the available tools and executes them by name."""

    def __init__(self, max_result_chars: int = MAX_RESULT_CHARS):
        self._tools: Dict[str, Tool] = {}
        self._max_result_chars = max_result_chars

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("a tool must have a name")
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def to_ollama_tools(self, user_input: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return tool definitions, optionally filtered for one user request."""
        tools = self._tools.values()
        if user_input is not None:
            tools = (tool for tool in tools if tool.is_relevant_for(user_input))
        return [tool.to_ollama_tool() for tool in tools]

    def claims_request(self, user_input: str) -> bool:
        """Return whether any tool explicitly claims this request."""
        return any(tool.claims_request(user_input) for tool in self._tools.values())

    def execute(self, name: str, arguments: Any) -> ToolResult:
        """Validate and run a tool. Never raises: failures come back as an error result."""
        tool = self.get(name)
        if tool is None:
            available = ", ".join(sorted(self._tools)) or "none"
            return ToolResult(f"Unknown tool '{name}'. Available tools: {available}", is_error=True)

        arguments = coerce_arguments(tool.parameters, arguments)
        error = validate_arguments(tool.parameters, arguments)
        if error:
            return ToolResult(f"Invalid arguments for '{name}': {error}", is_error=True)

        try:
            return ToolResult(truncate_result(str(tool.execute(arguments)), self._max_result_chars))
        except ToolError as e:
            return ToolResult(truncate_result(f"Tool '{name}' failed: {e}", self._max_result_chars), is_error=True)
        except Exception as e:
            # Details go to the log; the model and the user only get the error type.
            logger.exception("Tool '%s' raised an unexpected error", name)
            return ToolResult(f"Tool '{name}' failed unexpectedly ({type(e).__name__}).", is_error=True)


def format_tool_listing(tools: List[Tool]) -> str:
    """Human-readable list of tools with their origin (`native` or an MCP server)."""
    if not tools:
        return "No tools available."
    lines = []
    for tool in tools:
        origin = getattr(tool, "server", None) or "native"
        lines.append(f"- {tool.name} [{origin}]: {tool.description}")
    return "\n".join(lines)
