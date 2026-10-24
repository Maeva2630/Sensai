from typing import Any, Dict, List, Optional

from tools.base import Tool, ToolResult

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

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

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
            return ToolResult(f"Unknown tool '{name}'", is_error=True)

        error = validate_arguments(tool.parameters, arguments)
        if error:
            return ToolResult(f"Invalid arguments for '{name}': {error}", is_error=True)

        try:
            return ToolResult(str(tool.execute(arguments)))
        except Exception as e:
            return ToolResult(f"Tool '{name}' failed: {e}", is_error=True)


def format_tool_listing(tools: List[Tool]) -> str:
    """Human-readable list of tools with their origin (`native` or an MCP server)."""
    if not tools:
        return "No tools available."
    lines = []
    for tool in tools:
        origin = getattr(tool, "server", None) or "native"
        lines.append(f"- {tool.name} [{origin}]: {tool.description}")
    return "\n".join(lines)
