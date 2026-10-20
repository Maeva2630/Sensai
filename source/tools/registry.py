from typing import Any, Dict, List, Optional

from tools.base import Tool, ToolResult

_JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


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
            return f"unknown argument '{name}'"
        spec = properties[name]
        expected = _JSON_TYPES.get(spec.get("type"))
        if expected:
            # bool is a subclass of int, but is not a valid number
            is_bool_as_number = isinstance(value, bool) and spec.get("type") in ("integer", "number")
            if is_bool_as_number or not isinstance(value, expected):
                return f"argument '{name}' must be of type {spec['type']}"
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
