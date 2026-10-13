from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ToolResult:
    """Outcome of a tool execution, sent back to the model as text."""
    content: str
    is_error: bool = False


class Tool(ABC):
    """A function the model can call.

    Subclasses define `name`, `description`, `parameters` (a JSON schema of type
    "object") and implement `execute`.
    """
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {"type": "object", "properties": {}}

    @abstractmethod
    def execute(self, arguments: Dict[str, Any]) -> str:
        """Run the tool and return its result as text."""

    def to_ollama_tool(self) -> Dict[str, Any]:
        """Definition in the format expected by the Ollama `tools` parameter."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
