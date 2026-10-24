from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ToolResult:
    """Outcome of a tool execution, sent back to the model as text."""
    content: str
    is_error: bool = False


class ToolError(Exception):
    """A tool failure whose message is safe and useful to show to the model and the user."""


class Tool(ABC):
    """A function the model can call.

    Subclasses define `name`, `description`, `parameters` (a JSON schema of type
    "object") and implement `execute`.
    """
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {"type": "object", "properties": {}}
    # True for tools whose output comes from outside our code (MCP servers, the web).
    untrusted: bool = False

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

    def is_relevant_for(self, user_input: str) -> bool:
        """Return whether this tool should be offered for a user request.

        Tools without a routing rule stay available by default, which preserves
        the behavior of existing custom tools. Tools with a narrow purpose can
        override this method to avoid distracting the model on unrelated tasks.
        """
        return True

    def claims_request(self, user_input: str) -> bool:
        """Return whether the request is explicitly meant for this tool.

        A claimed request is answered by the tool instead of document retrieval
        when RAG mode is on. Tools without a precise routing rule never claim a
        request, so offering them does not switch RAG off.
        """
        return False
