import json
import re
from typing import Any, Dict, List, Optional

from mcp.client import MCPClient, MCPError, MCPTimeoutError, MCPToolError
from mcp.config import DEFAULT_CONFIG_PATH, MCPServerConfig, load_mcp_config
from tools.base import Tool, ToolError
from tools.registry import ToolRegistry
from tools.safety import personal_data_warning


def _object_schema(input_schema: Any) -> Dict[str, Any]:
    """MCP `inputSchema` as the object schema expected by Ollama."""
    schema = dict(input_schema) if isinstance(input_schema, dict) else {}
    schema["type"] = "object"
    schema.setdefault("properties", {})
    return schema


class MCPTool(Tool):
    """A tool exposed by an MCP server, usable like any native tool.

    The model sees it as `<server>__<tool>` so tools of different servers can
    never collide.
    """

    untrusted = True

    def __init__(self, server: str, client: MCPClient, definition: Dict[str, Any], network: bool = False,
                 keywords: Optional[List[str]] = None):
        self.server = server
        self.network = network
        self.keywords = [word.lower() for word in keywords or []]
        self.remote_name = definition["name"]
        self.name = f"{server}__{self.remote_name}"
        self.description = definition.get("description") or f"Tool '{self.remote_name}' from MCP server '{server}'."
        self.parameters = _object_schema(definition.get("inputSchema"))
        self._client = client

    def is_relevant_for(self, user_input: str) -> bool:
        """Offer the tool when the request uses one of its server's keywords.

        Offering every MCP tool on every message made the 3B model call a grade
        calculator or a timer on ordinary lesson requests.
        """
        if not self.keywords:
            return True
        text = user_input.lower()
        return any(re.search(rf"(?<!\w){re.escape(word)}", text) for word in self.keywords)

    def confirmation_prompt(self, arguments: Dict[str, Any]) -> str | None:
        if not self.network:
            return None
        shown = json.dumps(arguments, ensure_ascii=False)
        return f"Call {self.name} with {shown}? This sends a request to the internet.{personal_data_warning(shown)}"

    def execute(self, arguments: Dict[str, Any]) -> str:
        try:
            result = self._client.call_tool(self.remote_name, arguments)
            text = MCPClient.result_text(result)
            if result.get("isError"):
                raise MCPToolError(text or "the MCP tool reported an error")
            return text
        except MCPTimeoutError as e:
            raise ToolError(f"MCP server '{self.server}' timed out: {e}") from e
        except MCPError as e:
            raise ToolError(f"MCP server '{self.server}': {e}") from e


class MCPManager:
    """Starts the configured MCP servers and exposes their tools."""

    def __init__(self, servers: Optional[List[MCPServerConfig]] = None, warnings: Optional[List[str]] = None):
        self._servers = servers or []
        self._clients: List[MCPClient] = []
        self._tools: List[MCPTool] = []
        self._config_warnings = list(warnings or [])

    @classmethod
    def from_config_file(cls, path: str = DEFAULT_CONFIG_PATH, network_enabled: bool = False) -> "MCPManager":
        """Load the servers from a file; servers marked `network` are kept only if network access is on."""
        servers, warnings = load_mcp_config(path)
        return cls([server for server in servers if network_enabled or not server.network], warnings)

    @property
    def tools(self) -> List[MCPTool]:
        return list(self._tools)

    def start(self) -> List[str]:
        """Connect to every server and discover its tools.

        A server that fails to start is reported in the returned warnings and
        skipped: it never prevents the others (or the application) from starting.
        """
        warnings = list(self._config_warnings)
        for server in self._servers:
            client = MCPClient(server.command, server.args, server.env or None, server.timeout, server.cwd)
            # Registered before the handshake so Ctrl+C during a slow start still stops the process.
            self._clients.append(client)
            try:
                client.start()
                definitions = client.list_tools()
            except MCPError as e:
                self._clients.remove(client)
                client.close()
                warnings.append(f"MCP server '{server.name}' unavailable: {e}")
                continue
            for definition in definitions:
                if not isinstance(definition, dict) or not isinstance(definition.get("name"), str) or not definition["name"]:
                    warnings.append(f"MCP server '{server.name}' sent a tool without a valid name: skipped")
                    continue
                self._tools.append(MCPTool(server.name, client, definition, server.network, server.keywords))
        return warnings

    def register_tools(self, registry: ToolRegistry) -> List[str]:
        """Add the discovered tools to a registry. Returns warnings for tools it could not add."""
        warnings = []
        for tool in self._tools:
            try:
                registry.register(tool)
            except ValueError as e:
                warnings.append(f"Skipping MCP tool '{tool.name}': {e}")
        return warnings

    def close(self) -> None:
        """Stop every server. Safe to call several times."""
        clients, self._clients = self._clients, []
        self._tools = []
        for client in clients:
            client.close()
