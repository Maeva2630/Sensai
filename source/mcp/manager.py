from typing import Any, Dict, List, Optional

from mcp.client import MCPClient, MCPError
from mcp.config import DEFAULT_CONFIG_PATH, MCPServerConfig, load_mcp_config
from tools.base import Tool
from tools.registry import ToolRegistry


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

    def __init__(self, server: str, client: MCPClient, definition: Dict[str, Any]):
        self.server = server
        self.remote_name = definition["name"]
        self.name = f"{server}__{self.remote_name}"
        self.description = definition.get("description") or f"Tool '{self.remote_name}' from MCP server '{server}'."
        self.parameters = _object_schema(definition.get("inputSchema"))
        self._client = client

    def execute(self, arguments: Dict[str, Any]) -> str:
        result = self._client.call_tool(self.remote_name, arguments)
        text = MCPClient.result_text(result)
        if result.get("isError"):
            raise RuntimeError(text or "the MCP tool reported an error")
        return text


class MCPManager:
    """Starts the configured MCP servers and exposes their tools."""

    def __init__(self, servers: Optional[List[MCPServerConfig]] = None, warnings: Optional[List[str]] = None):
        self._servers = servers or []
        self._clients: List[MCPClient] = []
        self._tools: List[MCPTool] = []
        self._config_warnings = list(warnings or [])

    @classmethod
    def from_config_file(cls, path: str = DEFAULT_CONFIG_PATH) -> "MCPManager":
        servers, warnings = load_mcp_config(path)
        return cls(servers, warnings)

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
            client = MCPClient(server.command, server.args, server.env or None, server.timeout)
            try:
                client.start()
                definitions = client.list_tools()
            except MCPError as e:
                client.close()
                warnings.append(f"MCP server '{server.name}' unavailable: {e}")
                continue
            self._clients.append(client)
            self._tools.extend(MCPTool(server.name, client, definition) for definition in definitions)
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
