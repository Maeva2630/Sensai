import os
import shutil
import sys
import unittest
from unittest.mock import MagicMock

# Ensure `source` directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

# pyrefly: ignore [missing-import]
from mcp.config import load_mcp_config
# pyrefly: ignore [missing-import]
from mcp.manager import MCPManager, MCPTool
# pyrefly: ignore [missing-import]
from tools.registry import ToolRegistry

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Tool definition returned by the official `mcp-server-fetch` (version 2026.10.24), description shortened.
FETCH_DEFINITION = {
    "name": "fetch",
    "description": "Fetches a URL from the internet and optionally extracts its contents as markdown.",
    "annotations": {"destructiveHint": False, "idempotentHint": True, "openWorldHint": True, "readOnlyHint": True},
    "inputSchema": {
        "title": "fetchArguments",
        "type": "object",
        "required": ["url"],
        "properties": {
            "url": {"title": "Url", "type": "string", "format": "uri", "minLength": 1, "description": "URL to fetch"},
            "max_length": {"title": "Max Length", "type": "integer", "default": 5000, "minimum": 1, "maximum": 999999},
            "start_index": {"title": "Start Index", "type": "integer", "default": 0, "minimum": 0},
            "raw": {"title": "Raw", "type": "boolean", "default": False},
        },
    },
}


def fetch_tool(text="Contents of https://example.com/:\nHello", is_error=False):
    client = MagicMock()
    client.call_tool.return_value = {"content": [{"type": "text", "text": text}], "isError": is_error}
    registry = ToolRegistry()
    registry.register(MCPTool("web", client, FETCH_DEFINITION))
    return registry, client


class TestThirdPartySchemaCompatibility(unittest.TestCase):
    """The real definition of a third-party tool works with no special case in our code."""

    def test_definition_is_converted(self):
        tool = MCPTool("web", MagicMock(), FETCH_DEFINITION)
        self.assertEqual(tool.name, "web__fetch")
        self.assertEqual(tool.server, "web")
        self.assertEqual(tool.parameters["type"], "object")
        self.assertEqual(tool.parameters["required"], ["url"])
        self.assertIn("max_length", tool.parameters["properties"])
        self.assertTrue(tool.untrusted)

    def test_tool_is_offered_to_the_model_with_its_schema(self):
        registry, _ = fetch_tool()
        offered = registry.to_ollama_tools("anything")[0]["function"]
        self.assertEqual(offered["name"], "web__fetch")
        self.assertEqual(offered["parameters"]["properties"]["url"]["format"], "uri")

    def test_call_reaches_the_server_with_the_original_tool_name(self):
        registry, client = fetch_tool()
        result = registry.execute("web__fetch", {"url": "https://example.com"})
        self.assertFalse(result.is_error)
        self.assertIn("Hello", result.content)
        client.call_tool.assert_called_once_with("fetch", {"url": "https://example.com"})

    def test_numbers_sent_as_text_by_a_small_model_are_converted(self):
        registry, client = fetch_tool()
        registry.execute("web__fetch", {"url": "https://example.com", "max_length": "2000", "raw": "false"})
        client.call_tool.assert_called_once_with("fetch", {"url": "https://example.com", "max_length": 2000, "raw": False})

    def test_invalid_arguments_are_refused_before_reaching_the_server(self):
        registry, client = fetch_tool()
        self.assertIn("missing required argument 'url'", registry.execute("web__fetch", {}).content)
        self.assertIn("must be of type integer", registry.execute("web__fetch", {"url": "x", "max_length": "many"}).content)
        client.call_tool.assert_not_called()

    def test_server_refusal_reaches_the_model(self):
        registry, _ = fetch_tool("Refused to fetch http://localhost/: not a public address", is_error=True)
        result = registry.execute("web__fetch", {"url": "http://localhost/"})
        self.assertTrue(result.is_error)
        self.assertIn("not a public address", result.content)


class TestDefaultConfigDeclaresBothServers(unittest.TestCase):

    def test_teacher_and_third_party_servers_are_declared(self):
        servers, warnings = load_mcp_config(os.path.join(ROOT, "mcp_servers.json"))
        self.assertEqual(warnings, [])
        by_name = {server.name: server for server in servers}
        self.assertEqual(sorted(by_name), ["teacher", "web"])
        self.assertEqual((by_name["web"].command, by_name["web"].args), ("uvx", ["mcp-server-fetch"]))
        # The first start may download the package
        self.assertGreaterEqual(by_name["web"].timeout, 30)


@unittest.skipUnless(
    os.environ.get("SENSAI_THIRD_PARTY_TESTS") == "1" and shutil.which("uvx"),
    "set SENSAI_THIRD_PARTY_TESTS=1 and install uv to run the test against the real fetch server",
)
class TestRealThirdPartyServer(unittest.TestCase):
    """Opt-in: starts the real `mcp-server-fetch` next to our own server."""

    def test_both_servers_run_together(self):
        servers, _ = load_mcp_config(os.path.join(ROOT, "mcp_servers.json"))
        # Relative script paths in the config are relative to the repository root
        previous = os.getcwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(ROOT)
        manager = MCPManager(servers)
        self.addCleanup(manager.close)
        registry = ToolRegistry()

        warnings = manager.start() + manager.register_tools(registry)

        self.assertEqual(warnings, [])
        names = [tool.name for tool in registry.list_tools()]
        self.assertIn("web__fetch", names)
        self.assertIn("teacher__calculate_grade", names)
        self.assertFalse(registry.execute("teacher__calculate_grade", {"marks": [10, 20]}).is_error)
        # Private addresses are refused by the server itself, with no network needed
        refused = registry.execute("web__fetch", {"url": "http://localhost:1/"})
        self.assertTrue(refused.is_error)
        self.assertIn("not a public address", refused.content)


if __name__ == "__main__":
    unittest.main()
