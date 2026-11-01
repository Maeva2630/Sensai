import os
import sys
import unittest

# Ensure `source` directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

# pyrefly: ignore [missing-import]
from mcp.config import MCPServerConfig
# pyrefly: ignore [missing-import]
from mcp.manager import MCPManager
# pyrefly: ignore [missing-import]
from tools.registry import ToolRegistry, format_tool_listing
# pyrefly: ignore [missing-import]
from tools.time_tool import GetTimeTool

FAKE_SERVER = os.path.join(os.path.dirname(__file__), "fake_mcp_server.py")


def fake_server(name="fake", timeout=5.0):
    return MCPServerConfig(name, sys.executable, [FAKE_SERVER], {}, timeout)


class TestMCPManager(unittest.TestCase):

    def start(self, servers, warnings=None):
        manager = MCPManager(servers, warnings)
        self.addCleanup(manager.close)
        return manager, manager.start()

    def test_discovers_tools_with_server_prefix(self):
        manager, warnings = self.start([fake_server("alpha")])

        self.assertEqual(warnings, [])
        self.assertEqual(
            [t.name for t in manager.tools],
            ["alpha__echo", "alpha__fail", "alpha__hang", "alpha__crash"],
        )
        self.assertEqual(manager.tools[0].server, "alpha")
        self.assertEqual(manager.tools[0].description, "Echo a text.")

    def test_schema_is_converted_for_ollama(self):
        manager, _ = self.start([fake_server("alpha")])
        registry = ToolRegistry()
        manager.register_tools(registry)

        definition = next(
            d for d in registry.to_ollama_tools() if d["function"]["name"] == "alpha__echo"
        )
        self.assertEqual(definition["type"], "function")
        self.assertEqual(definition["function"]["parameters"], {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        })

    def test_missing_schema_becomes_empty_object_schema(self):
        manager, _ = self.start([fake_server("alpha")])
        crash = next(t for t in manager.tools if t.remote_name == "crash")
        self.assertEqual(crash.parameters, {"type": "object", "properties": {}})

    def test_tool_runs_through_the_registry(self):
        manager, _ = self.start([fake_server("alpha")])
        registry = ToolRegistry()
        manager.register_tools(registry)

        result = registry.execute("alpha__echo", {"text": "hello"})

        self.assertFalse(result.is_error)
        self.assertEqual(result.content, "hello")

    def test_tool_error_becomes_error_result(self):
        manager, _ = self.start([fake_server("alpha")])
        registry = ToolRegistry()
        manager.register_tools(registry)

        result = registry.execute("alpha__fail", {})

        self.assertTrue(result.is_error)
        self.assertIn("something went wrong", result.content)

    def test_invalid_arguments_are_rejected_before_reaching_the_server(self):
        manager, _ = self.start([fake_server("alpha")])
        registry = ToolRegistry()
        manager.register_tools(registry)

        result = registry.execute("alpha__echo", {})

        self.assertTrue(result.is_error)
        self.assertIn("missing required argument 'text'", result.content)

    def test_two_servers_do_not_collide(self):
        manager, warnings = self.start([fake_server("alpha"), fake_server("beta")])
        registry = ToolRegistry()
        registry.register(GetTimeTool())

        self.assertEqual(manager.register_tools(registry), [])
        names = [t.name for t in registry.list_tools()]
        self.assertIn("alpha__echo", names)
        self.assertIn("beta__echo", names)
        self.assertIn("get_time", names)

    def test_broken_server_does_not_prevent_the_others(self):
        broken = MCPServerConfig("broken", "this-command-does-not-exist-sensai", [], {}, 1.0)
        manager, warnings = self.start([broken, fake_server("alpha")])

        self.assertEqual(len(warnings), 1)
        self.assertIn("MCP server 'broken' unavailable", warnings[0])
        self.assertIn("alpha__echo", [t.name for t in manager.tools])

    def test_config_warnings_are_returned(self):
        _, warnings = self.start([], ["Ignoring MCP server 'x': nope"])
        self.assertEqual(warnings, ["Ignoring MCP server 'x': nope"])

    def test_duplicate_registration_is_reported(self):
        manager, _ = self.start([fake_server("alpha")])
        registry = ToolRegistry()
        manager.register_tools(registry)

        warnings = manager.register_tools(registry)

        self.assertEqual(len(warnings), 4)
        self.assertIn("Skipping MCP tool 'alpha__echo'", warnings[0])

    def test_close_stops_servers_and_is_idempotent(self):
        manager, _ = self.start([fake_server("alpha")])
        process = manager._clients[0]._process

        manager.close()
        manager.close()

        self.assertIsNotNone(process.poll())
        self.assertEqual(manager.tools, [])

    def test_no_servers_is_fine(self):
        manager, warnings = self.start([])
        self.assertEqual((manager.tools, warnings), ([], []))


class TestFormatToolListing(unittest.TestCase):

    def test_lists_native_and_mcp_tools_with_origin(self):
        manager = MCPManager([fake_server("alpha")])
        self.addCleanup(manager.close)
        manager.start()
        registry = ToolRegistry()
        registry.register(GetTimeTool())
        manager.register_tools(registry)

        listing = format_tool_listing(registry.list_tools())

        self.assertIn("- get_time [native]: Get the current local date, weekday, time and time zone of this computer.", listing)
        self.assertIn("- alpha__echo [alpha]: Echo a text.", listing)

    def test_no_tools(self):
        self.assertEqual(format_tool_listing([]), "No tools available.")


if __name__ == "__main__":
    unittest.main()


class TestMCPToolRouting(unittest.TestCase):

    def test_tools_are_offered_only_for_their_server_keywords(self):
        server = MCPServerConfig("teacher", sys.executable, [FAKE_SERVER], {}, 5.0, keywords=["moyenne", "minuteur"])
        manager = MCPManager([server])
        self.addCleanup(manager.close)
        manager.start()
        registry = ToolRegistry()
        manager.register_tools(registry)

        self.assertEqual(registry.to_ollama_tools("Écris une activité sur les fractions"), [])
        self.assertEqual(len(registry.to_ollama_tools("Calcule la moyenne de Léa")), 4)
        self.assertEqual(len(registry.to_ollama_tools("Lance un minuteur de 10 minutes")), 4)

    def test_tools_without_keywords_stay_always_offered(self):
        manager = MCPManager([fake_server("alpha")])
        self.addCleanup(manager.close)
        manager.start()

        self.assertTrue(all(tool.is_relevant_for("anything") for tool in manager.tools))

    def test_a_tool_definition_without_a_name_is_skipped_with_a_warning(self):
        from unittest.mock import patch

        with patch("mcp.manager.MCPClient") as client_class:
            client_class.return_value.list_tools.return_value = [{"description": "no name"}, {"name": "ok"}]
            manager = MCPManager([fake_server("alpha")])
            warnings = manager.start()

        self.assertEqual([tool.name for tool in manager.tools], ["alpha__ok"])
        self.assertIn("without a valid name", warnings[0])
