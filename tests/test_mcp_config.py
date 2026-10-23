import json
import os
import sys
import tempfile
import unittest

# Ensure `source` directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

# pyrefly: ignore [missing-import]
from mcp.config import load_mcp_config


class TestLoadMCPConfig(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, "mcp_servers.json")

    def write(self, content):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(content if isinstance(content, str) else json.dumps(content))

    def test_missing_file_means_no_servers(self):
        self.assertEqual(load_mcp_config(self.path), ([], []))

    def test_valid_configuration(self):
        self.write({"mcpServers": {
            "time": {"command": "uvx", "args": ["mcp-server-time"], "env": {"TZ": "UTC"}, "timeout": 20},
            "minimal": {"command": "python3"},
        }})
        servers, warnings = load_mcp_config(self.path)

        self.assertEqual(warnings, [])
        self.assertEqual([s.name for s in servers], ["time", "minimal"])
        self.assertEqual(servers[0].command, "uvx")
        self.assertEqual(servers[0].args, ["mcp-server-time"])
        self.assertEqual(servers[0].env, {"TZ": "UTC"})
        self.assertEqual(servers[0].timeout, 20.0)
        self.assertEqual(servers[1].args, [])
        self.assertEqual(servers[1].timeout, 10.0)

    def test_invalid_json_is_reported(self):
        self.write("{ not json")
        servers, warnings = load_mcp_config(self.path)
        self.assertEqual(servers, [])
        self.assertIn("Could not read MCP configuration", warnings[0])

    def test_missing_mcp_servers_key_is_reported(self):
        self.write({"servers": {}})
        servers, warnings = load_mcp_config(self.path)
        self.assertEqual(servers, [])
        self.assertIn("mcpServers", warnings[0])

    def test_invalid_entries_are_skipped_but_valid_ones_kept(self):
        self.write({"mcpServers": {
            "good": {"command": "python3"},
            "no-command": {"args": []},
            "bad args": {"command": "python3"},
            "bad-args": {"command": "python3", "args": "oops"},
            "bad-env": {"command": "python3", "env": {"K": 1}},
            "bad-timeout": {"command": "python3", "timeout": 0},
            "not-an-object": "python3",
        }})
        servers, warnings = load_mcp_config(self.path)

        self.assertEqual([s.name for s in servers], ["good"])
        self.assertEqual(len(warnings), 6)
        self.assertTrue(all(w.startswith("Ignoring MCP server") for w in warnings))


if __name__ == "__main__":
    unittest.main()
