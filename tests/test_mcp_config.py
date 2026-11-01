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


class TestMCPRoutingAndLocation(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, "mcp_servers.json")

    def write(self, content):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(json.dumps(content))

    def test_keywords_are_loaded_and_validated(self):
        self.write({"mcpServers": {
            "teacher": {"command": "python3", "keywords": [" moyenne ", "minuteur"]},
            "broken": {"command": "python3", "keywords": "moyenne"},
        }})
        servers, warnings = load_mcp_config(self.path)

        self.assertEqual(servers[0].keywords, ["moyenne", "minuteur"])
        self.assertEqual(len(warnings), 1)
        self.assertIn("'keywords' must be a list", warnings[0])

    def test_servers_run_in_the_folder_of_their_configuration(self):
        self.write({"mcpServers": {"teacher": {"command": "python3", "args": ["server.py"]}}})
        servers, _ = load_mcp_config(self.path)

        self.assertEqual(servers[0].cwd, os.path.dirname(os.path.abspath(self.path)))

    def test_default_configuration_is_the_project_file_not_the_working_directory(self):
        from mcp.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT

        self.assertTrue(os.path.isabs(DEFAULT_CONFIG_PATH))
        self.assertEqual(os.path.dirname(DEFAULT_CONFIG_PATH), PROJECT_ROOT)
        self.assertTrue(os.path.exists(os.path.join(PROJECT_ROOT, "source", "main.py")))

    def test_python_servers_use_the_interpreter_running_sensai(self):
        from mcp.client import _interpreter

        self.assertEqual(_interpreter("python3"), sys.executable)
        self.assertEqual(_interpreter("uvx"), "uvx")
