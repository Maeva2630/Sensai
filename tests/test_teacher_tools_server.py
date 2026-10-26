import json
import os
import subprocess
import sys
import unittest

# Ensure `source` directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

# pyrefly: ignore [missing-import]
from mcp.config import MCPServerConfig, load_mcp_config
# pyrefly: ignore [missing-import]
from mcp.manager import MCPManager
# pyrefly: ignore [missing-import]
from mcp.servers.teacher_tools import TOOLS, TeacherToolsServer
# pyrefly: ignore [missing-import]
from tools.registry import ToolRegistry

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SERVER_PATH = os.path.join(ROOT, "source", "mcp", "servers", "teacher_tools.py")


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def call(server, name, arguments, request_id=1):
    return server.handle({
        "jsonrpc": "2.0", "id": request_id, "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    })


def text_of(response):
    return response["result"]["content"][0]["text"]


class TestTeacherToolsProtocol(unittest.TestCase):

    def setUp(self):
        self.server = TeacherToolsServer()

    def test_initialize_returns_capabilities(self):
        response = self.server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        })
        self.assertEqual(response["id"], 1)
        self.assertEqual(response["result"]["protocolVersion"], "2024-11-05")
        self.assertIn("tools", response["result"]["capabilities"])

    def test_notifications_get_no_response(self):
        self.assertIsNone(self.server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_tools_list_describes_every_tool(self):
        response = self.server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = [tool["name"] for tool in response["result"]["tools"]]
        self.assertEqual(names, ["calculate_grade", "class_statistics", "start_timer", "timer_status"])
        for tool in TOOLS:
            self.assertEqual(tool["inputSchema"]["type"], "object")
            self.assertTrue(tool["description"])

    def test_unknown_method_is_a_jsonrpc_error(self):
        response = self.server.handle({"jsonrpc": "2.0", "id": 3, "method": "nope"})
        self.assertEqual(response["error"]["code"], -32601)

    def test_unknown_tool_is_a_jsonrpc_error(self):
        response = call(self.server, "nope", {})
        self.assertEqual(response["error"]["code"], -32602)

    def test_non_object_message_is_an_invalid_request(self):
        self.assertEqual(self.server.handle([1, 2])["error"]["code"], -32600)

    def test_arguments_must_be_an_object(self):
        response = self.server.handle({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "calculate_grade", "arguments": "12"},
        })
        self.assertTrue(response["result"]["isError"])


class TestCalculateGrade(unittest.TestCase):

    def setUp(self):
        self.server = TeacherToolsServer()

    def test_weighted_average(self):
        response = call(self.server, "calculate_grade", {"marks": [12, 15, 9], "coefficients": [2, 3, 1]})
        self.assertFalse(response["result"]["isError"])
        self.assertEqual(text_of(response), "Weighted average: 13/20 (3 marks)")

    def test_coefficients_default_to_one(self):
        response = call(self.server, "calculate_grade", {"marks": [10, 15]})
        self.assertIn("12.5/20", text_of(response))

    def test_custom_max_mark(self):
        response = call(self.server, "calculate_grade", {"marks": [80, 90], "max_mark": 100})
        self.assertIn("85/100", text_of(response))

    def test_invalid_arguments_are_reported_with_is_error(self):
        cases = [
            {},
            {"marks": []},
            {"marks": "12"},
            {"marks": [12, "x"]},
            {"marks": [True]},
            {"marks": [25]},
            {"marks": [-1]},
            {"marks": [10, 12], "coefficients": [1]},
            {"marks": [10], "coefficients": [-1]},
            {"marks": [10], "coefficients": [0]},
            {"marks": [10], "max_mark": 0},
            {"marks": [float("nan")]},
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                response = call(self.server, "calculate_grade", arguments)
                self.assertTrue(response["result"]["isError"])
                self.assertIn("Invalid arguments", text_of(response))

    def test_too_many_marks_are_refused(self):
        response = call(self.server, "calculate_grade", {"marks": [10] * 1001})
        self.assertTrue(response["result"]["isError"])


class TestClassStatistics(unittest.TestCase):

    def setUp(self):
        self.server = TeacherToolsServer()

    def test_statistics(self):
        response = call(self.server, "class_statistics", {"marks": [8, 12, 15, 19, 5]})
        self.assertEqual(
            text_of(response),
            "Students: 5; mean: 11.8/20; median: 12; min: 5; max: 19; pass rate: 60% (pass mark 10)",
        )

    def test_custom_pass_mark(self):
        response = call(self.server, "class_statistics", {"marks": [8, 12, 15, 19, 5], "pass_mark": 14})
        self.assertIn("pass rate: 40%", text_of(response))

    def test_invalid_arguments(self):
        for arguments in [{}, {"marks": []}, {"marks": [30]}, {"marks": [10], "pass_mark": 25}, {"marks": [10], "pass_mark": "x"}]:
            with self.subTest(arguments=arguments):
                self.assertTrue(call(self.server, "class_statistics", arguments)["result"]["isError"])


class TestTimers(unittest.TestCase):

    def setUp(self):
        self.clock = FakeClock()
        self.server = TeacherToolsServer(self.clock)

    def test_start_returns_immediately_with_an_id(self):
        response = call(self.server, "start_timer", {"minutes": 10, "label": "Exercise 2"})
        self.assertEqual(text_of(response), "Timer 1 started for 10 minutes (Exercise 2).")

    def test_status_reports_remaining_time_then_finished(self):
        call(self.server, "start_timer", {"minutes": 10})
        self.clock.now += 125
        self.assertEqual(text_of(call(self.server, "timer_status", {"timer_id": 1})), "Timer 1: 7 min 55 s remaining.")
        self.clock.now += 600
        self.assertEqual(text_of(call(self.server, "timer_status", {"timer_id": 1})), "Timer 1 is finished.")

    def test_timers_are_independent(self):
        call(self.server, "start_timer", {"minutes": 1})
        self.assertIn("Timer 2", text_of(call(self.server, "start_timer", {"minutes": 5})))

    def test_invalid_arguments(self):
        cases = [
            ("start_timer", {}),
            ("start_timer", {"minutes": 0}),
            ("start_timer", {"minutes": -3}),
            ("start_timer", {"minutes": 9999}),
            ("start_timer", {"minutes": "10"}),
            ("start_timer", {"minutes": 5, "label": 3}),
            ("timer_status", {}),
            ("timer_status", {"timer_id": "1"}),
            ("timer_status", {"timer_id": 42}),
        ]
        for name, arguments in cases:
            with self.subTest(name=name, arguments=arguments):
                self.assertTrue(call(self.server, name, arguments)["result"]["isError"])

    def test_number_of_timers_is_limited(self):
        for _ in range(50):
            call(self.server, "start_timer", {"minutes": 1})
        self.assertTrue(call(self.server, "start_timer", {"minutes": 1})["result"]["isError"])


class TestTeacherToolsProcess(unittest.TestCase):
    """The server run as a real process, as an MCP client would start it."""

    def exchange(self, lines):
        completed = subprocess.run(
            [sys.executable, SERVER_PATH], input="\n".join(lines) + "\n",
            capture_output=True, text=True, timeout=10, check=True,
        )
        return [json.loads(line) for line in completed.stdout.splitlines()]

    def test_server_answers_raw_jsonrpc_messages(self):
        responses = self.exchange([
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                        "params": {"name": "calculate_grade", "arguments": {"marks": [10, 20]}}}),
        ])
        self.assertEqual([r["id"] for r in responses], [1, 2])
        self.assertIn("15/20", text_of(responses[1]))

    def test_malformed_json_gets_a_parse_error_and_the_server_keeps_running(self):
        responses = self.exchange([
            "this is not json",
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}),
        ])
        self.assertEqual(responses[0]["error"]["code"], -32700)
        self.assertEqual(responses[1]["result"], {})


class TestTeacherToolsWithSensai(unittest.TestCase):
    """The server used through SensAI's own MCP client and tool registry."""

    def setUp(self):
        config = MCPServerConfig("teacher", sys.executable, [SERVER_PATH], {}, 10.0)
        self.manager = MCPManager([config])
        self.addCleanup(self.manager.close)
        self.registry = ToolRegistry()
        self.warnings = self.manager.start() + self.manager.register_tools(self.registry)

    def test_tools_are_discovered_and_registered(self):
        self.assertEqual(self.warnings, [])
        self.assertEqual(
            sorted(tool.name for tool in self.registry.list_tools()),
            ["teacher__calculate_grade", "teacher__class_statistics", "teacher__start_timer", "teacher__timer_status"],
        )

    def test_tool_call_through_the_registry(self):
        result = self.registry.execute("teacher__calculate_grade", {"marks": [12, 15, 9], "coefficients": [2, 3, 1]})
        self.assertFalse(result.is_error)
        self.assertEqual(result.content, "Weighted average: 13/20 (3 marks)")

    def test_invalid_arguments_come_back_as_an_error_result(self):
        result = self.registry.execute("teacher__class_statistics", {"marks": [99]})
        self.assertTrue(result.is_error)
        self.assertIn("between 0 and 20", result.content)

    def test_timer_keeps_its_state_between_calls(self):
        self.registry.execute("teacher__start_timer", {"minutes": 5})
        result = self.registry.execute("teacher__timer_status", {"timer_id": 1})
        self.assertFalse(result.is_error)
        self.assertIn("remaining", result.content)


class TestDefaultMcpConfig(unittest.TestCase):

    def test_repository_config_declares_the_teacher_server(self):
        servers, warnings = load_mcp_config(os.path.join(ROOT, "mcp_servers.json"))
        self.assertEqual(warnings, [])
        teacher = next(server for server in servers if server.name == "teacher")
        self.assertTrue(os.path.exists(os.path.join(ROOT, teacher.args[0])))


if __name__ == "__main__":
    unittest.main()
