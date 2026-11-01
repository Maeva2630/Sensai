import unittest
import sys
import os

# Ensure `source` directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

# pyrefly: ignore [missing-import]
from mcp.client import MCPClient, MCPError, MCPConnectionError, MCPTimeoutError

FAKE_SERVER = os.path.join(os.path.dirname(__file__), "fake_mcp_server.py")


def make_client(timeout=5.0):
    return MCPClient(sys.executable, [FAKE_SERVER], timeout=timeout)


class TestMCPClient(unittest.TestCase):

    def test_handshake_and_list_tools(self):
        with make_client() as client:
            tools = client.list_tools()
        self.assertEqual([t["name"] for t in tools], ["echo", "fail", "hang", "crash"])
        self.assertIn("inputSchema", tools[0])

    def test_call_tool_success(self):
        with make_client() as client:
            result = client.call_tool("echo", {"text": "hello"})
        self.assertFalse(result["isError"])
        self.assertEqual(MCPClient.result_text(result), "hello")

    def test_call_tool_error_result(self):
        with make_client() as client:
            result = client.call_tool("fail")
        self.assertTrue(result["isError"])
        self.assertEqual(MCPClient.result_text(result), "something went wrong")

    def test_jsonrpc_error_raises(self):
        with make_client() as client:
            with self.assertRaises(MCPError) as ctx:
                client.call_tool("unknown")
        self.assertIn("Unknown tool", str(ctx.exception))

    def test_request_timeout(self):
        with make_client(timeout=0.5) as client:
            with self.assertRaises(MCPTimeoutError):
                client.call_tool("hang")

    def test_server_crash_raises_connection_error(self):
        with make_client() as client:
            with self.assertRaises(MCPConnectionError):
                client.call_tool("crash")
            # The client keeps reporting the failure on later calls
            with self.assertRaises(MCPConnectionError):
                client.list_tools()

    def test_command_not_found(self):
        client = MCPClient("this-command-does-not-exist-sensai")
        with self.assertRaises(MCPConnectionError):
            client.start()

    def test_close_terminates_subprocess(self):
        client = make_client()
        client.start()
        process = client._process
        client.close()
        self.assertIsNotNone(process.poll())
        client.close()  # safe to call twice

    def test_exit_terminates_subprocess_after_error(self):
        process = None
        with self.assertRaises(MCPError):
            with make_client() as client:
                process = client._process
                client.call_tool("unknown")
        self.assertIsNotNone(process.poll())

    def test_calls_after_close_raise(self):
        client = make_client()
        client.start()
        client.close()
        with self.assertRaises(MCPConnectionError):
            client.list_tools()


if __name__ == "__main__":
    unittest.main()


class TestRequestDeadline(unittest.TestCase):

    def test_a_server_that_keeps_logging_cannot_hold_a_request_forever(self):
        import threading
        import time
        from unittest.mock import MagicMock

        from mcp.client import MCPClient, MCPTimeoutError

        client = MCPClient("unused", timeout=0.3)
        client._process = MagicMock()
        client._process.poll.return_value = None
        stop = threading.Event()

        def chatter():
            while not stop.is_set():
                client._lines.put("log line, not JSON-RPC\n")
                time.sleep(0.02)

        threading.Thread(target=chatter, daemon=True).start()
        self.addCleanup(stop.set)
        started = time.monotonic()
        with self.assertRaises(MCPTimeoutError):
            client.call_tool("anything")
        self.assertLess(time.monotonic() - started, 2.0)
