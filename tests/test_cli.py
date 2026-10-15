import json
import unittest
from unittest.mock import patch, MagicMock
import io
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

# pyrefly: ignore [missing-import]
from cli.cli import cli, MAX_TOOL_ITERATIONS
# pyrefly: ignore [missing-import]
from tools.base import Tool
# pyrefly: ignore [missing-import]
from tools.registry import ToolRegistry
# pyrefly: ignore [missing-import]
from utilities.ollama_client import (
    OllamaConnectionError,
    OllamaModelNotFoundError,
)
from rag.context import IndexingResult, PreparedContext
from rag.ingestion import IngestedDocument
from rag.storage import SearchResult

class TestCLIExceptionHandling(unittest.TestCase):
    """Unit tests for exception handling and flow control in CLI loop."""

    @patch("builtins.input", side_effect=["Hello", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    def test_cli_ollama_connection_error_does_not_exit(self, mock_client_cls, mock_input):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.stream_message.side_effect = OllamaConnectionError("Connection refused")

        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            cli()

        output = captured_output.getvalue()
        self.assertIn("Connection Error: Unable to connect to Ollama server", output)
        self.assertIn("Connection refused", output)
        self.assertIn("Please ensure Ollama is running.", output)

    @patch("builtins.input", side_effect=["Hello", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    def test_cli_generic_ollama_error_displays_warning(self, mock_client_cls, mock_input):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.stream_message.side_effect = OllamaModelNotFoundError("Model 'llama3.2' not found")

        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            cli()

        output = captured_output.getvalue()
        self.assertIn("Warning: Model 'llama3.2' not found", output)

    @patch("builtins.input", side_effect=["Hello", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    def test_cli_successful_stream(self, mock_client_cls, mock_input):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.stream_message.return_value = [
            b'{"message": {"content": "Hi"}}',
            b'{"message": {"content": " there!"}}',
        ]

        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            cli()

        output = captured_output.getvalue()
        self.assertIn("Assistant: Hi there!", output)

    @patch("builtins.input", side_effect=["Quelle heure est-il ?", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    def test_cli_offers_time_tool_for_a_time_request(self, mock_client_cls, mock_input):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.stream_message.return_value = [b'{"message": {"content": "Il est 10:00."}}']

        with patch("sys.stdout", io.StringIO()):
            cli()

        offered_tools = mock_client.stream_message.call_args.kwargs["tools"]
        self.assertEqual([tool["function"]["name"] for tool in offered_tools], ["get_time"])

    @patch("builtins.input", side_effect=[KeyboardInterrupt])
    @patch("cli.cli.OllamaHTTPClient")
    def test_cli_keyboard_interrupt_exits(self, mock_client_cls, mock_input):
        mock_client_cls.return_value = MagicMock()

        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            with self.assertRaises(SystemExit) as ctx:
                cli()
            self.assertEqual(ctx.exception.code, 0)

        output = captured_output.getvalue()
        self.assertIn("Exiting.", output)

    @patch("builtins.input", side_effect=["Hello", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    def test_cli_mid_stream_disconnection_does_not_exit(self, mock_client_cls, mock_input):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        def generator_with_error():
            yield b'{"message": {"content": "Starting response..."}}'
            raise OllamaConnectionError("Connection lost mid-stream")

        mock_client.stream_message.return_value = generator_with_error()

        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            cli()

        output = captured_output.getvalue()
        self.assertIn("Assistant: Starting response...", output)
        self.assertIn("Connection Error: Unable to connect to Ollama server", output)
        self.assertIn("Connection lost mid-stream", output)

    @patch("builtins.input", side_effect=["/ingest", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    def test_cli_ingest_command_without_path_displays_usage(self, mock_client_cls, mock_input):
        captured_output = io.StringIO()

        with patch("sys.stdout", captured_output):
            cli()

        self.assertIn("Usage: /ingest <path-to-document>", captured_output.getvalue())
        mock_client_cls.return_value.stream_message.assert_not_called()

    @patch("cli.cli.OllamaHTTPClient")
    def test_cli_ingest_command_reports_indexing_result(self, mock_client_cls):
        document = IngestedDocument("document-id", "knowledge.txt", "/tmp/knowledge.txt", "Knowledge", 9)
        indexer = MagicMock()
        indexer.index_document.side_effect = [
            IndexingResult(document, chunk_count=2, inserted_chunk_count=2),
            IndexingResult(document, chunk_count=2, inserted_chunk_count=0),
        ]
        captured_output = io.StringIO()
        with patch("builtins.input", side_effect=["/ingest knowledge.txt", "/ingest knowledge.txt", "exit"]), patch("sys.stdout", captured_output):
            cli(rag_indexer=indexer)

        output = captured_output.getvalue()
        self.assertIn("Document indexed: knowledge.txt (2 chunks; 2 new chunks stored).", output)
        self.assertIn("Document indexed: knowledge.txt (2 chunks; 0 new chunks stored).", output)
        mock_client_cls.return_value.stream_message.assert_not_called()

class AddTool(Tool):
    name = "add"
    description = "Add two integers."
    parameters = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    }

    def execute(self, arguments):
        return arguments["a"] + arguments["b"]


def tool_call_chunk(name, arguments):
    return json.dumps({"message": {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": name, "arguments": arguments}}
    ]}}).encode()


def text_chunk(text):
    return json.dumps({"message": {"content": text}}).encode()


class TestCLIToolCalling(unittest.TestCase):
    """Unit tests for the tool-calling loop in the CLI."""

    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(AddTool())
        self.histories = []

    def mock_client(self, mock_client_cls, responses):
        """Make stream_message return `responses` in order, recording the history it receives."""
        client = MagicMock()
        mock_client_cls.return_value = client
        remaining = list(responses)

        def stream_message(messages, tools=None):
            self.histories.append([dict(m) for m in messages])
            return remaining.pop(0)

        client.stream_message.side_effect = stream_message
        return client

    def run_cli(self, registry=None):
        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            cli(registry=registry or self.registry)
        return captured_output.getvalue()

    @patch("builtins.input", side_effect=["What is 17 + 25?", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    def test_tool_call_result_is_used_in_final_answer(self, mock_client_cls, mock_input):
        client = self.mock_client(mock_client_cls, [
            [tool_call_chunk("add", {"a": 17, "b": 25})],
            [text_chunk("It is 42.")],
        ])

        output = self.run_cli()

        self.assertIn('[tool] add({"a": 17, "b": 25})', output)
        self.assertIn("[tool] result: 42", output)
        self.assertIn("Assistant: It is 42.", output)
        # The second request carries the assistant tool call and the tool result
        second_request = self.histories[1]
        self.assertEqual(second_request[-2]["role"], "assistant")
        self.assertIn("tool_calls", second_request[-2])
        self.assertEqual(second_request[-1], {"role": "tool", "tool_name": "add", "content": "42"})
        # Tools are offered to the model on every request
        for call in client.stream_message.call_args_list:
            self.assertEqual(call.kwargs["tools"], self.registry.to_ollama_tools())

    @patch("builtins.input", side_effect=["Hello", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    def test_no_tool_call_answers_normally(self, mock_client_cls, mock_input):
        client = self.mock_client(mock_client_cls, [[text_chunk("Hi"), text_chunk(" there!")]])

        output = self.run_cli()

        self.assertIn("Assistant: Hi there!", output)
        self.assertNotIn("[tool]", output)
        self.assertEqual(client.stream_message.call_count, 1)

    @patch("builtins.input", side_effect=["Add things", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    def test_chained_tool_calls(self, mock_client_cls, mock_input):
        client = self.mock_client(mock_client_cls, [
            [tool_call_chunk("add", {"a": 1, "b": 2})],
            [tool_call_chunk("add", {"a": 3, "b": 4})],
            [text_chunk("Done.")],
        ])

        output = self.run_cli()

        self.assertIn("[tool] result: 3", output)
        self.assertIn("[tool] result: 7", output)
        self.assertIn("Assistant: Done.", output)
        self.assertEqual(client.stream_message.call_count, 3)

    @patch("builtins.input", side_effect=["Loop forever", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    def test_iteration_limit_stops_the_loop(self, mock_client_cls, mock_input):
        client = self.mock_client(mock_client_cls, [
            [tool_call_chunk("add", {"a": 1, "b": 1})] for _ in range(MAX_TOOL_ITERATIONS)
        ])

        output = self.run_cli()

        self.assertEqual(client.stream_message.call_count, MAX_TOOL_ITERATIONS)
        self.assertIn(f"stopped after {MAX_TOOL_ITERATIONS} tool calls", output)

    @patch("builtins.input", side_effect=["Use a tool", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    def test_unknown_tool_error_is_sent_back_to_the_model(self, mock_client_cls, mock_input):
        self.mock_client(mock_client_cls, [
            [tool_call_chunk("missing", {})],
            [text_chunk("Sorry, I cannot do that.")],
        ])

        output = self.run_cli()

        self.assertIn("[tool] error: Unknown tool 'missing'", output)
        self.assertIn("Unknown tool 'missing'", self.histories[1][-1]["content"])
        self.assertIn("Assistant: Sorry, I cannot do that.", output)

    @patch("builtins.input", side_effect=["Add", "Hello", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    def test_failed_turn_rolls_back_the_whole_turn(self, mock_client_cls, mock_input):
        def failing_second_request():
            raise OllamaConnectionError("Connection lost")
            yield  # pragma: no cover

        self.mock_client(mock_client_cls, [
            [tool_call_chunk("add", {"a": 1, "b": 2})],
            failing_second_request(),
            [text_chunk("Hi")],
        ])

        output = self.run_cli()

        self.assertIn("Connection Error", output)
        # The next turn starts from a clean history: system + the new user message only
        self.assertEqual([m["role"] for m in self.histories[2]], ["system", "user"])
        self.assertEqual(self.histories[2][-1]["content"], "Hello")


class TestCLIRAGIntegration(unittest.TestCase):
    @patch("cli.cli.OllamaHTTPClient")
    def test_cli_adds_rag_context_only_when_rag_is_enabled(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.stream_message.return_value = [b'{"message": {"content": "Grounded answer"}}']
        context_provider = MagicMock()
        context_provider.prepare.return_value = PreparedContext("RAG source context", [])
        captured_output = io.StringIO()

        with patch("builtins.input", side_effect=["Question", "exit"]), patch("sys.stdout", captured_output):
            cli(rag_enabled=True, rag_context_provider=context_provider, rag_indexer=MagicMock())

        sent_messages = mock_client.stream_message.call_args.args[0]
        self.assertEqual(sent_messages[1]["content"], "RAG source context")
        self.assertEqual(mock_client.stream_message.call_args.kwargs["tools"], [])
        context_provider.prepare.assert_called_once_with("Question")

    @patch("cli.cli.OllamaHTTPClient")
    def test_cli_routes_a_time_request_to_the_tool_when_rag_is_enabled(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.stream_message.return_value = [b'{"message": {"content": "Il est 10:00."}}']
        context_provider = MagicMock()

        with patch("builtins.input", side_effect=["Quelle heure est-il ?", "exit"]), patch(
            "sys.stdout", io.StringIO()
        ):
            cli(rag_enabled=True, rag_context_provider=context_provider, rag_indexer=MagicMock())

        context_provider.prepare.assert_not_called()
        offered_tools = mock_client.stream_message.call_args.kwargs["tools"]
        self.assertEqual([tool["function"]["name"] for tool in offered_tools], ["get_time"])

    @patch("cli.cli.OllamaHTTPClient")
    def test_cli_displays_sources_used_for_a_rag_answer(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.stream_message.return_value = [b'{"message": {"content": "Grounded answer"}}']
        context_provider = MagicMock()
        context_provider.prepare.return_value = PreparedContext(
            "RAG source context",
            [SearchResult("doc-1", "chunk-1", "guide.md", 0, "Evidence", 0.95)],
        )
        captured_output = io.StringIO()

        with patch("builtins.input", side_effect=["Question", "exit"]), patch("sys.stdout", captured_output):
            cli(rag_enabled=True, rag_context_provider=context_provider, rag_indexer=MagicMock())

        self.assertIn("Assistant: Grounded answer", captured_output.getvalue())
        self.assertIn("Sources: guide.md (chunks: chunk-1)", captured_output.getvalue())

if __name__ == "__main__":
    unittest.main()
