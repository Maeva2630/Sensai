import json
import unittest
from unittest.mock import patch, MagicMock
import io
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

# pyrefly: ignore [missing-import]
from cli.cli import CliDependencies, MAX_TOOL_ITERATIONS, _dispatch_command, cli
from cli.commands.rag import (
    _handle_rag_command,
    _handle_rag_filter_command,
    _handle_sources_command,
)
# pyrefly: ignore [missing-import]
from tools.base import Tool
# pyrefly: ignore [missing-import]
from tools.registry import ToolRegistry
# pyrefly: ignore [missing-import]
from utilities.ollama_client import (
    OllamaConnectionError,
    OllamaModelNotFoundError,
)
from rag.context import ArchiveIndexingResult, IndexingResult, PreparedContext
from rag.ingestion import ArchiveEntryFailure, ArchiveIngestionResult, IngestedDocument
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

    @patch("builtins.input", side_effect=["Hello", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    def test_cli_malformed_stream_response_displays_warning_without_crashing(
        self, mock_client_cls, mock_input
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.stream_message.return_value = [b"this is not JSON"]

        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            cli()

        self.assertIn("Warning: Ollama returned an invalid streaming response.", captured_output.getvalue())

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
            cli(dependencies=CliDependencies(rag_indexer=indexer))

        output = captured_output.getvalue()
        self.assertIn("Document indexed: knowledge.txt (2 chunks; 2 new chunks stored).", output)
        self.assertIn("Document indexed: knowledge.txt (2 chunks; 0 new chunks stored).", output)
        mock_client_cls.return_value.stream_message.assert_not_called()

    @patch("cli.cli.OllamaHTTPClient")
    def test_cli_ingest_command_reports_archive_summary(self, mock_client_cls):
        document = IngestedDocument("document-id", "project.zip::guide.md", "archive::guide.md", "Guide", 5)
        archive = ArchiveIngestionResult(
            "project.zip",
            "/tmp/project.zip",
            [document],
            ["image.png"],
            [ArchiveEntryFailure("broken.pdf", "Could not read PDF")],
        )
        indexer = MagicMock()
        indexer.index_document.return_value = ArchiveIndexingResult(
            archive,
            [IndexingResult(document, chunk_count=1, inserted_chunk_count=1)],
        )
        captured_output = io.StringIO()
        with patch("builtins.input", side_effect=["/ingest project.zip", "exit"]), patch("sys.stdout", captured_output):
            cli(dependencies=CliDependencies(rag_indexer=indexer))

        output = captured_output.getvalue()
        self.assertIn("Archive indexed: project.zip (1 files indexed; 1 chunks; 1 new chunks stored).", output)
        self.assertIn("Indexed files: project.zip::guide.md", output)
        self.assertIn("Skipped unsupported files: image.png", output)
        self.assertIn("Failed to extract broken.pdf: Could not read PDF", output)

    def test_rag_filter_commands_set_and_clear_metadata_restrictions(self):
        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            metadata_filter = _handle_rag_filter_command("/rag filter name guide.md,notes.md", None)
            category_filter = _handle_rag_filter_command("/rag filter category code", metadata_filter)
            cleared_filter = _handle_rag_filter_command("/rag filter clear", category_filter)

        self.assertEqual(metadata_filter.source_filenames, ("guide.md", "notes.md"))
        self.assertEqual(category_filter.source_types, ("code",))
        self.assertIsNone(cleared_filter)
        self.assertIn("RAG metadata filter cleared.", captured_output.getvalue())

    @patch("cli.cli.OllamaHTTPClient")
    def test_cli_ingest_command_keeps_a_path_with_spaces_intact(self, mock_client_cls):
        indexer = MagicMock()
        document = IngestedDocument(
            "document-id", "my guide.md", "/tmp/my guide.md", "Knowledge", 9
        )
        indexer.index_document.return_value = IndexingResult(document, chunk_count=1, inserted_chunk_count=1)

        with patch("builtins.input", side_effect=["/ingest docs/my guide.md", "exit"]), patch(
            "sys.stdout", io.StringIO()
        ):
            cli(dependencies=CliDependencies(rag_indexer=indexer))

        indexer.index_document.assert_called_once_with("docs/my guide.md")


class TestCLICommands(unittest.TestCase):
    def test_command_dispatcher_uses_the_first_matching_prefix(self):
        handled_commands = []
        handlers = (
            ("/rag filter", lambda command, _session, _context: handled_commands.append(command)),
            (
                "/rag",
                lambda command, _session, _context: handled_commands.append(f"generic:{command}"),
            ),
        )

        session = MagicMock()
        context = MagicMock()
        self.assertTrue(_dispatch_command("/rag filter category code", session, context, handlers))
        self.assertFalse(_dispatch_command("Hello", session, context, handlers))
        self.assertEqual(handled_commands, ["/rag filter category code"])

    def test_rag_commands_change_state_and_invalid_options_preserve_it(self):
        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            enabled = _handle_rag_command("/rag on", False)
            disabled = _handle_rag_command("/rag off", enabled)
            unchanged = _handle_rag_command("/rag unexpected", disabled)

        self.assertTrue(enabled)
        self.assertFalse(disabled)
        self.assertFalse(unchanged)
        self.assertIn("RAG mode enabled", captured_output.getvalue())
        self.assertIn("Usage: /rag", captured_output.getvalue())

    def test_source_commands_change_state_and_report_status(self):
        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            hidden = _handle_sources_command("/sources off", True)
            unchanged = _handle_sources_command("/sources status", hidden)
            restored = _handle_sources_command("/sources on", unchanged)

        self.assertFalse(hidden)
        self.assertFalse(unchanged)
        self.assertTrue(restored)
        self.assertIn("Source display is disabled.", captured_output.getvalue())

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
            cli(dependencies=CliDependencies(registry=registry or self.registry))
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

        with patch("builtins.input", side_effect=["/rag on", "Question", "exit"]), patch("sys.stdout", captured_output):
            cli(
                dependencies=CliDependencies(
                    rag_context_provider=context_provider,
                    rag_indexer=MagicMock(),
                ),
            )

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

        with patch("builtins.input", side_effect=["/rag on", "Quelle heure est-il ?", "exit"]), patch(
            "sys.stdout", io.StringIO()
        ):
            cli(
                dependencies=CliDependencies(
                    rag_context_provider=context_provider,
                    rag_indexer=MagicMock(),
                ),
            )

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

        with patch("builtins.input", side_effect=["/rag on", "Question", "exit"]), patch("sys.stdout", captured_output):
            cli(
                dependencies=CliDependencies(
                    rag_context_provider=context_provider,
                    rag_indexer=MagicMock(),
                ),
            )

        self.assertIn("Assistant: Grounded answer", captured_output.getvalue())
        self.assertIn("Sources: guide.md (chunks: chunk-1)", captured_output.getvalue())
        self.assertIn("Grounded answer\n\nSources:", captured_output.getvalue())

    @patch("cli.cli.OllamaHTTPClient")
    def test_cli_can_hide_source_display_without_disabling_rag(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.stream_message.return_value = [b'{"message": {"content": "Grounded answer"}}']
        context_provider = MagicMock()
        context_provider.prepare.return_value = PreparedContext(
            "RAG source context",
            [SearchResult("doc-1", "chunk-1", "guide.md", 0, "Evidence", 0.95)],
        )
        captured_output = io.StringIO()

        with patch("builtins.input", side_effect=["/rag on", "/sources off", "Question", "exit"]), patch(
            "sys.stdout", captured_output
        ):
            cli(
                dependencies=CliDependencies(
                    rag_context_provider=context_provider,
                    rag_indexer=MagicMock(),
                ),
            )

        output = captured_output.getvalue()
        self.assertIn("Source display disabled.", output)
        self.assertIn("Assistant: Grounded answer", output)
        self.assertNotIn("Sources:", output)

if __name__ == "__main__":
    unittest.main()
