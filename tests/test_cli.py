import json
import unittest
from unittest.mock import patch, MagicMock
import io
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

# pyrefly: ignore [missing-import]
from cli.cli import CliDependencies, MAX_TOOL_ITERATIONS, _dispatch_command, cli
from mcp.manager import MCPManager
from config import AppConfig, FeatureConfig, PersonaConfig, PersonasConfig
from cli.commands import GLOBAL_COMMANDS
from cli.commands.artifact import ARTIFACT_COMMANDS
from cli.commands.rag import (
    _handle_rag_command,
    _handle_rag_filter_command,
    _handle_rag_reset_command,
    _handle_sources_command,
)
from cli.session import CliCommandContext, CliSession, CommandDefinition
# pyrefly: ignore [missing-import]
from tools.base import Tool
from memory.artifact_manager import ArtifactManager
from memory.session_manager import SessionManager
# pyrefly: ignore [missing-import]
from tools.registry import ToolRegistry
from tools.time_tool import GetTimeTool
# pyrefly: ignore [missing-import]
from utilities.ollama_client import (
    OllamaConnectionError,
    OllamaModelNotFoundError,
)
from rag.context import ArchiveIndexingResult, IndexingResult, PreparedContext
from rag.ingestion import ArchiveEntryFailure, ArchiveIngestionResult, DocumentIngestionError, IngestedDocument
from rag.storage import SearchResult


def setUpModule():
    # The CLI would start the servers declared in the repository's mcp_servers.json.
    patcher = patch("cli.cli.MCPManager.from_config_file", return_value=MCPManager([]))
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


class TestCLIPromptInjection(unittest.TestCase):
    """Tests for ProfileManager integration and system prompt injection."""

    @patch("builtins.input", side_effect=["exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    @patch("cli.cli.ProfileManager")
    def test_cli_injects_profile_system_prompt_on_new_session(
        self, mock_pm_cls, mock_sm_cls, mock_client_cls, mock_input
    ):
        mock_pm = MagicMock()
        mock_pm.build_system_prompt.return_value = "Custom Profile System Prompt"
        mock_pm_cls.return_value = mock_pm

        mock_sm = MagicMock()
        mock_sm.load.return_value = None
        mock_sm_cls.return_value = mock_sm

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        with patch("sys.stdout", io.StringIO()):
            cli(
                profile_path="custom_profile.json",
                app_config=AppConfig(
                    personas=PersonasConfig(
                        active_persona="math-assistant",
                        profiles=(
                            PersonaConfig(
                                identifier="math-assistant",
                                display_name="Mathematics assistant",
                                role="Supports mathematics teachers",
                                tone="Precise and practical",
                                system_prompt="Configured persona prompt",
                                response_scope="Lesson preparation.",
                            ),
                        ),
                    )
                ),
            )

        mock_pm_cls.assert_called_once_with(profile_path="custom_profile.json")
        configured_instructions = mock_pm.build_system_prompt.call_args.args[0]
        self.assertIn("Mathematics assistant", configured_instructions)
        self.assertIn("Configured persona prompt", configured_instructions)


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
        self.assertIn("Connection error: unable to reach Ollama.", output)
        self.assertIn("Make sure Ollama is running", output)
        self.assertNotIn("Connection refused", output)
        self.assertIn("Connection error: unable to reach Ollama.", output)
        self.assertIn("Make sure Ollama is running", output)
        self.assertNotIn("Connection refused", output)

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
        self.assertIn("Model error: the configured Ollama model was not found.", output)
        self.assertIn("Model error: the configured Ollama model was not found.", output)

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

        self.assertIn("Ollama could not complete this request. Please try again.", captured_output.getvalue())
        self.assertIn("Ollama could not complete this request. Please try again.", captured_output.getvalue())

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
        self.assertIn("Connection error: unable to reach Ollama.", output)
        self.assertNotIn("Connection lost mid-stream", output)

    @patch("builtins.input", side_effect=["Hello", "Hello again", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    def test_cli_unexpected_error_is_hidden_and_the_next_turn_still_works(
        self, mock_client_cls, _mock_input
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.stream_message.side_effect = [
            RuntimeError("sensitive implementation detail"),
            [b'{"message": {"content": "Recovered"}}'],
        ]

        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            cli()

        output = captured_output.getvalue()
        self.assertIn("An unexpected error occurred and was handled safely.", output)
        self.assertNotIn("sensitive implementation detail", output)
        self.assertIn("Assistant: Recovered", output)
        self.assertIn("Connection error: unable to reach Ollama.", output)
        self.assertNotIn("Connection lost mid-stream", output)

    @patch("builtins.input", side_effect=["Hello", "Hello again", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    def test_cli_unexpected_error_is_hidden_and_the_next_turn_still_works(
        self, mock_client_cls, _mock_input
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.stream_message.side_effect = [
            RuntimeError("sensitive implementation detail"),
            [b'{"message": {"content": "Recovered"}}'],
        ]

        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            cli()

        output = captured_output.getvalue()
        self.assertIn("An unexpected error occurred and was handled safely.", output)
        self.assertNotIn("sensitive implementation detail", output)
        self.assertIn("Assistant: Recovered", output)

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

    @patch("cli.commands.rag.ask_confirmation", return_value=True)
    def test_rag_reset_requires_confirmation_and_clears_the_index(self, _confirmation):
        indexer = MagicMock()
        indexer.clear_index.return_value = 4
        captured_output = io.StringIO()

        with patch("sys.stdout", captured_output):
            reset = _handle_rag_reset_command("/rag reset", indexer)

        self.assertTrue(reset)
        indexer.clear_index.assert_called_once_with()
        self.assertIn("RAG index reset: 4 chunks deleted.", captured_output.getvalue())

    @patch("cli.commands.rag.ask_confirmation", return_value=False)
    def test_rag_reset_cancellation_keeps_the_index(self, _confirmation):
        indexer = MagicMock()
        captured_output = io.StringIO()

        with patch("sys.stdout", captured_output):
            reset = _handle_rag_reset_command("/rag reset", indexer)

        self.assertFalse(reset)
        indexer.clear_index.assert_not_called()
        self.assertIn("RAG index reset cancelled.", captured_output.getvalue())

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

    @patch("cli.cli.OllamaHTTPClient")
    def test_cli_ingest_multiple_indexes_quoted_paths_and_continues_after_a_failure(self, mock_client_cls):
        documents = [
            IngestedDocument("document-1", "first.md", "/tmp/first.md", "First", 5),
            IngestedDocument("document-2", "third.md", "/tmp/third.md", "Third", 5),
        ]
        indexer = MagicMock()
        indexer.index_document.side_effect = [
            IndexingResult(documents[0], chunk_count=1, inserted_chunk_count=1),
            DocumentIngestionError("missing file"),
            IndexingResult(documents[1], chunk_count=1, inserted_chunk_count=1),
        ]
        captured_output = io.StringIO()

        with patch(
            "builtins.input",
            side_effect=["/ingest --multiple first.md\t\"docs/second guide.md\" third.md", "exit"],
        ), patch("sys.stdout", captured_output):
            cli(dependencies=CliDependencies(rag_indexer=indexer))

        self.assertEqual(
            [call.args[0] for call in indexer.index_document.call_args_list],
            ["first.md", "docs/second guide.md", "third.md"],
        )
        self.assertIn("Document error: the file could not be ingested.", captured_output.getvalue())
        self.assertIn("Document error: the file could not be ingested.", captured_output.getvalue())
        self.assertIn("Multiple ingestion complete: 2/3 paths indexed.", captured_output.getvalue())


class TestCLICommands(unittest.TestCase):
    def test_command_dispatcher_uses_the_first_matching_command(self):
        handled_commands = []
        command_definitions = (
            CommandDefinition(
                "/rag filter",
                "/rag filter <...>",
                "Filter RAG retrieval.",
                "RAG",
                lambda command, _session, _context: handled_commands.append(command),
            ),
            CommandDefinition(
                "/rag",
                "/rag <...>",
                "Configure RAG.",
                "RAG",
                lambda command, _session, _context: handled_commands.append(f"generic:{command}"),
            ),
        )

        session = MagicMock()
        context = MagicMock()
        self.assertTrue(_dispatch_command("/rag filter category code", session, context, command_definitions))
        self.assertFalse(_dispatch_command("/ragged", session, context, command_definitions))
        self.assertFalse(_dispatch_command("Hello", session, context, command_definitions))
        self.assertEqual(handled_commands, ["/rag filter category code"])

    @patch("builtins.input", side_effect=["/help", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    def test_help_is_generated_from_the_registered_commands(self, mock_client_cls, _mock_input):
        captured_output = io.StringIO()

        with patch("sys.stdout", captured_output):
            cli()

        output = captured_output.getvalue()
        self.assertIn("Available commands:", output)
        self.assertIn("CLI:", output)
        self.assertIn("RAG:", output)
        for command in GLOBAL_COMMANDS:
            self.assertIn(command.usage, output)
        mock_client_cls.return_value.stream_message.assert_not_called()

    @patch("builtins.input", side_effect=["/metric", "/raggg", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    def test_unknown_slash_commands_show_help_without_contacting_the_model(
        self,
        mock_client_cls,
        _mock_input,
    ):
        captured_output = io.StringIO()

        with patch("sys.stdout", captured_output):
            cli()

        output = captured_output.getvalue()
        self.assertIn("Unknown command: /metric", output)
        self.assertIn("Did you mean: /metrics?", output)
        self.assertIn("Unknown command: /raggg", output)
        self.assertIn("Did you mean: /rag?", output)
        self.assertIn("Type /help to list available commands.", output)
        mock_client_cls.return_value.stream_message.assert_not_called()

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


class TestPersonaCommands(unittest.TestCase):
    def setUp(self):
        self.teacher = PersonaConfig(
            identifier="teacher-assistant",
            display_name="Teacher assistant",
            role="Supports teachers",
            tone="Practical",
            system_prompt="Teacher prompt",
            response_scope="Teaching support.",
        )
        self.mathematics = PersonaConfig(
            identifier="mathematics-assistant",
            display_name="Mathematics assistant",
            role="Supports mathematics teachers",
            tone="Precise",
            system_prompt="Mathematics prompt",
            response_scope="Mathematics teaching support.",
        )
        self.personas = PersonasConfig(
            active_persona=self.teacher.identifier,
            profiles=(self.teacher, self.mathematics),
        )

    def test_persona_commands_list_status_and_switch_without_erasing_history(self):
        profile_manager = MagicMock()
        profile_manager.build_system_prompt.side_effect = lambda prompt: f"Profile: {prompt}"
        messages = [
            {"role": "system", "content": "Profile: Teacher prompt"},
            {"role": "user", "content": "Existing question"},
            {"role": "assistant", "content": "Existing answer"},
        ]
        context = CliCommandContext(
            feature_config=FeatureConfig(),
            rag_indexer=MagicMock(),
            conversation_session=MagicMock(),
            command_definitions=GLOBAL_COMMANDS,
            personas=self.personas,
            profile_manager=profile_manager,
            messages=messages,
        )
        session = CliSession(active_persona_id=self.teacher.identifier)
        output = io.StringIO()

        with patch("sys.stdout", output):
            self.assertTrue(_dispatch_command("/persona list", session, context, GLOBAL_COMMANDS))
            self.assertTrue(_dispatch_command("/persona status", session, context, GLOBAL_COMMANDS))
            self.assertTrue(
                _dispatch_command("/persona use mathematics-assistant", session, context, GLOBAL_COMMANDS)
            )
            self.assertTrue(_dispatch_command("/persona use unknown", session, context, GLOBAL_COMMANDS))

        self.assertEqual(session.active_persona_id, self.mathematics.identifier)
        self.assertIn("Mathematics prompt", messages[0]["content"])
        self.assertIn("Supports mathematics teachers", messages[0]["content"])
        self.assertEqual(messages[1:], [
            {"role": "user", "content": "Existing question"},
            {"role": "assistant", "content": "Existing answer"},
        ])
        self.assertIn("Available personas:", output.getvalue())
        self.assertIn("Active persona: Teacher assistant", output.getvalue())
        self.assertIn("Unknown persona: unknown", output.getvalue())

    @patch("builtins.input", side_effect=["/persona use mathematics-assistant", "Hello", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    @patch("cli.cli.ProfileManager")
    def test_next_model_request_uses_the_runtime_selected_persona(
        self,
        profile_manager_class,
        session_manager_class,
        client_class,
        _mock_input,
    ):
        profile_manager = MagicMock()
        profile_manager.build_system_prompt.side_effect = lambda prompt: f"Profile: {prompt}"
        profile_manager_class.return_value = profile_manager
        session_manager_class.return_value.load.return_value = None
        client_class.return_value.stream_message.return_value = [
            b'{"message": {"content": "Hello."}}'
        ]
        configuration = AppConfig(personas=self.personas)
        mcp_manager = MagicMock()
        mcp_manager.start.return_value = []
        mcp_manager.register_tools.return_value = []

        with patch("sys.stdout", io.StringIO()):
            cli(app_config=configuration, mcp_manager=mcp_manager)

        request_messages = client_class.return_value.stream_message.call_args.args[0]
        self.assertIn("Profile: PERSONA:", request_messages[0]["content"])
        self.assertIn("Mathematics prompt", request_messages[0]["content"])
        self.assertEqual(request_messages[1], {"role": "user", "content": "Hello"})

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

        self.assertIn("Connection error: unable to reach Ollama.", output)
        self.assertIn("Connection error: unable to reach Ollama.", output)
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

    def run_rag_cli(self, mock_client_cls, user_input, registry):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.stream_message.return_value = [b'{"message": {"content": "Answer"}}']
        context_provider = MagicMock()
        context_provider.prepare.return_value = PreparedContext("RAG source context", [])

        with patch("builtins.input", side_effect=["/rag on", user_input, "exit"]), patch(
            "sys.stdout", io.StringIO()
        ):
            cli(
                dependencies=CliDependencies(
                    registry=registry,
                    rag_context_provider=context_provider,
                    rag_indexer=MagicMock(),
                ),
            )
        return mock_client, context_provider

    @patch("cli.cli.OllamaHTTPClient")
    def test_rag_still_runs_when_a_tool_without_routing_rule_is_offered(self, mock_client_cls):
        registry = ToolRegistry()
        registry.register(AddTool())

        mock_client, context_provider = self.run_rag_cli(mock_client_cls, "What does my course say?", registry)

        context_provider.prepare.assert_called_once_with("What does my course say?")
        sent_messages = mock_client.stream_message.call_args.args[0]
        self.assertTrue(sent_messages[1]["content"].startswith("RAG source context"))
        self.assertIn("Tools are available", sent_messages[1]["content"])
        offered_tools = mock_client.stream_message.call_args.kwargs["tools"]
        self.assertEqual([tool["function"]["name"] for tool in offered_tools], ["add"])

    @patch("cli.cli.OllamaHTTPClient")
    def test_tool_note_is_not_added_when_no_tool_is_offered(self, mock_client_cls):
        mock_client, _ = self.run_rag_cli(mock_client_cls, "What does my course say?", ToolRegistry())

        sent_messages = mock_client.stream_message.call_args.args[0]
        self.assertEqual(sent_messages[1]["content"], "RAG source context")

    @patch("cli.cli.OllamaHTTPClient")
    def test_claimed_request_skips_rag_even_with_a_default_tool(self, mock_client_cls):
        registry = ToolRegistry()
        registry.register(AddTool())
        registry.register(GetTimeTool())

        mock_client, context_provider = self.run_rag_cli(mock_client_cls, "Quelle heure est-il ?", registry)

        context_provider.prepare.assert_not_called()
        offered_tools = mock_client.stream_message.call_args.kwargs["tools"]
        self.assertEqual([tool["function"]["name"] for tool in offered_tools], ["add", "get_time"])

    @patch("cli.cli.OllamaHTTPClient")
    def test_non_matching_request_with_time_tool_still_uses_rag(self, mock_client_cls):
        registry = ToolRegistry()
        registry.register(GetTimeTool())

        _, context_provider = self.run_rag_cli(mock_client_cls, "What is the time complexity of quicksort?", registry)

        context_provider.prepare.assert_called_once()

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

class TestCLICommandDispatch(unittest.TestCase):
    """Tests for the command dispatch map (/save, /load, exit, quit)."""

    @patch("builtins.input", side_effect=["/save", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_slash_save_calls_session_save(self, mock_sm_cls, mock_client_cls, mock_input):
        mock_session = MagicMock()
        mock_session.load.return_value = None
        mock_sm_cls.return_value = mock_session
        mock_client_cls.return_value = MagicMock()

        with patch("sys.stdout", io.StringIO()):
            cli()

        # save() called at least once for /save (and once more for exit)
        self.assertGreaterEqual(mock_session.save.call_count, 1)

    @patch("builtins.input", side_effect=["exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_exit_command_saves_and_exits(self, mock_sm_cls, mock_client_cls, mock_input):
        mock_session = MagicMock()
        mock_session.load.return_value = None
        mock_sm_cls.return_value = mock_session
        mock_client_cls.return_value = MagicMock()

        with patch("sys.stdout", io.StringIO()):
            cli()

        mock_session.save.assert_called_once()

    @patch("builtins.input", side_effect=["quit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_quit_command_saves_and_exits(self, mock_sm_cls, mock_client_cls, mock_input):
        mock_session = MagicMock()
        mock_session.load.return_value = None
        mock_sm_cls.return_value = mock_session
        mock_client_cls.return_value = MagicMock()

        with patch("sys.stdout", io.StringIO()):
            cli()

    @patch("builtins.input", side_effect=["/load", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_slash_load_without_args_prints_usage(self, mock_sm_cls, mock_client_cls, mock_input):
        mock_session = MagicMock()
        mock_session.load.return_value = None
        mock_sm_cls.return_value = mock_session
        mock_client_cls.return_value = MagicMock()

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli()

        self.assertIn("Usage: /load", captured.getvalue())

    @patch("builtins.input", side_effect=["/load nonexistent-id", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_slash_load_with_missing_session_prints_error(self, mock_sm_cls, mock_client_cls, mock_input):
        mock_session = MagicMock()
        mock_session.load.side_effect = FileNotFoundError("Session nonexistent-id not found")
        mock_sm_cls.return_value = mock_session
        mock_client_cls.return_value = MagicMock()

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli()

        self.assertIn("File error: the requested file or session could not be found.", captured.getvalue())
        self.assertNotIn("nonexistent-id", captured.getvalue())
        self.assertIn("File error: the requested file or session could not be found.", captured.getvalue())
        self.assertNotIn("nonexistent-id", captured.getvalue())

    @patch("builtins.input", side_effect=["/load valid-id", "Follow-up question", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_slash_load_restores_conversation_context_for_next_turn(
        self, mock_sm_cls, mock_client_cls, mock_input
    ):
        mock_session = MagicMock()
        mock_session.load.return_value = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        mock_sm_cls.return_value = mock_session
        mock_client = MagicMock()
        mock_client.stream_message.return_value = [
            b'{"message": {"content": "Follow-up answer"}}'
        ]
        mock_client_cls.return_value = mock_client

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli()

        self.assertIn("Session valid-id loaded.", captured.getvalue())
        sent_messages = mock_client.stream_message.call_args.args[0]
        self.assertEqual(sent_messages[0]["role"], "system")
        self.assertEqual(
            sent_messages[1:],
            [
                {"role": "user", "content": "Previous question"},
                {"role": "assistant", "content": "Previous answer"},
                {"role": "user", "content": "Follow-up question"},
                {"role": "assistant", "content": "Follow-up answer"},
            ],
        )

    @patch("builtins.input", side_effect=["/save", "/load some-id", "/tokens", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_memory_commands_are_blocked_when_memory_is_disabled(
        self, mock_sm_cls, mock_client_cls, mock_input
    ):
        mock_session = MagicMock()
        mock_session.load.return_value = None
        mock_sm_cls.return_value = mock_session
        mock_client_cls.return_value = MagicMock()

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli(app_config=AppConfig(features=FeatureConfig(memory_enabled=False)))

        self.assertEqual(captured.getvalue().count("Memory is disabled in config.json."), 3)
        mock_session.save.assert_not_called()
        mock_session.load.assert_not_called()

    @patch("builtins.input", side_effect=["First question", "Second question", "/tokens", "/tokens extra", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_tokens_command_tracks_per_turn_and_cumulative_usage(
        self, mock_sm_cls, mock_client_cls, mock_input
    ):
        mock_session = MagicMock()
        mock_session.load.return_value = None
        mock_sm_cls.return_value = mock_session
        mock_client = MagicMock()
        mock_client.stream_message.side_effect = [
            [
                b'{"message": {"content": "First answer"}, "done": true, "prompt_eval_count": 100, "eval_count": 25}'
            ],
            [
                b'{"message": {"content": "Second answer"}, "done": true, "prompt_eval_count": 150, "eval_count": 50}'
            ],
        ]
        mock_client_cls.return_value = mock_client

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli()

        output = captured.getvalue()
        self.assertIn("- Last turn: 150 prompt + 50 completion = 200 tokens", output)
        self.assertIn("- Context window: 200 / 4096 tokens (4.9%)", output)
        self.assertIn("- Session total: 325 tokens (250 prompt, 75 completion)", output)
        self.assertIn("Usage: /tokens", output)

    @patch("builtins.input", side_effect=["/load valid-id", "Follow-up question", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_slash_load_restores_conversation_context_for_next_turn(
        self, mock_sm_cls, mock_client_cls, mock_input
    ):
        mock_session = MagicMock()
        mock_session.load.return_value = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        mock_sm_cls.return_value = mock_session
        mock_client = MagicMock()
        mock_client.stream_message.return_value = [
            b'{"message": {"content": "Follow-up answer"}}'
        ]
        mock_client_cls.return_value = mock_client

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli()

        self.assertIn("Session valid-id loaded.", captured.getvalue())
        sent_messages = mock_client.stream_message.call_args.args[0]
        self.assertEqual(sent_messages[0]["role"], "system")
        self.assertEqual(
            sent_messages[1:],
            [
                {"role": "user", "content": "Previous question"},
                {"role": "assistant", "content": "Previous answer"},
                {"role": "user", "content": "Follow-up question"},
                {"role": "assistant", "content": "Follow-up answer"},
            ],
        )

    @patch("builtins.input", side_effect=["/save", "/load some-id", "/tokens", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_memory_commands_are_blocked_when_memory_is_disabled(
        self, mock_sm_cls, mock_client_cls, mock_input
    ):
        mock_session = MagicMock()
        mock_session.load.return_value = None
        mock_sm_cls.return_value = mock_session
        mock_client_cls.return_value = MagicMock()

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli(app_config=AppConfig(features=FeatureConfig(memory_enabled=False)))

        self.assertEqual(captured.getvalue().count("Memory is disabled in config.json."), 3)
        mock_session.save.assert_not_called()
        mock_session.load.assert_not_called()

    @patch("builtins.input", side_effect=["First question", "Second question", "/tokens", "/tokens extra", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_tokens_command_tracks_per_turn_and_cumulative_usage(
        self, mock_sm_cls, mock_client_cls, mock_input
    ):
        mock_session = MagicMock()
        mock_session.load.return_value = None
        mock_sm_cls.return_value = mock_session
        mock_client = MagicMock()
        mock_client.stream_message.side_effect = [
            [
                b'{"message": {"content": "First answer"}, "done": true, "prompt_eval_count": 100, "eval_count": 25}'
            ],
            [
                b'{"message": {"content": "Second answer"}, "done": true, "prompt_eval_count": 150, "eval_count": 50}'
            ],
        ]
        mock_client_cls.return_value = mock_client

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli()

        output = captured.getvalue()
        self.assertIn("- Last turn: 150 prompt + 50 completion = 200 tokens", output)
        self.assertIn("- Context window: 200 / 4096 tokens (4.9%)", output)
        self.assertIn("- Session total: 325 tokens (250 prompt, 75 completion)", output)
        self.assertIn("Usage: /tokens", output)

    @patch("builtins.input", side_effect=["/load valid-id", "Follow-up question", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_slash_load_restores_conversation_context_for_next_turn(
        self, mock_sm_cls, mock_client_cls, mock_input
    ):
        mock_session = MagicMock()
        mock_session.load.return_value = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        mock_sm_cls.return_value = mock_session
        mock_client = MagicMock()
        mock_client.stream_message.return_value = [
            b'{"message": {"content": "Follow-up answer"}}'
        ]
        mock_client_cls.return_value = mock_client

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli()

        self.assertIn("Session valid-id loaded.", captured.getvalue())
        sent_messages = mock_client.stream_message.call_args.args[0]
        self.assertEqual(sent_messages[0]["role"], "system")
        self.assertEqual(
            sent_messages[1:],
            [
                {"role": "user", "content": "Previous question"},
                {"role": "assistant", "content": "Previous answer"},
                {"role": "user", "content": "Follow-up question"},
                {"role": "assistant", "content": "Follow-up answer"},
            ],
        )

    @patch("builtins.input", side_effect=["/save", "/load some-id", "/tokens", "/compress", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_memory_commands_are_blocked_when_memory_is_disabled(
        self, mock_sm_cls, mock_client_cls, mock_input
    ):
        mock_session = MagicMock()
        mock_session.load.return_value = None
        mock_sm_cls.return_value = mock_session
        mock_client_cls.return_value = MagicMock()

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli(app_config=AppConfig(features=FeatureConfig(memory_enabled=False)))

        self.assertEqual(captured.getvalue().count("Memory is disabled in config.json."), 4)
        mock_session.save.assert_not_called()
        mock_session.load.assert_not_called()

    @patch("builtins.input", side_effect=["First question", "Second question", "/tokens", "/tokens extra", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_tokens_command_tracks_per_turn_and_cumulative_usage(
        self, mock_sm_cls, mock_client_cls, mock_input
    ):
        mock_session = MagicMock()
        mock_session.load.return_value = None
        mock_sm_cls.return_value = mock_session
        mock_client = MagicMock()
        mock_client.stream_message.side_effect = [
            [
                b'{"message": {"content": "First answer"}, "done": true, "prompt_eval_count": 100, "eval_count": 25}'
            ],
            [
                b'{"message": {"content": "Second answer"}, "done": true, "prompt_eval_count": 150, "eval_count": 50}'
            ],
        ]
        mock_client_cls.return_value = mock_client

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli()

        output = captured.getvalue()
        self.assertIn("- Last turn: 150 prompt + 50 completion = 200 tokens", output)
        self.assertIn("- Context window: 200 / 4096 tokens (4.9%)", output)
        self.assertIn("- Session total: 325 tokens (250 prompt, 75 completion)", output)
        self.assertIn("Usage: /tokens", output)

    def test_cli_session_budget_warning_thresholds(self):
        session = CliSession(context_window=4096)
        session.record_turn_tokens(500, 200)
        self.assertAlmostEqual(session.context_usage_ratio, 700 / 4096)
        self.assertIsNone(session.get_budget_warning())

        session.record_turn_tokens(3277, 0)
        self.assertGreaterEqual(session.context_usage_ratio, 0.8)
        warning = session.get_budget_warning()
        self.assertIsNotNone(warning)
        self.assertIn("Warning: Context token usage is at 80.0% (3277/4096 tokens).", warning)

        session.record_turn_tokens(3000, 400)
        warning = session.get_budget_warning()
        self.assertIsNotNone(warning)
        self.assertIn("Warning: Context token usage is at 83.0% (3400/4096 tokens).", warning)
        self.assertIsNone(session.get_budget_warning(threshold=0.9))

    @patch("builtins.input", side_effect=["Heavy context turn", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_context_budget_warning_displayed_when_threshold_exceeded(
        self, mock_sm_cls, mock_client_cls, mock_input
    ):
        mock_session = MagicMock()
        mock_session.load.return_value = None
        mock_sm_cls.return_value = mock_session
        mock_client = MagicMock()
        mock_client.stream_message.return_value = [
            b'{"message": {"content": "Long answer"}, "done": true, "prompt_eval_count": 3000, "eval_count": 300}'
        ]
        mock_client_cls.return_value = mock_client

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli()

        output = captured.getvalue()
        self.assertIn("Warning: Context token usage is at 80.6% (3300/4096 tokens).", output)

    @patch("builtins.input", side_effect=["Light turn", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_context_budget_warning_not_displayed_when_under_threshold(
        self, mock_sm_cls, mock_client_cls, mock_input
    ):
        mock_session = MagicMock()
        mock_session.load.return_value = None
        mock_sm_cls.return_value = mock_session
        mock_client = MagicMock()
        mock_client.stream_message.return_value = [
            b'{"message": {"content": "Short answer"}, "done": true, "prompt_eval_count": 100, "eval_count": 25}'
        ]
        mock_client_cls.return_value = mock_client

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli()

        output = captured.getvalue()
        self.assertNotIn("Warning: Context token usage is at", output)

    @patch("builtins.input", side_effect=["Heavy turn without memory", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_context_budget_warning_not_displayed_when_memory_disabled(
        self, mock_sm_cls, mock_client_cls, mock_input
    ):
        mock_session = MagicMock()
        mock_session.load.return_value = None
        mock_sm_cls.return_value = mock_session
        mock_client = MagicMock()
        mock_client.stream_message.return_value = [
            b'{"message": {"content": "Long answer"}, "done": true, "prompt_eval_count": 3000, "eval_count": 300}'
        ]
        mock_client_cls.return_value = mock_client

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli(app_config=AppConfig(features=FeatureConfig(memory_enabled=False)))

        output = captured.getvalue()
        self.assertNotIn("Warning: Context token usage is at", output)

    @patch("builtins.input", side_effect=["/compress", "/compress extra", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_compress_command_when_not_enough_history(
        self, mock_sm_cls, mock_client_cls, mock_input
    ):
        mock_session = MagicMock()
        mock_session.load.return_value = None
        mock_sm_cls.return_value = mock_session
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli()

        output = captured.getvalue()
        self.assertIn("Compression skipped: Not enough dialogue turns to compress", output)
        self.assertIn("Usage: /compress", output)

    @patch("builtins.input", side_effect=["Q1", "Q2", "Q3", "/compress", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_compress_command_manual_success(
        self, mock_sm_cls, mock_client_cls, mock_input
    ):
        mock_session = MagicMock()
        mock_session.load.return_value = None
        mock_session.messages = []
        mock_sm_cls.return_value = mock_session
        mock_client = MagicMock()
        mock_client.stream_message.side_effect = [
            [b'{"message": {"content": "A1"}, "done": true, "prompt_eval_count": 50, "eval_count": 20}'],
            [b'{"message": {"content": "A2"}, "done": true, "prompt_eval_count": 70, "eval_count": 20}'],
            [b'{"message": {"content": "A3"}, "done": true, "prompt_eval_count": 90, "eval_count": 20}'],
        ]
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Summary of Q1 and A1 topics."}
        }
        mock_client.complete_message.return_value = mock_response
        mock_client_cls.return_value = mock_client

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli()

        output = captured.getvalue()
        self.assertIn("Context compressed successfully:", output)
        self.assertIn("Summary: Summary of Q1 and A1 topics.", output)

    @patch("builtins.input", side_effect=["Q1", "Q2", "Q3", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_auto_compression_triggered_when_budget_critical(
        self, mock_sm_cls, mock_client_cls, mock_input
    ):
        mock_session = MagicMock()
        mock_session.load.return_value = None
        mock_session.messages = []
        mock_sm_cls.return_value = mock_session
        mock_client = MagicMock()
        mock_client.stream_message.side_effect = [
            [b'{"message": {"content": "A1"}, "done": true, "prompt_eval_count": 50, "eval_count": 20}'],
            [b'{"message": {"content": "A2"}, "done": true, "prompt_eval_count": 70, "eval_count": 20}'],
            [b'{"message": {"content": "A3"}, "done": true, "prompt_eval_count": 3500, "eval_count": 100}'],
        ]
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Consolidated summary of prior turns."}
        }
        mock_client.complete_message.return_value = mock_response
        mock_client_cls.return_value = mock_client

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli()

        output = captured.getvalue()
        self.assertIn("Warning: Context token usage is at 87.9% (3600/4096 tokens).", output)
        self.assertIn("Auto-compression triggered: condensed older turns", output)

    @patch("builtins.input", side_effect=["/load valid-id", "Follow-up question", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_slash_load_restores_conversation_context_for_next_turn(
        self, mock_sm_cls, mock_client_cls, mock_input
    ):
        mock_session = MagicMock()
        mock_session.load.return_value = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        mock_sm_cls.return_value = mock_session
        mock_client = MagicMock()
        mock_client.stream_message.return_value = [
            b'{"message": {"content": "Follow-up answer"}}'
        ]
        mock_client_cls.return_value = mock_client

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli()

        self.assertIn("Session valid-id loaded.", captured.getvalue())
        sent_messages = mock_client.stream_message.call_args.args[0]
        self.assertEqual(sent_messages[0]["role"], "system")
        self.assertEqual(
            sent_messages[1:],
            [
                {"role": "user", "content": "Previous question"},
                {"role": "assistant", "content": "Previous answer"},
                {"role": "user", "content": "Follow-up question"},
                {"role": "assistant", "content": "Follow-up answer"},
            ],
        )

    @patch("builtins.input", side_effect=["/save", "/load some-id", "/tokens", "/compress", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_memory_commands_are_blocked_when_memory_is_disabled(
        self, mock_sm_cls, mock_client_cls, mock_input
    ):
        mock_session = MagicMock()
        mock_session.load.return_value = None
        mock_sm_cls.return_value = mock_session
        mock_client_cls.return_value = MagicMock()

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli(app_config=AppConfig(features=FeatureConfig(memory_enabled=False)))

        self.assertEqual(captured.getvalue().count("Memory is disabled in config.json."), 4)
        mock_session.save.assert_not_called()
        mock_session.load.assert_not_called()

    @patch("builtins.input", side_effect=["/compress", "/compress extra", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_compress_command_when_not_enough_history(
        self, mock_sm_cls, mock_client_cls, mock_input
    ):
        mock_session = MagicMock()
        mock_session.load.return_value = None
        mock_sm_cls.return_value = mock_session
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli()

        output = captured.getvalue()
        self.assertIn("Compression skipped: Not enough dialogue turns to compress", output)
        self.assertIn("Usage: /compress", output)

    @patch("builtins.input", side_effect=["Q1", "Q2", "Q3", "/compress", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_compress_command_manual_success(
        self, mock_sm_cls, mock_client_cls, mock_input
    ):
        mock_session = MagicMock()
        mock_session.load.return_value = None
        mock_session.messages = []
        mock_sm_cls.return_value = mock_session
        mock_client = MagicMock()
        mock_client.stream_message.side_effect = [
            [b'{"message": {"content": "A1"}, "done": true, "prompt_eval_count": 50, "eval_count": 20}'],
            [b'{"message": {"content": "A2"}, "done": true, "prompt_eval_count": 70, "eval_count": 20}'],
            [b'{"message": {"content": "A3"}, "done": true, "prompt_eval_count": 90, "eval_count": 20}'],
        ]
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Summary of Q1 and A1 topics."}
        }
        mock_client.complete_message.return_value = mock_response
        mock_client_cls.return_value = mock_client

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli()

        output = captured.getvalue()
        self.assertIn("Context compressed successfully:", output)
        self.assertIn("Summary: Summary of Q1 and A1 topics.", output)

    @patch("builtins.input", side_effect=["Q1", "Q2", "Q3", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_auto_compression_triggered_when_budget_critical(
        self, mock_sm_cls, mock_client_cls, mock_input
    ):
        mock_session = MagicMock()
        mock_session.load.return_value = None
        mock_session.messages = []
        mock_sm_cls.return_value = mock_session
        mock_client = MagicMock()
        mock_client.stream_message.side_effect = [
            [b'{"message": {"content": "A1"}, "done": true, "prompt_eval_count": 50, "eval_count": 20}'],
            [b'{"message": {"content": "A2"}, "done": true, "prompt_eval_count": 70, "eval_count": 20}'],
            [b'{"message": {"content": "A3"}, "done": true, "prompt_eval_count": 3500, "eval_count": 100}'],
        ]
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Consolidated summary of prior turns."}
        }
        mock_client.complete_message.return_value = mock_response
        mock_client_cls.return_value = mock_client

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cli()

        output = captured.getvalue()
        self.assertIn("Warning: Context token usage is at 87.9% (3600/4096 tokens).", output)
        self.assertIn("Auto-compression triggered: condensed older turns", output)

    @patch("builtins.input", side_effect=["", "   ", "exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_empty_input_is_ignored(self, mock_sm_cls, mock_client_cls, mock_input):
        mock_session = MagicMock()
        mock_session.load.return_value = None
        mock_sm_cls.return_value = mock_session
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        with patch("sys.stdout", io.StringIO()):
            cli()

        mock_client.stream_message.assert_not_called()


class TestCLISessionLoading(unittest.TestCase):
    """Tests for --session flag behaviour at startup."""

    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_invalid_session_id_at_startup_exits_with_error(self, mock_sm_cls, mock_client_cls):
        mock_session = MagicMock()
        mock_session.load.side_effect = FileNotFoundError("Session bad-id not found")
        mock_sm_cls.return_value = mock_session
        mock_client_cls.return_value = MagicMock()

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            with self.assertRaises(SystemExit) as ctx:
                cli(session_id="bad-id")
            self.assertEqual(ctx.exception.code, 1)

        self.assertIn("File error: the requested file or session could not be found.", captured.getvalue())
        self.assertIn("File error: the requested file or session could not be found.", captured.getvalue())

    @patch("builtins.input", side_effect=["exit"])
    @patch("cli.cli.OllamaHTTPClient")
    @patch("cli.cli.SessionManager")
    def test_valid_session_id_loads_messages(self, mock_sm_cls, mock_client_cls, mock_input):
        mock_session = MagicMock()
        mock_session.load.return_value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        mock_sm_cls.return_value = mock_session
        mock_client_cls.return_value = MagicMock()

        with patch("sys.stdout", io.StringIO()):
            cli(session_id="valid-id")

        mock_session.load.assert_called_with("valid-id")


class TestCLIMCPTools(unittest.TestCase):
    """Tests for MCP tool discovery wiring and the /tools command."""

    def run_cli(self, inputs, mcp_manager, registry=None):
        output = io.StringIO()
        with patch("builtins.input", side_effect=inputs), \
                patch("cli.cli.OllamaHTTPClient"), \
                patch("cli.cli.SessionManager"), \
                patch("cli.cli.atexit.register") as register, \
                patch("sys.stdout", output):
            cli(dependencies=CliDependencies(registry=registry or ToolRegistry()), mcp_manager=mcp_manager)
        return output.getvalue(), register

    def test_tools_command_lists_native_and_mcp_tools(self):
        # pyrefly: ignore [missing-import]
        from mcp.manager import MCPManager, MCPTool
        registry = ToolRegistry()
        registry.register(AddTool())
        manager = MagicMock(spec=MCPManager)
        manager.start.return_value = []
        remote = MCPTool("alpha", MagicMock(), {"name": "echo", "description": "Echo a text."})
        manager.register_tools.side_effect = lambda reg: reg.register(remote) or []

        output, _ = self.run_cli(["/tools", "exit"], manager, registry)

        self.assertIn("- add [native]: Add two integers.", output)
        self.assertIn("- alpha__echo [alpha]: Echo a text.", output)

    def test_mcp_warnings_are_displayed_and_do_not_stop_the_cli(self):
        # pyrefly: ignore [missing-import]
        from mcp.manager import MCPManager
        manager = MagicMock(spec=MCPManager)
        manager.start.return_value = ["MCP server 'broken' unavailable: boom"]
        manager.register_tools.return_value = ["Skipping MCP tool 'x': duplicate"]

        output, _ = self.run_cli(["exit"], manager)

        self.assertIn("Warning: MCP server 'broken' unavailable: boom", output)
        self.assertIn("Warning: Skipping MCP tool 'x': duplicate", output)

    def test_mcp_servers_are_closed_at_exit(self):
        # pyrefly: ignore [missing-import]
        from mcp.manager import MCPManager
        manager = MagicMock(spec=MCPManager)
        manager.start.return_value = []
        manager.register_tools.return_value = []

        _, register = self.run_cli(["exit"], manager)

        register.assert_any_call(manager.close)


class TestCLIArtifactCommands(unittest.TestCase):
    """Tests for /artifact and /doc CLI commands."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        from tools.permissions import AllowedDirectory, PermissionChecker

        self.temp_dir = tempfile.mkdtemp()
        self.root_path = Path(self.temp_dir).resolve()
        self.export_dir = self.root_path / "exports"
        self.export_dir.mkdir()
        self.readonly_dir = self.root_path / "readonly"
        self.readonly_dir.mkdir()

        self.checker = PermissionChecker(
            [
                AllowedDirectory(self.export_dir, ("read", "write")),
                AllowedDirectory(self.readonly_dir, ("read",)),
            ],
            base_dir=self.root_path,
        )
        self.artifact_manager = ArtifactManager(session_id="test_cli", storage_dir=self.temp_dir)
        self.conversation_session = MagicMock()
        self.context = CliCommandContext(
            feature_config=FeatureConfig(),
            rag_indexer=MagicMock(),
            conversation_session=self.conversation_session,
            artifact_manager=self.artifact_manager,
            permission_checker=self.checker,
        )
        self.session = CliSession()
        self.handler = ARTIFACT_COMMANDS[0].handler

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_artifact_view_empty(self):
        self.handler("/artifact", self.session, self.context)
        self.handler("/artifact view", self.session, self.context)

    def test_artifact_init_and_outline(self):
        self.handler("/artifact init Project X", self.session, self.context)
        self.assertTrue(self.artifact_manager.has_artifact())
        self.assertEqual(self.artifact_manager.artifact.title, "Project X")

        self.artifact_manager.update_section("Section 1", "Content 1")
        self.handler("/artifact outline", self.session, self.context)
        self.assertIn("Section 1", self.artifact_manager.get_outline())

    def test_artifact_export(self):
        self.artifact_manager.init_document("Export Test")
        self.artifact_manager.update_section("Overview", "Content to export.")

        export_file = str(self.export_dir / "exported_doc.md")
        self.handler(f"/artifact export {export_file}", self.session, self.context)

        self.assertTrue(os.path.exists(export_file))
        with open(export_file, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("# Export Test", content)
        self.assertIn("Content to export.", content)

    def test_artifact_export_with_spaces_in_quotes(self):
        self.artifact_manager.init_document("Export Test")
        self.artifact_manager.update_section("Overview", "Content to export.")

        export_file = self.export_dir / "my notes.md"
        self.handler(f'/artifact export "{export_file}"', self.session, self.context)

        self.assertTrue(export_file.exists())
        self.assertIn("Content to export.", export_file.read_text(encoding="utf-8"))

    def test_artifact_export_permission_denied(self):
        self.artifact_manager.init_document("Export Test")
        readonly_file = self.readonly_dir / "doc.md"

        self.handler(f"/artifact export {readonly_file}", self.session, self.context)
        self.assertFalse(readonly_file.exists())

    def test_artifact_export_without_overwrite_refuses(self):
        self.artifact_manager.init_document("Export Test")
        export_file = self.export_dir / "existing.md"
        export_file.write_text("Original content", encoding="utf-8")

        self.handler(f"/artifact export {export_file}", self.session, self.context)
        self.assertEqual(export_file.read_text(encoding="utf-8"), "Original content")

        self.handler(f"/artifact export {export_file} --overwrite", self.session, self.context)
        self.assertIn("# Export Test", export_file.read_text(encoding="utf-8"))

    def test_artifact_reset(self):
        self.artifact_manager.init_document("To Reset")
        self.assertTrue(self.artifact_manager.has_artifact())

        self.handler("/artifact reset", self.session, self.context)
        self.assertFalse(self.artifact_manager.has_artifact())


class TestArtifactSessionLifecycle(unittest.TestCase):
    """Tests for saving and restoring artifacts across session /save and /load."""

    def setUp(self):
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        self.session_id = "lifecycle_test_session"
        self.conversation_session = SessionManager(session_id=self.session_id, storage_dir=self.temp_dir)
        self.artifact_manager = ArtifactManager(session_id=self.session_id, storage_dir=self.temp_dir)
        self.context = CliCommandContext(
            feature_config=FeatureConfig(),
            rag_indexer=MagicMock(),
            conversation_session=self.conversation_session,
            artifact_manager=self.artifact_manager,
            messages=[],
        )
        self.session = CliSession()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_session_with_artifact(self):
        from cli.commands.memory import _dispatch_save, _dispatch_load

        # 1. Initialize artifact and add content
        self.artifact_manager.init_document("Living Spec")
        self.artifact_manager.update_section("Architecture", "Clean Architecture pattern.")

        # 2. Add message and save session
        self.conversation_session.add_message("user", "Hello")
        _dispatch_save("/save", self.session, self.context)

        # Verify artifact file exists on disk
        artifact_file = os.path.join(self.temp_dir, f"{self.session_id}_artifact.md")
        self.assertTrue(os.path.exists(artifact_file))

        # 3. Reset in-memory artifact to simulate new session
        self.artifact_manager.reset()
        self.assertFalse(self.artifact_manager.has_artifact())

        # 4. Load session back
        _dispatch_load(f"/load {self.session_id}", self.session, self.context)

        # 5. Verify artifact is restored
        self.assertTrue(self.artifact_manager.has_artifact())
        self.assertEqual(self.artifact_manager.artifact.title, "Living Spec")
        self.assertEqual(self.artifact_manager.artifact.sections.get("Architecture"), "Clean Architecture pattern.")

    def test_exit_command_auto_saves_artifact(self):
        from cli.commands.system import _dispatch_exit

        self.artifact_manager.init_document("Exit Auto Save Doc")
        self.artifact_manager.update_section("Notes", "Auto save content.")
        self.conversation_session.add_message("user", "Hello before exit")

        _dispatch_exit("exit", self.session, self.context)

        artifact_file = os.path.join(self.temp_dir, f"{self.session_id}_artifact.md")
        self.assertTrue(os.path.exists(artifact_file))
        content = Path(artifact_file).read_text(encoding="utf-8")
        self.assertIn("# Exit Auto Save Doc", content)
        self.assertIn("Auto save content.", content)
        self.assertTrue(self.session.exit_requested)
        self.assertTrue(self.session.persistence_saved_on_exit)

    def test_slash_exit_command_auto_saves_artifact(self):
        from cli.commands.system import _dispatch_exit

        self.artifact_manager.init_document("Slash Exit Doc")
        self.artifact_manager.update_section("Notes", "Slash exit content.")

        _dispatch_exit("/exit", self.session, self.context)

        artifact_file = os.path.join(self.temp_dir, f"{self.session_id}_artifact.md")
        self.assertTrue(os.path.exists(artifact_file))
        content = Path(artifact_file).read_text(encoding="utf-8")
        self.assertIn("# Slash Exit Doc", content)
        self.assertTrue(self.session.exit_requested)
        self.assertTrue(self.session.persistence_saved_on_exit)


if __name__ == "__main__":
    unittest.main()
