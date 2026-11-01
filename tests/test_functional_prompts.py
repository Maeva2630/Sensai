"""Prompt-level workflows exercised entirely with deterministic fake Ollama replies.

These tests complement unit tests: each one follows the same commands a CLI user
would enter, while keeping CI independent from an installed model or network.
"""

import io
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli.cli import CliDependencies, cli
from mcp.manager import MCPManager
from support import use_temporary_working_directory_for_module
from rag.context import PreparedContext
from rag.storage import MetadataFilter, SearchResult


def response_chunk(text):
    return ('{"message": {"content": "' + text + '"}}').encode()


def tool_call_chunk(name, arguments):
    return (
        '{"message": {"content": "", "tool_calls": ['
        '{"function": {"name": "' + name + '", "arguments": ' + arguments + "}}]}}"
    ).encode()


def setUpModule():
    # The CLI would start the servers declared in the repository's mcp_servers.json.
    patcher = patch("cli.cli.MCPManager.from_config_file", return_value=MCPManager([]))
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)
    # cli() saves sessions/, profile.json and data/ in the current directory: not in the repository.
    use_temporary_working_directory_for_module()


class TestPromptWorkflows(unittest.TestCase):
    """Functional CLI scenarios grouped by user-visible SensAI features."""

    @patch("cli.cli.OllamaHTTPClient")
    def test_normal_chat_prompt_streams_a_reply(self, mock_client_class):
        chat_client = MagicMock()
        sent_requests = []

        def reply(messages, **_kwargs):
            sent_requests.append([message.copy() for message in messages])
            return [response_chunk("Hello from SensAI.")]

        chat_client.stream_message.side_effect = reply
        mock_client_class.return_value = chat_client

        output = self._run_cli(["Hello", "exit"])

        self.assertIn("Assistant: Hello from SensAI.", output)
        self.assertEqual(sent_requests[0][-1], {"role": "user", "content": "Hello"})

    @patch("cli.cli.OllamaHTTPClient")
    def test_time_prompt_uses_a_tool_before_the_final_reply(self, mock_client_class):
        chat_client = MagicMock()
        chat_client.stream_message.side_effect = [
            [tool_call_chunk("get_time", "{}")],
            [response_chunk("The time tool completed.")],
        ]
        mock_client_class.return_value = chat_client

        output = self._run_cli(["Quelle heure est-il ?", "exit"])

        self.assertIn("[tool] get_time({})", output)
        self.assertIn("[tool] result:", output)
        self.assertIn("Assistant: The time tool completed.", output)
        self.assertEqual(chat_client.stream_message.call_count, 2)

    @patch("cli.cli.OllamaHTTPClient")
    def test_rag_prompt_applies_filter_and_can_hide_sources(self, mock_client_class):
        chat_client = MagicMock()
        chat_client.stream_message.return_value = [response_chunk("The function is in parser.py.")]
        mock_client_class.return_value = chat_client
        context_provider = MagicMock()
        context_provider.prepare.return_value = PreparedContext(
            "Retrieved context for parser.py",
            [SearchResult("document-1", "chunk-1", "parser.py", 0, "def parse(): ...", 0.98, "python", "code", 1, 2)],
        )

        output = self._run_cli(
            ["/rag on", "/rag filter category code", "/sources off", "Where is the parser?", "exit"],
            dependencies=CliDependencies(
                rag_indexer=MagicMock(),
                rag_context_provider=context_provider,
            ),
        )

        context_provider.prepare.assert_called_once_with(
            "Where is the parser?", MetadataFilter(source_types=("code",))
        )
        request_messages = chat_client.stream_message.call_args.args[0]
        self.assertEqual(request_messages[1]["content"], "Retrieved context for parser.py")
        self.assertIn("RAG filter set to category: code", output)
        self.assertIn("Source display disabled.", output)
        self.assertIn("Assistant: The function is in parser.py.", output)
        self.assertNotIn("Sources:", output)

    @patch("cli.cli.OllamaHTTPClient")
    def test_rag_prompt_without_relevant_context_does_not_claim_a_source(self, mock_client_class):
        chat_client = MagicMock()
        chat_client.stream_message.return_value = [
            response_chunk("I cannot find this information in the indexed documents.")
        ]
        mock_client_class.return_value = chat_client
        context_provider = MagicMock()
        context_provider.prepare.return_value = PreparedContext(
            "No relevant indexed excerpts were found.", []
        )

        output = self._run_cli(
            ["/rag on", "What is the deployment port?", "exit"],
            dependencies=CliDependencies(
                rag_indexer=MagicMock(),
                rag_context_provider=context_provider,
            ),
        )

        self.assertIn("I cannot find this information in the indexed documents.", output)
        self.assertNotIn("Sources:", output)

    @staticmethod
    def _run_cli(inputs, **kwargs):
        captured_output = io.StringIO()
        with patch("builtins.input", side_effect=inputs), patch("sys.stdout", captured_output):
            cli(**kwargs)
        return captured_output.getvalue()


if __name__ == "__main__":
    unittest.main()
