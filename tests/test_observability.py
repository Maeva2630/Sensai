"""Tests for file logging and explicit reconstructed-prompt inspection."""

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from support import use_temporary_working_directory
from cli.cli import cli
from cli.prompt_inspection import format_model_request
from config import LoggingConfig
from observability.app_logging import configure_logging, get_logger


def response_chunk(text):
    return ('{"message": {"content": "' + text + '"}}').encode()


def tool_call_chunk(name, arguments):
    return (
        '{"message": {"content": "", "tool_calls": ['
        '{"function": {"name": "' + name + '", "arguments": ' + arguments + "}}]}}"
    ).encode()


class TestApplicationLogging(unittest.TestCase):
    def test_logs_include_timestamp_level_and_component_without_console_output(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "sensai.log"
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                logger = configure_logging(LoggingConfig(level="DEBUG", file_path=str(log_path)))
                logger.debug("event=test_event")

            content = log_path.read_text(encoding="utf-8")
            self.assertIn("DEBUG | sensai | event=test_event", content)
            self.assertEqual(stderr.getvalue(), "")

    def test_child_component_logs_use_the_configured_file(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "sensai.log"
            configure_logging(LoggingConfig(level="INFO", file_path=str(log_path)))
            get_logger("tools").info("tool.completed name=get_time")

            self.assertIn("sensai.tools | tool.completed name=get_time", log_path.read_text(encoding="utf-8"))


class TestPromptInspection(unittest.TestCase):
    def test_prompt_formatter_identifies_rag_context(self):
        output = format_model_request(
            [
                {"role": "system", "content": "Base instructions"},
                {"role": "system", "content": "Retrieved context"},
                {"role": "user", "content": "Question"},
            ],
            "Retrieved context",
        )

        self.assertIn("System message", output)
        self.assertIn("RAG rules", output)
        self.assertIn("User message", output)
        self.assertNotIn("Base instructions", output)

    def test_prompt_formatter_displays_system_content_only_when_explicitly_requested(self):
        messages = [{"role": "system", "content": "Private profile instructions"}]

        hidden_output = format_model_request(messages, None)
        visible_output = format_model_request(messages, None, include_system=True)

        self.assertIn("<hidden; use /debug prompt --include-system to display it>", hidden_output)
        self.assertNotIn("Private profile instructions", hidden_output)
        self.assertIn("Private profile instructions", visible_output)

    @patch("cli.cli.OllamaHTTPClient")
    def test_debug_prompt_displays_user_assistant_and_tool_messages(self, mock_client_class):
        client = MagicMock()
        client.stream_message.side_effect = [
            [tool_call_chunk("get_time", "{}")],
            [response_chunk("It is noon.")],
        ]
        mock_client_class.return_value = client
        output = io.StringIO()
        use_temporary_working_directory(self)  # cli() saves sessions/ and data/ in the current directory

        with patch("builtins.input", side_effect=["What time is it?", "/debug prompt", "exit"]), patch(
            "sys.stdout", output
        ):
            cli()

        inspected_request = output.getvalue()
        self.assertIn("Reconstructed model request:", inspected_request)
        self.assertIn("User message", inspected_request)
        self.assertIn("Assistant history", inspected_request)
        self.assertIn("Tool message", inspected_request)
        self.assertIn("It is noon.", inspected_request)
