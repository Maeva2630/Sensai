import unittest
from unittest.mock import patch, MagicMock
import io
import sys
import os

# Ensure `source` directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

# pyrefly: ignore [missing-import]
from cli.cli import cli
# pyrefly: ignore [missing-import]
from utilities.ollama_client import (
    OllamaConnectionError,
    OllamaModelNotFoundError,
)


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
            with self.assertRaises(SystemExit):
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
            with self.assertRaises(SystemExit):
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
            with self.assertRaises(SystemExit):
                cli()

        output = captured_output.getvalue()
        self.assertIn("Assistant: Hi there!", output)

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
            with self.assertRaises(SystemExit):
                cli()

        output = captured_output.getvalue()
        self.assertIn("Assistant: Starting response...", output)
        self.assertIn("Connection Error: Unable to connect to Ollama server", output)
        self.assertIn("Connection lost mid-stream", output)


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
            with self.assertRaises(SystemExit):
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
            with self.assertRaises(SystemExit) as ctx:
                cli()
            self.assertEqual(ctx.exception.code, 0)

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
            with self.assertRaises(SystemExit) as ctx:
                cli()
            self.assertEqual(ctx.exception.code, 0)

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
            with self.assertRaises(SystemExit):
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
            with self.assertRaises(SystemExit):
                cli()

        self.assertIn("Error:", captured.getvalue())
        self.assertIn("nonexistent-id", captured.getvalue())

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
            with self.assertRaises(SystemExit):
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

        self.assertIn("Error:", captured.getvalue())

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
            with self.assertRaises(SystemExit):
                cli(session_id="valid-id")

        mock_session.load.assert_called_with("valid-id")


if __name__ == "__main__":
    unittest.main()
