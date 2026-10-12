import unittest
from unittest.mock import patch, MagicMock
import io
import sys
import os
import tempfile
from pathlib import Path

# Ensure `source` directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

# pyrefly: ignore [missing-import]
from cli.cli import cli
# pyrefly: ignore [missing-import]
from utilities.ollama_client import (
    OllamaConnectionError,
    OllamaModelNotFoundError,
)
from rag.ingestion import DocumentIngestor

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

        self.assertIn("Usage: /ingest <path-to-.txt-or-.md-file>", captured_output.getvalue())
        mock_client_cls.return_value.stream_message.assert_not_called()

    @patch("cli.cli.OllamaHTTPClient")
    def test_cli_ingest_command_reports_success_and_duplicate(self, mock_client_cls):
        with tempfile.TemporaryDirectory() as directory:
            document_path = Path(directory) / "knowledge.txt"
            document_path.write_text("Knowledge for the assistant", encoding="utf-8")
            inputs = [f"/ingest {document_path}", f"/ingest {document_path}", "exit"]

            captured_output = io.StringIO()
            with patch("builtins.input", side_effect=inputs), patch("sys.stdout", captured_output):
                cli(ingestor=DocumentIngestor())

        output = captured_output.getvalue()
        self.assertIn("Document ingested: knowledge.txt", output)
        self.assertIn("Ready for chunking and indexing.", output)
        self.assertIn("Document already ingested: knowledge.txt", output)
        mock_client_cls.return_value.stream_message.assert_not_called()


if __name__ == "__main__":
    unittest.main()
