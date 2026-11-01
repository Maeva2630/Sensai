import logging
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from cli.error_handling import format_cli_error, format_fatal_error
from rag.embeddings import EmbeddingConnectionError, EmbeddingModelError
from rag.ingestion import DocumentIngestionError
from utilities.ollama_client import OllamaConnectionError, OllamaModelNotFoundError, OllamaTimeoutError

LOGGER = logging.getLogger("sensai.tests.error_handling")
LOGGER.addHandler(logging.NullHandler())
LOGGER.propagate = False


def wrapped(outer: Exception, cause: Exception) -> Exception:
    outer.__cause__ = cause
    return outer


class TestActionableErrorMessages(unittest.TestCase):
    def test_missing_embedding_model_names_the_install_command(self):
        error = wrapped(EmbeddingModelError("unavailable"), OllamaModelNotFoundError("not found", model="nomic-embed-text"))

        message = format_cli_error(error, LOGGER)

        self.assertIn("'nomic-embed-text' is not installed", message)
        self.assertIn("ollama pull nomic-embed-text", message)

    def test_a_slow_model_is_not_reported_as_ollama_being_down(self):
        message = format_cli_error(OllamaTimeoutError("slow", timeout=120), LOGGER)

        self.assertIn("did not answer within 120 s", message)
        self.assertIn("request_timeout_seconds", message)
        self.assertNotIn("unable to reach Ollama", message)

    def test_ollama_down_behind_a_rag_error_says_to_start_ollama(self):
        error = wrapped(EmbeddingConnectionError("unavailable"), OllamaConnectionError("refused"))

        self.assertIn("unable to reach Ollama", format_cli_error(error, LOGGER))

    def test_document_errors_keep_the_reason_written_by_sensai(self):
        error = DocumentIngestionError("PDF support requires the 'pypdf' package.")

        self.assertEqual(format_cli_error(error, LOGGER), "Document error: PDF support requires the 'pypdf' package.")

    def test_unknown_errors_hide_their_details(self):
        message = format_cli_error(RuntimeError("secret path /home/x"), LOGGER)

        self.assertNotIn("secret path", message)
        self.assertIn("You can continue using SensAI", message)
        self.assertEqual(format_fatal_error(RuntimeError("secret"), LOGGER), "An unexpected error stopped SensAI.")


if __name__ == "__main__":
    unittest.main()
