"""Tests for translating application configuration into RAG settings."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from cli.cli import _rag_context_config
from config import RagConfig


class TestRagConfiguration(unittest.TestCase):
    def test_cli_translates_the_supported_rag_settings(self):
        settings = _rag_context_config(
            RagConfig(top_k=5, max_context_characters=1200, relevance_threshold=0.6)
        )

        self.assertEqual(settings.top_k, 5)
        self.assertEqual(settings.max_context_characters, 1200)
        self.assertEqual(settings.relevance_threshold, 0.6)


if __name__ == "__main__":
    unittest.main()
