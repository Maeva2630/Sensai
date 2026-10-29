"""Tests for controlled chat-model fallback behavior."""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from cli.fallback import FallbackChatClient
from utilities.ollama_client import OllamaConnectionError, OllamaModelNotFoundError


class TestFallbackChatClient(unittest.TestCase):
    def setUp(self):
        self.primary = MagicMock()
        self.primary.model = "qwen2.5:3b"
        self.fallback = MagicMock()
        self.fallback.model = "llama3.2:latest"
        self.fallback_events: list[tuple[str, str]] = []
        self.client = FallbackChatClient(
            (self.primary, self.fallback),
            on_fallback=lambda primary, fallback: self.fallback_events.append((primary, fallback)),
        )

    def test_stream_retries_the_next_model_when_primary_is_missing(self):
        self.primary.stream_message.side_effect = OllamaModelNotFoundError("missing")
        self.fallback.stream_message.return_value = iter([b'{"message": {"content": "Hello"}}'])

        response = list(self.client.stream_message([{"role": "user", "content": "Hi"}]))

        self.assertEqual(response, [b'{"message": {"content": "Hello"}}'])
        self.assertEqual(self.client.model, "llama3.2:latest")
        self.assertEqual(self.fallback_events, [("qwen2.5:3b", "llama3.2:latest")])

    def test_stream_does_not_retry_when_ollama_server_is_unreachable(self):
        self.primary.stream_message.side_effect = OllamaConnectionError("offline")

        with self.assertRaises(OllamaConnectionError):
            list(self.client.stream_message([{"role": "user", "content": "Hi"}]))

        self.fallback.stream_message.assert_not_called()
        self.assertEqual(self.fallback_events, [])

    def test_stream_does_not_retry_after_a_partial_response(self):
        def partial_stream():
            yield b'{"message": {"content": "Partial"}}'
            raise OllamaModelNotFoundError("lost model")

        self.primary.stream_message.return_value = partial_stream()

        with self.assertRaises(OllamaModelNotFoundError):
            list(self.client.stream_message([{"role": "user", "content": "Hi"}]))

        self.fallback.stream_message.assert_not_called()
        self.assertEqual(self.fallback_events, [])

    def test_complete_retries_the_next_model_when_primary_is_missing(self):
        self.primary.complete_message.side_effect = OllamaModelNotFoundError("missing")
        response = MagicMock()
        self.fallback.complete_message.return_value = response

        result = self.client.complete_message([{"role": "user", "content": "Summarize"}])

        self.assertIs(result, response)
        self.assertEqual(self.client.model, "llama3.2:latest")
        self.assertEqual(self.fallback_events, [("qwen2.5:3b", "llama3.2:latest")])
