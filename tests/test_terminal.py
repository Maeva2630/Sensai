"""Tests for terminal styling and non-interactive fallbacks."""

import io
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from cli.terminal import assistant_prefix, user_prompt


class TerminalStream(io.StringIO):
    def isatty(self) -> bool:
        return True


class TestTerminalPresentation(unittest.TestCase):
    def test_labels_are_colored_and_bold_in_an_interactive_terminal(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(assistant_prefix(TerminalStream()), "\033[1;31mAssistant:\033[0m ")
            self.assertIn("\033[1;32mYou:", user_prompt(False, TerminalStream()))

    def test_labels_stay_plain_when_output_is_not_a_terminal_or_colors_are_disabled(self):
        self.assertEqual(assistant_prefix(io.StringIO()), "Assistant: ")
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            self.assertEqual(user_prompt(False, TerminalStream()), "You: ")

    def test_readline_prompt_marks_ansi_sequences_as_non_printing(self):
        with patch.dict(os.environ, {}, clear=True):
            prompt = user_prompt(True, stream=TerminalStream())

        self.assertIn("\001\033[1;32m\002", prompt)
        self.assertIn("\001\033[0m\002", prompt)
