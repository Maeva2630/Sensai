"""Unit tests for semantic context compression via sliding-window LLM summarization."""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from memory.compression import (
    SUMMARIZATION_REQUEST,
    SUMMARIZATION_SYSTEM_PROMPT,
    assemble_compressed_history,
    compress_history,
    create_summary_message,
    format_dialogue_for_summary,
    generate_summary,
    is_summary_message,
    split_history_for_compression,
)
from utilities.ollama_client import OllamaError


class TestSemanticCompression(unittest.TestCase):
    """Test suite for memory/compression.py functions."""

    def test_is_summary_message(self):
        self.assertTrue(
            is_summary_message(
                {"role": "system", "content": "[Previous Conversation Summary: Key facts here]"}
            )
        )
        self.assertFalse(
            is_summary_message({"role": "system", "content": "You are a teacher assistant."})
        )
        self.assertFalse(
            is_summary_message(
                {"role": "user", "content": "[Previous Conversation Summary: Fake]"}
            )
        )
        self.assertFalse(is_summary_message({"role": "system", "content": ""}))
        self.assertFalse(is_summary_message({}))

    def test_format_dialogue_for_summary(self):
        messages = [
            {"role": "system", "content": "[Previous Conversation Summary: Initial facts]"},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "A high-level programming language."},
        ]
        formatted = format_dialogue_for_summary(messages)
        self.assertIn("Previous Summary: Initial facts", formatted)
        self.assertIn("User: What is Python?", formatted)
        self.assertIn("Assistant: A high-level programming language.", formatted)

    def test_compress_history_preserves_base_system_and_recent_turns(self):
        messages = [
            {"role": "system", "content": "You are a helpful teacher assistant."},
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Answer 2"},
            {"role": "user", "content": "Question 3"},
            {"role": "assistant", "content": "Answer 3"},
            {"role": "user", "content": "Question 4"},
            {"role": "assistant", "content": "Answer 4"},
            {"role": "user", "content": "Question 5"},
            {"role": "assistant", "content": "Answer 5"},
        ]
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "User asked about questions 1 to 3; solutions provided."}
        }
        mock_client.complete_message.return_value = mock_response

        updated_messages, result = compress_history(messages, mock_client, keep_recent_turns=2)

        self.assertTrue(result.compressed)
        self.assertEqual(result.messages_before, 11)
        # Expected: base system + summary + 4 recent messages = 6
        self.assertEqual(len(updated_messages), 6)
        self.assertEqual(result.messages_after, 6)

        # Base system prompt preserved at index 0
        self.assertEqual(updated_messages[0]["content"], "You are a helpful teacher assistant.")

        # Summary system prompt inserted at index 1
        self.assertTrue(is_summary_message(updated_messages[1]))
        self.assertIn("User asked about questions 1 to 3", updated_messages[1]["content"])

        # Recent turns preserved at the end
        self.assertEqual(updated_messages[2]["content"], "Question 4")
        self.assertEqual(updated_messages[3]["content"], "Answer 4")
        self.assertEqual(updated_messages[4]["content"], "Question 5")
        self.assertEqual(updated_messages[5]["content"], "Answer 5")

    def test_compress_history_when_not_enough_messages(self):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        mock_client = MagicMock()
        updated, result = compress_history(messages, mock_client)

        self.assertFalse(result.compressed)
        self.assertEqual(updated, messages)
        mock_client.complete_message.assert_not_called()

    def test_compress_history_with_existing_summary(self):
        messages = [
            {"role": "system", "content": "Teacher Assistant persona."},
            {"role": "system", "content": "[Previous Conversation Summary: Covered module 1.]"},
            {"role": "user", "content": "Module 2 question 1"},
            {"role": "assistant", "content": "Module 2 answer 1"},
            {"role": "user", "content": "Module 2 question 2"},
            {"role": "assistant", "content": "Module 2 answer 2"},
            {"role": "user", "content": "Module 3 recent question"},
            {"role": "assistant", "content": "Module 3 recent answer"},
            {"role": "user", "content": "Latest question"},
            {"role": "assistant", "content": "Latest answer"},
        ]
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Covered module 1 and module 2 questions."}
        }
        mock_client.complete_message.return_value = mock_response

        updated, result = compress_history(messages, mock_client, keep_recent_turns=2)

        self.assertTrue(result.compressed)
        self.assertEqual(updated[0]["content"], "Teacher Assistant persona.")
        self.assertTrue(is_summary_message(updated[1]))
        self.assertIn("Covered module 1 and module 2", updated[1]["content"])
        self.assertEqual(updated[-2]["content"], "Latest question")
        self.assertEqual(updated[-1]["content"], "Latest answer")

    def test_compress_history_handles_ollama_error_gracefully(self):
        messages = [
            {"role": "system", "content": "Persona"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "A3"},
            {"role": "user", "content": "Q4"},
            {"role": "assistant", "content": "A4"},
        ]
        mock_client = MagicMock()
        mock_client.complete_message.side_effect = OllamaError("Connection timeout")

        updated, result = compress_history(messages, mock_client)
        self.assertFalse(result.compressed)
        self.assertIn("Summarization call failed", result.reason)
        self.assertEqual(updated, messages)

    def test_compress_history_handles_empty_summary_response(self):
        messages = [
            {"role": "system", "content": "Persona"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "A3"},
            {"role": "user", "content": "Q4"},
            {"role": "assistant", "content": "A4"},
        ]
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "   "}}
        mock_client.complete_message.return_value = mock_response

        updated, result = compress_history(messages, mock_client)
        self.assertFalse(result.compressed)
        self.assertEqual(updated, messages)

    def test_split_history_for_compression(self):
        # Empty messages
        self.assertIsNone(split_history_for_compression([]))

        # Too few messages
        short_history = [
            {"role": "system", "content": "Base"},
            {"role": "user", "content": "Hi"},
        ]
        self.assertIsNone(split_history_for_compression(short_history))

        # Sufficient messages
        long_history = [
            {"role": "system", "content": "Persona"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "A3"},
        ]
        result = split_history_for_compression(long_history, keep_recent_turns=1)
        self.assertIsNotNone(result)
        base_system, older, recent = result
        self.assertEqual(len(base_system), 1)
        self.assertEqual(base_system[0]["content"], "Persona")
        self.assertEqual(len(recent), 2)  # 1 turn = 2 messages
        self.assertEqual(recent[-1]["content"], "A3")
        self.assertEqual(len(older), 4)

    def test_create_summary_message(self):
        msg = create_summary_message("Key outcomes")
        self.assertEqual(msg["role"], "system")
        self.assertTrue(is_summary_message(msg))
        self.assertEqual(msg["content"], "[Previous Conversation Summary: Key outcomes]")

    def test_assemble_compressed_history(self):
        base = [{"role": "system", "content": "Persona"}]
        recent = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
        assembled = assemble_compressed_history(base, "Summary text", recent)
        self.assertEqual(len(assembled), 4)
        self.assertEqual(assembled[0]["content"], "Persona")
        self.assertTrue(is_summary_message(assembled[1]))
        self.assertEqual(assembled[2]["content"], "Q")
        self.assertEqual(assembled[3]["content"], "A")

    def test_generate_summary_handles_success_and_error(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "Generated summary text"}}
        mock_client.complete_message.return_value = mock_resp

        res = generate_summary("User: Hello", mock_client)
        self.assertEqual(res, "Generated summary text")

        mock_client.complete_message.side_effect = OllamaError("Failed")
        self.assertIsNone(generate_summary("User: Hello", mock_client))

    def test_generate_summary_handles_unreadable_replies(self):
        not_json = MagicMock()
        not_json.json.side_effect = ValueError("Expecting value")
        replies = {
            "not JSON": not_json,
            "not an object": MagicMock(**{"json.return_value": ["a"]}),
            "message is not an object": MagicMock(**{"json.return_value": {"message": "text"}}),
            "content is not text": MagicMock(**{"json.return_value": {"message": {"content": 3}}}),
        }
        for name, reply in replies.items():
            with self.subTest(name):
                client = MagicMock()
                client.complete_message.return_value = reply
                self.assertIsNone(generate_summary("User: Hello", client))

    def test_unexpected_errors_are_not_hidden(self):
        client = MagicMock()
        client.complete_message.side_effect = RuntimeError("bug")
        with self.assertRaises(RuntimeError):
            generate_summary("User: Hello", client)

    def test_summary_prompt_keeps_the_teacher_facts(self):
        client = MagicMock()
        client.complete_message.return_value.json.return_value = {"message": {"content": "- Teacher: Mme Martin"}}

        generate_summary("User: I am Mme Martin, 5e B, 27 pupils, exam on 12 June.", client)

        system, user = client.complete_message.call_args.args[0]
        self.assertEqual(system["content"], SUMMARIZATION_SYSTEM_PROMPT)
        for expected in ("teacher: name", "grade level", "numbers", "dates", "decisions", "tasks still open",
                         "introducing themselves", "language of the conversation", "previous summary"):
            with self.subTest(expected=expected):
                self.assertIn(expected, SUMMARIZATION_SYSTEM_PROMPT.lower())
        # The dialogue is delimited and followed by the instruction, which small models follow best.
        self.assertIn("<<<\nUser: I am Mme Martin, 5e B, 27 pupils, exam on 12 June.\n>>>", user["content"])
        self.assertTrue(user["content"].endswith(SUMMARIZATION_REQUEST.split(">>>\n")[1]))
        self.assertIn("name, class, numbers, dates, decisions and open tasks", user["content"])

    def test_split_never_separates_tool_calls_from_their_results(self):
        tool_call = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_time"}}]}
        messages = [
            {"role": "system", "content": "Persona"},
            {"role": "user", "content": "I teach 5e B."},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "27 pupils."},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "What time is it?"},
            tool_call,
            {"role": "tool", "content": "10:00"},
            {"role": "tool", "content": "10:00 (second call)"},
            {"role": "assistant", "content": "It is 10:00."},
        ]

        _, older, recent = split_history_for_compression(messages, keep_recent_turns=1)

        self.assertIs(recent[0], tool_call)
        self.assertEqual([m["role"] for m in recent], ["assistant", "tool", "tool", "assistant"])
        self.assertNotIn("tool", [m["role"] for m in older])
        self.assertEqual(older[-1]["content"], "What time is it?")

    def test_compressed_history_starts_kept_turns_with_the_whole_tool_exchange(self):
        messages = [
            {"role": "system", "content": "Persona"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_time"}}]},
            {"role": "tool", "content": "10:00"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "A3"},
        ]
        client = MagicMock()
        client.complete_message.return_value.json.return_value = {"message": {"content": "Summary"}}

        updated, result = compress_history(messages, client, keep_recent_turns=2)

        self.assertTrue(result.compressed)
        roles = [m["role"] for m in updated]
        self.assertEqual(roles[2:], ["assistant", "tool", "assistant", "user", "assistant"])
        self.assertIn("tool_calls", updated[2])

    def test_a_single_tool_exchange_is_kept_whole_or_not_split(self):
        exchange = [
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "a"}}]},
            *({"role": "tool", "content": str(n)} for n in range(5)),
        ]

        _, older, recent = split_history_for_compression(
            [{"role": "user", "content": "Q"}, *exchange], keep_recent_turns=1, min_turns_to_compress=1
        )
        self.assertEqual((older, recent), ([{"role": "user", "content": "Q"}], exchange))
        self.assertIsNone(split_history_for_compression(exchange, keep_recent_turns=1, min_turns_to_compress=1))


if __name__ == "__main__":
    unittest.main()
