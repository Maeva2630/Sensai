import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from cli.context_budget import (
    DEFAULT_CHARACTERS_PER_TOKEN,
    MAXIMUM_CHARACTERS_PER_TOKEN,
    TokenEstimator,
    answer_reserve,
    over_budget_message,
    prompt_budget,
)


class TestContextBudget(unittest.TestCase):
    def test_a_quarter_of_the_window_is_kept_for_the_answer(self):
        self.assertEqual(answer_reserve(4096), 1024)
        self.assertEqual(prompt_budget(4096), 3072)
        # Small windows still keep a usable answer length.
        self.assertEqual(answer_reserve(512), 256)

    def test_estimate_counts_messages_and_tool_definitions(self):
        estimator = TokenEstimator()
        messages = [{"role": "user", "content": "x" * 320}]
        tools = [{"type": "function", "function": {"name": "get_time", "description": "", "parameters": {}}}]

        without_tools = estimator.estimate(messages)
        with_tools = estimator.estimate(messages, tools)

        self.assertEqual(without_tools, 100 + 4)
        self.assertGreater(with_tools, without_tools + 80)

    def test_the_ratio_is_learned_from_ollama_counts(self):
        estimator = TokenEstimator()
        messages = [{"role": "user", "content": "a" * 1040}]

        # Ollama reported 204 tokens: 200 for the text plus 4 for the role markers.
        estimator.learn(messages, None, 204)

        self.assertAlmostEqual(estimator.characters_per_token, 5.2)
        self.assertEqual(estimator.estimate(messages), 204)

    def test_learning_ignores_impossible_counts_and_clamps_the_ratio(self):
        estimator = TokenEstimator()
        estimator.learn([{"role": "user", "content": "abc"}], None, 0)
        self.assertEqual(estimator.characters_per_token, DEFAULT_CHARACTERS_PER_TOKEN)

        estimator.learn([{"role": "user", "content": "a" * 10_000}], None, 10)
        self.assertEqual(estimator.characters_per_token, MAXIMUM_CHARACTERS_PER_TOKEN)

    def test_refusal_message_explains_the_risk_and_the_remedies(self):
        message = over_budget_message(5000, 4096)

        self.assertIn("about 5000 tokens", message)
        self.assertIn("1024 are kept for the answer", message)
        self.assertIn("/ingest", message)
        self.assertIn("model.context_window", message)


if __name__ == "__main__":
    unittest.main()
