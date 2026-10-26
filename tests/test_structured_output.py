import io
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure `source` directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

# pyrefly: ignore [missing-import]
from cli.cli import CliDependencies, cli
# pyrefly: ignore [missing-import]
from mcp.manager import MCPManager
# pyrefly: ignore [missing-import]
from rag.context import PreparedContext
# pyrefly: ignore [missing-import]
from rag.storage import SearchResult
# pyrefly: ignore [missing-import]
from tools.structured_output import (
    QUIZ_SCHEMA,
    StructuredOutputError,
    check_quiz,
    format_quiz,
    generate_structured,
    quiz_messages,
    validate_json,
)
# pyrefly: ignore [missing-import]
from utilities.ollama_client import OllamaConnectionError


def setUpModule():
    # The CLI would start the servers declared in the repository's mcp_servers.json.
    patcher = patch("cli.cli.MCPManager.from_config_file", return_value=MCPManager([]))
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


def good_quiz(**overrides):
    quiz = {
        "title": "Pythagoras",
        "questions": [
            {"question": "What is a^2 + b^2?", "choices": ["c^2", "c"], "answer_index": 0, "explanation": "By the theorem."},
            {"question": "Which triangle?", "choices": ["Right", "Equilateral", "Obtuse"], "answer_index": 0},
        ],
    }
    quiz.update(overrides)
    return quiz


def response(content):
    reply = MagicMock()
    reply.json.return_value = {"message": {"role": "assistant", "content": content}}
    return reply


def client_answering(*contents):
    client = MagicMock()
    client.complete_message.side_effect = [response(c) for c in contents]
    return client


class TestValidateJson(unittest.TestCase):

    def test_valid_quiz(self):
        self.assertIsNone(validate_json(QUIZ_SCHEMA, good_quiz()))

    def test_nested_errors_are_located(self):
        quiz = good_quiz()
        quiz["questions"][1]["choices"] = ["only one"]
        self.assertEqual(validate_json(QUIZ_SCHEMA, quiz), "$.questions[1].choices must have at least 2 items")

    def test_invalid_values(self):
        cases = {
            "wrong type": ({"type": "object"}, [], "$ must be of type object"),
            "missing property": ({"type": "object", "required": ["a"]}, {}, "$ is missing required property 'a'"),
            "wrong property type": ({"type": "object", "properties": {"a": {"type": "integer"}}}, {"a": "1"}, "$.a must be of type integer"),
            "bool is not an integer": ({"type": "integer"}, True, "$ must be of type integer"),
            "enum": ({"enum": ["a", "b"]}, "c", "$ must be one of ['a', 'b']"),
            "too many items": ({"type": "array", "maxItems": 1}, [1, 2], "$ must have at most 1 items"),
            "below minimum": ({"type": "integer", "minimum": 0}, -1, "$ must be at least 0"),
            "above maximum": ({"type": "number", "maximum": 5}, 6, "$ must be at most 5"),
            "empty string": ({"type": "string", "minLength": 1}, "  ", "$ must not be empty"),
            "extra property": ({"type": "object", "additionalProperties": False}, {"x": 1}, "$ has unexpected property 'x'"),
        }
        for name, (schema, value, expected) in cases.items():
            with self.subTest(name):
                self.assertEqual(validate_json(schema, value), expected)

    def test_type_lists_and_unknown_keywords_are_accepted(self):
        self.assertIsNone(validate_json({"type": ["string", "null"]}, None))
        self.assertIsNone(validate_json({"format": "date"}, "anything"))


class TestGenerateStructured(unittest.TestCase):

    messages = [{"role": "user", "content": "Make a quiz"}]

    def test_valid_answer_is_returned_and_schema_is_sent_to_ollama(self):
        client = client_answering(json.dumps(good_quiz()))

        quiz = generate_structured(client, self.messages, QUIZ_SCHEMA)

        self.assertEqual(quiz["title"], "Pythagoras")
        self.assertEqual(client.complete_message.call_args.kwargs["output_format"], QUIZ_SCHEMA)

    def test_invalid_json_is_retried_with_the_error(self):
        client = client_answering("not json", json.dumps(good_quiz()))

        quiz = generate_structured(client, self.messages, QUIZ_SCHEMA)

        self.assertEqual(quiz["title"], "Pythagoras")
        retry_messages = client.complete_message.call_args_list[1].args[0]
        self.assertEqual(retry_messages[-2], {"role": "assistant", "content": "not json"})
        self.assertIn("not valid JSON", retry_messages[-1]["content"])

    def test_schema_violation_is_retried_with_the_error(self):
        client = client_answering(json.dumps({"title": "x"}), json.dumps(good_quiz()))

        generate_structured(client, self.messages, QUIZ_SCHEMA)

        retry_messages = client.complete_message.call_args_list[1].args[0]
        self.assertIn("missing required property 'questions'", retry_messages[-1]["content"])

    def test_gives_up_after_the_maximum_number_of_attempts(self):
        client = client_answering("a", "b", "c")

        with self.assertRaises(StructuredOutputError) as raised:
            generate_structured(client, self.messages, QUIZ_SCHEMA, max_attempts=3)

        self.assertEqual(client.complete_message.call_count, 3)
        self.assertIn("after 3 attempts", str(raised.exception))

    def test_extra_check_can_reject_an_answer(self):
        bad = good_quiz()
        bad["questions"][0]["answer_index"] = 5
        client = client_answering(json.dumps(bad), json.dumps(good_quiz()))

        quiz = generate_structured(client, self.messages, QUIZ_SCHEMA, extra_check=check_quiz)

        self.assertEqual(quiz["questions"][0]["answer_index"], 0)
        self.assertIn("answer_index 5", client.complete_message.call_args_list[1].args[0][-1]["content"])

    def test_unreadable_ollama_response(self):
        client = MagicMock()
        client.complete_message.return_value.json.side_effect = ValueError("bad")
        with self.assertRaises(StructuredOutputError):
            generate_structured(client, self.messages, QUIZ_SCHEMA)

    def test_connection_errors_are_not_swallowed(self):
        client = MagicMock()
        client.complete_message.side_effect = OllamaConnectionError("down")
        with self.assertRaises(OllamaConnectionError):
            generate_structured(client, self.messages, QUIZ_SCHEMA)


class TestQuizHelpers(unittest.TestCase):

    def test_check_quiz(self):
        self.assertIsNone(check_quiz(good_quiz()))
        bad = good_quiz()
        bad["questions"][1]["answer_index"] = 3
        self.assertIn("question 2", check_quiz(bad))

    def test_check_quiz_enforces_the_requested_number_of_questions(self):
        self.assertIsNone(check_quiz(good_quiz(), 2))
        self.assertIn("exactly 3 questions, not 2", check_quiz(good_quiz(), 3))

    def test_format_quiz_shows_questions_and_answer_key(self):
        text = format_quiz(good_quiz())
        self.assertIn("1. What is a^2 + b^2?", text)
        self.assertIn("   A) c^2", text)
        self.assertIn("Answer key:", text)
        self.assertIn("1. A - By the theorem.", text)
        self.assertIn("2. A", text)

    def test_quiz_messages_are_grounded_only_with_excerpts(self):
        plain = quiz_messages("Pythagoras", 3)
        grounded = quiz_messages("Pythagoras", 3, "[course.md]\nThe theorem says...")
        self.assertNotIn("Course excerpts", plain[0]["content"])
        self.assertIn("Course excerpts", grounded[0]["content"])
        self.assertIn("The theorem says", grounded[0]["content"])
        self.assertIn("3 questions about: Pythagoras", plain[1]["content"])
        self.assertIn("in English", plain[0]["content"])
        self.assertIn("in Français", quiz_messages("Pythagoras", 3, language="Français")[0]["content"])


class TestStructuredOutputCommands(unittest.TestCase):

    def run_cli(self, inputs, answers=(), context_provider=None):
        chat_client = MagicMock()
        chat_client.complete_message.side_effect = [response(a) for a in answers]
        dependencies = CliDependencies(rag_context_provider=context_provider or MagicMock(), rag_indexer=MagicMock())
        output = io.StringIO()
        with patch("cli.cli.OllamaHTTPClient", return_value=chat_client), \
                patch("builtins.input", side_effect=[*inputs, "exit"]), patch("sys.stdout", output):
            cli(dependencies=dependencies)
        return output.getvalue(), chat_client

    def test_quiz_command_prints_the_quiz(self):
        output, client = self.run_cli(["/quiz --questions 2 Pythagoras"], [json.dumps(good_quiz())])

        self.assertIn("1. What is a^2 + b^2?", output)
        self.assertIn("Answer key:", output)
        self.assertIn("2 questions about: Pythagoras", client.complete_message.call_args.args[0][1]["content"])

    def test_quiz_usage_is_shown_for_invalid_input(self):
        for command in ["/quiz", "/quiz --questions", "/quiz --questions 0 topic", "/quiz --questions 99 topic", "/quiz --questions abc topic"]:
            with self.subTest(command=command):
                output, client = self.run_cli([command])
                self.assertIn("Usage: /quiz", output)
                client.complete_message.assert_not_called()

    def test_quiz_failure_is_reported_without_crashing(self):
        output, _ = self.run_cli(["/quiz Pythagoras"], ["bad", "bad", "bad"])
        self.assertIn("Quiz generation failed", output)

    def test_quiz_is_built_from_the_course_when_rag_is_on(self):
        provider = MagicMock()
        provider.prepare.return_value = PreparedContext(
            "ctx", [SearchResult("d", "c1", "cours.md", 0, "Le carre de l'hypotenuse...", 0.9)]
        )

        output, client = self.run_cli(["/rag on", "/quiz --questions 2 Pythagoras"], [json.dumps(good_quiz())], provider)

        provider.prepare.assert_called_once_with("Pythagoras")
        system_message = client.complete_message.call_args.args[0][0]["content"]
        self.assertIn("[cours.md]", system_message)
        self.assertIn("Le carre de l'hypotenuse", system_message)
        self.assertIn("Answer key:", output)

    def test_quiz_with_rag_on_but_no_matching_excerpt(self):
        provider = MagicMock()
        provider.prepare.return_value = PreparedContext("ctx", [])

        output, client = self.run_cli(["/rag on", "/quiz Pythagoras"], [], provider)

        self.assertIn("No indexed course excerpt matches this topic", output)
        client.complete_message.assert_not_called()

    def test_quiz_is_listed_in_help(self):
        output, _ = self.run_cli(["/help"])
        self.assertIn("/quiz [--questions", output)


if __name__ == "__main__":
    unittest.main()
