import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure `source` directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pyrefly: ignore [missing-import]
from cli.cli import CliDependencies, cli
# pyrefly: ignore [missing-import]
from cli.commands.tools import QuizRequest, QuizUsageError, _parse_quiz_arguments
# pyrefly: ignore [missing-import]
from mcp.manager import MCPManager
# pyrefly: ignore [missing-import]
from support import use_temporary_working_directory_for_module
# pyrefly: ignore [missing-import]
from tools.base import ToolError
# pyrefly: ignore [missing-import]
from tools.permissions import AllowedDirectory, PermissionChecker
# pyrefly: ignore [missing-import]
from tools.quiz_export import QuizStore, SaveLastQuizTool, render_quiz, save_quiz
# pyrefly: ignore [missing-import]
from tools.registry import ToolRegistry
# pyrefly: ignore [missing-import]
from cli.cli import default_registry
# pyrefly: ignore [missing-import]
from tools.structured_output import ANSWER_KEY_NOTE, format_quiz


def setUpModule():
    # The CLI would start the servers declared in the repository's mcp_servers.json.
    patcher = patch("cli.cli.MCPManager.from_config_file", return_value=MCPManager([]))
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)
    # cli() saves sessions/, profile.json and data/ in the current directory: not in the repository.
    use_temporary_working_directory_for_module()


QUIZ = {
    "title": "Volcans",
    "questions": [
        {"question": "Comment s'appelle le magma en surface ?", "choices": ["La lave", "La cendre"],
         "answer_index": 0, "explanation": "C'est la lave."},
        {"question": "Quelle éruption projette des cendres ?", "choices": ["Effusive", "Explosive"], "answer_index": 1},
    ],
}


class ExportTestCase(unittest.TestCase):

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        (self.root / "courses").mkdir()
        (self.root / "exports").mkdir()
        self.checker = PermissionChecker(
            [AllowedDirectory(self.root / "courses", ("read",)),
             AllowedDirectory(self.root / "exports", ("read", "write"))],
            base_dir=self.root,
        )


class TestRenderQuiz(unittest.TestCase):

    def test_text_with_and_without_the_answer_key(self):
        with_answers = render_quiz(QUIZ, Path("quiz.md"))
        without = render_quiz(QUIZ, Path("quiz.md"), answers=False)
        self.assertEqual(with_answers, format_quiz(QUIZ))
        self.assertIn("Answer key:", with_answers)
        self.assertIn("A) La lave", without)
        self.assertNotIn("Answer key", without)
        self.assertNotIn("C'est la lave", without)

    def test_json_keeps_everything_by_default(self):
        data = json.loads(render_quiz(QUIZ, Path("quiz.JSON")))
        self.assertEqual(data, QUIZ)

    def test_json_without_answers_drops_the_answer_fields_only(self):
        text = render_quiz(QUIZ, Path("quiz.json"), answers=False)
        data = json.loads(text)
        self.assertEqual(data["title"], "Volcans")
        for question in data["questions"]:
            self.assertEqual(sorted(question), ["choices", "question"])
        self.assertIn("answer_index", QUIZ["questions"][0])  # the original quiz is not modified

    def test_accents_are_kept(self):
        self.assertIn("éruption", render_quiz(QUIZ, Path("quiz.json")))

    def test_text_export_reminds_to_check_the_key_but_json_data_does_not(self):
        self.assertIn(ANSWER_KEY_NOTE, render_quiz(QUIZ, Path("quiz.md")))
        self.assertNotIn(ANSWER_KEY_NOTE, render_quiz(QUIZ, Path("eleves.md"), answers=False))
        self.assertNotIn(ANSWER_KEY_NOTE, render_quiz(QUIZ, Path("quiz.json")))

    def test_labels_written_by_the_model_are_removed_from_every_export(self):
        labelled = {**QUIZ, "questions": [{**QUIZ["questions"][0], "choices": ["A) La lave", "B) La cendre"]}]}

        text = render_quiz(labelled, Path("quiz.md"))
        data = json.loads(render_quiz(labelled, Path("quiz.json")))

        self.assertIn("   A) La lave", text)
        self.assertNotIn("A) A)", text)
        self.assertEqual(data["questions"][0]["choices"], ["La lave", "La cendre"])


class TestSaveQuiz(ExportTestCase):

    def test_saves_text_and_json(self):
        text_path = save_quiz(QUIZ, "exports/quiz.md", self.checker)
        json_path = save_quiz(QUIZ, "exports/quiz.json", self.checker)
        self.assertEqual(text_path, self.root / "exports" / "quiz.md")
        self.assertIn("Answer key:", text_path.read_text(encoding="utf-8"))
        self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), QUIZ)

    def test_creates_missing_folders(self):
        save_quiz(QUIZ, "exports/6a/quiz.md", self.checker)
        self.assertTrue((self.root / "exports" / "6a" / "quiz.md").exists())

    def test_student_copy_has_no_answer_key(self):
        path = save_quiz(QUIZ, "exports/student.md", self.checker, answers=False)
        self.assertNotIn("Answer key", path.read_text(encoding="utf-8"))

    def test_refused_paths_write_nothing(self):
        for path in ["courses/quiz.md", "quiz.md", "../quiz.md", "/tmp/quiz.md", "exports/.." + "/courses/quiz.md"]:
            with self.subTest(path=path):
                with self.assertRaises(ToolError) as raised:
                    save_quiz(QUIZ, path, self.checker)
                self.assertIn("Access denied", str(raised.exception))
        self.assertEqual(list((self.root / "courses").iterdir()), [])
        self.assertEqual(list(self.root.glob("*.md")), [])

    def test_existing_file_is_kept_unless_overwrite(self):
        target = self.root / "exports" / "quiz.md"
        target.write_text("old", encoding="utf-8")

        with self.assertRaises(ToolError) as raised:
            save_quiz(QUIZ, "exports/quiz.md", self.checker)
        self.assertIn("--overwrite", str(raised.exception))
        self.assertEqual(target.read_text(encoding="utf-8"), "old")

        save_quiz(QUIZ, "exports/quiz.md", self.checker, overwrite=True)
        self.assertIn("Volcans", target.read_text(encoding="utf-8"))

    def test_directory_target(self):
        with self.assertRaises(ToolError) as raised:
            save_quiz(QUIZ, "exports", self.checker)
        self.assertIn("is a directory", str(raised.exception))

    def test_no_permissions_loaded(self):
        with self.assertRaises(ToolError) as raised:
            save_quiz(QUIZ, "exports/quiz.md", None)
        self.assertIn("not available", str(raised.exception))

    def test_os_error_is_reported_without_the_absolute_path(self):
        with patch.object(Path, "write_text", side_effect=OSError(28, "No space left on device")):
            with self.assertRaises(ToolError) as raised:
                save_quiz(QUIZ, "exports/quiz.md", self.checker)
        self.assertIn("No space left on device", str(raised.exception))
        self.assertNotIn(str(self.root), str(raised.exception))


class TestParseQuizArguments(unittest.TestCase):

    def test_valid_combinations(self):
        cases = {
            "/quiz volcans": QuizRequest("volcans"),
            "/quiz les volcans de France": QuizRequest("les volcans de France"),
            "/quiz --questions 3 volcans": QuizRequest("volcans", 3),
            "/quiz --save exports/q.md volcans": QuizRequest("volcans", save_path="exports/q.md"),
            "/quiz volcans --save exports/q.md": QuizRequest("volcans", save_path="exports/q.md"),
            "/quiz --save exports/q.md --no-answers volcans": QuizRequest("volcans", save_path="exports/q.md", answers=False),
            "/quiz --save exports/q.md --overwrite --questions 2 volcans": QuizRequest("volcans", 2, "exports/q.md", True, True),
            '/quiz --save "exports/my quiz.md" volcans': QuizRequest("volcans", save_path="exports/my quiz.md"),
            "/quiz --language Français volcans": QuizRequest("volcans", language="Français"),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(_parse_quiz_arguments(text), expected)

    def test_apostrophes_are_part_of_the_topic(self):
        self.assertEqual(_parse_quiz_arguments("/quiz les volcans de l'Auvergne").topic, "les volcans de l'Auvergne")
        request = _parse_quiz_arguments("/quiz --save exports/q.md l'éruption d'un volcan")
        self.assertEqual((request.topic, request.save_path), ("l'éruption d'un volcan", "exports/q.md"))

    def test_a_lone_double_dash_is_part_of_the_topic(self):
        self.assertEqual(_parse_quiz_arguments("/quiz Pythagore -- et Thalès").topic, "Pythagore -- et Thalès")

    def test_invalid_input_explains_what_is_wrong(self):
        cases = {
            "/quiz": "a topic is required",
            "/quiz --save": "--save needs a value",
            "/quiz --save --no-answers volcans": "--save needs a value",
            "/quiz --save exports/q.md": "a topic is required",
            "/quiz --no-answers volcans": "only make sense with --save",
            "/quiz --overwrite volcans": "only make sense with --save",
            "/quiz --unknown volcans": "unknown option --unknown",
            "/quiz --questions 0 volcans": "--questions must be a number",
            "/quiz --questions volcans": "--questions must be a number",
            '/quiz "unclosed': "double quote is not closed",
            "/quiz --save a.md --save b.md volcans": "--save is given twice",
            "/quiz --questions 2 --questions 3 volcans": "--questions is given twice",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                with self.assertRaises(QuizUsageError) as raised:
                    _parse_quiz_arguments(text)
                self.assertIn(expected, str(raised.exception))


class TestSaveLastQuizTool(ExportTestCase):
    """The model exports the quiz without retyping it."""

    def setUp(self):
        super().setUp()
        self.store = QuizStore()
        self.registry = ToolRegistry()
        self.registry.register(SaveLastQuizTool(self.checker, self.store))

    def run_tool(self, **arguments):
        return self.registry.execute("save_last_quiz", arguments)

    def test_exports_the_stored_quiz_exactly(self):
        self.store.quiz = QUIZ
        result = self.run_tool(path="exports/quiz.json")
        self.assertFalse(result.is_error)
        self.assertEqual(json.loads((self.root / "exports" / "quiz.json").read_text(encoding="utf-8")), QUIZ)

    def test_the_answer_key_is_always_kept(self):
        self.store.quiz = QUIZ
        self.run_tool(path="exports/a.md")
        self.assertIn("Answer key", (self.root / "exports" / "a.md").read_text(encoding="utf-8"))

    def test_the_model_cannot_drop_the_answers_or_replace_a_file(self):
        self.store.quiz = QUIZ
        self.assertEqual(sorted(self.registry.get("save_last_quiz").parameters["properties"]), ["path"])

        self.run_tool(path="exports/a.md")
        for extra in ({"include_answers": False}, {"overwrite": True}):
            result = self.run_tool(path="exports/a.md", **extra)
            self.assertTrue(result.is_error)
            self.assertIn("unknown argument", result.content)
        self.assertIn("Answer key", (self.root / "exports" / "a.md").read_text(encoding="utf-8"))

    def test_no_quiz_yet(self):
        result = self.run_tool(path="exports/quiz.md")
        self.assertTrue(result.is_error)
        self.assertIn("run /quiz first", result.content)
        self.assertFalse((self.root / "exports" / "quiz.md").exists())

    def test_permissions_and_existing_files_are_respected(self):
        self.store.quiz = QUIZ
        self.assertIn("Access denied", self.run_tool(path="courses/quiz.md").content)
        self.assertFalse(self.run_tool(path="exports/quiz.md").is_error)
        self.assertIn("already exists", self.run_tool(path="exports/quiz.md").content)

    def test_offered_for_export_requests_only(self):
        self.assertEqual(len(self.registry.to_ollama_tools("Exporte le quiz dans un fichier")), 1)
        self.assertEqual(self.registry.to_ollama_tools("Explique les volcans"), [])

    def test_default_registry_offers_it_only_with_a_store_and_permissions(self):
        with_store = [t.name for t in default_registry(True, self.checker, QuizStore()).list_tools()]
        self.assertIn("save_last_quiz", with_store)
        self.assertNotIn("save_last_quiz", [t.name for t in default_registry(True, self.checker).list_tools()])
        self.assertNotIn("save_last_quiz", [t.name for t in default_registry(False, self.checker, QuizStore()).list_tools()])

    def test_write_file_points_to_it_for_quizzes(self):
        description = next(t for t in default_registry(True, self.checker).list_tools() if t.name == "write_file").description
        self.assertIn("save_last_quiz", description)


class TestQuizCommand(ExportTestCase):

    def run_cli(self, inputs, answers=1):
        chat_client = MagicMock()
        reply = MagicMock()
        reply.json.return_value = {"message": {"role": "assistant", "content": json.dumps(QUIZ)}}
        chat_client.complete_message.return_value = reply
        chat_client.stream_message.return_value = [b'{"message": {"content": "Voici."}}']
        output = io.StringIO()
        dependencies = CliDependencies(rag_context_provider=MagicMock(), rag_indexer=MagicMock())
        with patch("cli.cli.OllamaHTTPClient", return_value=chat_client), \
                patch("cli.cli.load_permissions", return_value=(self.checker, [])), \
                patch("builtins.input", side_effect=[*inputs, "exit"]), patch("sys.stdout", output):
            cli(dependencies=dependencies)
        return output.getvalue(), chat_client

    def test_save_creates_the_file_and_still_displays_the_quiz(self):
        output, _ = self.run_cli(["/quiz --questions 2 --save exports/volcans.md volcans"])

        self.assertIn("1. Comment s'appelle le magma en surface ?", output)
        self.assertIn("Quiz saved to 'exports/volcans.md'", output)
        saved = (self.root / "exports" / "volcans.md").read_text(encoding="utf-8")
        self.assertIn("Answer key:", saved)
        self.assertIn(ANSWER_KEY_NOTE, saved)
        self.assertIn(ANSWER_KEY_NOTE, output)

    def test_student_copy(self):
        output, _ = self.run_cli(["/quiz --questions 2 --save exports/eleves.md --no-answers volcans"])

        self.assertIn("(without the answer key)", output)
        self.assertNotIn("Answer key", (self.root / "exports" / "eleves.md").read_text(encoding="utf-8"))

    def test_refused_path_reports_the_error_and_keeps_the_quiz_displayed(self):
        output, _ = self.run_cli(["/quiz --questions 2 --save courses/volcans.md volcans"])

        self.assertIn("Answer key:", output)
        self.assertIn("Quiz not saved: Access denied", output)
        self.assertFalse((self.root / "courses" / "volcans.md").exists())

    def test_existing_file_needs_overwrite(self):
        (self.root / "exports" / "volcans.md").write_text("old", encoding="utf-8")

        output, _ = self.run_cli(["/quiz --questions 2 --save exports/volcans.md volcans"])
        self.assertIn("already exists", output)
        self.assertEqual((self.root / "exports" / "volcans.md").read_text(encoding="utf-8"), "old")

        output, _ = self.run_cli(["/quiz --questions 2 --save exports/volcans.md --overwrite volcans"])
        self.assertIn("Quiz saved", output)

    def test_usage_for_bad_options(self):
        output, client = self.run_cli(["/quiz --no-answers volcans"])
        self.assertIn("Invalid /quiz command: --no-answers only make sense with --save.", output)
        self.assertIn("Usage: /quiz", output)
        self.assertIn("--save <path>", output)
        client.complete_message.assert_not_called()

    def test_apostrophe_in_the_topic_works(self):
        output, client = self.run_cli(["/quiz --questions 2 les volcans de l'Auvergne"])
        self.assertNotIn("Usage", output)
        self.assertIn("about: les volcans de l'Auvergne", client.complete_message.call_args.args[0][1]["content"])

    def test_quiz_language_follows_the_option_then_the_profile(self):
        _, client = self.run_cli(["/quiz --questions 2 volcans"])
        self.assertIn("in English", client.complete_message.call_args.args[0][0]["content"])

        _, client = self.run_cli(["/quiz --questions 2 --language Français volcans"])
        self.assertIn("in Français", client.complete_message.call_args.args[0][0]["content"])

        profile = self.root / "profile.json"
        profile.write_text(json.dumps({"preferred_language": "French"}), encoding="utf-8")
        chat_client = MagicMock()
        reply = MagicMock()
        reply.json.return_value = {"message": {"content": json.dumps(QUIZ)}}
        chat_client.complete_message.return_value = reply
        with patch("cli.cli.OllamaHTTPClient", return_value=chat_client), \
                patch("cli.cli.load_permissions", return_value=(self.checker, [])), \
                patch("builtins.input", side_effect=["/quiz --questions 2 volcans", "exit"]), patch("sys.stdout", io.StringIO()):
            cli(profile_path=str(profile), dependencies=CliDependencies(rag_context_provider=MagicMock(), rag_indexer=MagicMock()))
        self.assertIn("in French", chat_client.complete_message.call_args.args[0][0]["content"])

    def test_quiz_stays_in_the_conversation(self):
        _, client = self.run_cli(["/quiz --questions 2 volcans", "Exporte le quiz dans un fichier"])

        sent = client.stream_message.call_args.args[0]
        contents = [message["content"] for message in sent]
        self.assertIn("/quiz --questions 2 volcans", contents)
        self.assertTrue(any("Answer key:" in content for content in contents))
        # The quiz comes before the follow-up request
        self.assertLess(contents.index("/quiz --questions 2 volcans"), contents.index("Exporte le quiz dans un fichier"))

    def test_wrong_number_of_questions_is_retried(self):
        one = dict(QUIZ, questions=QUIZ["questions"][:1])
        replies = []
        for quiz in (one, QUIZ):
            reply = MagicMock()
            reply.json.return_value = {"message": {"content": json.dumps(quiz)}}
            replies.append(reply)
        chat_client = MagicMock()
        chat_client.complete_message.side_effect = replies
        output = io.StringIO()
        with patch("cli.cli.OllamaHTTPClient", return_value=chat_client), \
                patch("cli.cli.load_permissions", return_value=(self.checker, [])), \
                patch("builtins.input", side_effect=["/quiz --questions 2 volcans", "exit"]), patch("sys.stdout", output):
            cli(dependencies=CliDependencies(rag_context_provider=MagicMock(), rag_indexer=MagicMock()))

        self.assertEqual(chat_client.complete_message.call_count, 2)
        self.assertIn("exactly 2 questions, not 1", chat_client.complete_message.call_args.args[0][-1]["content"])
        self.assertIn("2. Quelle éruption", output.getvalue())

    def test_failed_generation_adds_nothing_to_the_conversation(self):
        bad = MagicMock()
        bad.json.return_value = {"message": {"content": "not json"}}
        chat_client = MagicMock()
        chat_client.complete_message.return_value = bad
        chat_client.stream_message.return_value = [b'{"message": {"content": "ok"}}']
        with patch("cli.cli.OllamaHTTPClient", return_value=chat_client), \
                patch("cli.cli.load_permissions", return_value=(self.checker, [])), \
                patch("builtins.input", side_effect=["/quiz volcans", "Bonjour", "exit"]), patch("sys.stdout", io.StringIO()):
            cli(dependencies=CliDependencies(rag_context_provider=MagicMock(), rag_indexer=MagicMock()))

        sent = chat_client.stream_message.call_args.args[0]
        self.assertFalse(any("/quiz" in message["content"] for message in sent))


if __name__ == "__main__":
    unittest.main()
