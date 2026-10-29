import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Ensure `source` directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

# pyrefly: ignore [missing-import]
from cli.commands.memory import _dispatch_forget, _dispatch_load
# pyrefly: ignore [missing-import]
from cli.session import CliSession
# pyrefly: ignore [missing-import]
from mcp.client import MCPClient, safe_environment
# pyrefly: ignore [missing-import]
from memory.profile_manager import ProfileManager
# pyrefly: ignore [missing-import]
from memory.session_manager import InvalidSessionIdError, SessionManager
# pyrefly: ignore [missing-import]
from tools.safety import personal_data_warning
# pyrefly: ignore [missing-import]
from tools.web_search import WebSearchTool

POSIX = os.name == "posix"


def mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


class TempDirCase(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.sessions = self.root / "sessions"


class SessionIdTests(TempDirCase):
    BAD_IDS = ("../../etc/passwd", "a/b", "a\\b", "..", "", "x" * 65, "id with space", "id.json")

    def test_unsafe_ids_are_rejected_at_creation(self):
        for bad in filter(None, self.BAD_IDS):
            with self.subTest(session_id=bad), self.assertRaises(InvalidSessionIdError):
                SessionManager(session_id=bad, storage_dir=str(self.sessions))

    def test_unsafe_ids_are_rejected_on_load_and_delete(self):
        manager = SessionManager(storage_dir=str(self.sessions))
        for bad in filter(None, self.BAD_IDS):
            with self.subTest(session_id=bad):
                with self.assertRaises(InvalidSessionIdError):
                    manager.load(bad)
                with self.assertRaises(InvalidSessionIdError):
                    manager.delete(bad)

    def test_nothing_is_written_outside_the_sessions_folder(self):
        with self.assertRaises(InvalidSessionIdError):
            SessionManager(session_id="../escape", storage_dir=str(self.sessions)).save()

        self.assertEqual(list(self.root.glob("*.json")), [])

    def test_normal_ids_still_work(self):
        manager = SessionManager(session_id="course_2025-a", storage_dir=str(self.sessions))
        manager.add_message("user", "hello")
        manager.save()

        self.assertEqual(SessionManager(storage_dir=str(self.sessions)).load("course_2025-a")[0]["content"], "hello")

    def test_load_command_reports_an_unsafe_id_without_crashing(self):
        context = SimpleNamespace(
            feature_config=SimpleNamespace(memory_enabled=True),
            conversation_session=SessionManager(storage_dir=str(self.sessions)),
            messages=[],
        )

        with patch("builtins.print") as shown:
            _dispatch_load("/load ../../secret", CliSession(), context)

        self.assertTrue(any("Session error" in str(call) for call in shown.call_args_list))


@unittest.skipUnless(POSIX, "file permissions are POSIX only")
class PrivateFileTests(TempDirCase):
    def test_session_files_and_folder_are_owner_only(self):
        manager = SessionManager(session_id="s1", storage_dir=str(self.sessions))
        manager.add_message("user", "private")
        path = manager.save()

        self.assertEqual(mode(path), 0o600)
        self.assertEqual(mode(self.sessions), 0o700)

    def test_existing_world_readable_file_is_tightened_on_save(self):
        self.sessions.mkdir()
        existing = self.sessions / "s1.json"
        existing.write_text("{}")
        os.chmod(existing, 0o644)

        SessionManager(session_id="s1", storage_dir=str(self.sessions)).save()

        self.assertEqual(mode(existing), 0o600)

    def test_profile_is_owner_only(self):
        path = self.root / "profile.json"
        ProfileManager(profile_path=path)

        self.assertEqual(mode(path), 0o600)
        self.assertEqual(json.loads(path.read_text())["user_name"], "User")


class DeletionTests(TempDirCase):
    def make_sessions(self, *ids):
        for session_id in ids:
            manager = SessionManager(session_id=session_id, storage_dir=str(self.sessions))
            manager.add_message("user", "x")
            manager.save()

    def test_delete_removes_only_that_session(self):
        self.make_sessions("a", "b")

        self.assertTrue(SessionManager(storage_dir=str(self.sessions)).delete("a"))

        self.assertEqual(sorted(p.name for p in self.sessions.iterdir()), ["b.json"])

    def test_delete_of_a_missing_session_is_reported(self):
        self.assertFalse(SessionManager(storage_dir=str(self.sessions)).delete("nope"))

    def test_delete_all_counts_and_removes_everything(self):
        self.make_sessions("a", "b", "c")

        self.assertEqual(SessionManager(storage_dir=str(self.sessions)).delete_all(), 3)
        self.assertEqual(list(self.sessions.iterdir()), [])

    def test_delete_all_without_a_folder_is_zero(self):
        self.assertEqual(SessionManager(storage_dir=str(self.sessions)).delete_all(), 0)


class ForgetCommandTests(TempDirCase):
    def setUp(self):
        super().setUp()
        self.manager = SessionManager(session_id="mine", storage_dir=str(self.sessions))
        self.manager.add_message("user", "secret grades")
        self.manager.save()
        SessionManager(session_id="other", storage_dir=str(self.sessions)).save()
        self.system = {"role": "system", "content": "persona"}
        self.context = SimpleNamespace(
            conversation_session=self.manager,
            messages=[self.system, {"role": "user", "content": "secret grades"}],
        )

    def forget(self, text, confirm=True):
        with patch("cli.commands.memory.ask_confirmation", return_value=confirm), patch("builtins.print"):
            _dispatch_forget(text, CliSession(), self.context)

    def test_forget_deletes_this_session_and_clears_memory(self):
        self.forget("/forget")

        self.assertFalse((self.sessions / "mine.json").exists())
        self.assertTrue((self.sessions / "other.json").exists())
        self.assertEqual(self.context.messages, [self.system])
        self.assertEqual(self.manager.get_messages(), [])

    def test_forget_all_needs_confirmation(self):
        self.forget("/forget all", confirm=False)

        self.assertEqual(len(list(self.sessions.iterdir())), 2)
        self.assertEqual(len(self.context.messages), 2)

    def test_forget_all_deletes_every_session(self):
        self.forget("/forget all")

        self.assertEqual(list(self.sessions.iterdir()), [])
        self.assertEqual(self.context.messages, [self.system])

    def test_unknown_argument_changes_nothing(self):
        self.forget("/forget everything")

        self.assertEqual(len(list(self.sessions.iterdir())), 2)
        self.assertEqual(len(self.context.messages), 2)


class PersonalDataWarningTests(unittest.TestCase):
    def test_detects_obvious_personal_data(self):
        for text, label in (
            ("contact lea.martin@example.org", "e-mail"),
            ("appelle le 06 12 34 56 78", "phone"),
            ("appelle le +33 6 12 34 56 78", "phone"),
            ("compte FR76 3000 6000 0112 3456 7890 189", "IBAN"),
            ("sécu 1 85 05 78 006 084 36", "identifier"),
        ):
            with self.subTest(text=text):
                self.assertIn(label, personal_data_warning(text))

    def test_ordinary_queries_have_no_warning(self):
        for text in ("nouveautés Python 3.13", "programme de maths 2025 cycle 4", "bac 2025 date des épreuves", ""):
            with self.subTest(text=text):
                self.assertEqual(personal_data_warning(text), "")

    def test_confirmation_question_carries_the_warning(self):
        question = WebSearchTool().confirmation_prompt({"query": "note de lea@example.org"})

        self.assertIn("e-mail address", question)
        self.assertIn('"note de lea@example.org"', question)

    def test_clean_query_has_no_warning_in_the_question(self):
        self.assertNotIn("Warning", WebSearchTool().confirmation_prompt({"query": "python"}))


class McpEnvironmentTests(unittest.TestCase):
    ENVIRON = {
        "PATH": "/usr/bin",
        "HOME": "/home/me",
        "LC_ALL": "fr_FR.UTF-8",
        "UV_CACHE_DIR": "/cache",
        "HTTPS_PROXY": "http://proxy",
        "OPENAI_API_KEY": "sk-secret",
        "GITHUB_TOKEN": "ghp_secret",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
    }

    def test_secrets_are_dropped_and_runtime_variables_kept(self):
        env = safe_environment(self.ENVIRON)

        self.assertEqual(set(env), {"PATH", "HOME", "LC_ALL", "UV_CACHE_DIR", "HTTPS_PROXY"})

    def test_server_starts_with_filtered_environment_plus_explicit_variables(self):
        client = MCPClient("server", [], {"MY_KEY": "explicit"})
        with patch.dict(os.environ, self.ENVIRON, clear=True), patch("mcp.client.subprocess.Popen") as popen:
            popen.side_effect = OSError("stop here")
            with self.assertRaises(Exception):
                client.start()

        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["MY_KEY"], "explicit")
        self.assertEqual(env["PATH"], "/usr/bin")
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("GITHUB_TOKEN", env)

    def test_server_without_explicit_env_still_gets_a_filtered_one(self):
        client = MCPClient("server", [])
        with patch.dict(os.environ, self.ENVIRON, clear=True), patch("mcp.client.subprocess.Popen") as popen:
            popen.side_effect = OSError("stop here")
            with self.assertRaises(Exception):
                client.start()

        self.assertNotIn("OPENAI_API_KEY", popen.call_args.kwargs["env"])


if __name__ == "__main__":
    unittest.main()
