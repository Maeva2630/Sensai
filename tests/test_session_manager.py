import json
import os
import shutil
import sys
import unittest
from pathlib import Path

# Ensure `source` directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

# pyrefly: ignore [missing-import]
from memory.session_manager import SessionManager


class TestSessionManagerInit(unittest.TestCase):
    """Tests for SessionManager initialisation."""

    def test_generates_session_id_when_none_given(self):
        session = SessionManager()
        self.assertIsNotNone(session.session_id)
        self.assertIsInstance(session.session_id, str)

    def test_uses_provided_session_id(self):
        session = SessionManager(session_id="my-id")
        self.assertEqual(session.session_id, "my-id")

    def test_messages_empty_on_init(self):
        session = SessionManager()
        self.assertEqual(session.messages, [])


class TestSessionManagerMessages(unittest.TestCase):
    """Tests for in-memory message management."""

    def setUp(self):
        self.session = SessionManager()

    def test_add_message_appends_correctly(self):
        self.session.add_message("user", "Hello")
        self.assertEqual(self.session.messages, [{"role": "user", "content": "Hello"}])

    def test_add_multiple_messages(self):
        self.session.add_message("user", "Hello")
        self.session.add_message("assistant", "Hi!")
        self.assertEqual(len(self.session.messages), 2)
        self.assertEqual(self.session.messages[1]["role"], "assistant")

    def test_get_messages_returns_list(self):
        self.session.add_message("user", "test")
        result = self.session.get_messages()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    def test_clear_empties_messages(self):
        self.session.add_message("user", "Hello")
        self.session.clear()
        self.assertEqual(self.session.messages, [])


class TestSessionManagerSave(unittest.TestCase):
    """Tests for session save to disk."""

    def setUp(self):
        self.tmp_dir = Path("/tmp/sensai_test_sessions")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.session = SessionManager(session_id="test-save-id", storage_dir=str(self.tmp_dir))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_creates_json_file(self):
        self.session.add_message("user", "Hello")
        path = self.session.save()
        self.assertTrue(Path(path).exists())

    def test_save_returns_correct_path(self):
        path = self.session.save()
        self.assertIn("test-save-id", path)
        self.assertTrue(path.endswith(".json"))

    def test_save_file_contains_correct_structure(self):
        self.session.add_message("user", "Hello")
        self.session.add_message("assistant", "Hi!")
        path = self.session.save()

        with open(path, "r") as f:
            data = json.load(f)

        self.assertEqual(data["session_id"], "test-save-id")
        self.assertIn("timestamp", data)
        self.assertEqual(len(data["messages"]), 2)
        self.assertEqual(data["messages"][0], {"role": "user", "content": "Hello"})

    def test_save_creates_storage_dir_if_missing(self):
        new_dir = self.tmp_dir / "subdir"
        session = SessionManager(session_id="sub-id", storage_dir=str(new_dir))
        session.save()
        self.assertTrue(new_dir.exists())


class TestSessionManagerLoad(unittest.TestCase):
    """Tests for session load from disk."""

    def setUp(self):
        self.tmp_dir = Path("/tmp/sensai_test_sessions_load")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write_session_file(self, session_id: str, messages: list) -> None:
        data = {"session_id": session_id, "timestamp": "2026-01-01T00:00:00", "messages": messages}
        path = self.tmp_dir / f"{session_id}.json"
        with open(path, "w") as f:
            json.dump(data, f)

    def test_load_returns_messages(self):
        self._write_session_file("abc123", [{"role": "user", "content": "Hey"}])
        session = SessionManager(session_id="abc123", storage_dir=str(self.tmp_dir))
        messages = session.load("abc123")
        self.assertEqual(messages, [{"role": "user", "content": "Hey"}])

    def test_load_updates_session_id(self):
        self._write_session_file("abc123", [])
        session = SessionManager(storage_dir=str(self.tmp_dir))
        session.load("abc123")
        self.assertEqual(session.session_id, "abc123")

    def test_load_updates_in_memory_messages(self):
        self._write_session_file("abc123", [{"role": "assistant", "content": "Hi"}])
        session = SessionManager(storage_dir=str(self.tmp_dir))
        session.load("abc123")
        self.assertEqual(session.messages, [{"role": "assistant", "content": "Hi"}])

    def test_load_raises_file_not_found_for_missing_session(self):
        session = SessionManager(storage_dir=str(self.tmp_dir))
        with self.assertRaises(FileNotFoundError) as ctx:
            session.load("nonexistent-id")
        self.assertIn("nonexistent-id", str(ctx.exception))

    def test_load_uses_own_session_id_when_none_given(self):
        self._write_session_file("my-id", [{"role": "user", "content": "test"}])
        session = SessionManager(session_id="my-id", storage_dir=str(self.tmp_dir))
        messages = session.load()
        self.assertEqual(len(messages), 1)


if __name__ == "__main__":
    unittest.main()
