import io
import os
import shutil
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Ensure `source` directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

# pyrefly: ignore [missing-import]
from cli.commands.memory import _dispatch_save
# pyrefly: ignore [missing-import]
from cli.commands.system import _dispatch_exit
# pyrefly: ignore [missing-import]
from cli.session import CliSession
# pyrefly: ignore [missing-import]
from memory.artifact_manager import ArtifactManager, is_artifact_file
# pyrefly: ignore [missing-import]
from memory.session_manager import SessionManager


class TestArtifactManagerInit(unittest.TestCase):
    """Tests for ArtifactManager initialization and empty state."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ArtifactManager(session_id="test_session", storage_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initial_state_empty(self):
        self.assertFalse(self.manager.has_artifact())
        self.assertEqual(self.manager.get_content(), "")
        self.assertEqual(self.manager.get_outline(), [])
        self.assertIn("No active artifact", self.manager.get_summary())

    def test_init_document_creates_empty_sections(self):
        artifact = self.manager.init_document("Project Spec")
        self.assertTrue(self.manager.has_artifact())
        self.assertEqual(artifact.title, "Project Spec")
        self.assertEqual(artifact.version, 1)
        self.assertEqual(artifact.sections, {})
        self.assertEqual(self.manager.get_content().strip(), "# Project Spec")

    def test_init_document_with_markdown_template(self):
        template = """# Software Plan
## Architecture
Microservices based with FastAPI.
## Database
PostgreSQL and Redis.
"""
        artifact = self.manager.init_document("Overridden Title", template=template)
        self.assertEqual(artifact.title, "Software Plan")
        self.assertIn("Architecture", artifact.sections)
        self.assertIn("Database", artifact.sections)
        self.assertEqual(artifact.sections["Architecture"], "Microservices based with FastAPI.")


class TestArtifactManagerEditing(unittest.TestCase):
    """Tests for incremental editing operations."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ArtifactManager(session_id="test_session", storage_dir=self.temp_dir)
        self.manager.init_document("Design Document")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_update_section_replace_mode(self):
        self.manager.update_section("Overview", "First version of overview.")
        self.assertEqual(self.manager.artifact.sections["Overview"], "First version of overview.")
        self.assertEqual(self.manager.artifact.version, 2)

        # Replace it
        self.manager.update_section("Overview", "Updated overview content.", mode="replace")
        self.assertEqual(self.manager.artifact.sections["Overview"], "Updated overview content.")
        self.assertEqual(self.manager.artifact.version, 3)

    def test_update_section_append_mode(self):
        self.manager.update_section("Notes", "Point 1")
        self.manager.update_section("Notes", "Point 2", mode="append")
        expected = "Point 1\n\nPoint 2"
        self.assertEqual(self.manager.artifact.sections["Notes"], expected)

    def test_update_section_prepend_mode(self):
        self.manager.update_section("Log", "Old entry")
        self.manager.update_section("Log", "New entry", mode="prepend")
        expected = "New entry\n\nOld entry"
        self.assertEqual(self.manager.artifact.sections["Log"], expected)

    def test_replace_text_in_section(self):
        self.manager.update_section("Config", "port: 8080\nhost: localhost")
        replaced = self.manager.replace_text("8080", "9090")
        self.assertTrue(replaced)
        self.assertIn("port: 9090", self.manager.artifact.sections["Config"])

    def test_replace_text_not_found(self):
        self.manager.update_section("Config", "port: 8080")
        replaced = self.manager.replace_text("nonexistent", "anything")
        self.assertFalse(replaced)

    def test_delete_section(self):
        self.manager.update_section("SectionA", "Content A")
        self.manager.update_section("SectionB", "Content B")
        self.assertEqual(len(self.manager.get_outline()), 2)

        deleted = self.manager.delete_section("SectionA")
        self.assertTrue(deleted)
        self.assertNotIn("SectionA", self.manager.artifact.sections)
        self.assertIn("SectionB", self.manager.artifact.sections)

        # Deleting non-existent section returns False
        self.assertFalse(self.manager.delete_section("NonExistent"))

    def test_render_markdown(self):
        self.manager.update_section("Intro", "This is an introduction.")
        self.manager.update_section("Details", "Technical details here.")
        markdown = self.manager.get_content()
        self.assertIn("# Design Document", markdown)
        self.assertIn("## Intro\n\nThis is an introduction.", markdown)
        self.assertIn("## Details\n\nTechnical details here.", markdown)


class TestArtifactManagerPersistence(unittest.TestCase):
    """Tests for saving and loading artifacts."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ArtifactManager(session_id="session_123", storage_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_artifact(self):
        self.manager.init_document("Session Report")
        self.manager.update_section("Summary", "All tasks completed.")

        saved_path = self.manager.save()
        self.assertTrue(saved_path.exists())
        json_path = saved_path.with_suffix(".json")
        self.assertTrue(json_path.exists())

        # Load into a new manager instance
        new_manager = ArtifactManager(session_id="session_123", storage_dir=self.temp_dir)
        loaded_artifact = new_manager.load()

        self.assertEqual(loaded_artifact.title, "Session Report")
        self.assertEqual(loaded_artifact.sections.get("Summary"), "All tasks completed.")
        self.assertEqual(loaded_artifact.version, 2)

    @unittest.skipUnless(os.name == "posix", "file permissions are POSIX only")
    def test_saved_files_and_folder_are_owner_only(self):
        storage = Path(self.temp_dir) / "sessions"
        manager = ArtifactManager(session_id="s1", storage_dir=str(storage))
        manager.init_document("Suivi PAP Tom Durand")
        md_path = manager.save()

        for path in (md_path, md_path.with_suffix(".json")):
            with self.subTest(path=path.name):
                self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(storage).st_mode), 0o700)

    @unittest.skipUnless(os.name == "posix", "file permissions are POSIX only")
    def test_existing_world_readable_artifact_is_tightened_on_save(self):
        self.manager.init_document("Notes")
        md_path = self.manager.save()
        os.chmod(md_path, 0o644)
        os.chmod(md_path.with_suffix(".json"), 0o644)

        self.manager.save()

        self.assertEqual(stat.S_IMODE(os.stat(md_path).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(md_path.with_suffix(".json")).st_mode), 0o600)
        self.assertIn("# Notes", md_path.read_text(encoding="utf-8"))

    def test_delete_removes_only_this_session_files(self):
        self.manager.init_document("Mine")
        self.manager.save()
        other = ArtifactManager(session_id="other", storage_dir=self.temp_dir)
        other.init_document("Other")
        other.save()

        self.assertTrue(self.manager.delete())
        self.assertFalse(self.manager.delete())
        self.assertEqual(sorted(p.name for p in Path(self.temp_dir).iterdir()), ["other_artifact.json", "other_artifact.md"])
        self.assertTrue(self.manager.has_artifact())  # memory is cleared by reset(), not by delete()

    def test_delete_all_counts_documents_not_files(self):
        for session_id in ("a", "b"):
            manager = ArtifactManager(session_id=session_id, storage_dir=self.temp_dir)
            manager.init_document(session_id)
            manager.save()
        (Path(self.temp_dir) / "a.json").write_text("{}", encoding="utf-8")  # a conversation, not an artifact

        self.assertEqual(self.manager.delete_all(), 2)
        self.assertEqual([p.name for p in Path(self.temp_dir).iterdir()], ["a.json"])
        self.assertEqual(ArtifactManager(storage_dir=str(Path(self.temp_dir) / "missing")).delete_all(), 0)

    def test_artifact_files_are_recognised_by_name(self):
        self.assertTrue(is_artifact_file(Path("s1_artifact.md")))
        self.assertTrue(is_artifact_file(Path("s1_artifact.json")))
        self.assertFalse(is_artifact_file(Path("s1.json")))
        self.assertFalse(is_artifact_file(Path("s1_artifact.txt")))

    def test_load_fallback_to_markdown_file(self):
        # Manually create a raw markdown file without json metadata
        md_file = Path(self.temp_dir) / "custom_doc.md"
        md_content = "# Architecture\n\n## Backend\nFastAPI app.\n\n## Frontend\nReact app."
        md_file.write_text(md_content, encoding="utf-8")

        manager = ArtifactManager(storage_dir=self.temp_dir)
        loaded = manager.load(filepath=md_file)
        self.assertEqual(loaded.title, "Architecture")
        self.assertEqual(loaded.sections.get("Backend"), "FastAPI app.")
        self.assertEqual(loaded.sections.get("Frontend"), "React app.")


class TestArtifactPersistenceCommands(unittest.TestCase):
    """/save and exit report an artifact that could not be written instead of hiding it."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.conversation = SessionManager(session_id="s1", storage_dir=folder.name)
        self.conversation.add_message("user", "Hello")
        self.artifacts = ArtifactManager(session_id="s1", storage_dir=folder.name)
        self.artifacts.init_document("Lesson plan")
        self.context = SimpleNamespace(
            feature_config=SimpleNamespace(memory_enabled=True),
            conversation_session=self.conversation,
            artifact_manager=self.artifacts,
        )
        self.session = CliSession()

    def run_command(self, handler, text):
        output = io.StringIO()
        with redirect_stdout(output):
            handler(text, self.session, self.context)
        return output.getvalue()

    def test_save_reports_both_files(self):
        output = self.run_command(_dispatch_save, "/save")

        self.assertIn("Session saved at", output)
        self.assertIn("Artifact saved at", output)

    def test_save_reports_a_failed_artifact_instead_of_success(self):
        with patch.object(self.artifacts, "save", side_effect=OSError("No space left on device")):
            output = self.run_command(_dispatch_save, "/save")

        self.assertIn("The artifact could not be saved: No space left on device", output)
        self.assertNotIn("Artifact saved", output)

    def test_save_of_an_empty_conversation_says_so(self):
        self.conversation.clear()
        self.artifacts.reset()

        output = self.run_command(_dispatch_save, "/save")

        self.assertIn("The conversation is empty: no session file was written.", output)
        self.assertNotIn("Session saved", output)

    def test_exit_logs_a_failed_artifact_and_still_exits(self):
        with patch.object(self.artifacts, "save", side_effect=OSError("disk full")), \
                self.assertLogs("sensai.commands.system", level="WARNING") as logs:
            output = self.run_command(_dispatch_exit, "exit")

        self.assertIn("artifact.save_failed_on_exit", logs.output[0])
        self.assertIn("the artifact could not be saved: disk full", output)
        self.assertTrue(self.session.exit_requested)
        self.assertTrue(self.session.persistence_saved_on_exit)


if __name__ == "__main__":
    unittest.main()
