import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure `source` directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

# pyrefly: ignore [missing-import]
from memory.artifact_manager import Artifact, ArtifactManager


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


if __name__ == "__main__":
    unittest.main()
