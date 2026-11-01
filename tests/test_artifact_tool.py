import os
import sys
import unittest

# Ensure `source` directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

# pyrefly: ignore [missing-import]
from memory.artifact_manager import ArtifactManager
# pyrefly: ignore [missing-import]
from tools.artifact_tool import ArtifactTool


class TestArtifactTool(unittest.TestCase):
    """Tests for the LLM Artifact editing tool."""

    def setUp(self):
        self.manager = ArtifactManager()
        self.tool = ArtifactTool(manager=self.manager)

    def test_tool_definition(self):
        self.assertEqual(self.tool.name, "edit_artifact")
        ollama_tool = self.tool.to_ollama_tool()
        self.assertEqual(ollama_tool["type"], "function")
        self.assertEqual(ollama_tool["function"]["name"], "edit_artifact")
        self.assertIn("action", self.tool.parameters["properties"])

    def test_execute_init(self):
        result = self.tool.execute({"action": "init", "title": "API Specification"})
        self.assertIn("Artifact initialized: 'API Specification'", result)
        self.assertTrue(self.manager.has_artifact())
        self.assertEqual(self.manager.artifact.title, "API Specification")

    def test_execute_update_section(self):
        self.tool.execute({"action": "init", "title": "API Specification"})
        result = self.tool.execute({
            "action": "update_section",
            "section_name": "Endpoints",
            "content": "GET /api/v1/users",
            "mode": "replace",
        })
        self.assertIn("Section 'Endpoints' updated", result)
        self.assertEqual(self.manager.artifact.sections["Endpoints"], "GET /api/v1/users")

    def test_execute_update_section_missing_name(self):
        self.tool.execute({"action": "init", "title": "Doc"})
        result = self.tool.execute({
            "action": "update_section",
            "content": "some content",
        })
        self.assertIn("Error: 'section_name' is required", result)

    def test_execute_replace_text(self):
        self.tool.execute({"action": "init", "title": "Doc"})
        self.tool.execute({
            "action": "update_section",
            "section_name": "Config",
            "content": "port: 3000",
        })
        result = self.tool.execute({
            "action": "replace_text",
            "target": "3000",
            "replacement": "8080",
        })
        self.assertIn("Replaced occurrences", result)
        self.assertIn("port: 8080", self.manager.artifact.sections["Config"])

    def test_execute_replace_text_not_found(self):
        self.tool.execute({"action": "init", "title": "Doc"})
        result = self.tool.execute({
            "action": "replace_text",
            "target": "missing_text",
            "replacement": "new_text",
        })
        self.assertIn("not found", result)

    def test_execute_delete_section(self):
        self.tool.execute({"action": "init", "title": "Doc"})
        self.tool.execute({
            "action": "update_section",
            "section_name": "Temporary",
            "content": "Delete me",
        })
        result = self.tool.execute({
            "action": "delete_section",
            "section_name": "Temporary",
        })
        self.assertIn("deleted", result)
        self.assertNotIn("Temporary", self.manager.artifact.sections)

    def test_execute_view(self):
        # View when empty
        self.assertIn("No active artifact", self.tool.execute({"action": "view"}))

        # View when initialized
        self.tool.execute({"action": "init", "title": "Notes"})
        self.tool.execute({
            "action": "update_section",
            "section_name": "Ideas",
            "content": "Idea 1",
        })
        view_result = self.tool.execute({"action": "view"})
        self.assertIn("# Notes", view_result)
        self.assertIn("## Ideas", view_result)

    def test_is_relevant_for(self):
        # Before init, matches keywords
        self.assertTrue(self.tool.is_relevant_for("Can you create an artifact for this spec?"))
        self.assertTrue(self.tool.is_relevant_for("Update the shared document please"))
        self.assertFalse(self.tool.is_relevant_for("What is the weather today?"))
        # Ordinary lesson words do not offer the artifact editor any more.
        self.assertFalse(self.tool.is_relevant_for("Help me write a lesson plan with notes and a summary"))

        # Once active, always relevant
        self.tool.execute({"action": "init", "title": "Notes"})
        self.assertTrue(self.tool.is_relevant_for("What is the weather today?"))


if __name__ == "__main__":
    unittest.main()
