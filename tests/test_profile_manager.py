import json
import tempfile
import unittest
from pathlib import Path
from memory.profile_manager import ProfileManager
from personas.teacher_assistant import TEACHER_ASSISTANT_BASE_PROMPT


class TestProfileManager(unittest.TestCase):
    """Unit tests for ProfileManager class."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_profile_creation_when_file_missing(self):
        profile_file = self.temp_path / "profile.json"
        self.assertFalse(profile_file.exists())

        manager = ProfileManager(profile_path=profile_file)

        self.assertTrue(profile_file.exists())
        self.assertEqual(manager.data, ProfileManager.DEFAULT_PROFILE)

    def test_load_existing_custom_profile(self):
        profile_file = self.temp_path / "custom_profile.json"
        custom_data = {
            "user_name": "Alice",
            "preferred_language": "French",
            "domain_expertise": "DevOps",
            "constraints": ["Keep responses concise", "No emojis"],
        }
        with open(profile_file, "w", encoding="utf-8") as f:
            json.dump(custom_data, f)

        manager = ProfileManager(profile_path=profile_file)

        self.assertEqual(manager.data["user_name"], "Alice")
        self.assertEqual(manager.data["preferred_language"], "French")
        self.assertEqual(len(manager.data["constraints"]), 2)

    def test_build_system_prompt_structure(self):
        profile_file = self.temp_path / "profile.json"
        manager = ProfileManager(profile_path=profile_file)
        manager.data = {
            "user_name": "Bob",
            "preferred_language": "English",
            "domain_expertise": "Backend Python",
            "constraints": ["Always provide unit tests"],
        }

        prompt = manager.build_system_prompt(base_instructions="You are a helpful assistant.")

        self.assertIn("You are a helpful assistant.", prompt)
        self.assertIn("Name: Bob", prompt)
        self.assertIn("Preferred Language: English", prompt)
        self.assertIn("Domain Expertise: Backend Python", prompt)
        self.assertIn("- Always provide unit tests", prompt)

    def test_build_system_prompt_empty_constraints(self):
        profile_file = self.temp_path / "profile.json"
        manager = ProfileManager(profile_path=profile_file)
        manager.data = {
            "user_name": "Charlie",
            "preferred_language": "French",
            "domain_expertise": "General",
            "constraints": [],
        }

        prompt = manager.build_system_prompt()

        self.assertIn("None", prompt)

    def test_default_system_prompt_uses_the_teacher_assistant_persona(self):
        profile_file = self.temp_path / "profile.json"
        manager = ProfileManager(profile_path=profile_file)

        prompt = manager.build_system_prompt()

        self.assertIn(TEACHER_ASSISTANT_BASE_PROMPT, prompt)
        self.assertIn("educational assistant that supports", prompt)
        self.assertIn("primary user is an educator, not a learner", prompt)
        self.assertIn("retrieved documents and RAG instructions", prompt)


if __name__ == "__main__":
    unittest.main()
