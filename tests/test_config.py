"""Tests for translating and validating application configuration."""

import json
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from cli.model_factory import create_rag_context_config
from config import AppConfig, ConfigurationError, FeatureConfig, RagConfig, load_config
from main import main
from rag.storage import VectorStoreError


class TestRagConfiguration(unittest.TestCase):
    def test_cli_translates_the_supported_rag_settings(self):
        settings = create_rag_context_config(
            RagConfig(top_k=5, max_context_characters=1200, relevance_threshold=0.6)
        )

        self.assertEqual(settings.top_k, 5)
        self.assertEqual(settings.max_context_characters, 1200)
        self.assertEqual(settings.relevance_threshold, 0.6)


class TestConfigurationLoading(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temporary_directory.name) / "config.json"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_config(self, changes=None):
        data = {
            "model": {
                "chat_model": "test-chat",
                "embedding_model": "test-embed",
                "base_url": "http://localhost:11434",
                "temperature": 0.2,
                "context_window": 4096,
            },
            "features": {
                "rag_enabled": True,
                "tools_enabled": True,
                "memory_enabled": False,
                "source_display_enabled": True,
            },
            "rag": {
                "top_k": 3,
                "max_context_characters": 4000,
                "relevance_threshold": 0.35,
            },
            "logging": {
                "level": "DEBUG",
                "file_path": "logs/test.log",
            },
        }
        for section, values in (changes or {}).items():
            data.setdefault(section, {}).update(values)
        self.config_path.write_text(json.dumps(data), encoding="utf-8")

    def test_loads_a_complete_valid_configuration(self):
        self.write_config()

        configuration = load_config(self.config_path)

        self.assertEqual(configuration.model.chat_model, "test-chat")
        self.assertEqual(configuration.model.context_window, 4096)
        self.assertTrue(configuration.features.rag_enabled)
        self.assertEqual(configuration.rag.top_k, 3)
        self.assertEqual(configuration.logging.level, "DEBUG")
        self.assertEqual(configuration.personas.active_persona, "teacher-assistant")

    def test_rejects_invalid_json(self):
        self.config_path.write_text("{ not valid JSON", encoding="utf-8")

        with self.assertRaisesRegex(ConfigurationError, "Invalid JSON"):
            load_config(self.config_path)

    def test_uses_safe_logging_defaults_when_the_optional_section_is_absent(self):
        self.write_config()
        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        del data["logging"]
        self.config_path.write_text(json.dumps(data), encoding="utf-8")

        configuration = load_config(self.config_path)

        self.assertEqual(configuration.logging.level, "INFO")
        self.assertEqual(configuration.logging.file_path, "logs/sensai.log")

    def test_rejects_missing_required_section(self):
        self.write_config()
        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        del data["features"]
        self.config_path.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaisesRegex(ConfigurationError, "features"):
            load_config(self.config_path)

    def test_rejects_invalid_values_before_startup(self):
        cases = (
            ({"model": {"temperature": 2.1}}, "temperature"),
            ({"model": {"context_window": 0}}, "context_window"),
            ({"model": {"chat_model": ""}}, "chat_model"),
            ({"features": {"rag_enabled": "yes"}}, "rag_enabled"),
            ({"rag": {"relevance_threshold": -1.1}}, "relevance_threshold"),
            ({"rag": {"top_k": 0}}, "top_k"),
            ({"logging": {"level": "TRACE"}}, "level"),
            ({"personas": {"active_persona": "missing", "profiles": []}}, "profiles"),
        )
        for changes, expected_field in cases:
            with self.subTest(changes=changes):
                self.write_config(changes)
                with self.assertRaisesRegex(ConfigurationError, expected_field):
                    load_config(self.config_path)

    def test_loads_and_selects_a_configured_persona(self):
        self.write_config(
            {
                "personas": {
                    "active_persona": "math-assistant",
                    "profiles": [
                        {
                            "identifier": "math-assistant",
                            "display_name": "Mathematics assistant",
                            "role": "Supports mathematics teachers",
                            "tone": "Precise and practical",
                            "system_prompt": "Support mathematics teachers.",
                            "response_scope": "Lesson preparation and assessment.",
                        },
                        {
                            "identifier": "history-assistant",
                            "display_name": "History assistant",
                            "role": "Supports history teachers",
                            "tone": "Clear and contextual",
                            "system_prompt": "Support history teachers.",
                            "response_scope": "Lesson preparation and assessment.",
                        },
                    ],
                }
            }
        )

        configuration = load_config(self.config_path)

        self.assertEqual(configuration.personas.active.display_name, "Mathematics assistant")
        self.assertEqual(configuration.personas.active.system_prompt, "Support mathematics teachers.")
        self.assertIn("Role: Supports mathematics teachers", configuration.personas.active.system_instructions)

    def test_loads_a_persona_prompt_from_a_file_relative_to_the_configuration(self):
        prompts_directory = self.config_path.parent / "prompts"
        prompts_directory.mkdir()
        (prompts_directory / "teacher.txt").write_text("Prompt loaded from a file.", encoding="utf-8")
        self.write_config(
            {
                "personas": {
                    "active_persona": "teacher-assistant",
                    "profiles": [
                        {
                            "identifier": "teacher-assistant",
                            "display_name": "Teacher assistant",
                            "role": "Supports teachers",
                            "tone": "Practical",
                            "prompt_file": "prompts/teacher.txt",
                            "response_scope": "Teaching support.",
                        }
                    ],
                }
            }
        )

        configuration = load_config(self.config_path)

        self.assertEqual(configuration.personas.active.system_prompt, "Prompt loaded from a file.")

    def test_rejects_personas_with_both_or_no_prompt_source(self):
        for prompt_fields in ({}, {"system_prompt": "Inline", "prompt_file": "prompt.txt"}):
            with self.subTest(prompt_fields=prompt_fields):
                self.write_config(
                    {
                        "personas": {
                            "active_persona": "teacher-assistant",
                            "profiles": [
                                {
                                    "identifier": "teacher-assistant",
                                    "display_name": "Teacher assistant",
                                    "role": "Supports teachers",
                                    "tone": "Practical",
                                    "response_scope": "Teaching support.",
                                    **prompt_fields,
                                }
                            ],
                        }
                    }
                )

                with self.assertRaisesRegex(ConfigurationError, "exactly one"):
                    load_config(self.config_path)


class TestApplicationStartup(unittest.TestCase):
    def test_main_reports_invalid_configuration_without_starting_the_cli(self):
        with patch("main.load_config", side_effect=ConfigurationError("invalid value")), patch(
            "main.cli"
        ) as mock_cli, patch("sys.stdout", new_callable=io.StringIO) as output:
            with patch("sys.argv", ["main.py"]):
                main()

        mock_cli.assert_not_called()
        self.assertIn("Configuration error: invalid value", output.getvalue())

    def test_main_reports_corrupted_rag_database_without_a_traceback(self):
        with patch("main.load_config", return_value=AppConfig()), patch("main.configure_logging"), patch(
            "main.cli", side_effect=VectorStoreError("database is corrupted")
        ), patch("sys.stdout", new_callable=io.StringIO) as output:
            with patch("sys.argv", ["main.py"]):
                main()

        self.assertIn("RAG startup error: database is corrupted", output.getvalue())

    def test_main_starts_cleanly_when_memory_is_enabled(self):
        configuration = AppConfig(features=FeatureConfig(memory_enabled=True))
        with patch("main.load_config", return_value=configuration), patch("main.configure_logging"), patch(
            "main.cli"
        ) as mock_cli, patch("sys.stdout", new_callable=io.StringIO) as output:
            with patch("sys.argv", ["main.py"]):
                main()

        mock_cli.assert_called_once_with(
            session_id=None,
            profile_path="profile.json",
            app_config=configuration,
        )
        self.assertEqual(output.getvalue(), "")
if __name__ == "__main__":
    unittest.main()
