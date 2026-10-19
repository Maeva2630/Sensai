import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from cli.cli import cli
from config import AppConfig, ConfigurationError, FeatureConfig, ModelConfig, RagConfig, load_config


class TestApplicationConfig(unittest.TestCase):
    def _write_config(self, data: dict) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_loads_all_configurable_sensai_settings(self):
        path = self._write_config(
            {
                "model": {
                    "chat_model": "custom-chat",
                    "embedding_model": "custom-embed",
                    "base_url": "http://localhost:9999",
                    "temperature": 0.4,
                    "context_window": 8192,
                },
                "features": {
                    "rag_enabled": True,
                    "tools_enabled": False,
                    "memory_enabled": False,
                    "source_display_enabled": False,
                },
                "rag": {
                    "top_k": 2,
                    "max_context_characters": 1000,
                    "relevance_threshold": 0.5,
                },
            }
        )

        config = load_config(path)

        self.assertEqual(config.model.chat_model, "custom-chat")
        self.assertEqual(config.model.embedding_model, "custom-embed")
        self.assertEqual(config.model.context_window, 8192)
        self.assertFalse(config.features.tools_enabled)
        self.assertEqual(config.rag.relevance_threshold, 0.5)

    def test_rejects_invalid_temperature(self):
        path = self._write_config(
            {
                "model": {
                    "chat_model": "llama3.2",
                    "embedding_model": "nomic-embed-text",
                    "base_url": "http://localhost:11434",
                    "temperature": 3,
                    "context_window": 4096,
                },
                "features": {
                    "rag_enabled": False,
                    "tools_enabled": True,
                    "memory_enabled": False,
                    "source_display_enabled": True,
                },
                "rag": {
                    "top_k": 3,
                    "max_context_characters": 4000,
                    "relevance_threshold": 0.35,
                },
            }
        )

        with self.assertRaises(ConfigurationError):
            load_config(path)

    @patch("cli.cli.OllamaHTTPClient")
    def test_cli_passes_model_options_and_disables_tools_from_config(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        config = AppConfig(
            model=ModelConfig(
                chat_model="configured-chat",
                embedding_model="configured-embed",
                temperature=0.7,
                context_window=8192,
            ),
            features=FeatureConfig(tools_enabled=False),
            rag=RagConfig(),
        )

        with patch("builtins.input", side_effect=["exit"]):
            cli(app_config=config)

        chat_call = mock_client_cls.call_args_list[0]
        self.assertEqual(chat_call.kwargs["model"], "configured-chat")
        self.assertEqual(chat_call.kwargs["options"], {"temperature": 0.7, "num_ctx": 8192})
        self.assertEqual(mock_client_cls.call_args_list[1].kwargs["model"], "configured-embed")

