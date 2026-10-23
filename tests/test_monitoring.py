"""Tests for privacy-preserving local monitoring and dashboard aggregation."""

import io
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from cli.cli import cli
from cli.commands.metrics import _format_dashboard, _parse_filters
from cli.tool_calls import run_tool_calls
from config import AppConfig, FeatureConfig, MonitoringConfig
from observability.monitoring import MetricsFilter, MonitoringStore, create_monitoring_store
from tools.registry import ToolRegistry
from tools.time_tool import GetTimeTool


def response_chunk(text):
    return ('{"message": {"content": "' + text + '"}}').encode()


class TestMonitoringStore(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "metrics.sqlite3"
        self.store = MonitoringStore(self.path, retention_days=30)

    def tearDown(self):
        self.directory.cleanup()

    def test_persists_aggregates_and_filters_safe_interaction_metadata(self):
        first_id = self.store.record_model_interaction(
            model="llama3.2",
            features=FeatureConfig(rag_enabled=True),
            prompt_version="teacher-v1",
            status="success",
            latency_ms=120,
            input_tokens=20,
            output_tokens=10,
            rag_chunk_count=2,
        )
        self.store.record_tool_interaction(
            model_interaction_id=first_id,
            tool_name="get_time",
            status="success",
            latency_ms=8,
        )
        self.store.record_model_interaction(
            model="llama3.2",
            features=FeatureConfig(tools_enabled=False),
            prompt_version="teacher-v2",
            status="error",
            latency_ms=300,
            input_tokens=30,
            output_tokens=None,
            rag_chunk_count=0,
            error_category="OllamaConnectionError",
        )

        all_metrics = self.store.dashboard()
        filtered_metrics = self.store.dashboard(MetricsFilter(prompt_version="teacher-v1"))

        self.assertEqual(all_metrics.model_calls, 2)
        self.assertEqual(all_metrics.rag_calls, 1)
        self.assertEqual(all_metrics.tool_calls, 1)
        self.assertEqual(all_metrics.tool_successes, 1)
        self.assertEqual(all_metrics.tool_failures, 0)
        self.assertEqual(all_metrics.input_tokens, 50)
        self.assertEqual(all_metrics.output_tokens, 10)
        self.assertEqual(all_metrics.error_rate, 50.0)
        self.assertEqual(filtered_metrics.model_calls, 1)
        self.assertEqual(filtered_metrics.average_latency_ms, 120.0)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(
                connection.execute("SELECT component FROM model_interactions").fetchone()[0],
                "model",
            )

    def test_database_schema_has_no_raw_prompt_or_response_columns(self):
        with sqlite3.connect(self.path) as connection:
            interaction_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(model_interactions)")
            }
            tool_columns = {row[1] for row in connection.execute("PRAGMA table_info(tool_interactions)")}

        forbidden = {"prompt", "response", "content", "arguments", "result", "profile", "secret"}
        self.assertTrue(forbidden.isdisjoint(interaction_columns))
        self.assertTrue(forbidden.isdisjoint(tool_columns))

    def test_empty_dashboard_is_safe_and_readable(self):
        output = _format_dashboard(self.store.dashboard(), MetricsFilter())

        self.assertIn("Model calls: 0", output)
        self.assertIn("No recorded interactions.", output)

    def test_disabled_monitoring_does_not_create_a_database(self):
        disabled_path = Path(self.directory.name) / "disabled.sqlite3"

        store = create_monitoring_store(
            MonitoringConfig(enabled=False, database_path=str(disabled_path))
        )
        store.record_model_interaction(model="llama3.2")

        self.assertFalse(disabled_path.exists())

    def test_tool_execution_records_only_name_status_and_duration(self):
        interaction_id = self.store.record_model_interaction(
            model="llama3.2",
            features=FeatureConfig(tools_enabled=True),
            prompt_version="teacher-v1",
            status="success",
            latency_ms=10,
            input_tokens=2,
            output_tokens=1,
            rag_chunk_count=0,
        )
        registry = ToolRegistry()
        registry.register(GetTimeTool())
        messages: list[dict[str, object]] = []

        run_tool_calls(
            registry,
            [{"function": {"name": "get_time", "arguments": {}}}],
            messages,
            self.store,
            interaction_id,
        )

        metrics = self.store.dashboard(MetricsFilter(tool_name="get_time"))
        self.assertEqual(metrics.tool_calls, 1)
        self.assertEqual(metrics.tool_successes, 1)

    def test_metrics_filters_are_parsed_without_accepting_unknown_options(self):
        filters = _parse_filters(
            "/metrics --from 2026-01-01 --model llama3.2 --feature rag"
        )

        self.assertEqual(filters.start_date, "2026-01-01")
        self.assertEqual(filters.model, "llama3.2")
        self.assertEqual(filters.feature, "rag")
        with self.assertRaises(ValueError):
            _parse_filters("/metrics --unknown value")
        with self.assertRaises(ValueError):
            _parse_filters("/metrics --from not-a-date")
        with self.assertRaises(ValueError):
            _parse_filters("/metrics --from 2026-02-01 --to 2026-01-01")


class TestMonitoringCliIntegration(unittest.TestCase):
    @patch("cli.cli.OllamaHTTPClient")
    def test_enabled_monitoring_records_each_model_request_without_storing_text(self, client_class):
        with tempfile.TemporaryDirectory() as directory:
            database_path = str(Path(directory) / "metrics.sqlite3")
            client = MagicMock()
            client.stream_message.return_value = [response_chunk("A private response")]
            client_class.return_value = client
            config = AppConfig(
                monitoring=MonitoringConfig(enabled=True, database_path=database_path, prompt_version="teacher-v1")
            )

            with patch("builtins.input", side_effect=["A private question", "exit"]), patch(
                "sys.stdout", io.StringIO()
            ):
                cli(app_config=config)

            with sqlite3.connect(database_path) as connection:
                stored_text = " ".join(str(value) for row in connection.execute("SELECT * FROM model_interactions") for value in row)
            self.assertNotIn("A private question", stored_text)
            self.assertNotIn("A private response", stored_text)
            self.assertEqual(MonitoringStore(database_path, 30).dashboard().model_calls, 1)
