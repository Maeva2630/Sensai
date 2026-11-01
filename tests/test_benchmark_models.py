"""Unit tests for local benchmark configuration variants."""

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).parent.parent
SCRIPT_PATH = ROOT / "scripts" / "benchmark_models.py"
SPEC = importlib.util.spec_from_file_location("benchmark_models", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark_models = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark_models
SPEC.loader.exec_module(benchmark_models)


class TestBenchmarkConfiguration(unittest.TestCase):
    def setUp(self):
        self.cases = json.loads((ROOT / "benchmarks" / "prompts.json").read_text(encoding="utf-8"))["cases"]

    def test_disabled_rag_removes_context_without_mutating_prompt_source(self):
        configured_cases = benchmark_models._configured_cases(self.cases, rag_enabled=False)
        configured_rag_case = next(case for case in configured_cases if case["id"] == "rag_context_compliance")
        source_rag_case = next(case for case in self.cases if case["id"] == "rag_context_compliance")

        self.assertIn("No RAG context is available", configured_rag_case["messages"][0]["content"])
        self.assertNotIn("RAG CONTEXT", configured_rag_case["messages"][0]["content"])
        self.assertIn("RAG CONTEXT", source_rag_case["messages"][0]["content"])

    def test_enabled_rag_preserves_context(self):
        configured_cases = benchmark_models._configured_cases(self.cases, rag_enabled=True)
        rag_case = next(case for case in configured_cases if case["id"] == "rag_context_compliance")

        self.assertIn("RAG CONTEXT", rag_case["messages"][0]["content"])

    def test_unavailable_measurement_is_not_replaced_with_zero(self):
        self.assertEqual(benchmark_models._format_measurement(None, "MiB"), "unavailable")
        self.assertEqual(benchmark_models._format_measurement(42, "MiB"), "42.00 MiB")


class TestBenchmarkPortability(unittest.TestCase):
    def test_ollama_memory_is_unavailable_without_proc(self):
        """macOS and Windows have no /proc: the sampler must not crash before the first request."""
        with patch.object(benchmark_models.Path, "iterdir", side_effect=FileNotFoundError("/proc")):
            self.assertIsNone(benchmark_models._ollama_rss_mib())
            sampler = benchmark_models.ResourceSampler()
            sampler._append_sample()

        self.assertIsNone(sampler.samples[0]["ollama_rss_mib"])
        self.assertIsNone(sampler.summary()["ollama_rss_peak_mib"])

    def test_script_loads_without_the_python_3_11_utc_alias(self):
        """SensAI supports Python 3.10, where datetime.UTC does not exist."""
        code = (
            "import datetime, runpy, sys; datetime.__dict__.pop('UTC', None); "
            "module = runpy.run_path(sys.argv[1]); module['_environment']()"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code, str(SCRIPT_PATH)], capture_output=True, text=True, timeout=60
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
