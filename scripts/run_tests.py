"""Run SensAI tests with a compact, feature-oriented summary.

Use ``python3 scripts/run_tests.py`` for the everyday overview.  Keep using
``unittest -v`` when investigating an individual failing test.
"""

from __future__ import annotations

import sys
import unittest
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIRECTORY = PROJECT_ROOT / "tests"

FEATURES = {
    "test_chunking": "Chunking",
    "test_cli": "CLI",
    "test_context": "RAG context",
    "test_embeddings": "Embeddings",
    "test_functional_prompts": "Functional prompts",
    "test_ingestion": "Document ingestion",
    "test_ollama_client": "Ollama client",
    "test_rag_integration": "RAG end-to-end",
    "test_storage": "Local storage",
    "test_tools": "Tools",
}


class FeatureTestResult(unittest.TestResult):
    """Collect test outcomes by feature without printing every successful test."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcomes: dict[str, list[str]] = defaultdict(list)

    def addSuccess(self, test):
        super().addSuccess(test)
        self.outcomes[_feature_name(test)].append("passed")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.outcomes[_feature_name(test)].append("failed")

    def addError(self, test, err):
        super().addError(test, err)
        self.outcomes[_feature_name(test)].append("error")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.outcomes[_feature_name(test)].append("skipped")


def _feature_name(test: unittest.TestCase) -> str:
    module_name = test.__class__.__module__.rsplit(".", maxsplit=1)[-1]
    return FEATURES.get(module_name, module_name.replace("test_", "").replace("_", " ").title())


def _print_failures(result: FeatureTestResult) -> None:
    problems = [*result.failures, *result.errors]
    if not problems:
        return

    print("\nFailures and errors:")
    for test, traceback in problems:
        last_line = next((line.strip() for line in reversed(traceback.splitlines()) if line.strip()), "")
        print(f"  ✗ {test.id()}")
        print(f"    {last_line}")
    print("\nFor the full traceback, run: python3 -m unittest discover -s tests -v")


def main() -> int:
    suite = unittest.defaultTestLoader.discover(str(TEST_DIRECTORY))
    result = FeatureTestResult()
    suite.run(result)

    print("\nSensAI test summary")
    print("=" * 52)
    for feature in sorted(result.outcomes):
        outcomes = result.outcomes[feature]
        passed = outcomes.count("passed")
        failed = outcomes.count("failed") + outcomes.count("error")
        skipped = outcomes.count("skipped")
        status = "✓" if failed == 0 else "✗"
        details = f"{passed}/{len(outcomes)} passed"
        if failed:
            details += f", {failed} failed"
        if skipped:
            details += f", {skipped} skipped"
        print(f"{status} {feature:<22} {details}")

    print("=" * 52)
    if result.wasSuccessful():
        print(f"✓ All {result.testsRun} tests passed.")
    else:
        print(f"✗ {result.testsRun} tests run: {len(result.failures)} failures, {len(result.errors)} errors.")
        _print_failures(result)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
