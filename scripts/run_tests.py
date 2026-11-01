"""Run SensAI tests with a compact, feature-oriented summary.

Use ``python3 scripts/run_tests.py`` for the everyday overview: one line per feature with
its pass rate, then a verdict. What the tests print is hidden, except for a failing test,
whose traceback and captured output are shown. Keep using ``unittest -v`` when
investigating an individual failing test.
"""

from __future__ import annotations

import io
import re
import sys
import unittest
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIRECTORY = PROJECT_ROOT / "tests"

# One label per test module, named after the feature it covers.
FEATURES = {
    "test_artifact_manager": "Artifact - living document",
    "test_artifact_tool": "Artifact - editing tool",
    "test_benchmark_models": "Model benchmark script",
    "test_chunking": "RAG - chunking",
    "test_cli": "CLI - chat loop and commands",
    "test_compression": "Memory - context compression",
    "test_config": "Configuration",
    "test_context": "RAG - context and indexing",
    "test_context_budget": "Context window budget",
    "test_evaluate_rag": "Evaluation - RAG A/B script",
    "test_probe_limits": "Evaluation - model limit probes",
    "test_error_handling": "Error messages",
    "test_embeddings": "RAG - embeddings",
    "test_filesystem_tools": "Tools - file access",
    "test_functional_prompts": "Functional prompt scenarios",
    "test_ingestion": "RAG - document ingestion",
    "test_mcp_client": "MCP - client protocol",
    "test_mcp_config": "MCP - server configuration",
    "test_mcp_manager": "MCP - server manager",
    "test_mcp_third_party": "MCP - third-party server",
    "test_model_fallback": "Model fallback",
    "test_monitoring": "Observability - metrics",
    "test_observability": "Observability - logs and prompt view",
    "test_ollama_client": "Ollama HTTP client",
    "test_permissions": "Tools - file permissions",
    "test_privacy": "Privacy and GDPR",
    "test_profile_manager": "Memory - user profile",
    "test_quiz_export": "Tools - quiz export",
    "test_rag_integration": "RAG - end to end",
    "test_session_manager": "Memory - sessions",
    "test_storage": "RAG - vector storage",
    "test_structured_output": "Tools - structured output (quiz)",
    "test_teacher_tools_server": "MCP - teacher tools server",
    "test_terminal": "CLI - terminal display",
    "test_tool_safety": "Tools - safety and limits",
    "test_tools": "Tools - registry and arguments",
    "test_web": "Web interface",
    "test_web_search": "Tools - web search",
}

PASSED, FAILED, SKIPPED = "passed", "failed", "skipped"
# A fixture error is reported on a placeholder whose id reads "setUpClass (test_x.Class)".
_FIXTURE_ID = re.compile(r"\((?P<target>[^)]+)\)")


class FeatureTestResult(unittest.TestResult):
    """Collect one outcome per test, grouped by feature.

    What a test prints is captured and kept only when the test fails, so a green run shows
    the summary alone.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcomes: dict[str, dict[str, str]] = defaultdict(dict)
        self.captured_output: dict[str, str] = {}
        self._capture = io.StringIO()
        self._streams = (sys.stdout, sys.stderr)

    def startTest(self, test):
        super().startTest(test)
        self._capture = io.StringIO()
        self._streams = (sys.stdout, sys.stderr)
        sys.stdout = sys.stderr = self._capture

    def stopTest(self, test):
        sys.stdout, sys.stderr = self._streams
        output = self._capture.getvalue()
        if output.strip() and self.outcomes.get(_feature_name(test), {}).get(test.id()) == FAILED:
            self.captured_output[test.id()] = output
        super().stopTest(test)

    def _record(self, test, outcome: str) -> None:
        test_outcomes = self.outcomes[_feature_name(test)]
        if test_outcomes.get(test.id()) != FAILED:  # a failed sub-test fails the whole test
            test_outcomes[test.id()] = outcome

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test, PASSED)

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._record(test, FAILED)

    def addError(self, test, err):
        super().addError(test, err)
        self._record(test, FAILED)

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self._record(test, FAILED)

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, SKIPPED)

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self._record(test, PASSED)

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self._record(test, FAILED)


def _module_name(test: unittest.TestCase) -> str:
    fixture = _FIXTURE_ID.search(test.id())
    target = fixture.group("target") if fixture else test.id()
    return target.split(".")[0]


def _feature_name(test: unittest.TestCase) -> str:
    module_name = _module_name(test)
    return FEATURES.get(module_name, module_name.replace("test_", "").replace("_", " ").title())


def _percentage(passed: int, run: int) -> str:
    return f"{passed / run * 100:.1f}%" if run else "n/a"


def _count(outcomes) -> tuple[int, int, int]:
    """Return (passed, failed, skipped)."""
    values = list(outcomes)
    return values.count(PASSED), values.count(FAILED), values.count(SKIPPED)


def _print_failures(result: FeatureTestResult) -> None:
    problems = [*result.failures, *result.errors, *((test, "Unexpected success") for test in result.unexpectedSuccesses)]
    if not problems:
        return

    print("\nFailures and errors:")
    for test, details in problems:
        print(f"\n  ✗ {test.id()}")
        for line in details.rstrip().splitlines():
            print(f"    {line}")
        output = result.captured_output.get(test.id())
        if output:
            print("    Output of the test:")
            for line in output.rstrip().splitlines():
                print(f"      {line}")
    print("\nFor a verbose run: python3 -m unittest discover -s tests -v")


def main() -> int:
    suite = unittest.defaultTestLoader.discover(str(TEST_DIRECTORY))
    result = FeatureTestResult()
    suite.run(result)

    width = max((len(feature) for feature in result.outcomes), default=20)
    rule = "=" * (width + 38)
    print("\nSensAI test summary")
    print(rule)
    for feature in sorted(result.outcomes):
        passed, failed, skipped = _count(result.outcomes[feature].values())
        run = passed + failed
        status = "✓" if failed == 0 else "✗"
        details = f"{passed:>3}/{run:<3} passed {_percentage(passed, run):>7}"
        if skipped:
            details += f"  ({skipped} skipped)"
        print(f"{status} {feature:<{width}}  {details}")
    print(rule)

    passed, failed, skipped = _count(
        outcome for outcomes in result.outcomes.values() for outcome in outcomes.values()
    )
    run = passed + failed
    skipped_note = f", {skipped} skipped" if skipped else ""
    features = len(result.outcomes)
    if result.wasSuccessful():
        print(f"✓ All {run} tests passed ({_percentage(passed, run)}) across {features} features{skipped_note}.")
    else:
        failing = sum(1 for outcomes in result.outcomes.values() if FAILED in outcomes.values())
        print(
            f"✗ {passed}/{run} tests passed ({_percentage(passed, run)}): "
            f"{failed} failing in {failing} of {features} features{skipped_note}."
        )
        _print_failures(result)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
