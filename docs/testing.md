# Testing and development checks

[Home](../README.md) · [User guide](user-guide.md)

The regular [`requirements.txt`](../requirements.txt) includes `coverage` and `ruff` as
well as runtime dependencies; there is no separate `requirements-dev.txt` in this
checkout. Activate your environment and install the requirements as described in the
[user guide](user-guide.md).

Run the compact feature-oriented test summary (offline: Ollama is mocked, nothing is
written in the repository, about 13 s):

```bash
python3 scripts/run_tests.py
```

```text
SensAI test summary
==========================================================================
✓ CLI - chat loop and commands           86/86  passed  100.0%
✓ Context window budget                   5/5   passed  100.0%
✓ Privacy and GDPR                       28/28  passed  100.0%
✓ Tools - safety and limits              23/23  passed  100.0%
...
==========================================================================
✓ All 728 tests passed (100.0%) across 38 features, 1 skipped.
```

Each test module is a feature; a failing test prints its name and traceback before the
summary, and the exit code is 1. For detailed test names:

```bash
python3 -m unittest discover -s tests -v
```

Prompt-level functional tests use mocked Ollama responses; they do not require a live
Ollama server or downloaded models:

```bash
python3 -m unittest discover -s tests -p 'test_functional_prompts.py' -v
```

Check coverage and linting (`ruff` also flags a test method defined twice, whose first copy
would silently never run):

```bash
python3 -m coverage run scripts/run_tests.py
python3 -m coverage report -m
python3 -m ruff check .
```

Tests that run the chat loop work in a temporary folder (`tests/support.py`), so sessions,
the RAG index and logs never land in the repository. Python 3.10 is the minimum version;
the suite was run on 3.10 and 3.14.

## Live checks against Ollama

These need Ollama running with the configured models.

```bash
python3 tests/test_ollama_client.py --integration
```

```text
Live Ollama check: http://localhost:11434
PASS  installed models: qwen2.5:3b, nomic-embed-text installed
PASS  complete_message: 'OK'
PASS  stream_message: 10 chunks streamed, 65.6 tokens/s
PASS  get_embedding: nomic-embed-text: 768 dimensions
VERDICT: PASS (4/4 steps)
```

Measured quality, with a PASS/FAIL verdict and every answer saved as JSON
([evaluation](evaluation.md), [limits](limits.md)):

```bash
python3 scripts/evaluate_rag.py --quick     # RAG on vs off on teacher questions, about 1 minute
python3 scripts/evaluate_rag.py             # 3 runs, about 7 minutes
python3 scripts/probe_limits.py --quick     # hallucination, codename, long prompt, injection
python3 scripts/benchmark_models.py --model qwen2.5:3b --runs 3   # speed, latency, memory
```

Model answers are not deterministic: compare orders of magnitude and verdicts, not exact
wording.
