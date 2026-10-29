# Testing and development checks

[Home](../README.md) · [User guide](user-guide.md)

The regular [`requirements.txt`](../requirements.txt) includes `coverage` and `ruff` as
well as runtime dependencies; there is no separate `requirements-dev.txt` in this
checkout. Activate your environment and install the requirements as described in the
[user guide](user-guide.md).

Run the compact feature-oriented test summary:

```bash
python3 scripts/run_tests.py
```

For detailed test names and failures:

```bash
python3 -m unittest discover -s tests -v
```

Prompt-level functional tests use mocked Ollama responses; they do not require a live
Ollama server or downloaded models:

```bash
python3 -m unittest discover -s tests -p 'test_functional_prompts.py' -v
```

Check coverage and linting:

```bash
python3 -m coverage run scripts/run_tests.py
python3 -m coverage report -m
python3 -m ruff check .
```

Coverage helps identify untested code but does not itself prove correctness. Add focused
tests for changes and prompt-level workflow tests when appropriate.

## Optional live Ollama smoke test

With Ollama running and the configured models installed, launch SensAI and follow the
[sample document workflow](user-guide.md#6-try-document-grounded-answers). Check that
an answer uses the document and that `/sources off` hides the CLI source line. Model
responses are not deterministic; review them rather than expecting exact wording.

Alternatively, run the manual HTTP integration check:

```bash
python3 tests/test_ollama_client.py --integration
```

Unlike the automated tests, this requires a running server and local models.
