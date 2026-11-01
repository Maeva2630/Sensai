# SensAI

SensAI is a local AI assistant (terminal or web page) for primary and middle-school
teachers. Camille, a *sixième* mathematics teacher, uses it to prepare lessons, activities
and assessments **from her own course documents, on her own laptop**: the model runs in
[Ollama](https://ollama.com/), documents and conversations never leave the computer unless
she approves a web request. Generated content must always be reviewed by the teacher.

Built directly on the Ollama HTTP API in Python, with no LLM framework (no LangChain,
LlamaIndex or Ollama SDK).

## Quick start

Requirements: Python **3.10+**, [Ollama](https://ollama.com/), about 4.2 GB of disk for the
models. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

ollama serve                     # in another terminal, if Ollama is not already running
ollama pull qwen2.5:3b           # chat model, 1.9 GB
ollama pull nomic-embed-text     # embedding model for documents, 274 MB
ollama pull llama3.2:latest      # fallback chat model, 2.0 GB (optional)

python3 source/main.py --profile demo/profile_camille.json
```

A first session with the fictional course shipped in `courses/` (real output, answers vary):

```text
You: /ingest --multiple courses/progression_6e_B.md courses/consignes_evaluation.md
Document indexed: progression_6e_B.md (3 chunks; 3 new chunks stored).
Document indexed: consignes_evaluation.md (2 chunks; 2 new chunks stored).
Multiple ingestion complete: 2/2 paths indexed.
You: /rag on
RAG mode enabled. Responses will use indexed documents only.
You: Quand a lieu le contrôle commun de fractions et dans quelle salle ?
Assistant: Le contrôle commun de fractions a lieu le jeudi 17 décembre 2026 à 10h30, en salle B12.

Sources: consignes_evaluation.md (chunks: 01d8880c8c0f, 0ca1b30cd804); progression_6e_B.md (chunks: 247a1f9866e1)
You: Quelle est la capitale de l'Australie ?
Assistant: Je ne trouve pas cette information dans les documents indexés.
(No indexed excerpt reached the relevance threshold, so the model was not asked: nothing could be invented. Use /rag off to ask the model without documents.)
You: Calcule la moyenne pondérée d'Inès : 16, 14 et 15 avec les coefficients 1, 1 et 2.
[tool] teacher__calculate_grade({"marks": [16, 14, 15], "coefficients": [1, 1, 2], "max_mark": 20})
[tool] result: Weighted average: 15/20 (computed from 3 marks)
Assistant: La moyenne pondérée de Inès est de 15/20.
You: /stats
Last answer:
- Settings: model=qwen2.5:3b, temperature=0.2, num_ctx=4096
- Generation: 24 tokens in 0.41 s = 58.9 tokens/s
...
You: exit
```

The same assistant runs as a local web page for teachers who prefer not to use a terminal:

```bash
python3 source/main.py --web          # http://localhost:8000
```

## Features

Catalogue codes refer to the project's feature catalogue; every feature has user stories
in [docs/user-stories.md](docs/user-stories.md).

| Code | Feature | Try it | Details |
| --- | --- | --- | --- |
| B1 | Streaming chat with history, clean errors, model chosen by configuration or flag | `python3 source/main.py --model llama3.2:latest` | [User guide](docs/user-guide.md) |
| R1 | RAG on local documents (PDF, DOCX, Markdown, text, code, ZIP) with sources named by SensAI | `/ingest courses/progression_6e_B.md`, `/rag on` | [RAG](docs/features/rag.md) |
| R2 (partial) | Metadata filters and a calibrated relevance threshold; "not found" without calling the model when nothing is relevant | `/rag filter name progression_6e_B.md` | [RAG](docs/features/rag.md#how-retrieved-text-reaches-the-model) |
| M1 | Saved conversations and a teacher profile | `/save`, `--session <id>`, `--profile` | [Sessions and profile](docs/features/sessions-and-profile.md) |
| M2 | Token budget checked before sending, warnings, summaries | `/tokens`, `/compress` | [Context engineering](docs/context-engineering.md) |
| M4 | A shared document the assistant drafts with the teacher | `/artifact init Séquence fractions` | [Sessions and profile](docs/features/sessions-and-profile.md) |
| T1 | Tool calling with a custom MCP server (grades, class statistics, timers) and a third-party one (`mcp-server-fetch`) | `Calcule la moyenne de Léa : 12, 15 et 9.` | [Tools and MCP](docs/features/tools-and-mcp.md) |
| T3 | Web search, off by default, each query approved | `network_enabled: true`, then `Cherche sur le web …` | [Tools and MCP](docs/features/tools-and-mcp.md) |
| T4 | File access limited by `permissions.json`; every write needs a yes | `Lis courses/progression_6e_B.md`, `/permissions` | [Tools and MCP](docs/features/tools-and-mcp.md#file-tools-and-permissions) |
| T5, X3 | Validated JSON quizzes and their export | `/quiz --questions 5 --save exports/quiz.md Les fractions` | [Tools and MCP](docs/features/tools-and-mcp.md#quiz-generation-and-controlled-export) |
| A5 | Configurable personas (teacher, mathematics) | `/persona use mathematics-assistant` | [Personas](docs/features/personas.md) |
| EV3, X1 | Logs, prompt inspection, speed, local metrics; web interface | `/debug prompt`, `/stats`, `/metrics`, `--web` | [Observability](docs/features/observability.md), [Web](docs/features/web-interface.md) |

## Configuration

Everything is configured in [`config.json`](config.json): models, feature switches (RAG,
tools, memory, network), temperature, context window, RAG settings, logging, monitoring,
personas. MCP servers are in `mcp_servers.json`, file permissions in `permissions.json`.

```bash
python3 source/main.py --config demo/config.json                # another configuration file
python3 source/main.py --temperature 0.8 --context-window 8192   # one run only
```

```text
You: /set temperature 1.2      # this session only: compare answers to the same request
You: /set context 8192
```

Reference: [configuration](docs/configuration.md). Why these values:
[context engineering](docs/context-engineering.md), [decisions](docs/decisions.md).

## Commands

| Area | Commands |
| --- | --- |
| Chat | `/help`, `exit`, `quit` |
| Documents | `/ingest [--multiple] <path>…`, `/rag on\|off\|status`, `/rag filter …`, `/rag reset`, `/sources on\|off` |
| Tools | `/tools`, `/permissions`, `/quiz [--questions N] [--language L] [--save path [--no-answers]] <topic>` |
| Memory | `/save`, `/load <id>`, `/forget [all]`, `/tokens`, `/compress`, `/artifact …` (alias `/doc`) |
| Persona | `/persona list\|status\|use <id>` |
| Inspection | `/debug prompt [--include-system]`, `/stats`, `/metrics [--feature rag] [--tool NAME] …`, `/set …` |

## Tests and measurements

```bash
python3 scripts/run_tests.py                        # 729 offline tests, result and % per feature
python3 tests/test_ollama_client.py --integration   # live check against Ollama, PASS/FAIL verdict
python3 scripts/evaluate_rag.py --quick             # RAG on vs off on teacher questions, verdict
python3 scripts/probe_limits.py --quick             # hallucination, codename, long prompt, injection
python3 scripts/benchmark_models.py --model qwen2.5:3b --runs 3   # tokens/s, latency, memory
```

```text
SensAI test summary
==========================================================================
✓ CLI - chat loop and commands           86/86  passed  100.0%
✓ Privacy and GDPR                       28/28  passed  100.0%
...
✓ All 728 tests passed (100.0%) across 38 features, 1 skipped.
```

Results and protocols: [evaluation](docs/evaluation.md), [model limits](docs/limits.md),
[benchmarks](docs/model-benchmarks.md), [testing](docs/testing.md).

## Documentation

| For | Read |
| --- | --- |
| Using SensAI | [User guide](docs/user-guide.md) · [Configuration](docs/configuration.md) · [Feature guides](docs/features/README.md) |
| The scenario and its traceability | [Product scenario](docs/products-scenario.md) · [User stories and traceability](docs/user-stories.md) · [Demo script](demo/demo-script.md) |
| Technical choices | [Decisions and alternatives](docs/decisions.md) · [Context engineering](docs/context-engineering.md) · [Model benchmarks](docs/model-benchmarks.md) |
| Measured quality | [Evaluation (RAG A/B)](docs/evaluation.md) · [Model limits](docs/limits.md) · [Testing](docs/testing.md) |
| Trust | [Security and resilience](docs/security.md) · [Local data and GDPR](docs/RGPD.md) · [Cost-benefit analysis](docs/cost-benefit.md) |

## Current limitations

- A 3B local model makes mistakes: quiz answer keys and fraction arithmetic must be checked
  ([limits](docs/limits.md)). SensAI shows its sources and keeps actions under the teacher's
  control, it does not guarantee facts.
- Prompt injection in a document is detected and mitigated, not prevented; writes and network
  calls always need a yes ([security](docs/security.md)).
- Scanned PDFs (OCR), legacy `.doc`, encryption at rest, accounts and cloud synchronisation
  are not supported.
- Speed and memory depend on the computer: about 59 tokens/s on an Apple M5, 111 on an
  RTX 4060 laptop; a computer without a GPU was not measured.
