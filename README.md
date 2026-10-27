# SensAI

SensAI is a local, command-line AI assistant for primary and secondary-school teachers.
It helps prepare lessons, activities, assessments, and educational resources while keeping
documents, conversations, and metrics on the local machine. Model-generated content must
always be reviewed by the teacher.

## Quick start

Requirements: Python **3.11+**, [Ollama](https://ollama.com/), and a local checkout of
this repository. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Start Ollama in another terminal, then download the configured default models once:

```bash
ollama serve
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

Start SensAI and type `/help` to see the available commands:

```bash
python3 source/main.py
```

Model names, feature switches, local paths, and generation settings are configured in
[`config.json`](config.json). Read the [user guide](docs/user-guide.md) before changing
the configuration. The [configuration reference](docs/configuration.md) explains the
file format, location, required sections, optional settings, defaults, and validation.

## Implemented features

| Feature | What it does | Details |
| --- | --- | --- |
| RAG | Indexes supported local documents and source files, then retrieves relevant excerpts for document-grounded answers. | [RAG guide](docs/features/rag.md) |
| Tools, quizzes, and MCP | Provides local tools, generated quizzes, controlled file export, and tools from configured MCP servers. | [Tools and MCP](docs/features/tools-and-mcp.md) |
| Personas | Configures assistant roles and switches between profiles during a conversation. | [Personas](docs/features/personas.md) |
| Sessions and profile | Saves conversations, loads previous sessions, applies a local user profile, and manages the context budget. | [Sessions and profile](docs/features/sessions-and-profile.md) |
| Observability | Provides local application logs, on-demand prompt inspection, and an optional metrics dashboard. | [Logging and monitoring](docs/features/observability.md) |

## Current limitations

- SensAI is a local assistant, not a source of guaranteed facts: check generated content,
  retrieved sources, quiz answers, and grading suggestions before using them with learners.
- Text-based PDFs are supported, but scanned or image-only PDFs require OCR, which is not
  implemented.
- Legacy Word `.doc` files, cloud synchronization, and remote telemetry are not supported.
- The available speed, context size, and memory use depend on the selected Ollama models and
  the local computer.

## Documentation

- [User guide](docs/user-guide.md): install, configure, launch, and troubleshoot SensAI.
- [Configuration reference](docs/configuration.md): `config.json` format, location,
  required fields, optional settings, defaults, and validation.
- [Product scenario and traceability](docs/products-scenario.md): target teachers, final
  demonstration workflow, user stories, acceptance criteria, and test evidence.
- [Local model benchmarks](docs/model-benchmarks.md): reproducible model-comparison
  protocol, measurements, candidate models, and decision record.
- [Feature documentation](docs/features/README.md): detailed guides, examples, limitations,
  and privacy considerations for every implemented feature.
- [Testing and development checks](docs/testing.md): automated tests, coverage, linting,
  and optional live Ollama checks.
