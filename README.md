# Sensai_Mirror

Mirror of the Sensai project.

## Running the project

### Prerequisites

- Python 3.10 or later
- [Ollama](https://ollama.com/) installed and running locally

### Setup

Create and activate a virtual environment, then install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure SensAI

SensAI reads [`config.json`](config.json) when the application starts. This file centralizes the local model, RAG, tools, and display settings.

```json
{
  "model": {
    "chat_model": "llama3.2",
    "embedding_model": "nomic-embed-text",
    "temperature": 0.2,
    "context_window": 4096
  },
  "features": {
    "rag_enabled": true,
    "tools_enabled": true,
    "memory_enabled": false
  }
}
```

- `temperature` controls how varied the chat model responses are. A low value such as `0.2` is recommended for document-grounded answers because it is more stable and factual. Higher values can make conversation more creative but also less predictable.
- `context_window` is sent to Ollama as `num_ctx`. A larger value lets the model consider more conversation and RAG context, but uses more RAM/VRAM and can make generation slower. `4096` is a reasonable local default for `llama3.2`; increase it only if your machine and model support it.
- `rag_enabled` and `tools_enabled` determine whether those features are available. RAG starts off for a normal chat experience, then can be enabled with `/rag on` when `rag_enabled` is `true`. `source_display_enabled` controls whether verified sources are shown by default.
- `memory_enabled` is a reserved setting. SensAI does not yet implement persistent conversation memory, so enabling it only displays a startup notice and does not store conversations.
- The `rag` section controls retrieved chunk count, context budget, similarity threshold, and other implemented RAG settings.

Invalid values are reported at startup instead of being silently ignored.

The project uses the `llama3.2` model by default. Download it once if it is not already available:

```bash
ollama pull llama3.2
```

Start the Ollama server in a separate terminal:

```bash
ollama serve
```

### Start the application

From the root of the repository, run:

```bash
python3 source/main.py
```

The application starts an interactive chat connected to the local Ollama API at `http://localhost:11434`.

### CLI command help

Type the following while SensAI is running to display its available commands, grouped by feature:

```text
/help
```

This help output is generated from the command registry, so a command registered by RAG, Tools, or Memory can provide its own syntax and description in one place.

### Ingest a document for RAG

While the application is running, use the following command to validate, chunk, embed, and index a local document:

```text
/ingest path/to/document.md
```

To ingest several files in one command, use `-m` or `--multiple`. Separate paths with spaces or tabs, and quote any path containing spaces:

```text
/ingest --multiple docs/guide.md docs/architecture.pdf "docs/project notes.docx"
```

The command supports UTF-8 encoded documents and source files, text-based PDFs, Word `.docx` files, and ZIP archives containing these supported formats. Supported extensions include `.txt`, `.md`, `.pdf`, `.docx`, `.py`, `.c`, `.h`, `.cpp`, `.hpp`, `.js`, `.ts`, `.java`, `.go`, `.rs`, `.html`, `.css`, `.xml`, `.svg`, `.json`, `.yaml`, `.yml`, `.toml`, `.ini`, `.sql`, and `.sh`. PDF extraction uses `pypdf` and extracts native PDF text only. Scanned or image-only PDFs require OCR and are not supported yet. Word extraction uses `python-docx` to read paragraphs and headings in document order. Legacy `.doc` files, embedded image OCR, comments, tracked changes, and complex layout are not supported. ZIP archives are inspected without extracting their paths to the filesystem; unsupported entries are skipped and reported. SensAI rejects archives with path traversal entries, password protection, excessive size, file counts, nesting depth, or suspicious compression ratios. SensAI reads text and source files as text only: it never executes, imports, renders, or interprets their contents. The command validates the path, file type, encoding, and content, then stores its chunks and embeddings in the local RAG database. Enter `exit` or `quit` to leave the application.

### Use documents in the chat

Enable document-grounded answers after indexing at least one document:

```text
/rag on
```

While RAG mode is enabled, SensAI retrieves the most relevant excerpts before every question, sends them as bounded context to Ollama, and cites the source filename in its answer. If the information cannot be found in the indexed documents, it must say so. Use `/rag off` to restore the normal chatbot behavior, or `/rag status` to display the current mode.

### Filter RAG sources

Limit retrieval to one or more exact source filenames, or to an ingestion category. Categories are `document`, `code`, `configuration`, and `markup`:

```text
/rag filter name guide.md,notes.md
/rag filter category code
/rag filter clear
```

Only chunks matching the active filter are considered before semantic similarity ranking. If no indexed chunk matches, SensAI reports that the active filter has no matching chunks. With no active filter, RAG keeps its normal retrieval behavior.

### Source display

SensAI prints verified source files and chunk identifiers after a RAG answer by default. The model is instructed not to repeat those citations in its generated text. Control this display with:

```text
/sources on
/sources off
/sources status
```

### Example RAG workflow

Start the application, then index a document and enable RAG mode:

```text
/ingest docs/guide.md
/rag on
How are embeddings stored?
```

After a response that used retrieved context, the CLI prints the verified document source and chunk identifier, for example `Sources: guide.md (chunks: ...)`. These sources come from the chunks retrieved by SensAI, not from a citation generated by the model. No source line is printed when no document context was used.

### Tools and RAG together

With `/rag on`, document retrieval runs for every question, and the registered tools stay available in the same request. For example, "make a quiz from my course" uses the indexed documents, and "what is the weighted average of these marks?" can call a tool.

A tool decides how it is routed with two optional methods:

- `is_relevant_for(user_input)`: whether the tool is offered to the model for this request. The default is `True`, so a tool without a rule is always offered.
- `claims_request(user_input)`: whether the request is explicitly meant for this tool. The default is `False`. When a tool claims a request, RAG is skipped and the tool answers (this is how "what time is it?" reaches `get_time` instead of the documents).

### Structured answers: `/quiz`

SensAI can force the model to answer with JSON that matches a schema. The schema is sent to Ollama (`format` parameter), the answer is validated, and an invalid answer is sent back to the model with the error, up to 3 attempts.

```text
You: /quiz --questions 3 Le théorème de Pythagore
```

`/quiz` prints a multiple-choice quiz followed by its answer key. With `/rag on`, the questions are built only from the indexed course; if no excerpt matches the topic, nothing is generated. Without RAG, the model uses its own knowledge.

A small model can write a well-formed quiz with a wrong answer key. The format is checked by the program, the content is not: read the answer key before using a quiz.

### Chunking configuration

The RAG chunking module splits normalized document text at paragraph and sentence boundaries. Its default configuration is an 800-character maximum chunk size with a 120-character overlap. These values are defined in `source/rag/chunking.py` and can be adjusted by creating a `ChunkingConfig` instance.

### RAG embedding model

The RAG embedding module uses the Ollama model `nomic-embed-text` by default. Install it with:

```bash
ollama pull nomic-embed-text
```

The chat model and the embedding model are separate: a dedicated embedding model produces vectors intended for semantic search. The embedding model can be changed in Python with `EmbeddingConfig(model="your-model-name")`.

### RAG context configuration

By default, RAG retrieves the top three matching chunks, rejects chunks with a cosine similarity score below `0.35`, and keeps the complete context message under 4,000 characters. Adjust these settings through `RAGContextConfig(top_k=..., max_context_characters=..., relevance_threshold=...)` when creating a `RAGContextProvider`. The threshold must be between `-1` and `1`; increase it to be more selective, or decrease it when relevant excerpts are being rejected too often.

### Local RAG storage and search

Embeddings are persisted locally in the SQLite database `data/sensai_rag.sqlite3`. This database is created automatically and is excluded from Git because it contains local, generated knowledge-base data. The `LocalVectorStore` retrieves the most semantically similar chunks with cosine similarity. By default, it returns the top three results; this can be adjusted with the `top_k` argument.

### Test the project

Install the development dependencies once in the activated virtual environment:

```bash
pip install -r requirements-dev.txt
```

Run the compact test summary for the everyday overview. It reports one line per feature, such as `✓ Document ingestion 20/20 passed`, and only shows individual test details when something fails.

```bash
python3 scripts/run_tests.py
```

For debugging a failure, run all tests with detailed test names and outcomes:

```bash
python3 -m unittest discover -s tests -v
```

Run only the prompt-level functional workflows. They simulate Ollama responses, so they are deterministic and do not require Ollama, a downloaded model, or Internet access:

```bash
python3 -m unittest discover -s tests -p 'test_functional_prompts.py' -v
```

Generate a branch-aware coverage report by module, including uncovered line numbers:

```bash
python3 -m coverage run scripts/run_tests.py
python3 -m coverage report -m
```

Coverage is a signal for identifying untested code, not a reason to add low-value tests solely to reach 100%. New behaviour should include a focused unit test and, when it changes a user flow, a mocked prompt-level functional test.

Check likely runtime mistakes and unused imports with Ruff:

```bash
python3 -m ruff check .
```

### Optional manual Ollama smoke test

The automated suite deliberately never calls a live model. To validate the local integration after changing prompts or model configuration, start Ollama, run `python3 source/main.py`, then try the following flow with a small Markdown file:

```text
/ingest docs/guide.md
/rag on
Ask one question answered by guide.md
/sources off
Ask the question again
```

Confirm that the first response is grounded in the document and displays its source, and that the second retains the answer but hides the CLI source line.

For a separate live HTTP smoke test of the Ollama client, after starting Ollama and downloading the configured model, run:

```bash
python3 tests/test_ollama_client.py --integration
```

This command is intentionally manual: unlike the automated tests, it requires a running Ollama server and local models.
