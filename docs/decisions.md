# Technical decisions

[Home](../README.md) · [Product scenario](products-scenario.md) · [User stories](user-stories.md) · [Model benchmarks](model-benchmarks.md)

Each record follows the same structure: **Context**, **Options considered**, **Decision**, **Trade-off accepted**, **Evidence**, **What we would revisit**. Every figure is either measured (the source is given), computed from the code (the file is given), or labelled as an estimate or assumption.

## Constraints the decisions rely on

| ID | Constraint | Source |
| --- | --- | --- |
| C1 | No LLM framework (LangChain, LlamaIndex, Haystack, or equivalent). Utility packages are allowed for non-LLM concerns. | Subject |
| C2 | Every model call goes through the Ollama HTTP API; no Ollama SDK. | Subject |
| C3 | Team hardware: an Apple M5 laptop with 16 GB unified memory, and a laptop with an AMD Ryzen 7 8845HS, 14 GiB RAM, and an RTX 4060 with 8 GB VRAM. Target teacher laptops: 8 to 16 GB of memory (assumption). | [Model benchmarks](model-benchmarks.md#benchmark-environment), team measurement |
| C4 | Scenario: Camille, a *sixième* mathematics teacher, prepares lessons from her own course on her own laptop. | [Product scenario](products-scenario.md) |
| C5 | Pupil data stays on the laptop; network access is off by default and every outgoing request is approved. | [Product scenario](products-scenario.md#why-it-must-run-locally), [GDPR assessment](RGPD.md) |
| C6 | Every runtime dependency is something a teacher installs with `pip install -r requirements.txt`. Today the runtime dependencies are `requests`, `pypdf`, and `python-docx`. | `requirements.txt` |

## Arbitrations at a glance

| Contradictory constraints | Where it is decided | What we chose |
| --- | --- | --- |
| Latency versus answer quality | [ADR-01](#adr-01-chat-model) | The 3B model: twice the generation speed of the 7B, quality advantage of the 7B not demonstrated. |
| Chunk granularity versus context cost | [ADR-03](#adr-03-chunk-size) | 800-character chunks: three excerpts fit entirely in the 4,000-character RAG budget, about a fifth of the 4,096-token window. |
| Search speed versus installation effort | [ADR-02](#adr-02-vector-store) | Brute-force search in SQLite: no dependency, 0.2 s for 1,000 chunks, linear growth accepted. |
| Fresh information versus privacy | [Product scenario](products-scenario.md#why-it-must-run-locally), story [T3-02](user-stories.md#t3--web-search) | Web search exists but is off by default and asks before each request, showing the exact query. |
| Precision of retrieval versus latency | [ADR-07](#adr-07-features-deliberately-not-implemented) | No reranking; metadata filter and relevance threshold instead. |
| Recall versus invented answers | [ADR-03c](#adr-03c-relevance-threshold-calibration) | Threshold 0.6 calibrated on measured scores; below it, "not found" without calling the model. |
| Grounding in documents versus obeying them | [ADR-03b](#adr-03b-rag-context-placement) | Excerpts are data in the user turn; actions with side effects need a human yes. |
| Window size versus memory | [Context engineering](context-engineering.md#4-size-management) | 4,096 tokens, a quarter kept for the answer, budget checked before sending; 8,192 costs +144 MiB. |

## ADR-01: Chat model

**Context.** C2, C3, C4. The model must write French, handle mathematical notation, emit valid tool calls (teacher MCP tools, `edit_artifact`, file tools), and produce JSON for `/quiz`. It runs next to the embedding model on an 8 GB GPU or 16 GB of unified memory.

**Options considered.** Measured on the RTX 4060 laptop on 2026-10-27 with the fixed prompt set `teacher-scenario-v1` (four cases, three repetitions, temperature 0.2, context 4,096). Parameter counts and quantization come from Ollama's `/api/tags` on the M5 machine.

| Model | Size | Warm output speed | Cold first text token | Warm total latency | Peak VRAM (warm) | Valid `get_time` calls |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `llama3.2:latest` | 3.2B, Q4_K_M | 109.0 tok/s | 4,370 ms | 1,528 ms | 2,786 MiB | 3 calls emitted, 0/3 with valid arguments |
| `qwen2.5:3b` | 3.1B, Q4_K_M | 111.0 tok/s | 2,255 ms | 1,205 ms | 2,444 MiB | 3/3 |
| `qwen2.5:7b` | 4.7 GB download | 53.7 tok/s | 7,573 ms | 3,199 ms | 4,858 MiB | 3/3 |
| `qwen2.5:14b` | 9.0 GB download | not run | — | — | larger than the 8,188 MiB of VRAM | — |

**Decision.** `qwen2.5:3b` is the default chat model; `llama3.2:latest` is the automatic fallback when the default is not installed (`cli/fallback.py`). Temperature 0.2 and `num_ctx` 4,096 (`config.json`).

**Trade-off accepted.** This is the latency-versus-quality arbitration. The 7B model also produced valid tool calls, but it generates at half the speed and uses twice the VRAM, and its manual quality review is still marked "pending" in the benchmark: we had no evidence that it writes better activities or answer keys. On the M5, the 3B model generates at about 58 tok/s, and the demo's 15-minute activity (517 to 622 output tokens) took 9.4 to 10.7 s. If the 7B kept its RTX speed ratio on the M5 (53.7 / 111.0, an extrapolation, not a measurement), the same activity would take about 20 s, above the 15 s target for a short answer. The cost of this choice is the quality ceiling of a 3B model: an answer key can be wrong, so every document tells the teacher to review generated material. Small models also send numbers as strings in tool arguments; the registry converts them (`tools/registry.py`, `test_tools.py::TestArgumentCoercion`). Finally, `qwen2.5:3b` is distributed under a non-commercial research license, which fits this project but would have to be reconsidered before a real deployment in schools.

**Evidence.** [docs/model-benchmarks.md](model-benchmarks.md), including the [Apple M5 measurements](model-benchmarks.md#second-machine-apple-m5-unified-memory); `scripts/benchmark_models.py`, `benchmarks/prompts.json`.

**What we would revisit.**
- Finish the quality review for `qwen2.5:7b` and `llama3.2`; if the 7B writes clearly better answer keys, document it as a per-machine choice for 16 GB machines (a `config.json` change only).
- `qwen2.5:3b` supports a 32,768-token context, and the benchmark measured only +144 MiB of VRAM and no latency loss on short prompts with `num_ctx` 8,192. We kept 4,096 because long-context answer quality of a 3B model was not measured.
- Measure a laptop without a GPU.
- Check the license of every candidate before any deployment outside the project.

## ADR-02: Vector store

**Context.** R1 requires a local vector store. C1 rules out the vector-store layers of LLM frameworks; C6 makes every extra package an installation step for the teacher. The corpus is one teacher's material: the demo course is a single chunk, and a year of *sixième* courses, worksheets, and tests is estimated at about 1,000 chunks (assumption: 12 chapters, a few documents each).

**Options considered.**

| Option | What it requires | Why it was not chosen |
| --- | --- | --- |
| SQLite table and brute-force cosine in Python | Python standard library only | Chosen |
| FAISS (`faiss-cpu`) | A native wheel, and an index file kept in sync with the chunk metadata | Approximate search is not needed at our size (see measurement); a second store must be kept consistent on reset. |
| Chroma | An embedded database with its own embedding-function layer and many transitive dependencies | It overlaps with the pipeline we must build and explain ourselves (C1), and makes installation heavier (C6). |
| sqlite-vec | A loadable SQLite extension | Needs a Python build with SQLite extension loading and a binary per operating system; not needed below about 10,000 chunks. |
| pgvector | A PostgreSQL server | A database server on a teacher's laptop. |

**Decision.** `rag/storage.py` stores each chunk in one SQLite table `chunks` with its text, source metadata, embedding model, dimension, and vector (as JSON text). A query selects the rows of the same embedding model and dimension, applies the metadata filters in the SQL `WHERE` clause, computes the cosine similarity in Python, sorts, and keeps `top_k`.

**Trade-off accepted.** Search is linear in the number of chunks. We measured it with the `main` version of `LocalVectorStore.search`, 768-dimension random vectors (the dimension of `nomic-embed-text`), Python 3.14.3 on the Apple M5, median of five queries, on 2026-11-01:

| Indexed chunks | Search time | Database size |
| ---: | ---: | ---: |
| 300 | 60 ms | 5.1 MB |
| 1,000 | 200 ms | 17.0 MB |
| 3,000 | 601 ms | 50.8 MB |
| 10,000 | 2,026 ms | 169.4 MB |

About 0.2 ms per chunk. At the estimated 1,000 chunks, retrieval adds 0.2 s to an answer that takes 1.5 to 11 s to generate. Two further limits: vectors stored as JSON text take about 17 KB per chunk, and there is no per-document delete. A document's identifier is the SHA-256 of its text (`rag/ingestion.py`), so re-indexing a modified file adds new chunks next to the old ones until `/rag reset`.

<details>
<summary>Measurement script</summary>

Run from a scratch folder with the repository's virtual environment, passing the path to `source/`:

```python
import random, statistics, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from rag.chunking import DocumentChunk
from rag.embeddings import EmbeddedChunk, QueryEmbedding
from rag.storage import LocalVectorStore

random.seed(1)
for n in (300, 1000, 3000, 10000):
    with tempfile.TemporaryDirectory() as folder:
        store = LocalVectorStore(Path(folder) / "rag.sqlite3")
        store.store_chunks(
            EmbeddedChunk(DocumentChunk(f"c{i}", f"d{i // 10}", f"d{i // 10}.md", i % 10, 0, 800, "x" * 800),
                          tuple(random.uniform(-1, 1) for _ in range(768)), "nomic-embed-text")
            for i in range(n)
        )
        query = QueryEmbedding("q", tuple(random.uniform(-1, 1) for _ in range(768)), "nomic-embed-text")
        times = []
        for _ in range(5):
            start = time.perf_counter(); store.search(query, top_k=3); times.append(time.perf_counter() - start)
        print(n, f"{statistics.median(times) * 1000:.0f} ms")
```

</details>

**Evidence.** The measurement above; `test_storage.py::TestLocalVectorStore::test_search_orders_results_by_cosine_similarity`, `test_storage.py::TestLocalVectorStore::test_search_filters_chunks_by_source_filename_or_category`, `test_storage.py::TestLocalVectorStore::test_chunks_remain_available_after_reopening_database`.

**What we would revisit.** Store vectors as 32-bit float blobs with the standard `array` module (768 × 4 bytes instead of about 17 KB of JSON). Above about 10,000 chunks (2 s per query measured), move to sqlite-vec. Add a per-document delete keyed on the file name.

## ADR-03: Chunk size

**Context.** C4: the course is French prose, organised as definition, example, rule, example (`demo/fractions_6e.md`). Chunk size sets both retrieval precision and the cost in the 4,096-token window, which is shared by the persona and profile, the retrieved excerpts, the conversation history, and the answer.

**Current values.** `DEFAULT_CHUNK_SIZE = 800` and `DEFAULT_CHUNK_OVERLAP = 120` characters (`rag/chunking.py`). Chunks end at paragraph and sentence boundaries; source-code files use structural boundaries. These are Python constants, not `config.json` settings ([RAG guide](features/rag.md#configuration-and-limitations)). Retrieval keeps `top_k = 3` excerpts within `max_context_characters = 4000` (`config.json`).

**The cost in the context window, computed from the code.**

| Part of a RAG request | Size | Basis |
| --- | ---: | --- |
| RAG instructions (`RAGContextProvider._INSTRUCTIONS`) | 406 characters | Measured on the string |
| Source header per excerpt (file, 64-character chunk identifier, lines, score) | about 130 characters | Format in `rag/context.py` |
| Three excerpts of at most 800 characters with their headers | about 2,800 characters | 3 × (130 + 800) |
| **RAG block** | **about 3,200 characters, below the 4,000 budget: about 820 tokens** | At 3.89 characters per token, measured for French prose with `qwen2.5:3b` |
| Mathematics persona prompt and a short question | 465 to 477 tokens | Measured |
| Answer: a 15-minute activity | 506 to 866 tokens | Measured (same source) |

The fixed part of a RAG request with the mathematics persona is therefore about 1,300 tokens with three full excerpts, and about 1,500 tokens if the excerpts fill the 4,000-character budget: 32 to 37 % of the 4,096-token window, before the profile, the history, and the answer.

**Options considered.**

| Chunk size | Effect on context | Effect on retrieval |
| --- | --- | --- |
| 400 characters | Three excerpts take about 1,600 characters (about 400 tokens): half the cost. | A definition and its worked example are more often split into different chunks, and three excerpts carry less surrounding text. Twice as many chunks to embed at ingestion and to scan per query (+0.2 ms per chunk, ADR-02). |
| **800 characters (chosen)** | Three complete excerpts fit in the 4,000-character budget. | About one section of a *sixième* course per chunk. |
| 1,500 characters | Three excerpts need about 4,900 characters: `_build_message` cuts the third excerpt to fit the 4,000-character budget. | Each chunk mixes several notions, which dilutes its embedding for a precise question. `nomic-embed-text` accepts 2,048 tokens per input (Ollama `/api/tags`), so the size is not limited by the embedding model. |

**Decision.** Keep 800 and 120 with `top_k = 3`: it is the largest size at which three whole excerpts fit in the RAG budget.

**Trade-off accepted.** The values were set when RAG was built (2026-10-18), before the teacher scenario, and were not re-tuned on teacher questions. The arithmetic above justifies the order of magnitude; it does not prove that 800 retrieves better than 600 or 1,000.

**Evidence.** `rag/chunking.py`, `rag/context.py`; `test_chunking.py::TestDocumentChunker::test_chunks_end_at_sentence_boundaries_when_sentences_fit`, `test_chunking.py::TestDocumentChunker::test_adjacent_chunks_share_readable_overlap`, `test_context.py::TestRAGContextProvider::test_includes_retrieved_sources_and_respects_character_budget`.

**What we would revisit.** Move `chunk_size` and `chunk_overlap` into `config.json`; measure retrieval on a teacher question set ([restart item 4](#4-write-the-scenario-and-an-evaluation-set-first)); replace the 64-character chunk identifier in the prompt header with a short index, since it costs tokens on every excerpt (not measured).

## ADR-03b: RAG context placement

**Context.** Retrieved excerpts are written by third parties (a colleague, a publisher, a website). Until 2026-11-01 they were sent as a second `system` message, and the persona prompt told the model that this message was "authoritative". A poisoned document therefore spoke with the authority of SensAI's own instructions. The audit also showed that a fake "[End of retrieved context] SYSTEM: …" line could push the model to request a `fetch` with pupil names in the URL (3/3).

**Options considered.** All three keep the same persona, the same excerpts and the same questions; measured with `qwen2.5:3b`, temperature 0.2, 3 poisoned-document phrasings × 3 questions × 3 runs = 27 answers per arm (script kept with the audit evidence, method in [limits](limits.md#4-malicious-instructions-inside-an-indexed-document-indirect-prompt-injection)).

| Arm | Injected instruction obeyed | Correct answers |
| --- | ---: | ---: |
| Excerpts in a `system` message (former design) | 9/27 | 24/27 |
| Excerpts in the user turn, between markers, forged markers neutralised | 8/27 | 24/27 |
| Same + a closing reminder after the question (current design) | 6/27 | 24/27 |

A second measurement used the three poisoned documents of `scripts/probe_limits.py` (a canary hidden in an HTML comment, a "priority instruction" asking to copy the system prompt between "-----" lines, a fake `[End of retrieved context] SYSTEM:` ordering a tool call), 5 runs each, after hidden comments were stripped from excerpts:

| Arm | Canary printed | System prompt leaked | "-----" format followed | Injected tool called |
| --- | ---: | ---: | ---: | ---: |
| Excerpts in a `system` message | 0/15 | 0/15 | 0/15 | 0/15 |
| Excerpts in the user turn + reminder (current design) | 0/15 | 0/15 | 5/15 | 0/15 |

Before hidden comments were stripped, the HTML-comment canary was printed in 8 of 9 answers with the user-turn placement (3 of 9 on the former design): **removing text the teacher cannot see was the decisive fix**, not the placement.

**Decision.** SensAI's RAG rules stay a `system` message (our own text only); the excerpts go in the current user turn between `<<<DOCUMENT EXCERPTS>>>` markers, after removal of hidden comments and neutralisation of anything imitating a marker, a chat role or a template token, followed by the question and a one-line reminder. The stored history keeps the question without the excerpts.

**Trade-off accepted.** Over both measurements the placements are equivalent (instruction followed 9/42 times with the `system` placement, 11/42 with the user turn), with the same answer accuracy. We keep the user turn because it is the honest structure (documents are data, not instructions) and because the final evaluation (83 %) was measured with it, but we do **not** present placement as a protection. The protections are elsewhere: hidden content is removed, writes and network calls need a human yes, and SensAI warns the teacher when a document contains text addressed to an AI ([security](security.md)).

**What we would revisit.** Re-run the 27-trial comparison with each new chat model: instruction-following of larger models can change the result in both directions.

## ADR-03c: Relevance threshold calibration

**Context.** The threshold decides which retrieved chunks reach the model. It was 0.35, chosen without a measurement. The audit found that "What is the capital of Australia?" retrieved course chunks scored 0.42 and 0.41, so the model answered "Canberra" and SensAI printed three unrelated course files as sources.

**Options considered.** Measured with `nomic-embed-text` (cosine similarity between a question and the demo course, 2026-11-01):

| Questions | Scores |
| --- | --- |
| On-topic French questions about the fractions course (5) | 0.688, 0.689, 0.723, 0.819, 0.850 |
| Unrelated questions (7: geography, biology, sport, cooking, engines, small talk, English) | 0.396 to 0.539 |

**Decision.** `rag.relevance_threshold` = **0.6**, in the gap between the two groups (≈ 0.09 margin below the lowest on-topic score, ≈ 0.06 above the highest unrelated one). When no chunk passes and no tool fits the question, SensAI answers "not found" without calling the model at all (`cli/cli.py: no_evidence_answer`), so a weak match can no longer become an invented answer with a false source line.

**Trade-off accepted.** Precision against recall: a genuinely relevant but vaguely worded question can fall under 0.6 and get "not found", where the model would have found something in a 0.55 chunk. For a teacher, a visible "not found" is cheaper than a confident answer with a wrong source. The calibration rests on a small sample and one embedding model.

**Evidence and replay.** `python3 scripts/evaluate_rag.py` measures accuracy and abstention with RAG on and off on a fixed teacher question set ([evaluation](evaluation.md)). Nomic's documentation recommends task prefixes (`search_query:` / `search_document:`), which SensAI does not add; adding them would require re-indexing and a new calibration — listed as future work.

## ADR-04: Memory format

**Context.** M1 and M2. One teacher, one laptop, a few conversations per chapter. A session is always read in full at startup or with `/load`; it is never queried. C5: the teacher must be able to erase what she said about pupils.

**Options considered.**

| Option | Strength | Weakness for this scenario |
| --- | --- | --- |
| One JSON file per session, `profile.json`, and a summary message for compression (chosen) | Readable, one file per conversation, erasure is a file deletion. | No search across sessions; the whole file is rewritten at each save. |
| SQLite tables for sessions and messages (catalogue M3 territory) | Queries across sessions, partial updates. | A deleted row is not physically erased before a `VACUUM`; nothing in the scenario needs to query old conversations. |
| Vector memory (embed each turn, retrieve relevant past turns) | Recalls old facts beyond the window. | One more embedding call per turn and a reordered history; a preparation thread is short enough to fit after compression. |

**Decision.**
- `memory/session_manager.py` writes `sessions/<session id>.json`. The identifier must match `^[A-Za-z0-9_-]{1,64}$`, so it cannot point outside the folder. Files are created with mode `0600` in a `0700` folder (`memory/private_files.py`). `/forget` and `/forget all` delete files.
- `memory/profile_manager.py` injects `profile.json` (name, language, expertise, constraints) into the system message at startup.
- `memory/compression.py` keeps the system message and the last two exchanges, asks the model to summarise the older ones, and stores the summary as one system message starting with `[Previous Conversation Summary:`. A warning appears at 80 % of the window (`cli/session.py`) and compression runs at 85 %.

**Trade-off accepted.** Compression costs one extra model call and depends on the 3B model's summary. Tool calls and tool results are not saved in the session file: only the user message and the final answer are (`cli/cli.py`), so a reloaded conversation no longer contains tool results. The lesson-plan artifact is saved next to the session as `sessions/<session id>_artifact.md` and `.json`, but with the default file permissions (`memory/artifact_manager.py` uses `open()`, not `write_private_json`), and `/forget` removes only `<session id>.json`: the artifact files of a forgotten session stay on disk (`/forget all` removes the `.json` files, not the `.md`).

**Evidence.** `test_privacy.py::SessionIdTests::test_nothing_is_written_outside_the_sessions_folder`, `test_privacy.py::PrivateFileTests::test_session_files_and_folder_are_owner_only`, `test_compression.py::TestSemanticCompression::test_compress_history_preserves_base_system_and_recent_turns`, `test_cli.py::TestArtifactSessionLifecycle::test_save_and_load_session_with_artifact`.

**What we would revisit.** If teachers keep one conversation per chapter for weeks, store sessions in SQLite with an FTS5 index to search past preparations (no dependency). Save tool results with the session.

## ADR-05: MCP client and server

**Context.** T1 requires tool calling through MCP, tested with our own server and a preexisting one. C1, C6.

**Options considered.**
- **The official Python SDK** (`mcp` on PyPI). Version 2.3.0 declares 13 direct runtime dependencies, including `anyio`, `pydantic`, `starlette`, `uvicorn`, `httpx2`, `sse-starlette`, `pyjwt`, and `opentelemetry-api` (PyPI metadata, checked 2026-11-01). Its client API is asynchronous.
- **An in-house client and server** over stdio.

**Decision.** In-house, standard library only:
- `mcp/client.py` (219 lines) starts the server process and exchanges newline-delimited JSON-RPC 2.0 messages: `initialize` with protocol version `2024-11-05`, `notifications/initialized`, `tools/list`, `tools/call`. A reader thread and a queue give each request a timeout. The server receives only an allow-list of environment variables.
- `mcp/manager.py` exposes each remote tool to the model as `<server>__<tool>`, converts its input schema for Ollama, and isolates a failing server.
- `mcp/servers/teacher_tools.py` (285 lines) is our server: weighted average, class statistics, non-blocking timers.
- `mcp_servers.json` declares our server and the third-party `mcp-server-fetch` (started with `uvx`, only when network access is enabled).

The CLI loop is synchronous; the SDK would have required an event loop bridge and a dozen packages for three JSON-RPC methods over one transport.

**Trade-off accepted.** Only the stdio transport and tools are implemented: no HTTP transport, resources, prompts, or sampling. The protocol version is fixed. Compatibility is proven with our server and with `mcp-server-fetch` only. Our package is named `mcp`, the same name as the SDK: if the SDK were installed, the names would collide.

**Evidence.** `test_mcp_client.py::TestMCPClient::test_handshake_and_list_tools`, `test_mcp_client.py::TestMCPClient::test_request_timeout`, `test_mcp_manager.py::TestMCPManager::test_broken_server_does_not_prevent_the_others`, `test_teacher_tools_server.py::TestTeacherToolsProcess::test_server_answers_raw_jsonrpc_messages`; opt-in `test_mcp_third_party.py::TestRealThirdPartyServer::test_both_servers_run_together`.

**What we would revisit.** Rename the package (for example `sensai_mcp`); test against a second third-party server.

## ADR-06: Web interface on the standard library

**Context.** X1, added on 2026-11-01 for teachers who prefer not to use a terminal. One user, on her own computer. C5: nobody else on the school network may reach it.

**Options considered.**

| Option | What it adds | Assessment |
| --- | --- | --- |
| FastAPI | `fastapi`, `uvicorn`, `pydantic`, `starlette` | Allowed by the subject, and the natural choice for a multi-user API. For one local user and 13 routes, it adds a server stack to every teacher's installation (C6). |
| Flask | `flask`, `werkzeug`, `jinja2`, `itsdangerous`, `click`, `blinker` | Same assessment; templates are not needed for one static page. |
| `http.server` from the standard library | Nothing | Chosen. |

**Decision.** `web/server.py` (237 lines) runs a `ThreadingHTTPServer` bound to `127.0.0.1`. It refuses a `Host` header other than `localhost:<port>` or `127.0.0.1:<port>` (protection against DNS rebinding), and accepts a `POST` only as JSON or raw bytes, which a form on another site cannot send without a CORS preflight that the server refuses. Answers stream as JSON lines (`application/x-ndjson`). `web/app.py` drives the same services as the CLI: client, RAG, tools, MCP, memory, and the CLI command handlers. The page is one static `index.html` with no external resources.

**Trade-off accepted.** Routing and validation are written by hand. A confirmation during a streamed answer needs a second request (`POST /api/confirm`) while the stream waits, up to two minutes. `ThreadingHTTPServer` is not designed for many users, which this scenario does not have. The web engine duplicates the CLI's turn loop ([restart item 1](#1-one-turn-engine-for-the-terminal-and-the-web-page)).

**Evidence.** `test_web.py::ServerTests::test_foreign_host_is_refused`, `test_web.py::ServerTests::test_post_must_be_json_or_raw_bytes`, `test_web.py::ServerTests::test_chat_streams_json_lines`, `test_web.py::ToolConfirmationTests::test_refused_tool_does_not_run`.

**What we would revisit.** If SensAI were deployed on a school server for several teachers, FastAPI with authentication would become the right choice; this is outside the current scenario.

## ADR-07: Features deliberately not implemented

### Sandboxed code execution and self-correction (T2, tickets #36 and #37)

**Context.** T2 lets the agent run code it wrote, read the traceback, and fix it. Tickets #36 and #37 were marked out of scope "for the teacher scenario" on 2026-10-22 and remain open.

**Why not, architecturally.**
- **Security surface on the wrong machine.** SensAI runs on the teacher's personal laptop, next to her session files, profile, and exports, which can contain pupil data (C5). Running model-written code there needs real isolation. A plain Python subprocess cannot block file or network access by itself; operating-system sandboxes differ between macOS, Linux, and Windows; Docker would add a virtual machine on macOS that competes for memory with the 2.2 GB model on a 16 GB laptop (C3), and is not something a teacher installs.
- **No need in the scenario (C4).** Camille's calculations are a fixed set: weighted averages, class statistics, timers. The `teacher` MCP server implements them as deterministic, tested functions (`test_teacher_tools_server.py`). The model chooses the tool and its arguments; it never writes code that runs. This gives exact results with none of T2's risk.

**What would change the decision.** A scenario where the teacher analyses arbitrary data files (for example a spreadsheet of results), on a machine where an isolated runtime can be installed by the school.

### Reranking (part of R2, ticket #17)

**What happened.** Ticket #17 was closed as completed on 2026-10-19, but no reranking code exists, neither on `main` nor on any branch of the repository (checked with `git ls-remote` and `git log --all` on 2026-11-01). The ticket's criteria (wider candidate set, reranking switch in the configuration, fallback, tests) are not met. R2 is therefore declared **partial** in the [feature inventory](user-stories.md#feature-inventory).

**Why we kept it out.**
- **Little to reorder.** With one teacher's corpus and the file filter (`/rag filter name ...`), a question retrieves three candidates out of a small set; reordering three candidates changes little in what the model reads.
- **Cost.** Ollama's HTTP API has no reranking endpoint (`POST /api/rerank` returns 404 on Ollama 0.40.7, checked 2026-11-01), so a reranker would be either an extra chat call per question that scores the candidates with the same 3B model, or a cross-encoder outside Ollama (against C2). The chat-call option adds a full prompt of candidates and a second generation before every answer.
- **No way to prove a gain.** Without an evaluation set we could not show that reranking improved answers ([restart item 4](#4-write-the-scenario-and-an-evaluation-set-first)).

The zero-cost parts of R2 were kept instead: metadata filtering before scoring and a per-chunk relevance threshold.

### Other catalogue features left out

| Feature | Reason, tied to the scenario and constraints |
| --- | --- |
| Public share link (part of X3) | A public, unauthenticated link would publish teaching material and possibly pupil information outside the laptop (C5). Export to files covers printing and sending. |
| PDF export (part of X3) | Needs a PDF library (C6); Markdown and text open in the teacher's word processor. |
| M3 External structured memory | See ADR-04: nothing in a preparation workflow queries past conversations. |
| EV1 LLM-as-judge evaluation | The judge would be the same 3B model judging itself; model quality was reviewed manually instead ([benchmarks](model-benchmarks.md#quality-review-protocol)). |
| X5 Scheduling | Preparation is done on demand; no task in the scenario recurs on a timer. |
| A2 Multi-agent orchestration | One model fits in memory at a time on the target laptops (C3); personas (A5) already change role and tone. |

## If we restarted tomorrow

Each proposal answers a bottleneck we hit in this project and stays within the rules: local Ollama over HTTP, no framework, the same laptops.

### 1. One turn engine for the terminal and the web page

**Bottleneck.** On `main` (commit `6487dd6`), a conversation turn (RAG retrieval, tool loop, token count, metrics, sources, budget warning, compression) is written twice: in `cli()` (`cli/cli.py`, lines 298 to 423, about 125 lines) and in `WebApp._run_turn` (`web/app.py`, lines 200 to 288). The web engine imports six private `_` helpers from `cli.cli`, captures the command handlers' `print` output with `redirect_stdout`, and temporarily replaces the terminal confirmation function inside the command modules so that `/forget all` works in a browser. On that commit only the memory commands were patched, so `/rag reset` typed in the web page called `input()` in the server's terminal; the fix in progress patches the RAG module too. Every command that asks a question, and every feature merged after the web page, has to be wired twice.

**Alternative.** A `source/engine/turn.py` module with one function that runs a turn and emits events (`token`, `tool`, `confirm`, `sources`, `notice`, `error`), and a confirmation callback passed in by the caller. The CLI prints the events; the web server writes them as JSON lines. Command handlers return text instead of printing it.

**Why it is realistic.** Plain Python refactoring, no dependency; the existing CLI and web tests describe the expected behaviour.

### 2. Check the token budget before sending

**Bottleneck.** The context usage is computed after the answer, from Ollama's `prompt_eval_count` (`cli/session.py`, `record_turn_tokens`). The 80 % warning and the 85 % compression therefore act one turn late: the request that crosses the limit has already been sent. Ollama does not reject such a request: its `chatPrompt` function drops the oldest non-system messages until the prompt fits (Ollama's `server/prompt.go`; measured in [context engineering](context-engineering.md#4-size-management)), so the conversation silently loses its oldest turns before SensAI's own warning or compression has run. The fixed part of a RAG request with the mathematics persona already takes 32 to 37 % of the 4,096-token window ([ADR-03](#adr-03-chunk-size)).

**Alternative.** Estimate the request before sending it, with `estimate_tokens` calibrated by the ratio observed on the previous turn. If it exceeds the budget, drop the lowest-scoring excerpt, then compress, then send. Move the two thresholds, now constants in two modules, into `config.json`.

**Why it is realistic.** The estimate function and the compression already exist; no dependency. A pre-send check of this kind is being added at the time of writing; the point of this item is that the budget should have been designed this way from the start.

### 3. Hybrid lexical and embedding retrieval

**Bottleneck.** Retrieval relies on cosine similarity alone, and a single global threshold must separate relevant from irrelevant chunks (see [ADR-03c](#adr-03c-relevance-threshold-calibration)). Mathematics questions often hinge on exact terms ("dénominateur", "3/5") that a lexical match finds directly.

**Alternative.** Add an SQLite FTS5 index on the chunk text in the same database file, rank candidates by BM25 and by cosine, and merge the two rankings (reciprocal rank fusion) before applying a cosine floor. This also covers part of what reranking was meant to do.

**Why it is realistic.** FTS5 and its `bm25()` function ship with the SQLite of Python's standard library on our machines (checked: SQLite 3.51.3 in the project's virtual environment); no dependency and no extra model call.

### 4. Write the scenario and an evaluation set first

**Bottleneck.** We planned the feature epics by catalogue area on 2026-10-12, built RAG on 2026-10-18 with code-oriented chunking and source-code formats, and chose the teacher scenario only on 2026-10-22 ([scenario history](products-scenario.md#scenario-history)). The fixed benchmark prompt set (`teacher-scenario-v1`, four cases) came on 2026-10-27. As a result, the chunk size, the relevance threshold, and the reranking decision were taken without a measurement on teacher questions.

**Alternative.** In the first week: fix the persona and write about twenty teacher questions on the course, each with the chunk that should be retrieved and the facts the answer must contain, stored in `benchmarks/`. Every change to chunking, threshold, or retrieval then reports hit@3 and answer checks before merging.

**Why it is realistic.** It is a JSON file and a script in the style of `scripts/benchmark_models.py`; it needs Ollama only, and the team already has the hardware to run it.
