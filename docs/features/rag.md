# RAG: documents and source code

[Feature index](README.md) · [User guide](../user-guide.md)

Retrieval-augmented generation (RAG) lets SensAI search locally indexed material and
supply relevant passages to Ollama. Ingestion extracts text, divides it into chunks,
generates embeddings, and stores them in a local SQLite database. At question time,
the query is embedded, similar chunks are retrieved, and selected excerpts are added
to the model request. The database is `data/sensai_rag.sqlite3` and contains document
text and embeddings; treat it as private.

## Index and ask

Pull the configured embedding model and follow the [setup guide](../user-guide.md),
then try the `planets.md` example there. In the CLI:

```text
/ingest planets.md
/rag on
According to the document, why is Earth called the Blue Planet?
/rag status
/rag off
```

The response may identify the oceans as the reason and display `Sources: planets.md
(chunks: ...)`. This line identifies retrieved chunks, not a verified answer. Re-indexing
an unchanged document should store zero new chunks. Document identity is based on its
normalized content, so an edited document is treated as a different version; the old
indexed content is not automatically removed.

Index several files by separating paths with spaces or tabs; quote paths containing spaces:

```text
/ingest --multiple docs/guide.md docs/course.pdf "docs/lesson notes.docx"
```

`-m` is the short form. `/ingest project.zip` inspects a ZIP and reports indexed,
unsupported, and extraction-failed entries. Archive security limits reject unsafe paths,
encrypted entries, and excessive sizes, nesting, file counts, or compression ratios.

## Supported files and limits

- UTF-8 text and source files: `.txt`, `.md`, `.py`, `.c`, `.h`, `.cpp`, `.hpp`,
  `.js`, `.ts`, `.java`, `.go`, `.rs`, `.html`, `.css`, `.xml`, `.svg`, `.json`,
  `.yaml`, `.yml`, `.toml`, `.ini`, `.sql`, and `.sh`.
- Text-based PDFs via `pypdf`: no OCR for scans or images, and complex visual layouts
  and tables are not guaranteed to be preserved.
- `.docx` via `python-docx`: paragraphs and headings; legacy `.doc`, image OCR, comments,
  tracked changes, and complex layouts are not supported.
- ZIP archives of supported files. Code is read as text, not executed or imported.

## Filters and source display

```text
/rag filter name guide.md,notes.md
/rag filter category code
/rag filter clear
/rag reset
/sources status
/sources off
/sources on
```

Categories are `document`, `code`, `configuration`, and `markup`. Only matching chunks
are considered when a filter is active. The source commands control the CLI-generated
attribution line, not retrieval or citations the model might itself generate.

`/rag reset` asks for an explicit confirmation, then permanently deletes every indexed
chunk, extracted text, and embedding from `data/sensai_rag.sqlite3`. The SQLite file and
its schema remain so that the next `/ingest` starts with a clean index. It also disables
RAG for the current session and clears any metadata filter. This action cannot be undone:
keep the original source documents if you may need to index them again.

## Configuration and limitations

In `config.json`, `features.rag_enabled` makes RAG mode available;
`features.source_display_enabled` sets initial attribution visibility. RAG mode itself
starts off each session. The `rag` section controls `top_k` (default 3),
`relevance_threshold` (default 0.35, range -1 to 1), and `max_context_characters`
(default 4000). Adjust these values based on your own tests; they do not guarantee
answer quality. `model.embedding_model` selects the Ollama embedding model.

Chunks default to a maximum of 800 characters with a 120-character overlap. These are
Python defaults in `source/rag/chunking.py`, not CLI configuration options. The
chunker prefers readable boundaries and considers source structure for supported code
and markup. Search only considers embeddings generated with the configured model and
matching dimensions. Re-ingesting unchanged chunks after changing models does not replace
their existing vectors in the current database. Use `/rag reset` before a full re-index
when changing embedding models; it deletes the current index rather than removing or
modifying the original source documents.

The assistant is instructed to say when the answer is missing, but a model can still
ignore instructions or generate inaccurate text. RAG context remains available when tools
are offered, except for requests explicitly claimed by a narrow tool such as a time query.
A separate reranking step is not currently implemented. See [tools and MCP](tools-and-mcp.md).
