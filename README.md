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

### Ingest a document for RAG

While the application is running, use the following command to validate, chunk, embed, and index a local text or Markdown document:

```text
/ingest path/to/document.md
```

The command currently supports UTF-8 encoded `.txt` and `.md` files. It validates the path, file type, encoding, and content, then stores its chunks and embeddings in the local RAG database. Enter `exit` or `quit` to leave the application.

### Use documents in the chat

Enable document-grounded answers after indexing at least one document:

```text
/rag on
```

While RAG mode is enabled, SensAI retrieves the most relevant excerpts before every question, sends them as bounded context to Ollama, and cites the source filename in its answer. If the information cannot be found in the indexed documents, it must say so. Use `/rag off` to restore the normal chatbot behavior, or `/rag status` to display the current mode.

### Chunking configuration

The RAG chunking module splits normalized document text at paragraph and sentence boundaries. Its default configuration is an 800-character maximum chunk size with a 120-character overlap. These values are defined in `source/rag/chunking.py` and can be adjusted by creating a `ChunkingConfig` instance.

### RAG embedding model

The RAG embedding module uses the Ollama model `nomic-embed-text` by default. Install it with:

```bash
ollama pull nomic-embed-text
```

The chat model and the embedding model are separate: a dedicated embedding model produces vectors intended for semantic search. The embedding model can be changed in Python with `EmbeddingConfig(model="your-model-name")`.

### RAG context configuration

By default, RAG retrieves the top three matching chunks and keeps the complete context message under 4,000 characters. Adjust these settings through `RAGContextConfig(top_k=..., max_context_characters=...)` when creating a `RAGContextProvider`.

### Local RAG storage and search

Embeddings are persisted locally in the SQLite database `data/sensai_rag.sqlite3`. This database is created automatically and is excluded from Git because it contains local, generated knowledge-base data. The `LocalVectorStore` retrieves the most semantically similar chunks with cosine similarity. By default, it returns the top three results; this can be adjusted with the `top_k` argument.

### Run tests

```bash
python3 -m unittest discover -s tests
```
