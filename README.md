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

While the application is running, use the following command to validate and extract a local text or Markdown document:

```text
/ingest path/to/document.md
```

The command currently supports UTF-8 encoded `.txt` and `.md` files. It validates the path, file type, encoding, and content before preparing the document for the upcoming chunking and indexing stages. Enter `exit` or `quit` to leave the application.

### Chunking configuration

The RAG chunking module splits normalized document text at paragraph and sentence boundaries. Its default configuration is an 800-character maximum chunk size with a 120-character overlap. These values are defined in `source/rag/chunking.py` and can be adjusted by creating a `ChunkingConfig` instance.

### Run tests

```bash
python3 -m unittest discover -s tests
```
