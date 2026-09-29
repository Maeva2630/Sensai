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

The script runs streaming-chat and model-listing tests against the local Ollama API at `http://localhost:11434`. An embedding test is included in the code and can be enabled when needed.
