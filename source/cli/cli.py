import sys
import json
from utilities.ollama_client import OllamaHTTPClient, OllamaConnectionError, OllamaError
from rag.ingestion import DocumentIngestionError, DocumentIngestor, IngestedDocument


def cli(model: str = "llama3.2", ingestor: DocumentIngestor | None = None):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    client = OllamaHTTPClient(model=model)
    ingestor = ingestor or DocumentIngestor()
    ingested_documents: dict[str, IngestedDocument] = {}

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue

            if user_input.startswith("/ingest"):
                _handle_ingest_command(user_input, ingestor, ingested_documents)
                continue

            messages.append({"role": "user", "content": user_input})
            
            response = client.stream_message(messages)

            full_response = ""
            print("Assistant: ", end="", flush=True)

            for line in response:
                if line:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    print(content, end="", flush=True)
                    full_response += content

            print() # Print '\n'
            messages.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)

        except OllamaConnectionError as e:
            if messages and messages[-1].get("role") == "user":
                messages.pop()
            print(f"\nConnection Error: Unable to connect to Ollama server ({e}). Please ensure Ollama is running.")

        except OllamaError as e:
            if messages and messages[-1].get("role") == "user":
                messages.pop()
            print(f"\nWarning: {e}")


def _handle_ingest_command(
    user_input: str,
    ingestor: DocumentIngestor,
    ingested_documents: dict[str, IngestedDocument],
) -> None:
    """Ingest a local document without sending the command to the chat model."""
    command, separator, document_path = user_input.partition(" ")
    if command != "/ingest" or not separator or not document_path.strip():
        print("Usage: /ingest <path-to-.txt-or-.md-file>")
        return

    try:
        document = ingestor.ingest(document_path.strip())
    except DocumentIngestionError as error:
        print(f"Document ingestion failed: {error}")
        return

    if document.document_id in ingested_documents:
        print(f"Document already ingested: {document.source_filename}")
        return

    ingested_documents[document.document_id] = document
    print(
        f"Document ingested: {document.source_filename} "
        f"({document.character_count} characters, id: {document.document_id[:12]}...). "
        "Ready for chunking and indexing."
    )
