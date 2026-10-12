import json
import sys
from typing import Any, Dict, List, Optional, Tuple

from rag.ingestion import DocumentIngestionError, DocumentIngestor, IngestedDocument
from tools.registry import ToolRegistry
from tools.time_tool import GetTimeTool
from utilities.ollama_client import (
    OllamaConnectionError,
    OllamaError,
    OllamaHTTPClient,
    build_tool_message,
    parse_tool_calls,
)


MAX_TOOL_ITERATIONS = 5


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(GetTimeTool())
    return registry


def stream_response(
    client: OllamaHTTPClient,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Stream one model response to the terminal.

    Returns the text printed and the raw tool calls requested by the model.
    """
    full_response = ""
    raw_tool_calls: List[Dict[str, Any]] = []
    printed_prefix = False

    for line in client.stream_message(messages, tools=tools):
        if not line:
            continue
        message = json.loads(line).get("message", {})
        content = message.get("content", "")
        if content:
            if not printed_prefix:
                print("Assistant: ", end="", flush=True)
                printed_prefix = True
            print(content, end="", flush=True)
            full_response += content
        raw_tool_calls.extend(message.get("tool_calls") or [])

    if printed_prefix:
        print()
    return full_response, raw_tool_calls


def run_tool_calls(
    registry: ToolRegistry,
    raw_tool_calls: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
) -> None:
    """Execute each tool call and append its result to the history."""
    for call in parse_tool_calls({"tool_calls": raw_tool_calls}):
        print(f"[tool] {call['name']}({json.dumps(call['arguments'])})")
        result = registry.execute(call["name"], call["arguments"])
        label = "error" if result.is_error else "result"
        print(f"[tool] {label}: {result.content}")
        messages.append(build_tool_message(call["name"], result.content))


def cli(
    model: str = "llama3.2",
    registry: Optional[ToolRegistry] = None,
    ingestor: DocumentIngestor | None = None,
) -> None:
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    client = OllamaHTTPClient(model=model)
    registry = registry if registry is not None else default_registry()
    tools = registry.to_ollama_tools()
    ingestor = ingestor or DocumentIngestor()
    ingested_documents: dict[str, IngestedDocument] = {}

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue

            if user_input.startswith("/ingest"):
                _handle_ingest_command(user_input, ingestor, ingested_documents)
                continue

            turn_start = len(messages)
            messages.append({"role": "user", "content": user_input})

            for _ in range(MAX_TOOL_ITERATIONS):
                full_response, raw_tool_calls = stream_response(client, messages, tools)

                assistant_message: Dict[str, Any] = {"role": "assistant", "content": full_response}
                if raw_tool_calls:
                    assistant_message["tool_calls"] = raw_tool_calls
                messages.append(assistant_message)

                if not raw_tool_calls:
                    break
                run_tool_calls(registry, raw_tool_calls, messages)
            else:
                print(f"\nWarning: stopped after {MAX_TOOL_ITERATIONS} tool calls without a final answer.")

        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)

        except OllamaConnectionError as error:
            del messages[turn_start:]
            print(
                f"\nConnection Error: Unable to connect to Ollama server ({error}). "
                "Please ensure Ollama is running."
            )

        except OllamaError as error:
            del messages[turn_start:]
            print(f"\nWarning: {error}")


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
