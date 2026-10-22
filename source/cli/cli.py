import json
import sys
from dataclasses import dataclass
from typing import Any

from config import AppConfig, RagConfig
from memory.profile_manager import ProfileManager
from memory.session_manager import SessionManager
from rag.context import (
    ArchiveIndexingResult,
    RAGContextConfig,
    RAGContextProvider,
    RAGIndexer,
    format_source_attribution,
)
from rag.embeddings import EmbeddingConfig, EmbeddingError, EmbeddingGenerator
from rag.ingestion import DocumentIngestionError, DocumentIngestor
from rag.storage import MetadataFilter, VectorStoreError
from tools.registry import ToolRegistry
from tools.time_tool import GetTimeTool
from utilities.ollama_client import (
    OllamaConnectionError,
    OllamaError,
    OllamaHTTPClient,
    OllamaResponseError,
    build_tool_message,
    parse_tool_calls,
)

MAX_TOOL_ITERATIONS = 5


# Dependency injection for tests and alternative entry points: callers can replace
# external services with deterministic implementations without changing CLI settings.
@dataclass(frozen=True)
class CliDependencies:
    """Optional infrastructure overrides used by tests and alternate frontends."""

    registry: ToolRegistry | None = None
    rag_indexer: RAGIndexer | None = None
    rag_context_provider: RAGContextProvider | None = None


def default_registry(enabled: bool = True) -> ToolRegistry:
    registry = ToolRegistry()
    if enabled:
        registry.register(GetTimeTool())
    return registry


def _rag_context_config(config: RagConfig) -> RAGContextConfig:
    """Convert application configuration into the RAG component settings."""
    return RAGContextConfig(
        top_k=config.top_k,
        max_context_characters=config.max_context_characters,
        relevance_threshold=config.relevance_threshold,
    )


def stream_response(
    client: OllamaHTTPClient,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Stream one model response and return its text and requested tool calls."""
    full_response = ""
    raw_tool_calls: list[dict[str, Any]] = []
    printed_prefix = False

    for line in client.stream_message(messages, tools=tools):
        if not line:
            continue
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise OllamaResponseError("Ollama returned an invalid streaming response.") from error
        if not isinstance(payload, dict):
            raise OllamaResponseError("Ollama returned an invalid streaming response.")

        message = payload.get("message", {})
        if not isinstance(message, dict):
            raise OllamaResponseError("Ollama returned an invalid streaming response.")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise OllamaResponseError("Ollama returned a response with invalid text content.")
        if content:
            if not printed_prefix:
                print("Assistant: ", end="", flush=True)
                printed_prefix = True
            print(content, end="", flush=True)
            full_response += content
        tool_calls = message.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            raise OllamaResponseError("Ollama returned invalid tool calls in a streaming response.")
        raw_tool_calls.extend(tool_calls)

    if printed_prefix:
        print()
    return full_response, raw_tool_calls


def run_tool_calls(
    registry: ToolRegistry,
    raw_tool_calls: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> None:
    """Execute each tool call and append its result to the history."""
    for call in parse_tool_calls({"tool_calls": raw_tool_calls}):
        print(f"[tool] {call['name']}({json.dumps(call['arguments'])})")
        result = registry.execute(call["name"], call["arguments"])
        label = "error" if result.is_error else "result"
        print(f"[tool] {label}: {result.content}")
        messages.append(build_tool_message(call["name"], result.content))


def _request_messages(
    messages: list[dict[str, Any]],
    rag_system_message: str | None,
) -> list[dict[str, Any]]:
    """Inject temporary RAG context after the base system message when enabled."""
    if rag_system_message is None:
        return messages
    return [messages[0], {"role": "system", "content": rag_system_message}, *messages[1:]]


def cli(
    app_config: AppConfig | None = None,
    dependencies: CliDependencies | None = None,
    *,
    session_id: str | None = None,
    profile_path: str | None = None,
) -> None:
    """Run the interactive CLI using application settings and optional wiring overrides."""
    app_config = app_config or AppConfig()
    dependencies = dependencies or CliDependencies()
    model_config = app_config.model
    feature_config = app_config.features
    rag_config = app_config.rag
    try:
        session = SessionManager(session_id=session_id)
        profile_manager = ProfileManager(profile_path=profile_path)
        loaded = session.load(session_id) if session_id else None
    except FileNotFoundError as error:
        print(f"Error: {error}")
        sys.exit(1)
    messages = list(loaded) if loaded else [
        {"role": "system", "content": profile_manager.build_system_prompt()}
    ]
    metadata_filter: MetadataFilter | None = None
    show_sources = feature_config.source_display_enabled
    rag_enabled = False

    client = OllamaHTTPClient(
        base_url=model_config.base_url,
        model=model_config.chat_model,
        options={"temperature": model_config.temperature, "num_ctx": model_config.context_window},
    )
    registry = (
        dependencies.registry
        if dependencies.registry is not None
        else default_registry(feature_config.tools_enabled)
    )
    rag_indexer = dependencies.rag_indexer
    if rag_indexer is None:
        embedding_client = OllamaHTTPClient(
            base_url=model_config.base_url,
            model=model_config.embedding_model,
        )
        rag_indexer = RAGIndexer(
            ingestor=DocumentIngestor(),
            embedding_generator=EmbeddingGenerator(
                EmbeddingConfig(model=model_config.embedding_model),
                client=embedding_client,
            ),
        )
    rag_context_provider = dependencies.rag_context_provider
    if rag_context_provider is None:
        rag_context_provider = RAGContextProvider(
            embedding_generator=rag_indexer.embedding_generator,
            vector_store=rag_indexer.vector_store,
            config=_rag_context_config(rag_config),
        )
    def cmd_exit(_args: list[str]) -> bool:
        session.save()
        return True

    def cmd_save(_args: list[str]) -> None:
        session.save()

    def cmd_load(args: list[str]) -> None:
        if not args:
            print("Usage: /load <session_id>")
            return
        try:
            session.load(args[0])
            print(f"Session {args[0]} loaded.")
        except FileNotFoundError as error:
            print(f"Error: {error}")

    commands = {
        "/save": cmd_save,
        "/load": cmd_load,
        "exit": cmd_exit,
        "quit": cmd_exit,
    }

    while True:
        try:
            user_input = input("You: ")
            if not user_input.strip():
                continue

            command, *command_args = user_input.strip().split()
            if command.lower() in commands:
                if commands[command.lower()](command_args):
                    break
                continue

            if user_input.startswith("/rag filter"):
                metadata_filter = _handle_rag_filter_command(user_input, metadata_filter)
                continue

            if user_input.startswith("/rag"):
                if not feature_config.rag_enabled:
                    print("RAG is disabled in config.json.")
                    continue
                rag_enabled = _handle_rag_command(user_input, rag_enabled)
                continue

            if user_input.startswith("/ingest"):
                _handle_ingest_command(user_input, rag_indexer)
                continue

            if user_input.startswith("/sources"):
                show_sources = _handle_sources_command(user_input, show_sources)
                continue

            turn_start = len(messages)
            messages.append({"role": "user", "content": user_input})
            tools = registry.to_ollama_tools(user_input)
            rag_system_message = None
            prepared_context = None
            # A tool-specific request should be handled by its matching tool,
            # not by document retrieval. For all other requests, RAG keeps its
            # normal document-grounded behavior.
            if rag_enabled and not tools:
                try:
                    if metadata_filter is None:
                        prepared_context = rag_context_provider.prepare(user_input)
                    else:
                        prepared_context = rag_context_provider.prepare(user_input, metadata_filter)
                    rag_system_message = prepared_context.system_message
                    if metadata_filter and not prepared_context.results:
                        print("No indexed chunks match the active RAG filter.")
                except (EmbeddingError, VectorStoreError) as error:
                    del messages[turn_start:]
                    print(f"\nRAG Error: {error}")
                    continue

            for _ in range(MAX_TOOL_ITERATIONS):
                request_messages = _request_messages(messages, rag_system_message)
                full_response, raw_tool_calls = stream_response(client, request_messages, tools)

                assistant_message: dict[str, Any] = {"role": "assistant", "content": full_response}
                if raw_tool_calls:
                    assistant_message["tool_calls"] = raw_tool_calls
                messages.append(assistant_message)

                if not raw_tool_calls:
                    if show_sources and prepared_context is not None:
                        attribution = format_source_attribution(prepared_context.results)
                        if attribution:
                            print(f"\n{attribution}")
                    break
                run_tool_calls(registry, raw_tool_calls, messages)
            else:
                print(f"\nWarning: stopped after {MAX_TOOL_ITERATIONS} tool calls without a final answer.")

            session.add_message("user", user_input)
            session.add_message("assistant", full_response)


        except (KeyboardInterrupt, EOFError):
            session.save()
            print("Exiting.")
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


def _handle_ingest_command(user_input: str, rag_indexer: RAGIndexer) -> None:
    """Index a local document without sending the command to the chat model."""
    command, separator, document_path = user_input.partition(" ")
    if command != "/ingest" or not separator or not document_path.strip():
        print("Usage: /ingest <path-to-document>")
        return

    try:
        result = rag_indexer.index_document(document_path.strip())
    except (DocumentIngestionError, EmbeddingError, VectorStoreError) as error:
        print(f"Document indexing failed: {error}")
        return

    if isinstance(result, ArchiveIndexingResult):
        _print_archive_indexing_summary(result)
        return

    print(
        f"Document indexed: {result.document.source_filename} "
        f"({result.chunk_count} chunks; {result.inserted_chunk_count} new chunks stored)."
    )


def _print_archive_indexing_summary(result: ArchiveIndexingResult) -> None:
    """Display successful, skipped, and failed archive entries separately."""
    archive = result.archive
    print(
        f"Archive indexed: {archive.archive_filename} "
        f"({len(result.indexed_documents)} files indexed; {result.chunk_count} chunks; "
        f"{result.inserted_chunk_count} new chunks stored)."
    )
    if result.indexed_documents:
        indexed_files = ", ".join(
            document_result.document.archive_entry_path or document_result.document.source_filename
            for document_result in result.indexed_documents
        )
        print(f"Indexed files: {indexed_files}")
    if archive.skipped_entries:
        print(f"Skipped unsupported files: {', '.join(archive.skipped_entries)}")
    for failure in archive.failed_entries:
        print(f"Failed to extract {failure.entry_path}: {failure.reason}")


def _handle_rag_command(user_input: str, rag_enabled: bool) -> bool:
    """Enable, disable, or report the status of document-grounded chat mode."""
    command, _, option = user_input.partition(" ")
    normalized_option = option.strip().lower()
    if command != "/rag" or normalized_option not in {"on", "off", "status"}:
        print("Usage: /rag <on|off|status>")
        return rag_enabled
    if normalized_option == "on":
        print("RAG mode enabled. Responses will use indexed documents only.")
        return True
    if normalized_option == "off":
        print("RAG mode disabled. The normal chatbot behavior is restored.")
        return False
    print(f"RAG mode is {'enabled' if rag_enabled else 'disabled'}.")
    return rag_enabled


def _handle_rag_filter_command(
    user_input: str,
    current_filter: MetadataFilter | None,
) -> MetadataFilter | None:
    """Set or clear the metadata restriction used by subsequent RAG questions."""
    _, _, arguments = user_input.partition("/rag filter")
    option, _, value = arguments.strip().partition(" ")
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if option == "clear" and not value.strip():
        print("RAG metadata filter cleared.")
        return None
    if option == "name" and values:
        metadata_filter = MetadataFilter(source_filenames=values)
        print(f"RAG filter set to source name: {', '.join(values)}")
        return metadata_filter
    if option == "category" and values:
        allowed_categories = {"document", "code", "configuration", "markup"}
        normalized_values = tuple(item.lower() for item in values)
        invalid_values = sorted(set(normalized_values) - allowed_categories)
        if not invalid_values:
            metadata_filter = MetadataFilter(source_types=normalized_values)
            print(f"RAG filter set to category: {', '.join(normalized_values)}")
            return metadata_filter
        print(f"Unknown category: {', '.join(invalid_values)}")
        return current_filter
    print("Usage: /rag filter <name <source>[,<source>...]|category <type>[,<type>...]|clear>")
    return current_filter


def _handle_sources_command(user_input: str, show_sources: bool) -> bool:
    """Enable, disable, or report CLI-generated source attribution lines."""
    command, _, option = user_input.partition(" ")
    normalized_option = option.strip().lower()
    if command != "/sources" or normalized_option not in {"on", "off", "status"}:
        print("Usage: /sources <on|off|status>")
        return show_sources
    if normalized_option == "on":
        print("Source display enabled.")
        return True
    if normalized_option == "off":
        print("Source display disabled.")
        return False
    print(f"Source display is {'enabled' if show_sources else 'disabled'}.")
    return show_sources
