import json
import sys
from dataclasses import dataclass
from typing import Any

from cli.commands import GLOBAL_COMMAND_HANDLERS
from cli.model_factory import create_model_runtime
from cli.session import CliCommandContext, CliSession, CommandHandler
from cli.tool_calls import run_tool_calls
from config import AppConfig
from rag.context import RAGContextProvider, RAGIndexer, format_source_attribution
from rag.embeddings import EmbeddingError
from rag.storage import VectorStoreError
from tools.registry import ToolRegistry
from tools.time_tool import GetTimeTool
from utilities.ollama_client import (
    OllamaConnectionError,
    OllamaError,
    OllamaHTTPClient,
    OllamaResponseError,
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


def _dispatch_command(
    user_input: str,
    session: CliSession,
    context: CliCommandContext,
    command_handlers: tuple[tuple[str, CommandHandler], ...],
) -> bool:
    """Run the first matching command handler and report whether input was a command."""
    for prefix, handler in command_handlers:
        if user_input.startswith(prefix):
            handler(user_input, session, context)
            return True
    return False


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
) -> None:
    """Run the interactive CLI using application settings and optional wiring overrides."""
    app_config = app_config or AppConfig()
    dependencies = dependencies or CliDependencies()
    model_config = app_config.model
    feature_config = app_config.features
    rag_config = app_config.rag
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    session = CliSession(show_sources=feature_config.source_display_enabled)

    registry = (
        dependencies.registry
        if dependencies.registry is not None
        else default_registry(feature_config.tools_enabled)
    )
    runtime = create_model_runtime(
        model_config,
        rag_config,
        rag_indexer=dependencies.rag_indexer,
        rag_context_provider=dependencies.rag_context_provider,
        client_factory=OllamaHTTPClient,
    )
    command_context = CliCommandContext(feature_config, runtime.rag_indexer)

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue

            if _dispatch_command(user_input, session, command_context, GLOBAL_COMMAND_HANDLERS):
                continue

            turn_start = len(messages)
            messages.append({"role": "user", "content": user_input})
            tools = registry.to_ollama_tools(user_input)
            rag_system_message = None
            prepared_context = None
            # A tool-specific request should be handled by its matching tool,
            # not by document retrieval. For all other requests, RAG keeps its
            # normal document-grounded behavior.
            if session.rag_enabled and not tools:
                try:
                    if session.metadata_filter is None:
                        prepared_context = runtime.rag_context_provider.prepare(user_input)
                    else:
                        prepared_context = runtime.rag_context_provider.prepare(
                            user_input,
                            session.metadata_filter,
                        )
                    rag_system_message = prepared_context.system_message
                    if session.metadata_filter and not prepared_context.results:
                        print("No indexed chunks match the active RAG filter.")
                except (EmbeddingError, VectorStoreError) as error:
                    del messages[turn_start:]
                    print(f"\nRAG Error: {error}")
                    continue

            for _ in range(MAX_TOOL_ITERATIONS):
                request_messages = _request_messages(messages, rag_system_message)
                full_response, raw_tool_calls = stream_response(runtime.chat_client, request_messages, tools)

                assistant_message: dict[str, Any] = {"role": "assistant", "content": full_response}
                if raw_tool_calls:
                    assistant_message["tool_calls"] = raw_tool_calls
                messages.append(assistant_message)

                if not raw_tool_calls:
                    if session.show_sources and prepared_context is not None:
                        attribution = format_source_attribution(prepared_context.results)
                        if attribution:
                            print(f"\n{attribution}")
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
