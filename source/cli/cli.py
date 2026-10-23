import atexit
import json
import sqlite3
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from cli.commands import GLOBAL_COMMANDS
from cli.commands.help import display_unknown_command
from cli.model_factory import create_model_runtime
from cli.session import CliCommandContext, CliSession, CommandDefinition
from cli.tool_calls import run_tool_calls
from config import AppConfig, FeatureConfig
from memory.profile_manager import ProfileManager
from memory.session_manager import SessionManager
from observability.app_logging import get_logger
from observability.monitoring import (
    DisabledMonitoringStore,
    MonitoringStore,
    create_monitoring_store,
    estimate_tokens,
)
from rag.context import RAGContextProvider, RAGIndexer, format_source_attribution
from rag.embeddings import EmbeddingError
from mcp.manager import MCPManager
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
logger = get_logger("cli")

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
    command_definitions: tuple[CommandDefinition, ...],
) -> bool:
    """Run the first matching command handler and report whether input was a command."""
    for command in command_definitions:
        if command.matches(user_input):
            logger.info("command.executed name=%s feature=%s", command.name, command.feature)
            command.handler(user_input, session, context)
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
    *,
    session_id: str | None = None,
    profile_path: str | None = None,
    mcp_manager: MCPManager | None = None,
) -> None:
    """Run the interactive CLI using application settings and optional wiring overrides."""
    app_config = app_config or AppConfig()
    dependencies = dependencies or CliDependencies()
    model_config = app_config.model
    feature_config = app_config.features
    rag_config = app_config.rag
    try:
        conversation_session = SessionManager(session_id=session_id)
        profile_manager = ProfileManager(profile_path=profile_path)
        loaded = conversation_session.load(session_id) if session_id else None
    except FileNotFoundError as error:
        print(f"Error: {error}")
        sys.exit(1)
    messages = list(loaded) if loaded else [
        {"role": "system", "content": profile_manager.build_system_prompt(app_config.personas.active.system_prompt)}
    ]
    session = CliSession(show_sources=feature_config.source_display_enabled)

    registry = (
        dependencies.registry
        if dependencies.registry is not None
        else default_registry(feature_config.tools_enabled)
    )
    mcp_manager = mcp_manager if mcp_manager is not None else MCPManager.from_config_file()
    for warning in mcp_manager.start() + mcp_manager.register_tools(registry):
        print(f"Warning: {warning}")
    atexit.register(mcp_manager.close)
    runtime = create_model_runtime(
        model_config,
        rag_config,
        rag_indexer=dependencies.rag_indexer,
        rag_context_provider=dependencies.rag_context_provider,
        client_factory=OllamaHTTPClient,
    )
    command_context = CliCommandContext(
        feature_config,
        runtime.rag_indexer,
        conversation_session,
        GLOBAL_COMMANDS,
        registry,
        _create_monitoring_store(app_config),
    )

    while True:
        try:
            user_input = input("You: ")
            if not user_input.strip():
                continue

            if _dispatch_command(user_input, session, command_context, GLOBAL_COMMANDS):
                if session.exit_requested:
                    break
                continue
            if user_input.startswith("/"):
                display_unknown_command(user_input, GLOBAL_COMMANDS)
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
                    logger.warning("rag.request_error error_type=%s", type(error).__name__)
                    del messages[turn_start:]
                    print(f"\nRAG Error: {error}")
                    continue

            for _ in range(MAX_TOOL_ITERATIONS):
                request_messages = _request_messages(messages, rag_system_message)
                session.last_request_messages = [message.copy() for message in request_messages]
                session.last_rag_context = rag_system_message
                logger.debug(
                    "model.request_prepared message_count=%s rag_context=%s tool_definitions=%s",
                    len(request_messages),
                    rag_system_message is not None,
                    len(tools),
                )
                started_at = perf_counter()
                try:
                    full_response, raw_tool_calls = stream_response(
                        runtime.chat_client,
                        request_messages,
                        tools,
                    )
                except OllamaError as error:
                    _record_model_interaction(
                        command_context,
                        model_config.chat_model,
                        app_config.monitoring.prompt_version,
                        feature_config,
                        request_messages,
                        "",
                        prepared_context,
                        (perf_counter() - started_at) * 1000,
                        "error",
                        type(error).__name__,
                    )
                    raise
                interaction_id = _record_model_interaction(
                    command_context,
                    model_config.chat_model,
                    app_config.monitoring.prompt_version,
                    feature_config,
                    request_messages,
                    full_response,
                    prepared_context,
                    (perf_counter() - started_at) * 1000,
                    "success",
                )

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
                run_tool_calls(
                    registry,
                    raw_tool_calls,
                    messages,
                    command_context.monitoring_store,
                    interaction_id,
                )
            else:
                print(f"\nWarning: stopped after {MAX_TOOL_ITERATIONS} tool calls without a final answer.")

            conversation_session.add_message("user", user_input)
            conversation_session.add_message("assistant", full_response)


        except (KeyboardInterrupt, EOFError):
            logger.info("cli.interrupted")
            conversation_session.save()
            print("Exiting.")
            sys.exit(0)

        except OllamaConnectionError as error:
            logger.warning("ollama.connection_error error_type=%s", type(error).__name__)
            del messages[turn_start:]
            print(
                f"\nConnection Error: Unable to connect to Ollama server ({error}). "
                "Please ensure Ollama is running."
            )

        except OllamaError as error:
            logger.warning("ollama.request_error error_type=%s", type(error).__name__)
            del messages[turn_start:]
            print(f"\nWarning: {error}")

        finally:
            print()


def _create_monitoring_store(app_config: AppConfig) -> MonitoringStore | DisabledMonitoringStore:
    """Create optional local metrics storage without preventing the CLI from starting."""
    try:
        return create_monitoring_store(app_config.monitoring)
    except (OSError, sqlite3.Error) as error:
        logger.warning("monitoring.disabled error_type=%s", type(error).__name__)
        print(f"Warning: monitoring is disabled because its database could not be opened: {error}")
        return DisabledMonitoringStore()


def _record_model_interaction(
    command_context: CliCommandContext,
    model: str,
    prompt_version: str,
    feature_config: FeatureConfig,
    request_messages: list[dict[str, Any]],
    response_text: str,
    prepared_context: Any,
    latency_ms: float,
    status: str,
    error_category: str | None = None,
) -> int:
    """Record safe request metadata without persisting user or model content."""
    if command_context.monitoring_store is None:
        return 0
    try:
        return command_context.monitoring_store.record_model_interaction(
            model=model,
            features=feature_config,
            prompt_version=prompt_version,
            status=status,
            latency_ms=latency_ms,
            input_tokens=estimate_tokens(request_messages),
            output_tokens=estimate_tokens(response_text) if status == "success" else None,
            rag_chunk_count=len(prepared_context.results) if prepared_context is not None else 0,
            error_category=error_category,
        )
    except (OSError, sqlite3.Error) as error:
        # Metrics are optional: an unavailable local database must not interrupt chat.
        logger.warning("monitoring.write_failed error_type=%s", type(error).__name__)
        return 0
