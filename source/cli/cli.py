import atexit
import json
import sqlite3
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

from cli.commands import GLOBAL_COMMANDS
from cli.error_handling import report_cli_error
from cli.commands.help import display_unknown_command
from cli.context_budget import answer_limit_notice, answer_reserve, over_budget_message, prompt_budget
from cli.model_factory import create_model_runtime
from cli.session import CliCommandContext, CliSession, CommandDefinition
from cli.terminal import ask_confirmation, assistant_prefix, configure_line_editing, read_user_input
from cli.tool_calls import run_tool_calls
from config import AppConfig, FeatureConfig
from memory.compression import (
    DEFAULT_AUTO_COMPRESSION_THRESHOLD,
    compress_history,
    is_summary_message,
)
from memory.artifact_manager import ArtifactManager
from memory.profile_manager import ProfileManager
from memory.session_manager import SessionManager
from observability.app_logging import get_logger
from observability.monitoring import (
    DisabledMonitoringStore,
    MonitoringStore,
    create_monitoring_store,
    estimate_tokens,
)
from rag.context import PreparedContext, RAGContextProvider, RAGIndexer, format_source_attribution
from rag.embeddings import EmbeddingError
from mcp.manager import MCPManager
from rag.storage import VectorStoreError
from tools.filesystem import filesystem_tools
from tools.permissions import PermissionChecker, load_permissions
from tools.quiz_export import QuizStore, SaveLastQuizTool
from tools.artifact_tool import ArtifactTool
from tools.registry import ToolRegistry
from tools.safety import suspicious_instructions
from tools.time_tool import GetTimeTool
from tools.web_search import WebSearchTool
from utilities.ollama_client import (
    OllamaError,
    OllamaHTTPClient,
    OllamaResponseError,
)

MAX_TOOL_ITERATIONS = 5
logger = get_logger("cli")

# The RAG instructions forbid outside knowledge. When tools are offered in the same
# request, the model must still be allowed to call them, but only for what the excerpts
# cannot answer: a date written in a course document is not "today's date".
RAG_TOOLS_NOTE = (
    "\n\nTools are available in this conversation. Call one only for what the excerpts "
    "cannot answer (today's date or time, a calculation on given marks, a timer). Answer "
    "questions about the indexed documents from the excerpts."
)

# Dependency injection for tests and alternative entry points: callers can replace
# external services with deterministic implementations without changing CLI settings.
@dataclass(frozen=True)
class CliDependencies:
    """Optional infrastructure overrides used by tests and alternate frontends."""

    registry: ToolRegistry | None = None
    rag_indexer: RAGIndexer | None = None
    rag_context_provider: RAGContextProvider | None = None


def default_registry(
    enabled: bool = True,
    permission_checker: PermissionChecker | None = None,
    quiz_store: QuizStore | None = None,
    network_enabled: bool = False,
    artifact_manager: ArtifactManager | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    if enabled:
        registry.register(GetTimeTool())
        if network_enabled:
            registry.register(WebSearchTool())
        if artifact_manager is not None:
            registry.register(ArtifactTool(artifact_manager))
        if permission_checker is not None:
            for tool in filesystem_tools(permission_checker):
                registry.register(tool)
            if quiz_store is not None:
                registry.register(SaveLastQuizTool(permission_checker, quiz_store))
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
    token_counts: dict[str, int] | None = None,
    on_token: Callable[[str], None] | None = None,
    max_answer_tokens: int | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Stream one model response and return its text and requested tool calls.

    Text is printed on the terminal, or handed to `on_token` when a frontend supplies one.
    `max_answer_tokens` is sent as Ollama's `num_predict`: without it, a 3B model can
    loop for minutes (one evaluation answer ran 11,464 tokens in 211 s).
    """
    full_response = ""
    raw_tool_calls: list[dict[str, Any]] = []
    printed_prefix = False

    options = {"options": {"num_predict": max_answer_tokens}} if max_answer_tokens else {}
    for line in client.stream_message(messages, tools=tools, **options):
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
            if on_token is not None:
                on_token(content)
            else:
                if not printed_prefix:
                    print(assistant_prefix(), end="", flush=True)
                    printed_prefix = True
                print(content, end="", flush=True)
            full_response += content
        tool_calls = message.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            raise OllamaResponseError("Ollama returned invalid tool calls in a streaming response.")
        raw_tool_calls.extend(tool_calls)
        if token_counts is not None:
            # The final chunk carries Ollama's counters; durations are in nanoseconds.
            for field_name, key in _OLLAMA_COUNTERS:
                value = payload.get(field_name)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    token_counts[key] = value
            if payload.get("done_reason") == "length":
                token_counts["stopped_at_limit"] = 1

    if printed_prefix:
        print()
    return full_response, raw_tool_calls


_OLLAMA_COUNTERS = (
    ("prompt_eval_count", "prompt_tokens"),
    ("eval_count", "completion_tokens"),
    ("prompt_eval_duration", "prompt_duration_ns"),
    ("eval_duration", "completion_duration_ns"),
    ("load_duration", "load_duration_ns"),
    ("total_duration", "total_duration_ns"),
)


def _request_messages(
    messages: list[dict[str, Any]],
    prepared_context: PreparedContext | None = None,
    rag_rules: str | None = None,
    turn_start: int | None = None,
) -> list[dict[str, Any]]:
    """Build the messages of one model call without changing the stored history.

    Order: persona and profile, then SensAI's RAG rules (our own text only), then the
    conversation, then the current user turn. The retrieved excerpts travel inside that
    last user turn, framed as data: a document is written by someone else, so it never
    gets the authority of a system message, and it sits right next to the question it
    answers, where a small model pays the most attention.
    """
    if prepared_context is None:
        return messages
    rules = {"role": "system", "content": rag_rules or prepared_context.system_message}
    request = [messages[0], rules, *messages[1:]]
    index = _current_user_index(request, None if turn_start is None else turn_start + 1)
    if index is not None:
        question = request[index]
        request[index] = {**question, "content": prepared_context.user_message(str(question.get("content", "")))}
    return request


def _current_user_index(request: list[dict[str, Any]], hint: int | None) -> int | None:
    """Position of the user message that started the current turn."""
    if hint is not None and 0 <= hint < len(request) and request[hint].get("role") == "user":
        return hint
    for index in range(len(request) - 1, -1, -1):
        if request[index].get("role") == "user":
            return index
    return None


def _record_request(
    session: CliSession,
    request_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    rag_rules: str | None,
    chat_client: Any,
) -> None:
    """Keep what is about to be sent, for /debug prompt."""
    session.last_request_messages = [message.copy() for message in request_messages]
    session.last_rag_context = rag_rules
    session.last_request_tools = [tool["function"]["name"] for tool in tools]
    session.last_request_options = {"model": chat_client.model, **dict(getattr(chat_client, "options", {}) or {})}


def _check_context_budget(
    session: CliSession,
    request_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> str | None:
    """Return a refusal message when the request would not fit in the context window."""
    estimated = session.token_estimator.estimate(request_messages, tools)
    if estimated <= prompt_budget(session.context_window):
        return None
    return over_budget_message(estimated, session.context_window)


def _compress_before_sending(
    messages: list[dict[str, Any]],
    turn_start: int,
    conversation_session: SessionManager,
    chat_client: Any,
) -> bool:
    """Summarise the turns before the current question to make room. Return True if done."""
    history, current_turn = messages[:turn_start], messages[turn_start:]
    compressed, result = compress_history(history, chat_client)
    if not result.compressed:
        return False
    messages[:] = [*compressed, *current_turn]
    conversation_session.messages[:] = [
        dict(message) for message in compressed if message.get("role") != "system" or is_summary_message(message)
    ]
    print(
        "Context nearly full: older turns were summarised before sending "
        f"({result.messages_before} -> {result.messages_after} messages)."
    )
    return True


def _run_claimed_tool(
    registry: ToolRegistry,
    user_input: str,
    messages: list[dict[str, Any]],
    **run_options: Any,
) -> bool:
    """Run the argument-free tool an explicit request asks for, before the model answers.

    "Quelle heure est-il ?" must not depend on the model remembering to call the
    clock: SensAI calls it and the model only phrases the result. Returns True when
    a tool was run.
    """
    tool = registry.claimed_tool_without_arguments(user_input)
    if tool is None:
        return False
    call = {"function": {"name": tool.name, "arguments": {}}}
    messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
    run_tool_calls(registry, [call], messages, **run_options)
    return True


def injection_warning(prepared_context: PreparedContext | None) -> str | None:
    """Warn the teacher when an excerpt sent with this question talks to the AI.

    Prompt wording cannot stop a 3B model from sometimes obeying such text, so the
    teacher is told which document to check instead of trusting the answer blindly.
    """
    if prepared_context is None:
        return None
    sources = sorted(
        {result.source_filename for result in prepared_context.results if suspicious_instructions(str(result.text))}
    )
    if not sources:
        return None
    return (
        f"Warning: an excerpt used for this answer ({', '.join(sources)}) contains text addressed to an AI "
        "assistant. The answer may be influenced by it: check that document."
    )


def no_evidence_answer(preferred_language: Any) -> str:
    """The answer given in RAG mode when no excerpt is relevant and no tool is offered.

    The model is not called at all, so it cannot fill the gap with invented facts.
    """
    if isinstance(preferred_language, str) and preferred_language.strip().lower().startswith(("fr", "fran")):
        return "Je ne trouve pas cette information dans les documents indexés."
    return "I cannot find this information in the indexed documents."


# A small model sometimes ends its turn with no text at all, often after a long tool
# result or at a high temperature; without this line the teacher sees nothing happen.
EMPTY_ANSWER_NOTICE = (
    "The model returned an empty answer. Ask again more simply, split the request, or lower "
    "the temperature (/set temperature 0.2)."
)

NO_EVIDENCE_NOTE = (
    "(No indexed excerpt reached the relevance threshold, so the model was not asked: "
    "nothing could be invented. Use /rag off to ask the model without documents.)"
)


def _record_generation(session: CliSession, token_counts: dict[str, int]) -> None:
    """Remember Ollama's timings of the last answer for /stats."""
    session.last_generation_stats = dict(token_counts)


def _display_model_fallback(primary_model: str, fallback_model: str) -> None:
    """Keep a model switch visible instead of silently changing an answer source."""
    logger.warning("model.fallback primary=%s fallback=%s", primary_model, fallback_model)
    print(
        f"Model '{primary_model}' is unavailable. "
        f"Retrying with fallback '{fallback_model}'."
    )


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
    except (FileNotFoundError, ValueError) as error:
        report_cli_error(error, logger)
        sys.exit(1)
    messages = list(loaded) if loaded else []
    _ensure_base_system_message(messages, profile_manager, app_config.personas.active.system_instructions)
    session = CliSession(
        show_sources=feature_config.source_display_enabled,
        active_persona_id=app_config.personas.active.identifier,
        context_window=model_config.context_window,
    )

    artifact_manager = ArtifactManager(session_id=conversation_session.session_id)
    if session_id:
        try:
            artifact_manager.load()
            logger.info("artifact.loaded_at_startup session_id=%s", session_id)
        except FileNotFoundError:
            pass
    quiz_store = QuizStore()
    permission_checker, permission_warnings = load_permissions()
    for warning in permission_warnings:
        print(f"Warning: {warning}")
    if feature_config.network_enabled:
        print("Network access is enabled: web search and web servers can send requests to the internet.")
    registry = (
        dependencies.registry
        if dependencies.registry is not None
        else default_registry(
            feature_config.tools_enabled,
            permission_checker,
            quiz_store,
            feature_config.network_enabled,
            artifact_manager,
        )
    )
    mcp_manager = mcp_manager if mcp_manager is not None else _default_mcp_manager(feature_config)
    # Registered before the servers start: Ctrl+C during a slow start must still stop them.
    atexit.register(mcp_manager.close)
    for warning in mcp_manager.start() + mcp_manager.register_tools(registry):
        print(f"Warning: {warning}")

    def _auto_save() -> None:
        if feature_config.memory_enabled and not session.persistence_saved_on_exit:
            try:
                conversation_session.save()
            except Exception as error:
                logger.warning("session.autosave_failed error=%s", error)
            if artifact_manager and artifact_manager.has_artifact():
                try:
                    artifact_manager.session_id = conversation_session.session_id
                    artifact_manager.save()
                except Exception as error:
                    logger.warning("artifact.autosave_failed error=%s", error)

    atexit.register(_auto_save)
    runtime = create_model_runtime(
        model_config,
        rag_config,
        rag_indexer=dependencies.rag_indexer,
        rag_context_provider=dependencies.rag_context_provider,
        client_factory=OllamaHTTPClient,
        on_fallback=_display_model_fallback,
    )
    command_context = CliCommandContext(
        feature_config,
        runtime.rag_indexer,
        conversation_session,
        GLOBAL_COMMANDS,
        registry,
        _create_monitoring_store(app_config),
        app_config.personas,
        profile_manager,
        artifact_manager=artifact_manager,
        messages=messages,
        chat_client=runtime.chat_client,
        rag_context_provider=runtime.rag_context_provider,
        permission_checker=permission_checker,
        quiz_store=quiz_store,
    )
    line_editing_enabled = configure_line_editing()

    while True:
        turn_start: int | None = None
        try:
            user_input = read_user_input(line_editing_enabled)
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
            _take_web_sources(registry)
            messages.append({"role": "user", "content": user_input})
            tools = registry.to_ollama_tools(user_input)
            rag_rules = None
            prepared_context = None
            # A request explicitly claimed by a tool (for example "what time is it")
            # is handled by that tool, not by document retrieval. Any other request
            # keeps RAG's document-grounded behavior, even when tools are offered.
            if session.rag_enabled and not registry.claims_request(user_input):
                try:
                    if session.metadata_filter is None:
                        prepared_context = runtime.rag_context_provider.prepare(user_input)
                    else:
                        prepared_context = runtime.rag_context_provider.prepare(
                            user_input,
                            session.metadata_filter,
                        )
                    rag_rules = prepared_context.system_message
                    if tools:
                        rag_rules += RAG_TOOLS_NOTE
                    if session.metadata_filter and not prepared_context.results:
                        print("No indexed chunks match the active RAG filter.")
                    warning = injection_warning(prepared_context)
                    if warning:
                        print(warning)
                except (EmbeddingError, VectorStoreError) as error:
                    del messages[turn_start:]
                    report_cli_error(error, logger)
                    continue

            # Checked before sending: above num_ctx, Ollama would silently cut the
            # beginning of the prompt, where the persona and the instructions are.
            request_messages = _request_messages(messages, prepared_context, rag_rules, turn_start)
            refusal = _check_context_budget(session, request_messages, tools)
            if refusal and feature_config.memory_enabled and _compress_before_sending(
                messages, turn_start, conversation_session, runtime.chat_client
            ):
                turn_start = len(messages) - 1
                request_messages = _request_messages(messages, prepared_context, rag_rules, turn_start)
                refusal = _check_context_budget(session, request_messages, tools)
            if refusal:
                del messages[turn_start:]
                print(refusal)
                continue

            if prepared_context is not None and not prepared_context.results and not tools:
                full_response = no_evidence_answer(profile_manager.data.get("preferred_language"))
                print(f"{assistant_prefix()}{full_response}\n{NO_EVIDENCE_NOTE}")
                messages.append({"role": "assistant", "content": full_response})
                conversation_session.add_message("user", user_input)
                conversation_session.add_message("assistant", full_response)
                continue

            tools_used = _run_claimed_tool(
                registry,
                user_input,
                messages,
                monitoring_store=command_context.monitoring_store,
                confirm=ask_confirmation,
            )
            full_response, raw_tool_calls = "", []
            for _ in range(MAX_TOOL_ITERATIONS):
                request_messages = _request_messages(messages, prepared_context, rag_rules, turn_start)
                if _check_context_budget(session, request_messages, tools):
                    print("\nWarning: the tool results filled the context window; stopping this answer here.")
                    break
                _record_request(session, request_messages, tools, rag_rules, runtime.chat_client)
                logger.debug(
                    "model.request_prepared message_count=%s rag_context=%s tool_definitions=%s",
                    len(request_messages),
                    prepared_context is not None,
                    len(tools),
                )
                started_at = perf_counter()
                token_counts: dict[str, int] = {}
                try:
                    full_response, raw_tool_calls = stream_response(
                        runtime.chat_client,
                        request_messages,
                        tools,
                        token_counts=token_counts,
                        max_answer_tokens=answer_reserve(session.context_window),
                    )
                except OllamaError as error:
                    _record_model_interaction(
                        command_context,
                        runtime.chat_client.model,
                        app_config.monitoring.prompt_version,
                        replace(feature_config, rag_enabled=prepared_context is not None),
                        request_messages,
                        "",
                        prepared_context,
                        (perf_counter() - started_at) * 1000,
                        "error",
                        type(error).__name__,
                    )
                    raise
                prompt_tokens = token_counts.get("prompt_tokens", estimate_tokens(request_messages))
                completion_tokens = token_counts.get("completion_tokens", estimate_tokens(full_response))
                session.record_turn_tokens(prompt_tokens, completion_tokens)
                if token_counts.get("stopped_at_limit"):
                    print(f"\n{answer_limit_notice(session.context_window)}")
                if "prompt_tokens" in token_counts:
                    session.token_estimator.learn(request_messages, tools, token_counts["prompt_tokens"])
                _record_generation(session, token_counts)
                interaction_id = _record_model_interaction(
                    command_context,
                    runtime.chat_client.model,
                    app_config.monitoring.prompt_version,
                    replace(feature_config, rag_enabled=prepared_context is not None),
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
                    # After a tool call the answer rests on the tool result, which the
                    # [tool] lines already show; listing the excerpts would be a false claim.
                    if session.show_sources and prepared_context is not None and not tools_used:
                        attribution = format_source_attribution(prepared_context.results)
                        if attribution:
                            print(f"\n{attribution}")
                    web_sources = _take_web_sources(registry)
                    if web_sources:
                        print("\nWeb sources consulted:\n" + "\n".join(f"- {url}" for url in web_sources))
                    break
                run_tool_calls(
                    registry,
                    raw_tool_calls,
                    messages,
                    command_context.monitoring_store,
                    interaction_id,
                    confirm=ask_confirmation,
                )
                tools_used = True
            else:
                print(f"\nWarning: stopped after {MAX_TOOL_ITERATIONS} tool calls without a final answer.")
            if not full_response.strip() and not raw_tool_calls:
                print(f"\n{EMPTY_ANSWER_NOTICE}")

            conversation_session.add_message("user", user_input)
            conversation_session.add_message("assistant", full_response)

            if feature_config.memory_enabled:
                warning = session.get_budget_warning()
                if warning:
                    print(f"\n{warning}")
                if session.context_usage_ratio >= DEFAULT_AUTO_COMPRESSION_THRESHOLD:
                    compressed_msgs, result = compress_history(
                        messages,
                        runtime.chat_client,
                    )
                    if result.compressed:
                        messages[:] = compressed_msgs
                        persisted_messages = [
                            dict(msg)
                            for msg in compressed_msgs
                            if msg.get("role") != "system" or is_summary_message(msg)
                        ]
                        conversation_session.messages[:] = persisted_messages
                        print(
                            f"Auto-compression triggered: condensed older turns "
                            f"({result.messages_before} -> {result.messages_after} messages)."
                        )


        except (KeyboardInterrupt, EOFError):
            logger.info("cli.interrupted")
            if feature_config.memory_enabled:
                conversation_session.save()
                session.persistence_saved_on_exit = True
                if artifact_manager and artifact_manager.has_artifact():
                    try:
                        artifact_manager.session_id = conversation_session.session_id
                        artifact_manager.save()
                    except Exception as error:
                        logger.warning("artifact.save_failed error=%s", error)
            print("Exiting.")
            sys.exit(0)

        except Exception as error:
            if turn_start is not None:
                del messages[turn_start:]
            report_cli_error(error, logger)

        finally:
            print()


def _default_mcp_manager(feature_config: FeatureConfig) -> MCPManager:
    """MCP servers provide tools, so `tools_enabled: false` starts none of them."""
    if not feature_config.tools_enabled:
        return MCPManager()
    return MCPManager.from_config_file(network_enabled=feature_config.network_enabled)


def _take_web_sources(registry: ToolRegistry) -> list[str]:
    """Return the URLs the web search tool offered during this turn, and reset its list."""
    tool = registry.get("web_search")
    return tool.take_sources() if isinstance(tool, WebSearchTool) else []


def _create_monitoring_store(app_config: AppConfig) -> MonitoringStore | DisabledMonitoringStore:
    """Create optional local metrics storage without preventing the CLI from starting."""
    try:
        return create_monitoring_store(app_config.monitoring)
    except (OSError, sqlite3.Error) as error:
        logger.warning("monitoring.disabled error_type=%s", type(error).__name__)
        print(f"Warning: monitoring is disabled because its database could not be opened: {error}")
        return DisabledMonitoringStore()


def _ensure_base_system_message(
    messages: list[dict[str, Any]],
    profile_manager: ProfileManager,
    persona_prompt: str,
) -> None:
    """Insert the active persona when a loaded history has no system message."""
    system_message = {"role": "system", "content": profile_manager.build_system_prompt(persona_prompt)}
    if messages and messages[0].get("role") == "system":
        messages[0] = system_message
    else:
        messages.insert(0, system_message)


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
