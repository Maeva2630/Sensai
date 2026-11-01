"""Engine of the web interface.

It drives the same services as the terminal chat (model client, RAG, tools, MCP, memory,
profile, commands) but exchanges events with a browser instead of printing. Nothing here
talks HTTP: `web/server.py` only translates requests into calls on `WebApp`.
"""

from __future__ import annotations

import atexit
import io
import json
import queue
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter
from typing import Any

import cli.commands.memory as memory_commands
from cli.cli import (
    MAX_TOOL_ITERATIONS,
    RAG_TOOLS_NOTE,
    CliDependencies,
    _create_monitoring_store,
    _dispatch_command,
    _ensure_base_system_message,
    _record_model_interaction,
    _request_messages,
    _take_web_sources,
    default_registry,
    stream_response,
)
from cli.commands import GLOBAL_COMMANDS
from cli.commands.help import display_unknown_command
from cli.error_handling import format_cli_error
from cli.model_factory import create_model_runtime
from cli.session import CliCommandContext, CliSession
from cli.tool_calls import run_tool_calls
from config import AppConfig
from mcp.manager import MCPManager
from memory.artifact_manager import ArtifactManager
from memory.compression import DEFAULT_AUTO_COMPRESSION_THRESHOLD, compress_history, is_summary_message
from memory.profile_manager import ProfileManager
from memory.session_manager import SessionManager
from observability.monitoring import estimate_tokens
from rag.context import format_source_attribution
from rag.storage import VectorStoreError
from tools.permissions import load_permissions
from tools.quiz_export import QuizStore, render_quiz
from tools.registry import ToolRegistry
from tools.structured_output import MAX_QUIZ_QUESTIONS
from observability.app_logging import get_logger
from utilities.ollama_client import OllamaHTTPClient, OllamaError

logger = get_logger("web")

CONFIRMATION_TIMEOUT_SECONDS = 120
MAX_PREVIEW_CHARACTERS = 70
MAX_LISTED_SESSIONS = 30
PROFILE_FIELDS = ("user_name", "preferred_language", "domain_expertise")


class WebApp:
    """One teacher's local SensAI session, shared by every browser tab on this machine."""

    def __init__(
        self,
        app_config: AppConfig,
        *,
        session_id: str | None = None,
        profile_path: str | None = None,
        mcp_manager: MCPManager | None = None,
        dependencies: CliDependencies | None = None,
    ) -> None:
        dependencies = dependencies or CliDependencies()
        self.config = app_config
        features = app_config.features
        self.conversation_session = SessionManager(session_id=session_id)
        self.profile_manager = ProfileManager(profile_path=profile_path)
        loaded = self.conversation_session.load(session_id) if session_id else None
        self.messages: list[dict[str, Any]] = [dict(message) for message in loaded] if loaded else []
        _ensure_base_system_message(
            self.messages, self.profile_manager, app_config.personas.active.system_instructions
        )
        self.session = CliSession(
            show_sources=features.source_display_enabled,
            active_persona_id=app_config.personas.active.identifier,
            context_window=app_config.model.context_window,
        )

        self.artifact_manager = ArtifactManager(session_id=self.conversation_session.session_id)
        if session_id:
            try:
                self.artifact_manager.load()
            except FileNotFoundError:
                pass
        # Problems found at startup are shown in the page instead of the terminal.
        self.startup_warnings: list[str] = []
        self._fallback_notices: list[str] = []
        quiz_store = QuizStore()
        permission_checker, warnings = load_permissions()
        self.startup_warnings += warnings
        self.registry: ToolRegistry = (
            dependencies.registry
            if dependencies.registry is not None
            else default_registry(
                features.tools_enabled, permission_checker, quiz_store, features.network_enabled, self.artifact_manager
            )
        )
        mcp_manager = (
            mcp_manager
            if mcp_manager is not None
            else MCPManager.from_config_file(network_enabled=features.network_enabled)
        )
        self.startup_warnings += mcp_manager.start() + mcp_manager.register_tools(self.registry)
        atexit.register(mcp_manager.close)
        runtime = create_model_runtime(
            app_config.model,
            app_config.rag,
            rag_indexer=dependencies.rag_indexer,
            rag_context_provider=dependencies.rag_context_provider,
            client_factory=OllamaHTTPClient,
            on_fallback=lambda primary, fallback: self._fallback_notices.append(
                f"Model '{primary}' is unavailable. Retrying with fallback '{fallback}'."
            ),
        )
        self.runtime = runtime
        self._ollama = OllamaHTTPClient(base_url=app_config.model.base_url, model=app_config.model.chat_model)
        self.context = CliCommandContext(
            features,
            runtime.rag_indexer,
            self.conversation_session,
            GLOBAL_COMMANDS,
            self.registry,
            _create_monitoring_store(app_config),
            app_config.personas,
            self.profile_manager,
            artifact_manager=self.artifact_manager,
            messages=self.messages,
            chat_client=runtime.chat_client,
            rag_context_provider=runtime.rag_context_provider,
            permission_checker=permission_checker,
            quiz_store=quiz_store,
        )
        # One request at a time: the model, the messages list and stdout capture are shared.
        self._lock = threading.Lock()
        self._pending: dict[str, tuple[threading.Event, list[bool]]] = {}

    # ----------------------------------------------------------------- chat

    def stream_chat(self, user_input: str) -> Iterator[dict[str, Any]]:
        """Yield events (token, tool, confirm, notice, error, done) while a message is handled."""
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()

        def work() -> None:
            with self._lock:
                try:
                    if user_input.startswith("/"):
                        events.put({"type": "notice", "text": self._execute_command(user_input)})
                    else:
                        self._run_turn(user_input, events.put)
                except Exception as error:  # the page must keep working whatever fails
                    events.put({"type": "error", "text": format_cli_error(error, logger)})
                events.put({"type": "done", "state": self.state()})
                events.put(None)

        threading.Thread(target=work, daemon=True).start()
        while (event := events.get()) is not None:
            yield event

    def resolve_confirmation(self, confirmation_id: str, approved: bool) -> bool:
        """Answer a pending confirmation. Returns False if it is unknown or already answered."""
        pending = self._pending.get(confirmation_id)
        if pending is None:
            return False
        event, answer = pending
        answer.append(approved)
        event.set()
        return True

    def _confirm(self, emit: Callable[[dict[str, Any]], None]) -> Callable[[str], bool]:
        """A confirm function for tools: ask the browser and wait for its answer."""

        def confirm(question: str) -> bool:
            confirmation_id = uuid.uuid4().hex
            event, answer = threading.Event(), []
            self._pending[confirmation_id] = (event, answer)
            emit({"type": "confirm", "id": confirmation_id, "question": question})
            try:
                # No answer in time means no: nothing leaves the machine by default.
                return event.wait(CONFIRMATION_TIMEOUT_SECONDS) and bool(answer and answer[0])
            finally:
                del self._pending[confirmation_id]

        return confirm

    def _run_turn(self, user_input: str, emit: Callable[[dict[str, Any]], None]) -> None:
        """One user message: retrieval, streaming, tool calls, memory. Mirrors the CLI loop."""
        session, messages, registry = self.session, self.messages, self.registry
        model_config, features = self.config.model, self.config.features
        turn_start = len(messages)
        try:
            _take_web_sources(registry)
            messages.append({"role": "user", "content": user_input})
            tools = registry.to_ollama_tools(user_input)
            rag_system_message = None
            prepared_context = None
            if session.rag_enabled and not registry.claims_request(user_input):
                provider = self.runtime.rag_context_provider
                if session.metadata_filter is None:
                    prepared_context = provider.prepare(user_input)
                else:
                    prepared_context = provider.prepare(user_input, session.metadata_filter)
                rag_system_message = prepared_context.system_message
                if tools:
                    rag_system_message += RAG_TOOLS_NOTE
                if session.metadata_filter and not prepared_context.results:
                    emit({"type": "notice", "text": "No indexed chunks match the active RAG filter."})

            for _ in range(MAX_TOOL_ITERATIONS):
                request_messages = _request_messages(messages, rag_system_message)
                session.last_request_messages = [message.copy() for message in request_messages]
                session.last_rag_context = rag_system_message
                started_at = perf_counter()
                token_counts: dict[str, int] = {}
                try:
                    full_response, raw_tool_calls = stream_response(
                        self.runtime.chat_client,
                        request_messages,
                        tools,
                        token_counts=token_counts,
                        on_token=lambda text: emit({"type": "token", "text": text}),
                    )
                except OllamaError as error:
                    self._record(request_messages, "", prepared_context, started_at, "error", type(error).__name__)
                    raise
                while self._fallback_notices:
                    emit({"type": "notice", "text": self._fallback_notices.pop(0)})
                session.record_turn_tokens(
                    token_counts.get("prompt_tokens", estimate_tokens(request_messages)),
                    token_counts.get("completion_tokens", estimate_tokens(full_response)),
                )
                interaction_id = self._record(request_messages, full_response, prepared_context, started_at, "success")

                assistant_message: dict[str, Any] = {"role": "assistant", "content": full_response}
                if raw_tool_calls:
                    assistant_message["tool_calls"] = raw_tool_calls
                messages.append(assistant_message)

                if not raw_tool_calls:
                    if session.show_sources and prepared_context is not None:
                        attribution = format_source_attribution(prepared_context.results)
                        if attribution:
                            emit({"type": "sources", "text": attribution})
                    web_sources = _take_web_sources(registry)
                    if web_sources:
                        emit({"type": "sources", "text": "Web sources consulted:\n" + "\n".join(web_sources)})
                    break
                run_tool_calls(
                    registry,
                    raw_tool_calls,
                    messages,
                    self.context.monitoring_store,
                    interaction_id,
                    confirm=self._confirm(emit),
                    emit=lambda line: emit({"type": "tool", "text": line}),
                )
            else:
                emit({"type": "notice", "text": f"Stopped after {MAX_TOOL_ITERATIONS} tool calls without a final answer."})
        except Exception:
            del messages[turn_start:]
            raise

        self.conversation_session.add_message("user", user_input)
        self.conversation_session.add_message("assistant", full_response)
        self._autosave()
        if features.memory_enabled:
            warning = session.get_budget_warning()
            if warning:
                emit({"type": "notice", "text": warning})
            if session.context_usage_ratio >= DEFAULT_AUTO_COMPRESSION_THRESHOLD:
                result = self._compress()
                if result is not None:
                    emit({"type": "notice", "text": f"Older turns were summarised to free space ({result})."})

    def _record(
        self,
        request_messages: list[dict[str, Any]],
        response_text: str,
        prepared_context: Any,
        started_at: float,
        status: str,
        error_category: str | None = None,
    ) -> int:
        return _record_model_interaction(
            self.context,
            self._active_model(),
            self.config.monitoring.prompt_version,
            self.config.features,
            request_messages,
            response_text,
            prepared_context,
            (perf_counter() - started_at) * 1000,
            status,
            error_category,
        )

    def _compress(self) -> str | None:
        """Summarise older turns. Returns a short description, or None when nothing changed."""
        compressed, result = compress_history(self.messages, self.runtime.chat_client)
        if not result.compressed:
            return None
        self.messages[:] = compressed
        self.conversation_session.messages[:] = [
            dict(message) for message in compressed if message.get("role") != "system" or is_summary_message(message)
        ]
        return f"{result.messages_before} -> {result.messages_after} messages"

    def _autosave(self) -> None:
        """Keep the conversation on disk after each turn, as the CLI does on exit."""
        if self.config.features.memory_enabled:
            with redirect_stdout(io.StringIO()):
                self.conversation_session.save()
            if self.artifact_manager.has_artifact():
                self.artifact_manager.session_id = self.conversation_session.session_id
                self.artifact_manager.save()

    # ------------------------------------------------------------- commands

    def run_command(self, text: str, *, confirmed: bool = False) -> str:
        """Run a slash command and return what it would have printed."""
        with self._lock:
            return self._execute_command(text, confirmed=confirmed)

    def _execute_command(self, text: str, *, confirmed: bool = False) -> str:
        """Reuse the terminal command handlers and capture their output.

        Their confirmation prompt (`/forget all`) reads the terminal, so it is replaced by the
        answer the browser already collected: refused unless `confirmed`.
        """
        buffer = io.StringIO()
        original_confirm = memory_commands.ask_confirmation
        memory_commands.ask_confirmation = lambda _question: confirmed
        try:
            with redirect_stdout(buffer):
                try:
                    if not _dispatch_command(text, self.session, self.context, GLOBAL_COMMANDS):
                        display_unknown_command(text, GLOBAL_COMMANDS)
                except Exception as error:
                    print(format_cli_error(error, logger))
        finally:
            memory_commands.ask_confirmation = original_confirm
        if self.session.exit_requested:  # /exit and /quit mean nothing in a browser tab
            self.session.exit_requested = False
            buffer.write("\nThe web page stays open: just close the tab when you are done.")
        return buffer.getvalue().strip()

    def quiz_command(self, topic: str, questions: int, language: str) -> str:
        """Build and run a `/quiz` command from the quiz form."""
        topic = topic.replace('"', " ").strip()
        language = language.replace('"', " ").strip()
        if not topic:
            return "Invalid quiz: a topic is required."
        if not 1 <= questions <= MAX_QUIZ_QUESTIONS:
            return f"Invalid quiz: choose between 1 and {MAX_QUIZ_QUESTIONS} questions."
        command = f"/quiz --questions {questions}"
        if language:
            command += f' --language "{language}"'
        return self.run_command(f"{command} {topic}")

    def render_last_quiz(self, answers: bool, as_json: bool) -> tuple[str, str] | None:
        """The last quiz as (filename, content), or None when none was generated."""
        quiz = self.context.quiz_store.quiz
        if quiz is None:
            return None
        name = "quiz.json" if as_json else "quiz.txt"
        return name, render_quiz(quiz, Path(name), answers)

    # ------------------------------------------------------------ documents

    def ingest_file(self, filename: str, data: bytes, upload_dir: Path) -> str:
        """Index an uploaded document. The temporary copy is removed right after indexing."""
        safe_name = "".join(c for c in Path(filename).name if c.isalnum() or c in "._- ").strip(". ")
        if not safe_name:
            return "Document error: the file name is not valid."
        upload_dir.mkdir(parents=True, exist_ok=True)
        path = upload_dir / safe_name
        with self._lock:
            try:
                path.write_bytes(data)
                return self._execute_command(f'/ingest --multiple "{path}"')
            finally:
                path.unlink(missing_ok=True)

    # -------------------------------------------------------------- profile

    def update_profile(self, values: dict[str, Any]) -> None:
        """Save the profile and apply it to the system prompt of the current conversation."""
        data = dict(self.profile_manager.data)
        for field in PROFILE_FIELDS:
            value = values.get(field)
            if isinstance(value, str) and value.strip():
                data[field] = value.strip()
        constraints = values.get("constraints")
        if isinstance(constraints, list):
            data["constraints"] = [item.strip() for item in constraints if isinstance(item, str) and item.strip()]
        with self._lock:
            self.profile_manager.save(data)
            persona = self.config.personas.find(self.session.active_persona_id) or self.config.personas.active
            _ensure_base_system_message(self.messages, self.profile_manager, persona.system_instructions)

    # ------------------------------------------------------------- sessions

    def new_conversation(self) -> None:
        """Save the current conversation, then start an empty one."""
        with self._lock:
            if self.conversation_session.messages:
                self._autosave()
            self.conversation_session.session_id = str(uuid.uuid4())
            self.artifact_manager.session_id = self.conversation_session.session_id
            self.artifact_manager.reset()
            self.conversation_session.clear()
            self.messages[:] = self.messages[:1]
            self.session.last_request_messages = None
            self.session.last_rag_context = None

    def forget_all(self) -> str:
        """Delete every saved conversation. The browser asks the user first."""
        return self.run_command("/forget all", confirmed=True)

    def saved_sessions(self) -> list[dict[str, str]]:
        """The most recent saved conversations, newest first, with the first question as a preview."""
        storage = self.conversation_session.storage_dir
        paths = sorted(storage.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True) if storage.is_dir() else []
        sessions = []
        for path in paths[:MAX_LISTED_SESSIONS]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                first = next((m["content"] for m in data.get("messages", []) if m.get("role") == "user"), "")
            except (OSError, ValueError, KeyError, TypeError):
                continue
            preview = " ".join(str(first).split())
            if len(preview) > MAX_PREVIEW_CHARACTERS:
                preview = preview[:MAX_PREVIEW_CHARACTERS] + "..."
            sessions.append(
                {"id": path.stem, "timestamp": str(data.get("timestamp", "")), "preview": preview or "(empty)"}
            )
        return sorted(sessions, key=lambda item: item["timestamp"], reverse=True)

    # ---------------------------------------------------------------- state

    def history(self) -> list[dict[str, str]]:
        """The messages worth showing: what the teacher wrote and what the assistant answered."""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.messages
            if m.get("role") in ("user", "assistant") and m.get("content") and not is_summary_message(m)
        ]

    def state(self) -> dict[str, Any]:
        """Everything the page needs to draw itself."""
        session, features = self.session, self.config.features
        try:
            documents = [{"name": name, "chunks": count} for name, count in self.runtime.rag_indexer.vector_store.list_sources()]
        except VectorStoreError:
            documents = []
        return {
            "model": self._active_model(),
            "artifact": self._artifact_state(),
            "session_id": self.conversation_session.session_id,
            "features": {
                "rag": features.rag_enabled,
                "tools": features.tools_enabled,
                "memory": features.memory_enabled,
                "network": features.network_enabled,
            },
            "rag_enabled": session.rag_enabled,
            "show_sources": session.show_sources,
            "persona": session.active_persona_id,
            "personas": [{"id": p.identifier, "name": p.display_name} for p in self.config.personas.profiles],
            "profile": self.profile_manager.data,
            "documents": documents,
            "has_quiz": self.context.quiz_store.quiz is not None,
            "tools": [
                {"name": t.name, "origin": getattr(t, "server", None) or "native", "description": t.description}
                for t in self.registry.list_tools()
            ],
            "tokens": {
                "context": session.last_turn_tokens,
                "window": session.context_window,
                "total": session.total_session_tokens,
            },
            "warnings": self.startup_warnings,
        }

    def _active_model(self) -> str:
        """The chat model that answered last (it differs from the configured one after a fallback)."""
        model = getattr(self.runtime.chat_client, "model", None)
        return model if isinstance(model, str) else self.config.model.chat_model

    def _artifact_state(self) -> dict[str, Any] | None:
        artifact = self.artifact_manager.artifact
        if artifact is None:
            return None
        return {"title": artifact.title, "version": artifact.version, "sections": artifact.get_outline()}

    def artifact_markdown(self) -> tuple[str, str] | None:
        """The living document as (filename, Markdown), or None when there is none."""
        if not self.artifact_manager.has_artifact():
            return None
        return "document.md", self.artifact_manager.get_content()

    def check_model(self) -> str | None:
        """A readable problem if Ollama or every configured chat model is unavailable, else None."""
        try:
            installed = self._ollama.list_models()
        except OllamaError as error:
            return format_cli_error(error, logger)
        wanted = (self.config.model.chat_model, *self.config.model.fallback_chat_models)
        if not any(name == model or name.split(":")[0] == model for model in wanted for name in installed):
            return f"The model '{wanted[0]}' is not installed. Run: ollama pull {wanted[0]}"
        return None
