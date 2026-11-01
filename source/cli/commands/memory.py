"""CLI commands owned by the conversation-memory feature."""

from cli.error_handling import report_cli_error
from cli.terminal import ask_confirmation
from cli.session import CliCommandContext, CliSession, CommandDefinition
from observability.app_logging import get_logger
from memory.compression import compress_history, is_summary_message


logger = get_logger("commands.memory")


def _dispatch_save(
    _user_input: str,
    _session: CliSession,
    context: CliCommandContext,
) -> None:
    """Persist the current conversation and active artifact without leaving the CLI."""
    if not context.feature_config.memory_enabled:
        print("Memory is disabled in config.json.")
        return
    if context.conversation_session.save() is None:
        print("The conversation is empty: no session file was written.")
    if context.artifact_manager and context.artifact_manager.has_artifact():
        try:
            context.artifact_manager.session_id = context.conversation_session.session_id
            saved_artifact_path = context.artifact_manager.save()
        except (OSError, ValueError) as error:
            logger.warning("artifact.save_failed error_type=%s error=%s", type(error).__name__, error)
            print(f"The artifact could not be saved: {error}")
            return
        print(f"Artifact saved at {saved_artifact_path}")


def _dispatch_load(
    user_input: str,
    session: CliSession,
    context: CliCommandContext,
) -> None:
    """Load a saved conversation by its identifier."""
    if not context.feature_config.memory_enabled:
        print("Memory is disabled in config.json.")
        return
    _, *arguments = user_input.split(maxsplit=1)
    if not arguments or not arguments[0].strip():
        print("Usage: /load <session_id>")
        return
    session_id = arguments[0].strip()
    try:
        loaded = context.conversation_session.load(session_id)
    except (FileNotFoundError, ValueError) as error:
        report_cli_error(error, logger)
        return
    system_prefix = (
        [context.messages[0]]
        if context.messages and context.messages[0].get("role") == "system"
        else []
    )
    history = [dict(msg) for msg in loaded if msg.get("role") != "system" or is_summary_message(msg)]
    context.messages[:] = [*system_prefix, *history]
    session.last_request_messages = None
    session.last_rag_context = None

    if context.artifact_manager:
        context.artifact_manager.session_id = session_id
        try:
            context.artifact_manager.load()
            print(f"Artifact for session {session_id} loaded.")
        except FileNotFoundError:
            context.artifact_manager.reset()

    print(f"Session {session_id} loaded.")

def _dispatch_forget(
    user_input: str,
    session: CliSession,
    context: CliCommandContext,
) -> None:
    """Erase conversation data: this session, or every saved session with `/forget all`."""
    _, *arguments = user_input.split()
    if arguments not in ([], ["all"]):
        print("Usage: /forget [all]")
        return
    artifacts = context.artifact_manager
    if arguments:
        if not ask_confirmation("Delete ALL saved sessions and their artifacts? This cannot be undone."):
            print("Nothing deleted.")
            return
        sessions_deleted = context.conversation_session.delete_all()
        artifacts_deleted = artifacts.delete_all() if artifacts else 0
        print(f"Deleted {sessions_deleted} saved session(s) and {artifacts_deleted} saved artifact(s).")
    else:
        deleted = context.conversation_session.delete()
        print("Saved copy of this session deleted." if deleted else "This session had no saved copy.")
        if artifacts:
            # The artifact is saved under the identifier of the conversation it belongs to.
            artifacts.session_id = context.conversation_session.session_id
            if artifacts.delete():
                print("Saved artifact of this session deleted.")
    system_prefix = (
        [context.messages[0]]
        if context.messages and context.messages[0].get("role") == "system"
        else []
    )
    context.messages[:] = system_prefix
    context.conversation_session.clear()
    had_artifact = artifacts is not None and artifacts.has_artifact()
    if artifacts:
        artifacts.reset()
    session.last_request_messages = None
    session.last_rag_context = None
    print(
        "The current conversation and its artifact were cleared from memory."
        if had_artifact
        else "The current conversation was cleared from memory."
    )


def _dispatch_tokens(
    user_input: str,
    session: CliSession,
    context: CliCommandContext,
) -> None:
    """Display token usage for the last turn, active context window, and session."""
    if not context.feature_config.memory_enabled:
        print("Memory is disabled in config.json.")
        return
    if user_input.strip() != "/tokens":
        print("Usage: /tokens")
        return
    context_tokens = session.last_turn_tokens
    window = max(1, session.context_window)
    percentage = (context_tokens / window) * 100
    print("Token usage:")
    print(
        f"- Last turn: {session.last_prompt_tokens} prompt + "
        f"{session.last_completion_tokens} completion = {session.last_turn_tokens} tokens"
    )
    print(f"- Context window: {context_tokens} / {session.context_window} tokens ({percentage:.1f}%)")
    print(
        f"- Session total: {session.total_session_tokens} tokens "
        f"({session.total_prompt_tokens} prompt, {session.total_completion_tokens} completion)"
    )


def _dispatch_compress(
    user_input: str,
    session: CliSession,
    context: CliCommandContext,
) -> None:
    """Condense older conversation history into a factual summary block."""
    if not context.feature_config.memory_enabled:
        print("Memory is disabled in config.json.")
        return
    if user_input.strip() != "/compress":
        print("Usage: /compress")
        return
    if context.chat_client is None:
        print("Error: Model client is unavailable for compression.")
        return

    updated_messages, result = compress_history(context.messages, context.chat_client)
    if not result.compressed:
        print(f"Compression skipped: {result.reason}")
        return

    context.messages[:] = updated_messages
    persisted_messages = [
        dict(msg)
        for msg in updated_messages
        if msg.get("role") != "system" or is_summary_message(msg)
    ]
    context.conversation_session.messages[:] = persisted_messages
    session.last_request_messages = None
    session.last_rag_context = None

    print("Context compressed successfully:")
    print(f"- Messages reduced: {result.messages_before} -> {result.messages_after}")
    print(f"- Summary: {result.summary}")


MEMORY_COMMANDS: tuple[CommandDefinition, ...] = (
    CommandDefinition(
        name="/save",
        usage="/save",
        description="Save the current conversation.",
        feature="Memory",
        handler=_dispatch_save,
    ),
    CommandDefinition(
        name="/load",
        usage="/load <session_id>",
        description="Load a saved conversation by its identifier.",
        feature="Memory",
        handler=_dispatch_load,
    ),
    CommandDefinition(
        name="/forget",
        usage="/forget [all]",
        description="Delete this session's saved copy and clear it, or every saved session with 'all'.",
        feature="Memory",
        handler=_dispatch_forget,
    ),
    CommandDefinition(
        name="/tokens",
        usage="/tokens",
        description="Display per-turn, context window, and cumulative session token usage.",
        feature="Memory",
        handler=_dispatch_tokens,
    ),
    CommandDefinition(
        name="/compress",
        usage="/compress",
        description="Condense older conversation turns into a compact summary block.",
        feature="Memory",
        handler=_dispatch_compress,
    ),
)
