"""CLI commands owned by the conversation-memory feature."""

from cli.error_handling import report_cli_error
from cli.session import CliCommandContext, CliSession, CommandDefinition
from observability.app_logging import get_logger


logger = get_logger("commands.memory")


def _dispatch_save(
    _user_input: str,
    _session: CliSession,
    context: CliCommandContext,
) -> None:
    """Persist the current conversation without leaving the CLI."""
    if not context.feature_config.memory_enabled:
        print("Memory is disabled in config.json.")
        return
    context.conversation_session.save()


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
    except FileNotFoundError as error:
        report_cli_error(error, logger)
        return
    system_prefix = (
        [context.messages[0]]
        if context.messages and context.messages[0].get("role") == "system"
        else []
    )
    history = [dict(msg) for msg in loaded if msg.get("role") != "system"]
    context.messages[:] = [*system_prefix, *history]
    session.last_request_messages = None
    session.last_rag_context = None
    print(f"Session {session_id} loaded.")

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
        name="/tokens",
        usage="/tokens",
        description="Display per-turn, context window, and cumulative session token usage.",
        feature="Memory",
        handler=_dispatch_tokens,
    ),
)
