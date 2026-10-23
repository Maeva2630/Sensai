"""CLI commands owned by the conversation-memory feature."""

from cli.session import CliCommandContext, CliSession, CommandDefinition


def _dispatch_save(
    _user_input: str,
    _session: CliSession,
    context: CliCommandContext,
) -> None:
    """Persist the current conversation without leaving the CLI."""
    context.conversation_session.save()


def _dispatch_load(
    user_input: str,
    session: CliSession,
    context: CliCommandContext,
) -> None:
    """Load a saved conversation by its identifier."""
    _, *arguments = user_input.split(maxsplit=1)
    if not arguments or not arguments[0].strip():
        print("Usage: /load <session_id>")
        return
    session_id = arguments[0].strip()
    try:
        loaded = context.conversation_session.load(session_id)
    except FileNotFoundError as error:
        print(f"Error: {error}")
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
)
