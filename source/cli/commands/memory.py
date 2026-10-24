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
    context.conversation_session.save()


def _dispatch_load(
    user_input: str,
    _session: CliSession,
    context: CliCommandContext,
) -> None:
    """Load a saved conversation by its identifier."""
    _, *arguments = user_input.split(maxsplit=1)
    if not arguments or not arguments[0].strip():
        print("Usage: /load <session_id>")
        return
    try:
        context.conversation_session.load(arguments[0].strip())
    except FileNotFoundError as error:
        report_cli_error(error, logger)
        return
    print(f"Session {arguments[0].strip()} loaded.")


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
