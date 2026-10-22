"""Developer-only inspection commands that never contact the chat model."""

from cli.prompt_inspection import format_model_request
from cli.session import CliCommandContext, CliSession, CommandDefinition


def _display_prompt(
    session: CliSession,
    *,
    include_system: bool,
) -> None:
    """Print the most recent reconstructed request with the requested visibility."""
    print(
        format_model_request(
            session.last_request_messages or [],
            session.last_rag_context,
            include_system=include_system,
        )
    )


def _dispatch_debug_prompt(
    _user_input: str,
    session: CliSession,
    _context: CliCommandContext,
) -> None:
    """Inspect a request while protecting the base system prompt by default."""
    _display_prompt(session, include_system=False)


def _dispatch_debug_prompt_with_system(
    _user_input: str,
    session: CliSession,
    _context: CliCommandContext,
) -> None:
    """Inspect a complete request, including the base system prompt and profile."""
    _display_prompt(session, include_system=True)


DEBUG_COMMANDS: tuple[CommandDefinition, ...] = (
    CommandDefinition(
        name="/debug prompt --include-system",
        usage="/debug prompt --include-system",
        description="Display the last complete request, including the system prompt and profile.",
        feature="Debug",
        handler=_dispatch_debug_prompt_with_system,
    ),
    CommandDefinition(
        name="/debug prompt",
        usage="/debug prompt",
        description="Display the last request while hiding the system prompt and profile.",
        feature="Debug",
        handler=_dispatch_debug_prompt,
    ),
)
