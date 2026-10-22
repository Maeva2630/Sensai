"""Developer-only inspection commands that never contact the chat model."""

from cli.prompt_inspection import format_model_request
from cli.session import CliCommandContext, CliSession, CommandDefinition


def _dispatch_debug_prompt(
    user_input: str,
    session: CliSession,
    _context: CliCommandContext,
) -> None:
    """Print the most recent reconstructed request on explicit developer demand."""
    if user_input.strip() != "/debug prompt":
        print("Usage: /debug prompt")
        return
    print(format_model_request(session.last_request_messages or [], session.last_rag_context))


DEBUG_COMMANDS: tuple[CommandDefinition, ...] = (
    CommandDefinition(
        name="/debug prompt",
        usage="/debug prompt",
        description="Display the last complete request sent to the model.",
        feature="Debug",
        handler=_dispatch_debug_prompt,
    ),
)
