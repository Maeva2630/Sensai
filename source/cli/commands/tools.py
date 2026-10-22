"""CLI commands owned by the Tools feature."""

from cli.session import CommandHandler


# Reserved for future commands such as `/tools status`.
TOOL_COMMAND_HANDLERS: tuple[tuple[str, CommandHandler], ...] = ()
