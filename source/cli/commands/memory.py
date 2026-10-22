"""CLI commands owned by the Memory feature."""

from cli.session import CommandHandler


# Reserved for future commands such as `/memory clear`.
MEMORY_COMMAND_HANDLERS: tuple[tuple[str, CommandHandler], ...] = ()
