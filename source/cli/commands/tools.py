"""CLI commands owned by the Tools feature."""

from cli.session import CommandDefinition


# Reserved for future commands such as `/tools status`.
TOOL_COMMANDS: tuple[CommandDefinition, ...] = ()
