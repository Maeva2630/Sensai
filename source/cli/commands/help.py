"""CLI help command generated from the registered command definitions."""

from collections import defaultdict

from cli.session import CliCommandContext, CliSession, CommandDefinition


def _dispatch_help(
    user_input: str,
    _session: CliSession,
    context: CliCommandContext,
) -> None:
    """Display all commands grouped by their owning feature."""
    if user_input.strip() != "/help":
        print("Usage: /help")
        return

    commands_by_feature: dict[str, list[CommandDefinition]] = defaultdict(list)
    for command in context.command_definitions:
        commands_by_feature[command.feature].append(command)

    print("Available commands:")
    for feature, commands in commands_by_feature.items():
        print(f"\n{feature}:")
        for command in commands:
            print(f"  {command.usage}")
            print(f"    {command.description}")


HELP_COMMANDS: tuple[CommandDefinition, ...] = (
    CommandDefinition(
        name="/help",
        usage="/help",
        description="Display all available CLI commands.",
        feature="CLI",
        handler=_dispatch_help,
    ),
)
