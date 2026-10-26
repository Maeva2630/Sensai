"""CLI help command generated from the registered command definitions."""

from collections import defaultdict
from difflib import get_close_matches

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


def display_unknown_command(
    user_input: str,
    command_definitions: tuple[CommandDefinition, ...],
) -> None:
    """Explain an unrecognised slash command without sending it to the model."""
    entered_command = user_input.split(maxsplit=1)[0]
    command_names = [command.name for command in command_definitions]
    suggestion = get_close_matches(entered_command, command_names, n=1, cutoff=0.6)

    print(f"Unknown command: {entered_command}")
    if suggestion:
        print(f"Did you mean: {suggestion[0]}?")
    print("Type /help to list available commands.")


HELP_COMMANDS: tuple[CommandDefinition, ...] = (
    CommandDefinition(
        name="/help",
        usage="/help",
        description="Display all available CLI commands.",
        feature="CLI",
        handler=_dispatch_help,
    ),
)
