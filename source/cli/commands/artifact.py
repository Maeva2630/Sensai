"""CLI commands for managing the living artifact deliverable."""

import re
from pathlib import Path

from cli.session import CliCommandContext, CliSession, CommandDefinition
from tools.base import ToolError

_TOKEN = re.compile(r'"([^"]*)"|(\S+)')


def _parse_tokens(user_input: str) -> list[str]:
    """Parse user input into tokens, respecting double quotes."""
    return [quoted if quoted or not word else word for quoted, word in _TOKEN.findall(user_input)]


def _dispatch_artifact(
    user_input: str,
    _session: CliSession,
    context: CliCommandContext,
) -> None:
    """Display the current living artifact or its subcommands."""
    manager = context.artifact_manager
    if manager is None:
        print("Artifact management is not available.")
        return

    if user_input.count('"') % 2:
        print("Invalid command: a double quote is not closed.")
        return

    tokens = _parse_tokens(user_input)
    subcommand = tokens[1] if len(tokens) > 1 else "view"

    if subcommand in ("view", "show"):
        if not manager.has_artifact():
            print("No active artifact. You can ask SensAI to draft a document or use '/artifact init <title>'.")
            return
        print("\n" + "=" * 50)
        print(manager.get_content().rstrip())
        print("=" * 50 + "\n")
        return

    if subcommand == "outline":
        if not manager.has_artifact():
            print("No active artifact.")
            return
        outline = manager.get_outline()
        print(f"\nArtifact: '{manager.artifact.title}' (v{manager.artifact.version})")
        if not outline:
            print("  (no sections yet)")
        for idx, sec in enumerate(outline, start=1):
            print(f"  {idx}. {sec}")
        print()
        return

    if subcommand == "reset":
        manager.reset()
        print("Active artifact reset.")
        return

    if subcommand == "export":
        args = tokens[2:]
        overwrite = False
        filepath_tokens = []
        for arg in args:
            if arg == "--overwrite":
                overwrite = True
            else:
                filepath_tokens.append(arg)

        if len(filepath_tokens) != 1:
            print("Usage: /artifact export <filepath> [--overwrite]")
            return

        if not manager.has_artifact():
            print("Cannot export: No active artifact.")
            return

        target_str = filepath_tokens[0]
        checker = context.permission_checker
        if checker is None:
            print("Cannot export: Permission checker is not available.")
            return

        try:
            dest_path = checker.require(target_str, "write")
        except ToolError as e:
            print(f"Cannot export: {e}")
            return

        if dest_path.is_dir():
            print(f"Cannot export: '{target_str}' is a directory.")
            return

        if dest_path.exists() and not overwrite:
            print(f"Cannot export: File '{target_str}' already exists. Use --overwrite to replace it.")
            return

        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_text(manager.get_content(), encoding="utf-8")
            print(f"Artifact exported successfully to {dest_path}")
        except OSError as e:
            print(f"Failed to export artifact: {e}")
        return

    if subcommand == "init":
        title = " ".join(tokens[2:]).strip() or "Working Document"
        manager.init_document(title=title)
        print(f"Artifact initialized with title: '{title}'")
        return

    print("Unknown artifact subcommand. Usage: /artifact [view | outline | export <path> [--overwrite] | reset | init <title>]")


ARTIFACT_COMMANDS: tuple[CommandDefinition, ...] = (
    CommandDefinition(
        name="/artifact",
        usage="/artifact [view | outline | export <path> [--overwrite] | reset | init <title>]",
        description="Inspect, outline, export, or reset the living artifact document.",
        feature="Artifact",
        handler=_dispatch_artifact,
    ),
    CommandDefinition(
        name="/doc",
        usage="/doc [view | outline | export <path> [--overwrite] | reset | init <title>]",
        description="Alias for /artifact.",
        feature="Artifact",
        handler=_dispatch_artifact,
    ),
)
