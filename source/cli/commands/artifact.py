"""CLI commands for managing the living artifact deliverable."""

from pathlib import Path

from cli.session import CliCommandContext, CliSession, CommandDefinition


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

    tokens = user_input.strip().split()
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
        if len(tokens) < 3:
            print("Usage: /artifact export <filepath>")
            return
        if not manager.has_artifact():
            print("Cannot export: No active artifact.")
            return
        dest_path = Path(tokens[2])
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

    print("Unknown artifact subcommand. Usage: /artifact [view | outline | export <path> | reset | init <title>]")


ARTIFACT_COMMANDS: tuple[CommandDefinition, ...] = (
    CommandDefinition(
        name="/artifact",
        usage="/artifact [view | outline | export <path> | reset | init <title>]",
        description="Inspect, outline, export, or reset the living artifact document.",
        feature="Artifact",
        handler=_dispatch_artifact,
    ),
    CommandDefinition(
        name="/doc",
        usage="/doc [view | outline | export <path> | reset | init <title>]",
        description="Alias for /artifact.",
        feature="Artifact",
        handler=_dispatch_artifact,
    ),
)
