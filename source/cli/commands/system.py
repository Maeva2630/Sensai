"""General CLI commands that are not owned by an application feature."""

from cli.session import CliCommandContext, CliSession, CommandDefinition


def _dispatch_exit(
    _user_input: str,
    session: CliSession,
    context: CliCommandContext,
) -> None:
    """Save the current conversation, then ask the interactive loop to stop."""
    if context.feature_config.memory_enabled:
        context.conversation_session.save()
        if context.artifact_manager and context.artifact_manager.has_artifact():
            try:
                context.artifact_manager.session_id = context.conversation_session.session_id
                context.artifact_manager.save()
            except Exception:
                pass
    session.exit_requested = True


SYSTEM_COMMANDS: tuple[CommandDefinition, ...] = (
    CommandDefinition(
        name="exit",
        usage="exit",
        description="Save the current conversation and leave the CLI.",
        feature="CLI",
        handler=_dispatch_exit,
    ),
    CommandDefinition(
        name="/exit",
        usage="/exit",
        description="Save the current conversation and leave the CLI.",
        feature="CLI",
        handler=_dispatch_exit,
    ),
    CommandDefinition(
        name="quit",
        usage="quit",
        description="Save the current conversation and leave the CLI.",
        feature="CLI",
        handler=_dispatch_exit,
    ),
    CommandDefinition(
        name="/quit",
        usage="/quit",
        description="Save the current conversation and leave the CLI.",
        feature="CLI",
        handler=_dispatch_exit,
    ),
)
