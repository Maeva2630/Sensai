"""General CLI commands that are not owned by an application feature."""

from cli.session import CliCommandContext, CliSession, CommandDefinition
from observability.app_logging import get_logger


logger = get_logger("commands.system")


def _dispatch_exit(
    _user_input: str,
    session: CliSession,
    context: CliCommandContext,
) -> None:
    """Save the current conversation, then ask the interactive loop to stop."""
    if context.feature_config.memory_enabled:
        context.conversation_session.save()
        session.persistence_saved_on_exit = True
        if context.artifact_manager and context.artifact_manager.has_artifact():
            try:
                context.artifact_manager.session_id = context.conversation_session.session_id
                context.artifact_manager.save()
            except (OSError, ValueError) as error:
                # Leaving must still work: the failure is reported instead of hidden.
                logger.warning("artifact.save_failed_on_exit error_type=%s error=%s", type(error).__name__, error)
                print(f"Warning: the artifact could not be saved: {error}")
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
