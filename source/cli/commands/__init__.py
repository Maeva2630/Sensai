"""Feature-specific command registries combined for the interactive CLI."""

from cli.commands.artifact import ARTIFACT_COMMANDS
from cli.commands.debug import DEBUG_COMMANDS
from cli.commands.help import HELP_COMMANDS
from cli.commands.memory import MEMORY_COMMANDS
from cli.commands.metrics import METRICS_COMMANDS
from cli.commands.persona import PERSONA_COMMANDS
from cli.commands.rag import RAG_COMMANDS
from cli.commands.system import SYSTEM_COMMANDS
from cli.commands.tools import TOOL_COMMANDS
from cli.session import CommandDefinition


# The CLI dispatches every feature registry through this single ordered registry.
GLOBAL_COMMANDS: tuple[CommandDefinition, ...] = (
    *HELP_COMMANDS,
    *DEBUG_COMMANDS,
    *METRICS_COMMANDS,
    *SYSTEM_COMMANDS,
    *PERSONA_COMMANDS,
    *RAG_COMMANDS,
    *TOOL_COMMANDS,
    *MEMORY_COMMANDS,
    *ARTIFACT_COMMANDS,
)
