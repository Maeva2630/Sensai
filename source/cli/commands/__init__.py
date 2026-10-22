"""Feature-specific command registries combined for the interactive CLI."""

from cli.commands.memory import MEMORY_COMMAND_HANDLERS
from cli.commands.rag import RAG_COMMAND_HANDLERS
from cli.commands.tools import TOOL_COMMAND_HANDLERS


# The CLI dispatches every feature registry through this single ordered table.
GLOBAL_COMMAND_HANDLERS = (
    *RAG_COMMAND_HANDLERS,
    *TOOL_COMMAND_HANDLERS,
    *MEMORY_COMMAND_HANDLERS,
)
