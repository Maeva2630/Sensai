"""CLI commands owned by the Tools feature."""

from cli.session import CliCommandContext, CliSession, CommandDefinition
from tools.registry import format_tool_listing


def _dispatch_tools(
    _user_input: str,
    _session: CliSession,
    context: CliCommandContext,
) -> None:
    """List the tools available to the model, native and MCP."""
    tools = context.registry.list_tools() if context.registry is not None else []
    print(format_tool_listing(tools))


TOOL_COMMANDS: tuple[CommandDefinition, ...] = (
    CommandDefinition(
        name="/tools",
        usage="/tools",
        description="List the tools available to the model.",
        feature="Tools",
        handler=_dispatch_tools,
    ),
)
