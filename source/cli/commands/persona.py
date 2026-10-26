"""CLI commands for selecting the active assistant persona."""

from cli.session import CliCommandContext, CliSession, CommandDefinition


def _dispatch_persona_list(
    user_input: str,
    session: CliSession,
    context: CliCommandContext,
) -> None:
    """List configured personas without exposing their prompt content."""
    if user_input.strip() != "/persona list":
        print("Usage: /persona list")
        return
    print("Available personas:")
    for persona in context.personas.profiles:
        active_marker = " (active)" if persona.identifier == session.active_persona_id else ""
        print(f"- {persona.identifier}: {persona.display_name}{active_marker}")


def _dispatch_persona_status(
    user_input: str,
    session: CliSession,
    context: CliCommandContext,
) -> None:
    """Display safe metadata about the active persona."""
    if user_input.strip() != "/persona status":
        print("Usage: /persona status")
        return
    persona = context.personas.find(session.active_persona_id)
    if persona is None:
        print("No active persona is available.")
        return
    print(f"Active persona: {persona.display_name} ({persona.identifier})")
    print(f"Role: {persona.role}")
    print(f"Tone: {persona.tone}")
    print(f"Scope: {persona.response_scope}")


def _dispatch_persona_use(
    user_input: str,
    session: CliSession,
    context: CliCommandContext,
) -> None:
    """Switch persona while retaining prior user and assistant messages."""
    parts = user_input.split()
    if len(parts) != 3:
        print("Usage: /persona use <id>")
        return
    persona = context.personas.find(parts[2])
    if persona is None:
        available = ", ".join(profile.identifier for profile in context.personas.profiles)
        print(f"Unknown persona: {parts[2]}. Available personas: {available}")
        return
    if context.profile_manager is None:
        print("Persona switching is unavailable because the user profile is not loaded.")
        return

    system_message = {
        "role": "system",
        "content": context.profile_manager.build_system_prompt(persona.system_instructions),
    }
    if context.messages and context.messages[0].get("role") == "system":
        context.messages[0] = system_message
    else:
        context.messages.insert(0, system_message)
    session.active_persona_id = persona.identifier
    session.last_request_messages = None
    session.last_rag_context = None
    print(f"Active persona changed to: {persona.display_name} ({persona.identifier})")


def _dispatch_persona(
    _user_input: str,
    _session: CliSession,
    _context: CliCommandContext,
) -> None:
    """Display usage for incomplete persona commands."""
    print("Usage: /persona <list|status|use <id>>")


# Specific commands must be registered before the generic /persona command.
PERSONA_COMMANDS: tuple[CommandDefinition, ...] = (
    CommandDefinition(
        name="/persona list",
        usage="/persona list",
        description="List configured assistant personas.",
        feature="Persona",
        handler=_dispatch_persona_list,
    ),
    CommandDefinition(
        name="/persona status",
        usage="/persona status",
        description="Display the active persona.",
        feature="Persona",
        handler=_dispatch_persona_status,
    ),
    CommandDefinition(
        name="/persona use",
        usage="/persona use <id>",
        description="Switch persona without clearing the current chat history.",
        feature="Persona",
        handler=_dispatch_persona_use,
    ),
    CommandDefinition(
        name="/persona",
        usage="/persona <list|status|use <id>>",
        description="Manage the active assistant persona.",
        feature="Persona",
        handler=_dispatch_persona,
    ),
)
