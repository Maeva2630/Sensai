"""Developer-only inspection commands that never contact the chat model."""

from cli.prompt_inspection import format_model_request
from cli.session import CliCommandContext, CliSession, CommandDefinition


def _display_prompt(
    session: CliSession,
    *,
    include_system: bool,
) -> None:
    """Print the most recent reconstructed request with the requested visibility."""
    print(
        format_model_request(
            session.last_request_messages or [],
            session.last_rag_context,
            include_system=include_system,
            tools=session.last_request_tools if session.last_request_messages else None,
            options=session.last_request_options,
        )
    )


def _dispatch_debug_prompt(
    _user_input: str,
    session: CliSession,
    _context: CliCommandContext,
) -> None:
    """Inspect a request while protecting the base system prompt by default."""
    _display_prompt(session, include_system=False)


def _dispatch_debug_prompt_with_system(
    _user_input: str,
    session: CliSession,
    _context: CliCommandContext,
) -> None:
    """Inspect a complete request, including the base system prompt and profile."""
    _display_prompt(session, include_system=True)


def _dispatch_stats(
    _user_input: str,
    session: CliSession,
    _context: CliCommandContext,
) -> None:
    """Show the speed Ollama measured for the last answer."""
    print(format_generation_stats(session.last_generation_stats, session.last_request_options))


def format_generation_stats(stats: dict[str, int], options: dict | None = None) -> str:
    """Tokens per second of the last answer, from the durations Ollama reports.

    Generation (decode) speed is limited by memory bandwidth: every new token reads
    all the model weights once. Prompt reading (prefill) processes many tokens at
    once and is much faster.
    """
    if not stats.get("completion_duration_ns"):
        return "No answer with timing information yet: send a message first."
    lines = ["Last answer:"]
    if options:
        lines.append("- Settings: " + ", ".join(f"{name}={value}" for name, value in options.items()))
    completion, completion_ns = stats.get("completion_tokens", 0), stats["completion_duration_ns"]
    lines.append(
        f"- Generation: {completion} tokens in {completion_ns / 1e9:.2f} s = "
        f"{completion / (completion_ns / 1e9):.1f} tokens/s"
    )
    prompt, prompt_ns = stats.get("prompt_tokens", 0), stats.get("prompt_duration_ns", 0)
    if prompt and prompt_ns:
        lines.append(
            f"- Prompt reading: {prompt} tokens in {prompt_ns / 1e9:.2f} s = "
            f"{prompt / (prompt_ns / 1e9):.0f} tokens/s (tokens already in Ollama's cache are not counted)"
        )
    if "load_duration_ns" in stats:
        lines.append(f"- Model load: {stats['load_duration_ns'] / 1e9:.2f} s (close to 0 when the model is already in memory)")
    if "total_duration_ns" in stats:
        lines.append(f"- Total measured by Ollama: {stats['total_duration_ns'] / 1e9:.2f} s")
    return "\n".join(lines)


DEBUG_COMMANDS: tuple[CommandDefinition, ...] = (
    CommandDefinition(
        name="/debug prompt --include-system",
        usage="/debug prompt --include-system",
        description="Display the last complete request, including the system prompt and profile.",
        feature="Debug",
        handler=_dispatch_debug_prompt_with_system,
    ),
    CommandDefinition(
        name="/debug prompt",
        usage="/debug prompt",
        description="Display the last request while hiding the system prompt and profile.",
        feature="Debug",
        handler=_dispatch_debug_prompt,
    ),
    CommandDefinition(
        name="/stats",
        usage="/stats",
        description="Display the speed of the last answer (tokens per second, prompt reading, model load).",
        feature="Debug",
        handler=_dispatch_stats,
    ),
)
