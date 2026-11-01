"""Runtime change of generation settings, to observe their effect without restarting."""

from __future__ import annotations

import math

from cli.context_budget import prompt_budget
from cli.session import CliCommandContext, CliSession, CommandDefinition

_USAGE = "Usage: /set temperature <0-2> | /set context <tokens, at least 512> | /set"


def _dispatch_set(
    user_input: str,
    session: CliSession,
    context: CliCommandContext,
) -> None:
    """Change temperature or context window for the next requests of this session.

    config.json stays the source of the defaults; /set only lasts until SensAI stops,
    which keeps a live comparison from silently changing the teacher's setup.
    """
    client = context.chat_client
    parts = user_input.split()
    if client is None:
        print("No chat model is configured in this session.")
        return
    if len(parts) == 1:
        options = dict(getattr(client, "options", {}) or {})
        print(
            f"Current settings: model={client.model}, temperature={options.get('temperature')}, "
            f"context window={session.context_window} tokens (prompt budget {prompt_budget(session.context_window)})."
        )
        print(_USAGE)
        return
    if len(parts) != 3:
        print(_USAGE)
        return

    name, raw_value = parts[1].lower(), parts[2]
    if name == "temperature":
        try:
            value = float(raw_value)
        except ValueError:
            value = math.nan
        if not math.isfinite(value) or not 0.0 <= value <= 2.0:
            print("The temperature must be a number between 0 and 2.")
            return
        client.update_options({"temperature": value})
        print(
            f"Temperature set to {value:g} for this session. Low values give stable, repeatable answers "
            "(quizzes, facts); higher values give more varied ideas."
        )
    elif name in ("context", "num_ctx"):
        if not raw_value.isdigit() or int(raw_value) < 512:
            print("The context window must be a whole number of tokens, at least 512.")
            return
        value = int(raw_value)
        client.update_options({"num_ctx": value})
        session.context_window = value
        print(
            f"Context window set to {value} tokens for this session. Ollama reloads the model with the new "
            "size on the next request, and a larger window uses more memory (about 36 KiB per token with qwen2.5:3b)."
        )
    else:
        print(_USAGE)


SETTINGS_COMMANDS: tuple[CommandDefinition, ...] = (
    CommandDefinition(
        name="/set",
        usage="/set [temperature <0-2> | context <tokens>]",
        description="Show or change the temperature and context window for this session (config.json keeps the defaults).",
        feature="Model settings",
        handler=_dispatch_set,
    ),
)
