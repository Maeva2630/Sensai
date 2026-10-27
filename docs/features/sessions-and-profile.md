# Sessions and user profile

[Feature index](README.md) · [User guide](../user-guide.md)

## Save and resume a conversation

SensAI keeps the active conversation in memory and records user and assistant messages
in `sessions/<session_id>.json` when you run `/save`, `exit`, or `quit`:

```text
You: Help me prepare a lesson about fractions.
Assistant: ...
You: /save
Session saved at sessions/<session_id>.json
You: exit
```

For example, if the saved file is named
`sessions/2cb94e54-3573-4b0e-973b-428328d3c282.json`, resume it from the shell with:

```bash
python3 source/main.py --session 2cb94e54-3573-4b0e-973b-428328d3c282
```

Replace the example identifier with your own, without `.json`. `/load <session_id>`
replaces the active user and assistant history sent to the model while keeping the active
persona and profile system prompt. Loading also changes the session manager's active
identifier, so a later save updates the loaded session file.

## Profile

`profile.json` is read at startup and a default is created if it is missing. For example:

```json
{
  "user_name": "Camille",
  "preferred_language": "French",
  "domain_expertise": "Mathematics teaching",
  "constraints": ["Use examples appropriate for secondary school students"]
}
```

The profile is added to the system prompt. A different profile can be selected with
`python3 source/main.py --profile path/to/profile.json`. Changes are read when SensAI
starts, not dynamically during a session.

When `features.memory_enabled` is `true`, `/tokens` displays token use for the last turn,
the active context window, and the session total. `/compress` asks the model to summarize
older turns when the conversation becomes long; automatic compression can also run when
context use approaches the configured limit. This is conversation management, not a
long-term learner-memory system that automatically infers facts about a person.

```text
You: /tokens
You: /compress
```

Sessions and profiles can contain personal information. Review them before sharing or
committing files; the `sessions` directory is Git-ignored, but do not assume every profile
file is. Disable `features.memory_enabled` if session persistence and context-management
commands must not be available.
