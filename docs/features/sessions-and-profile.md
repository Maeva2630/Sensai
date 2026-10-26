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
loads data into the session manager during a running session, but currently does **not**
replace the active history sent to the model. Use `--session` at startup to resume a
saved conversation in chat. Because `/load` also changes the session manager's active
identifier, a later save can update the loaded session file; avoid using it when you need
the current chat and saved file to remain independent.

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

Sessions and profiles can contain personal information. Review them before sharing or
committing files; the `sessions` directory is Git-ignored, but do not assume every profile
file is. These mechanisms are separate from the reserved `memory_enabled` setting:
enabling that setting currently displays a notice rather than enabling an additional
long-term memory feature.
