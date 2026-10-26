# Personas

[Feature index](README.md) · [User guide](../user-guide.md)

A persona defines the assistant's name, role, tone, response scope, and system prompt.
The default `teacher-assistant` supports educators; it is not intended to replace a
teacher. `mathematics-assistant` is also configured in the repository.

## Switch during a session

```text
You: /persona list
Available personas:
- teacher-assistant: Teacher assistant (active)
- mathematics-assistant: Mathematics assistant

You: /persona status
Active persona: Teacher assistant (teacher-assistant)
Role: Educational assistant for teachers and educators
Tone: Clear, practical, supportive, and pedagogical
Scope: Support educators with lesson preparation, teaching resources, and assessment.

You: /persona use mathematics-assistant
Active persona changed to: Mathematics assistant (mathematics-assistant)

You: Suggest a quadratic-equation exercise for my students.
```

Switching replaces the base system message for subsequent requests and keeps the
existing conversation history. The runtime selection is not persisted across restarts:
the configured `personas.active_persona` is used for a new session.

## Add a persona

Add an object to the `personas.profiles` array in `config.json`:

```json
{
  "identifier": "history-assistant",
  "display_name": "History assistant",
  "role": "Assistant for history teachers",
  "tone": "Clear and pedagogical",
  "system_prompt": "Help teachers prepare history lessons and acknowledge uncertainty.",
  "response_scope": "History teaching resources and activities."
}
```

Restart SensAI, then use `/persona use history-assistant`. To make this the default,
also set `personas.active_persona` to `history-assistant`.

Each persona must specify exactly one prompt source: `system_prompt` for inline text,
or `prompt_file` for a UTF-8 file relative to the configuration directory, for example
`"prompt_file": "prompts/teacher-assistant.txt"`. The prompt file must remain inside
that directory. Persona instructions guide the model but do not guarantee that it will
follow them or produce correct information. RAG and tool instructions remain separate.
