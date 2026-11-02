# Sessions and user profile

[Feature index](README.md) · [User guide](../user-guide.md) · [Context engineering](../context-engineering.md) · [GDPR](../RGPD.md)

## What is persisted, exactly

| Item | Stored where | Content | Written when |
| --- | --- | --- | --- |
| Conversation | `sessions/<session_id>.json` | `session_id`, `timestamp`, and the list of user and assistant messages (text only). The persona, the profile, retrieved excerpts and tool calls are **not** stored: they are rebuilt for each request. A conversation summary replaces the older turns after compression. | `/save`, `exit`, `quit`, Ctrl+D, Ctrl+C (CLI); after every answer (web page). An empty conversation is never written. |
| Shared document (artifact) | `sessions/<session_id>_artifact.md` and `.json` | The document edited with `/artifact` or `/doc` | Same moments, when an artifact exists |
| Profile | `profile.json` or the file given to `--profile` | `user_name`, `preferred_language`, `domain_expertise`, `constraints` | Created with neutral values if missing; edited by hand or from the web page |
| Active persona | Not persisted | - | Each start uses `personas.active_persona` from `config.json`; `/persona use` lasts for the session |

All these files are owner-only (`0600`), see [GDPR](../RGPD.md).

## Save and resume a conversation

```text
You: Prépare une explication de deux phrases sur les fractions équivalentes pour une classe de sixième.
Assistant: Deux fractions sont équivalentes quand elles représentent la même part ...
You: /save
Session saved at sessions/2cb94e54-3573-4b0e-973b-428328d3c282.json
You: exit
Session saved at sessions/2cb94e54-3573-4b0e-973b-428328d3c282.json
```

Resume from the shell with the identifier, without `.json`:

```bash
python3 source/main.py --session 2cb94e54-3573-4b0e-973b-428328d3c282 --profile demo/profile_camille.json
```

```text
You: Peux-tu transformer l'explication précédente en une question de vérification ?
Assistant: Question : 2/3 et 4/6 sont-elles équivalentes ? Justifie ...
```

The restored user and assistant messages are sent to the model again after the current
persona and profile, so "the previous explanation" is understood. `/load <session_id>`
does the same during a conversation and makes later saves update that file. An unknown
or invalid identifier gives a one-line error (identifiers may only contain letters,
digits, `-` and `_`); a corrupted session file is reported, never a traceback.

**Demonstration-safe workflow.** Start from a clean state (`scripts/clean_sessions.sh`),
run `/save` and copy the identifier, then type the restart command in a second terminal
before the presentation. Do not reuse a session from rehearsal: the model can answer from
the old history instead of calling a tool again.

## Forget

```text
You: /forget
Saved copy of this session deleted.
You: /forget all
Delete ALL saved sessions? This cannot be undone. [y/N] y
Deleted 3 saved session(s) and 1 saved artifact(s).
```

`/forget` deletes the current conversation file and its artifact files; leaving SensAI
afterwards does not write them again unless new messages are added. `/forget all` deletes
every conversation and artifact after a confirmation. From a shell:
`scripts/clean_sessions.sh` (macOS and Linux).

## Profile

`profile.json` is read at startup and a default is created if it is missing. Camille's
demonstration profile:

```json
{
  "user_name": "Camille",
  "preferred_language": "French",
  "domain_expertise": "Middle-school mathematics teaching",
  "constraints": [
    "Use examples appropriate for sixth-grade pupils",
    "Prefer concise classroom-ready activities"
  ]
}
```

The profile is appended to the persona in the first system message (387 tokens with the
teacher-assistant persona, measured). It describes the teacher, not the pupils: keep pupil
information out of it. `preferred_language` also sets the language of SensAI's own
"not found" answer in RAG mode. Select another file with
`python3 source/main.py --profile path/to/profile.json`; keep a real profile outside the
repository so it is never committed.

## Context window management

When `features.memory_enabled` is `true`:

```text
You: /tokens
Token usage:
- Last turn: 768 prompt + 63 completion = 831 tokens
- Context window: 831 / 4096 tokens (20.3%)
- Session total: 831 tokens (768 prompt, 63 completion)
```

- Counts come from Ollama (`prompt_eval_count`, `eval_count`).
- Every request is checked **before** it is sent: if it would not fit in the window minus a
  quarter kept for the answer, older turns are summarised first, and a message that is too
  long on its own is refused with an explanation (Ollama would otherwise cut it silently).
- After a turn, a warning appears at 80 % of the window and automatic compression runs at
  85 %; `/compress` runs it on demand. The summary keeps names, class, dates, numbers,
  decisions and open tasks (measured on qwen2.5:3b: facts kept went from 21/50 to 48/50
  after the summary prompt was rewritten).

Details: [context engineering](../context-engineering.md#4-size-management).

## What this is not

`memory_enabled` covers conversation persistence and context management. It is not a
long-term learner memory: SensAI does not infer or store facts about pupils across
conversations. Disable `features.memory_enabled` if persistence and context-management
commands must not be available.
