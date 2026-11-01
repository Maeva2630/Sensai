# Web interface

[Home](../../README.md) · [Features](README.md) · [User guide](../user-guide.md)

Teachers are not expected to use a terminal, so SensAI also runs as a local web page. It is the
same assistant (same models, documents, tools, memory and commands): the page only replaces the
terminal.

## Start it

```bash
python3 source/main.py --web                       # opens http://localhost:8000
python3 source/main.py --web --port 9000 --session <session_id> --profile profile.json
```

Ollama must be running, as for the terminal. A banner at the top of the page says so when Ollama
or every configured chat model is unavailable. Stop the server with `Ctrl+C`.

## What the page offers

| Sidebar section | What it does | Terminal equivalent |
| --- | --- | --- |
| My course documents | Upload PDF, Word, text, Markdown or ZIP files; switch "Answer from my documents" and source display on or off | `/ingest`, `/rag`, `/sources` |
| Create a quiz | Topic, number of questions, language; download with answers, for students, or as JSON | `/quiz` |
| Assistant style | Choose a persona | `/persona use` |
| My profile | Name, language, subject, constraints, applied immediately | `profile.json` |
| Conversations | New, resume, delete one or all saved conversations | `/save`, `/load`, `/forget` |
| Working document | Title and sections of the living document the assistant drafts, with a Markdown download | `/artifact` |
| Memory | Context usage bar and a summarise button | `/tokens`, `/compress` |
| What the assistant can do | The tools available, native and MCP | `/tools` |

Any slash command can also be typed in the message box and its output is shown in the chat.

- **Streaming:** answers appear progressively, as in the terminal.
- **Web confirmation:** when the model wants to search the web, the page shows the exact text or
  URL with **Allow** and **Cancel**, like the `[y/N]` question in the terminal. No answer within
  two minutes counts as a refusal, so nothing leaves the machine by default.
- **Model fallback:** if the primary model is missing and a fallback answers, a notice says so.
- **Saving:** the conversation (and the working document) is saved after every answer. The
  terminal saves on exit only.
- **`/exit` and `/quit`:** they do nothing in the page, which stays open.

## How it is built

- `source/web/server.py`: a standard-library `http.server` (no new dependency, nothing to
  install). It only translates HTTP requests into calls on `WebApp`.
- `source/web/app.py`: `WebApp` holds the same services as the terminal chat. Slash commands run
  through the existing command handlers and their printed output is captured, so there is no
  second implementation of any feature. A chat turn mirrors the loop in `cli/cli.py`.
- `source/web/static/index.html`: one page of HTML, CSS and JavaScript, with no framework and no
  build step. It follows the system light or dark theme.
- Answers are sent as a stream of JSON lines (`application/x-ndjson`). A tool confirmation is an
  event in that stream, answered by a separate `POST /api/confirm`.

Small changes in the terminal code made this possible without duplicating it: `stream_response`
accepts an `on_token` callback, `run_tool_calls` an `emit` function, and `format_cli_error`
returns the safe message instead of printing it.

## Security and privacy

- The server listens on `127.0.0.1` only. Never expose it on a network: there is **no login**,
  it serves one local user.
- The `Host` header must be this machine (against DNS rebinding), and a `POST` must be JSON or
  raw bytes. A page from another site open in the same browser cannot send either without a CORS
  preflight, which the server refuses.
- Text is displayed with `textContent`, never as HTML.
- Uploaded files are written to `data/uploads/`, indexed, then deleted right away. File names are
  reduced to safe characters, size is limited to 25 MB, and the existing archive protections apply
  to ZIP files.
- The privacy rules in [RGPD](../RGPD.md) are unchanged: conversations are stored locally with
  owner-only permissions and web access always asks first.

## Limits

- One conversation is active at a time and it is shared by every tab. Requests are handled one
  after the other.
- The interface is in English and shows answers as plain text (no Markdown rendering).
- The saved-conversation list shows the 30 most recent conversations.
