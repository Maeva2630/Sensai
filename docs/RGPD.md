# Personal data (RGPD)

SensAI is used by teachers, so conversations can contain student names, grades and comments. This page says what is stored, what can leave the machine, and what to do after a leak. It is a working summary for the project, not legal advice.

## What is stored and where

| Data | Where | Protection |
|---|---|---|
| Conversations | `sessions/<id>.json` | Owner-only permissions (folder 700, files 600). Session ids are validated, so a file can never be written outside `sessions/`. |
| Profile (name, language, constraints) | `profile.json` | Owner-only permissions (600). |
| Course documents and their embeddings | `data/` (local vector store) | Local only. |
| Usage metrics | `data/sensai_monitoring.sqlite3` | Off by default. Stores no prompt or answer, only counts and latencies. Purged after `retention_days` (30). |
| Logs | `logs/sensai.log` | Local file, never printed in the terminal. |

Everything runs on the user's machine with a local Ollama model. No account, no cloud service.

## What can leave the machine

Only network features: `web_search` and the `web` MCP server (`fetch`).

- Off by default (`features.network_enabled`: `false`).
- Each call shows the exact text or URL and needs a yes before it is sent.
- The confirmation warns when the text looks like an e-mail, phone number, IBAN or long identifier. Names cannot be detected reliably, so the question itself is the protection: do not accept a search that contains a student's name.
- MCP servers get a filtered environment: API keys and tokens of the user's shell are not passed on.
- Web results and pages are treated as data, never as instructions, to reduce prompt injection that tries to push private data into a query.

## Rights and retention

- **Erasure:** `/forget` deletes this session's saved copy and clears it from memory. `/forget all` deletes every saved session (asks first).
- **Access:** conversations are plain JSON files the user can open, copy or print.
- **Retention:** SensAI does not delete sessions by itself. The user decides with `/forget` or `clean_sessions.sh`. Metrics are purged automatically.
- **Minimisation:** the profile holds a few preferences. Metrics hold no content.

## If a leak happens

1. **Contain:** stop using the machine or account concerned, revoke shared access, keep the evidence (logs, dates).
2. **Assess:** which data (grades, names, minors?), how many people, was it readable (files are not encrypted, only protected by permissions).
3. **Record:** write it in the breach register (what, when, who, effect, measures), even if no notification follows.
4. **Notify:** the data protection officer of the school or organisation, which decides on notifying the CNIL within 72 hours when people are at risk, and on informing the students or their guardians when the risk is high.
5. **Fix:** change what failed, then check `sessions/` and `profile.json` permissions and delete what is no longer needed with `/forget all`.

## Known limits

- Files are protected by permissions, not encrypted. Full-disk encryption of the machine is recommended.
- Sessions saved before this protection keep their old permissions until they are saved again.
- Personal data detection only catches obvious formats. It does not detect names.
- DuckDuckGo receives the query when the user accepts a search.
