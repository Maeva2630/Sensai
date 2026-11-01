# Local data, privacy, and GDPR-oriented assessment

SensAI is designed to run locally: the default configuration uses a local Ollama server, and RAG documents, conversations, and metrics stay on the computer. This reduces routine sharing with a cloud AI provider, but does not protect against an insecure, shared, unlocked, or compromised device.

Do not use real pupil names, grades, health information, or other sensitive information unless the school has approved the workflow and the device is properly protected.

## What SensAI stores

| Data | Location | Retention and deletion |
| --- | --- | --- |
| Indexed documents, chunks, and embeddings | `data/sensai_rag.sqlite3` | Kept until `/rag reset` or deletion of the database file. `/rag reset` does not delete the original document. |
| Conversation history and generated artefacts | `sessions/` | No automatic expiry. Use `/forget`, `/forget all`, or `scripts/clean_sessions.sh`. |
| Teacher profile | `profile.json`, or the path passed to `--profile` | Kept until manually edited or deleted. Keep only teaching preferences, not pupil data. |
| Logs | `logs/sensai.log` | No automatic expiry; delete or rotate it manually. Logs can contain technical details or paths. |
| Monitoring metadata | `data/sensai_monitoring.sqlite3` when enabled | Disabled by default. Rows older than `retention_days` are deleted when monitoring starts; default: 30 days. |
| Configuration, prompts, and MCP settings | `config.json`, `prompts/*.txt`, `mcp_servers.json` | Stored until manually changed. Never put credentials or personal data in versioned files. |
| Quiz exports | An allowed directory such as `exports/` | Kept until manually deleted. They may reproduce information from an entered prompt or source document. |

The generated SQLite databases, session directory, and log files are Git-ignored. The
repository's `profile.json` is only a non-sensitive example; a real profile should not be
committed.

## Current protections and limits

- **Local by default:** `features.network_enabled` is `false` in the root configuration.
- **Controlled external calls:** when network tools are enabled, SensAI requests
  confirmation before web or network-MCP calls. This is a warning, not a guarantee: names
  and most educational identifiers cannot be detected reliably.
- **Private local files:** new session files and profiles use `0700`/`0600` permissions on
  systems that support them. They are not encrypted.
- **Safer RAG ingestion:** ZIP archives have size, file-count, nesting, compression-ratio,
  and path-traversal limits. These prevent resource abuse, not confidential-data exposure.
- **Restricted tools:** filesystem tools are limited by `permissions.json`; tool outputs are
  truncated and presented to the model as untrusted data.
- **Privacy-preserving monitoring:** monitoring stores metrics such as latency and token
  estimates, but not raw prompts, responses, RAG excerpts, tool arguments, or profiles.
- **MCP caution:** configured MCP servers run commands on the local computer. SensAI filters
  inherited environment variables, but explicitly configured server environment values are
  passed through. Only use reviewed servers and never commit secrets.

Local storage isolates teachers from one another: a malicious document indexed on one
computer cannot affect another teacher’s separate installation. However, that document
can still mislead the model on the computer where it was indexed.

## Main risks and practical response

| Risk | What to do |
| --- | --- |
| Shared account, stolen device, or malware | Use separate accounts, screen lock, full-disk encryption, updates, and protected backups. Permissions alone are insufficient. |
| Accidental Git commit or shared screenshot/archive | Check `git status`; review attachments; never force-add sessions, logs, databases, real profiles, or course documents. |
| Enabled web search or network MCP | Keep network disabled for pupil-related work. Refuse any request that would send names or identifiers outside the computer. |
| Untrusted document or MCP server | Verify sources before ingestion; inspect source attribution; review MCP code, arguments, permissions, and environment values. |
| Data incident | Preserve evidence, notify the school/DPO, identify the affected data and people, contain the exposure, then remove unnecessary copies. |

If a personal-data breach presents a risk to people’s rights and freedoms, the organisation
responsible for the data decides whether to notify the CNIL. The CNIL states that a
notification should be made promptly and, where possible, within 72 hours after the
organisation becomes aware of the incident. See [CNIL breach guidance](https://www.cnil.fr/fr/violations-de-donnees-personnelles-les-regles-suivre).

## Recommended usage checklist

- Prefer anonymised or fictional examples for tests, demos, and screenshots.
- Keep network access disabled unless an external search is genuinely necessary.
- Delete no-longer-needed RAG data with `/rag reset` and conversations with `/forget all`.
- Delete or rotate logs and review monitoring retention regularly.
- Treat every model answer and retrieved source as material to verify before use with pupils.

Related documentation: [RAG](features/rag.md), [sessions and profile](features/sessions-and-profile.md),
[tools and MCP](features/tools-and-mcp.md), [observability](features/observability.md), and
[configuration](configuration.md).
