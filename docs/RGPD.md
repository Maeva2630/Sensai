# Local data, privacy, and GDPR-oriented assessment

[Home](../README.md) · [Security and resilience](security.md) · [Cost-benefit analysis](cost-benefit.md) · [Configuration](configuration.md)

SensAI is designed to run locally: the default configuration uses a local Ollama server, and
RAG documents, conversations, and metrics stay on the computer. This reduces routine sharing
with a cloud AI provider, but does not protect against an insecure, shared, unlocked, or
compromised device.

This document is an engineering assessment, not legal advice. In a real deployment the
school's data protection officer (DPO) validates the processing.

Do not use real pupil names, grades, health information, or other sensitive information
unless the school has approved the workflow and the device is properly protected. The demo
and test data in `courses/` and `demo/` are fictional.

## Who is responsible, and on which basis

| Question | Answer for a teacher using SensAI at school |
| --- | --- |
| Data controller | The school (for a French public *collège*, the head of the establishment), not the teacher personally and not the SensAI team. |
| Processor | None by default: SensAI runs on the teacher's computer and sends nothing to a third party. With `network_enabled: true`, the web search provider receives each approved query. |
| Purpose | Preparing lessons, activities, assessments and feedback. Not profiling or grading pupils automatically. |
| Legal basis | Public-interest task of the school (GDPR art. 6(1)(e)), within the teacher's mission. |
| Data subjects | Teachers (profile, conversations) and, if the teacher types them, pupils, who are **minors**. |
| Special categories | A support plan (PAP, PPS), a disability or a "dys" disorder is **health data** (art. 9). It should not be typed in SensAI; if a school allows it, a data protection impact assessment (DPIA, art. 35) is expected because it concerns vulnerable people. SensAI flags these words before any network call (see below). |
| Minimisation | The profile holds teaching preferences only. Quizzes, plans and questions do not need pupil names: use first names or "élève A" when an example is needed. |

## What SensAI stores, and how to delete it

| Data | Location | Permissions | Retention and deletion |
| --- | --- | --- | --- |
| Indexed documents (full text), chunks, embeddings | `data/sensai_rag.sqlite3` | Owner only | Kept until `/rag reset`, which deletes every chunk and runs `VACUUM`, so the deleted text cannot be read back from the file. The original documents are not touched. |
| Conversations | `sessions/<id>.json` | Owner only | No automatic expiry. `/forget` deletes the current conversation; `/forget all` deletes every conversation; `scripts/clean_sessions.sh` does the same from a shell (macOS and Linux). An empty conversation is never written. |
| Shared documents (artifacts) | `sessions/<id>_artifact.md` and `.json` | Owner only | Deleted together with their conversation by `/forget`, `/forget all` and `scripts/clean_sessions.sh`. |
| Teacher profile | `profile.json`, or the file passed to `--profile` | Owner only | Kept until edited or deleted. The repository's `profile.json` is a neutral example: give a real profile its own path (`--profile ~/sensai/camille.json`) so it is never committed. |
| Logs | `logs/sensai.log` | Owner only | No automatic expiry; delete or rotate manually. They contain technical events, file names of indexed documents and error tracebacks, not conversation text. |
| Monitoring metadata | `data/sensai_monitoring.sqlite3` | Owner only | Enabled in the repository configuration for `/metrics`. Stores latency, token counts, feature flags and tool names, **never text**. Rows older than `retention_days` (30) are deleted at startup. |
| Quiz and file exports | An allowed folder such as `exports/` | Owner only | Kept until manually deleted; they may reproduce a prompt or a course document. Each file written by the model required a "yes". |
| Configuration and prompts | `config.json`, `prompts/*.txt`, `mcp_servers.json`, `permissions.json` | — | Never put credentials or personal data in versioned files. |

"Owner only" means `0600` files and `0700` folders: SensAI sets `umask 077` at startup
(`source/main.py`), so every file it creates is private to the user account, on macOS and
Linux. Nothing is encrypted by SensAI; use the operating system's disk encryption.

The generated SQLite databases, the sessions folder and the log files are Git-ignored.

## What can leave the computer

| Path | Default | Control |
| --- | --- | --- |
| Chat and embedding requests to Ollama | Local (`http://localhost:11434`) | If `model.base_url` is changed to a remote server, every prompt leaves the computer: keep the default. |
| Web search (`web_search`) | Off (`features.network_enabled: false`) | When enabled, each query is shown and needs a yes. The search provider (DuckDuckGo, a US company) receives the query and the computer's IP address. |
| Network MCP servers (`"network": true`, e.g. `mcp-server-fetch`) | Off | Started only with `network_enabled: true`; each call shows the exact arguments (URL) and needs a yes. |
| Other MCP servers | Local process | They run with the teacher's rights; only configure reviewed servers. Their inherited environment is filtered (no tokens or keys). |

Before a network call, SensAI warns when the text looks like personal data: e-mail
address, phone number, IBAN, long identifier, French pupil identifier (INE), date of birth,
support plan or disability words (PAP, PPS, PAI, dyslexia, ADHD…). **Names cannot be
detected**; the yes/no question remains the barrier.

## What happens in case of a leak

| Scenario | What is exposed | Technical limit today |
| --- | --- | --- |
| Laptop stolen, disk not encrypted | Everything in the table above: indexed course text, conversations (possibly with pupil first names or marks), artifacts, exports, profile, logs | No application-level encryption |
| Another account on the same computer | Nothing created by SensAI (owner-only files) | Original documents keep their own permissions |
| Teacher's session left unlocked, or malware running as the teacher | Everything SensAI can read, including through the local web UI (no login) | The web UI listens on 127.0.0.1 only and checks the `Host` header, which stops other web pages, not local programs |
| Accidental `git commit` of a real profile or document | That file | `.gitignore` covers sessions, databases and logs, not a real profile saved as `profile.json` |
| A poisoned document pushes the model to send data out | Only what the teacher approves in the y/N question | Prompt injection is mitigated, not prevented ([security](security.md)) |

Response procedure (the school is the controller and decides on notifications):

1. **Contain**: stop SensAI, disconnect the device if malware is suspected, change the
   passwords of the account.
2. **Identify what was on the device** with the inventory above: `ls sessions/ exports/`,
   the list of indexed files (`/rag` sources in the log), the profile file.
3. **Inform the school's DPO** without delay, with that inventory, the date and the
   circumstances.
4. **Notify**: if the breach presents a risk to people's rights and freedoms, the controller
   notifies the CNIL, if possible within 72 hours after becoming aware of it (GDPR art. 33,
   [CNIL guidance](https://www.cnil.fr/fr/violations-de-donnees-personnelles-les-regles-suivre)).
   If the risk is high (for example health data of identifiable pupils), the families are
   informed directly (art. 34).
5. **Erase what is no longer needed**: `/forget all`, `/rag reset`, delete logs and exports.

## Technical constraints that shape these choices

- **Local model, local data**: the reason SensAI runs Ollama on the teacher's machine. The
  cost is a 4.2 GB download and a 3B model whose answers must be checked
  ([cost-benefit](cost-benefit.md)).
- **No accounts, one user per installation**: isolation between teachers is physical (one
  computer each). A malicious document indexed by one teacher cannot affect another
  teacher's installation.
- **No encryption at rest in SensAI**: delegated to full-disk encryption (FileVault,
  BitLocker, LUKS), which protects every file, including the original course documents.
- **Retention is manual** for conversations and logs: automatic expiry would delete a
  teacher's lesson plans without warning. The commands above make deletion one step.

## Recommended usage checklist

- Use fictional or anonymised examples for tests, demos, and screenshots.
- Keep `network_enabled` false for pupil-related work; refuse any request that would send
  names or identifiers outside the computer.
- Store a real profile outside the repository (`--profile`).
- Enable full-disk encryption and lock the session.
- Delete no-longer-needed data with `/forget all` and `/rag reset`; rotate logs.
- Treat every model answer and retrieved source as material to verify before use with pupils.

Related documentation: [security and resilience](security.md), [RAG](features/rag.md),
[sessions and profile](features/sessions-and-profile.md), [tools and MCP](features/tools-and-mcp.md),
[observability](features/observability.md), and [configuration](configuration.md).
