# Logging, prompt inspection, and monitoring

[Feature index](README.md) · [User guide](../user-guide.md)

## Logs

SensAI writes operational logs locally without displaying debug logs during normal chat.
The `logging` section of `config.json` selects a file and level:

```json
{
  "level": "DEBUG",
  "file_path": "logs/sensai.log"
}
```

This object is the value of the `logging` key, not a complete `config.json` file.
Levels are `DEBUG`, `INFO`, `WARNING`, and `ERROR`. For example, inspect recent lines
from a separate terminal using:

```bash
tail -n 30 logs/sensai.log
```

Logs include component, level, timestamps and diagnostic information. Error details can
contain technical or potentially sensitive data, so review logs before sharing them.

## Inspect the model request

After a question, run:

```text
/debug prompt
/debug prompt --include-system
```

The first command hides the base system message and profile. The second deliberately
shows them. Output labels system, RAG context, user, assistant history, and tool messages
when present. Example, after a RAG question:

```text
Reconstructed model request:

--- 1. System message ---
<hidden; use /debug prompt --include-system to display it>

--- 2. RAG context ---
... retrieved excerpts ...

--- 3. User message ---
What colour is Earth according to the document?
```

This display is illustrative and can include private document or conversation content.
Prompt inspection is shown on demand rather than copied into the operational log.

## Local monitoring dashboard

Monitoring is disabled by default. To enable it, edit the `monitoring` object in
`config.json` and restart:

```json
{
  "enabled": true,
  "database_path": "data/sensai_monitoring.sqlite3",
  "retention_days": 30,
  "prompt_version": "teacher-v1"
}
```

`retention_days` controls cleanup of old events when the store is initialized.
`prompt_version` is a manually maintained label, useful for comparing changes; it does
not change or detect the prompt automatically.

```text
/metrics
/metrics --from 2026-10-01 --to 2026-10-31 --model qwen2.5:3b
/metrics --prompt-version teacher-v1 --feature rag
/metrics --tool get_time
/metrics --error OllamaConnectionError
```

Filters are combined: only interactions matching all supplied filters contribute to the
dashboard. The error filter matches a recorded category, such as an Ollama exception
class name. A sample output is:

```text
Local monitoring dashboard
Filters: model=qwen2.5:3b
Model calls: 2
Latency: average 2234.1 ms | p95 3360.2 ms
Error rate: 0.0%
RAG retrievals: 1
Estimated tokens: input 192 | output 556
Tool calls: 1 (1 succeeded, 0 failed)
Calls over time:
  2026-10-23: 2
```

Values are illustrative. `p95` is the 95th-percentile latency; RAG retrievals count
model calls with at least one retrieved chunk, not all attempted searches. Token counts
are estimates, not billing-grade usage. The local metrics store does not persist raw
prompts, document excerpts, responses, tool arguments, or profile content by default.
Even metadata can be sensitive: protect the local database and review it before sharing.
