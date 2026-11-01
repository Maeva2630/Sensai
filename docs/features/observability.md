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
shows them. The output starts with the generation settings and the tools offered, then
labels each message: system message, RAG rules, conversation summary, user message (with
document excerpts framed as data when RAG is on), assistant history and tool messages.
Real output after a RAG question:

```text
Reconstructed model request:
Settings: model=qwen2.5:3b, temperature=0.2, num_ctx=4096
Tools offered (0): none

--- 1. System message ---
<hidden; use /debug prompt --include-system to display it>

--- 2. RAG rules (SensAI's own instructions) ---
RAG mode is on. The user's latest message starts with excerpts retrieved from the teacher's indexed documents, ...

--- 3. User message with document excerpts (framed as data) ---
<<<DOCUMENT EXCERPTS>>>
[Source: fractions_exam.md | Lines: 1-3 | Language: markdown | Score: 0.818]
Le contrôle de fractions de la 6e B a lieu le 14 mars 2027 à 10h30 en salle 204.
<<<END OF DOCUMENT EXCERPTS>>>

Question: Quelle est la date du contrôle de fractions de la 6e B ?
...
```

The display can include private document or conversation content. Prompt inspection is
shown on demand rather than copied into the operational log. Why the request has this
order and size: [context engineering](../context-engineering.md).

## Speed of the last answer

`/stats` shows what Ollama measured for the last answer:

```text
You: /stats
Last answer:
- Settings: model=qwen2.5:3b, temperature=0.2, num_ctx=4096
- Generation: 24 tokens in 0.41 s = 58.9 tokens/s
- Prompt reading: 728 tokens in 0.33 s = 2209 tokens/s (tokens already in Ollama's cache are not counted)
- Model load: 0.00 s (close to 0 when the model is already in memory)
- Total measured by Ollama: 3.12 s
```

Generation speed is bounded by memory bandwidth: each new token reads all the model weights
once (1.9 GB for qwen2.5:3b in Q4_K_M), so an Apple M5 (153 GB/s) generates about 59
tokens/s and the team's RTX 4060 laptop (256 GB/s) about 111. Prompt reading processes many
tokens at once and is 20 to 50 times faster. Comparable runs: [benchmarks](../model-benchmarks.md).

## Local monitoring dashboard

Monitoring is enabled in the repository's `config.json` (it is off when the section is
absent). It stores metadata only: latency, token counts, whether RAG was used for the
request, tool names and status, never text. To change it, edit the `monitoring` object and
restart:

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

For measured output tokens per second, first-token latency, process RAM, VRAM, and
configuration comparisons, use the [local model benchmark protocol](../model-benchmarks.md).
Unlike `/metrics`, its token throughput comes from Ollama's reported generation counts
and durations rather than SensAI's character-based estimate.
