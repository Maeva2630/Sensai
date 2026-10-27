# Configuration reference

[Home](../README.md) · [User guide](user-guide.md) · [Feature documentation](features/README.md)

SensAI loads one application configuration file at startup:

```text
<repository-root>/config.json
```

The current CLI always reads this location. `python3 source/main.py` has no `--config`
option, so edit the repository's `config.json` and restart SensAI after each change.

## File format

The file is a UTF-8 encoded **JSON object**. JSON needs double-quoted keys and strings;
it does not allow comments or trailing commas. This is invalid JSON:

```json
{
  "temperature": 0.2,
}
```

`model`, `features`, and `rag` are required objects. `logging`, `monitoring`, and
`personas` are optional. Unknown fields are currently ignored, so field names must be
spelled exactly as documented.

## Complete example

This is a complete valid starting point. The two model names must be available from the
Ollama server specified by `base_url`.

```json
{
  "model": {
    "chat_model": "qwen2.5:3b",
    "embedding_model": "nomic-embed-text",
    "base_url": "http://localhost:11434",
    "temperature": 0.2,
    "context_window": 4096
  },
  "features": {
    "rag_enabled": true,
    "tools_enabled": true,
    "memory_enabled": true,
    "source_display_enabled": true
  },
  "rag": {
    "top_k": 3,
    "max_context_characters": 4000,
    "relevance_threshold": 0.35
  },
  "logging": {
    "level": "INFO",
    "file_path": "logs/sensai.log"
  },
  "monitoring": {
    "enabled": false,
    "database_path": "data/sensai_monitoring.sqlite3",
    "retention_days": 30,
    "prompt_version": "default-v1"
  },
  "personas": {
    "active_persona": "teacher-assistant",
    "profiles": [
      {
        "identifier": "teacher-assistant",
        "display_name": "Teacher assistant",
        "role": "Educational assistant for teachers and educators",
        "tone": "Clear, practical, supportive, and pedagogical",
        "prompt_file": "prompts/teacher-assistant.txt",
        "response_scope": "Support educators with lesson preparation, teaching resources, and assessment."
      }
    ]
  }
}
```

## Required sections

### `model`

| Field | Accepted value | Default | Meaning |
| --- | --- | --- | --- |
| `chat_model` | Non-empty string | `qwen2.5:3b` | Ollama model generating chat responses. |
| `embedding_model` | Non-empty string | `nomic-embed-text` | Ollama model used by RAG search. |
| `base_url` | Non-empty string | `http://localhost:11434` | Ollama API address. |
| `temperature` | Finite number from `0` to `2` | `0.2` | Response variation; lower values are usually more stable. |
| `context_window` | Integer, at least `512` | `4096` | Context sent as Ollama `num_ctx`; larger values can use more RAM/VRAM. |

The chat and embedding models serve different purposes. Pull both configured models with
`ollama pull <model-name>`. Read the [RAG guide](features/rag.md#configuration-and-limitations)
before changing the embedding model.

### `features`

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `rag_enabled` | Boolean | `true` | Makes `/rag on` available; RAG starts off each session. |
| `tools_enabled` | Boolean | `true` | Registers native time and permitted filesystem tools, including controlled quiz export. |
| `memory_enabled` | Boolean | `true` | Enables session saving/loading, token usage, and conversation compression commands. |
| `source_display_enabled` | Boolean | `true` | Initial visibility of RAG source lines. |

### `rag`

| Field | Accepted value | Default | Meaning |
| --- | --- | --- | --- |
| `top_k` | Integer, at least `1` | `3` | Maximum matching chunks considered. |
| `max_context_characters` | Integer, at least `300` | `4000` | Character budget for retrieved context. |
| `relevance_threshold` | Finite number from `-1` to `1` | `0.35` | Chunks below this score are discarded. |

Raise the threshold to reject more weak matches; lower it if relevant passages are being
excluded. Test settings with representative documents: they do not guarantee correctness.

## Optional sections

### `logging`

If omitted, SensAI uses `INFO` and `logs/sensai.log`.

| Field | Accepted value | Meaning |
| --- | --- | --- |
| `level` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` | Minimum log severity; case is normalized to uppercase. |
| `file_path` | Non-empty string | Local log destination. |

### `monitoring`

If omitted, monitoring is disabled by default.

| Field | Accepted value | Default | Meaning |
| --- | --- | --- | --- |
| `enabled` | Boolean | `false` | Enables local interaction metrics. |
| `database_path` | Non-empty string | `data/sensai_monitoring.sqlite3` | Local metrics database. |
| `retention_days` | Integer, at least `1` | `30` | Age after which events are removed on store initialization. |
| `prompt_version` | Non-empty string | `default-v1` | Manual label for comparing prompt versions. |

See [logging and monitoring](features/observability.md) for `/metrics` and privacy notes.

### `personas`

If omitted, SensAI uses a built-in teacher-assistant persona. If included,
`active_persona` must match a unique `profiles[].identifier`; `profiles` must be a
non-empty array. Each profile needs non-empty `identifier`, `display_name`, `role`,
`tone`, and `response_scope` strings, plus **exactly one** prompt source:

- `system_prompt`: an inline, non-empty string; or
- `prompt_file`: a non-empty UTF-8 file path relative to `config.json`.

A prompt file must remain within the configuration directory. For example:

```json
{
  "identifier": "mathematics-assistant",
  "display_name": "Mathematics assistant",
  "role": "Educational assistant for mathematics teachers",
  "tone": "Precise and pedagogical",
  "prompt_file": "prompts/mathematics-assistant.txt",
  "response_scope": "Support mathematics lessons and assessment preparation."
}
```

See [personas](features/personas.md) for runtime commands.

## Separate MCP configuration

MCP server definitions do **not** go in `config.json`. The optional `mcp_servers.json`
file is read from the directory where SensAI starts. Its `mcpServers` object defines a
local command, arguments, optional string environment values, and a positive timeout.
See [tools and MCP](features/tools-and-mcp.md#local-mcp-servers).

Only configure MCP servers you trust: SensAI starts their commands locally. Do not commit
passwords, tokens, personal data, or production credentials in configuration examples.

## Validation errors

SensAI validates this file at startup. Common errors are malformed JSON, missing required
sections, booleans written as strings (`true`, not `"true"`), invalid numeric ranges, a
persona defining both prompt fields, or a missing prompt file. Fix the reported field and
restart SensAI.
