# Tools and MCP

[Feature index](README.md) · [User guide](../user-guide.md)

Tools allow a model to request an action and use its result in a subsequent response.
SensAI validates tool arguments before execution. The model decides whether to request
an offered tool; results and model output can vary.

## Native time tool

With `features.tools_enabled` set to `true`, SensAI registers `get_time`. List tools
using `/tools` and ask an explicit date or time question:

```text
You: /tools
- get_time [native]: Get the current local date and time.
You: What time is it?
[tool] get_time({})
[tool] result: 2026-10-23 14:30:00
Assistant: The local time is 14:30.
```

The exchange is illustrative. The time is local to the machine running the tool.
SensAI selects relevant tools for a request. RAG context can still be included alongside
tools; only requests explicitly claimed by a narrow tool, such as `get_time`, bypass
document retrieval.

## Quiz generation and controlled export

`/quiz` asks the model for a validated multiple-choice quiz. With RAG enabled, it uses
matching indexed course excerpts; without RAG, it uses the model's general knowledge.

```text
You: /quiz --questions 3 --language French Le théorème de Pythagore
You: /quiz --questions 5 --save exports/volcans.md Les volcans
You: /quiz --questions 5 --save exports/volcans_eleves.md --no-answers Les volcans
```

The topic is required. `--questions` accepts the range printed by `/help`; quote a save
path containing spaces. SensAI can only write exports to directories allowed by
`permissions.json`; use `/permissions` to display those rules. The model may also call a
tool to save the last generated quiz, but `/quiz --save` is the reliable way to choose
whether answers are included or an existing file may be overwritten.

## Local MCP servers

MCP (Model Context Protocol) allows external local servers to expose additional tools.
SensAI reads `mcp_servers.json` from the repository root. The repository configuration
starts a teacher-tools server and can also start the optional `mcp-server-fetch` server:

```json
{
  "mcpServers": {
    "example": {
      "command": "python3",
      "args": ["source/mcp/servers/teacher_tools.py"],
      "timeout": 10
    }
  }
}
```

Start SensAI from the repository root and use `/tools` to see the names exposed by each
server. MCP tool names are prefixed with the server name, for example
`teacher__...`, to avoid collisions with native tools.

Only configure servers you trust: SensAI starts their commands on your machine. Review
their arguments, permissions, and data access. Avoid storing secrets in example files.
Configured MCP servers are loaded separately from the native `tools_enabled` switch.
The model must support tool calling for this workflow to work reliably.
