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
SensAI selects relevant tools for a request; when any tool is offered, the current chat
flow does not retrieve RAG context for that request.

## Local MCP servers

MCP allows external local servers to expose additional tools. SensAI reads an optional
`mcp_servers.json` from the directory where it is started. For a local demonstration,
create this file in the repository root:

```json
{
  "mcpServers": {
    "example": {
      "command": "python3",
      "args": ["tests/fake_mcp_server.py"],
      "timeout": 10
    }
  }
}
```

Start SensAI from the repository root and use `/tools`. One listed tool is
`example__echo [example]`. The server in this example is a test fixture: some of its
tools deliberately fail, wait, or terminate the server. Remove its configuration entry
when finished.

Only configure servers you trust: SensAI starts their commands on your machine. Review
their arguments, permissions, and data access. Avoid storing secrets in example files.
Configured MCP servers are loaded separately from the native `tools_enabled` switch.
The model must support tool calling for this workflow to work reliably.
