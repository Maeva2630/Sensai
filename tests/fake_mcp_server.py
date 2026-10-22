"""Tiny MCP server over stdio used by the MCP client tests."""
import json
import sys
import time

TOOLS = [
    {"name": "echo", "description": "Echo a text.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "fail", "description": "Always fails.", "inputSchema": {"type": "object"}},
    {"name": "hang", "description": "Never answers in time.", "inputSchema": {"type": "object"}},
    {"name": "crash", "description": "Kills the server.", "inputSchema": {"type": "object"}},
]


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        message = json.loads(line)
        method, request_id, params = message.get("method"), message.get("id"), message.get("params", {})
        if request_id is None:
            continue  # notification

        if method == "initialize":
            print("log line that is not JSON-RPC", flush=True)
            send({"jsonrpc": "2.0", "method": "notifications/message", "params": {"data": "hello"}})
            send({"jsonrpc": "2.0", "id": request_id, "result": {
                "protocolVersion": params["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1"},
            }})
        elif method == "tools/list":
            # Two pages, to exercise pagination
            if params.get("cursor") == "page2":
                send({"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS[2:]}})
            else:
                send({"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS[:2], "nextCursor": "page2"}})
        elif method == "tools/call":
            name, arguments = params["name"], params.get("arguments", {})
            if name == "echo":
                send({"jsonrpc": "2.0", "id": request_id, "result": {
                    "content": [{"type": "text", "text": arguments["text"]}], "isError": False}})
            elif name == "fail":
                send({"jsonrpc": "2.0", "id": request_id, "result": {
                    "content": [{"type": "text", "text": "something went wrong"}], "isError": True}})
            elif name == "hang":
                time.sleep(30)
            elif name == "crash":
                sys.exit(1)
            else:
                send({"jsonrpc": "2.0", "id": request_id,
                      "error": {"code": -32602, "message": f"Unknown tool: {name}"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "Method not found"}})


if __name__ == "__main__":
    main()
