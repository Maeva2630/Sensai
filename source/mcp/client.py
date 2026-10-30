import json
import os
import queue
import subprocess
import threading
from typing import Any, Dict, List, Optional

PROTOCOL_VERSION = "2024-11-05"
CLIENT_INFO = {"name": "sensai", "version": "0.1.0"}

# Variables a server needs to start and reach the network. Everything else (API keys,
# tokens, credentials in the user's environment) is not handed to third-party servers;
# a server that needs one gets it explicitly through the `env` field of its config.
_SAFE_ENV_NAMES = {
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "TERM", "TMPDIR", "TEMP", "TMP",
    "SYSTEMROOT", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PATHEXT", "COMSPEC",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "PYTHONPATH",
}
_SAFE_ENV_PREFIXES = ("LC_", "XDG_", "UV_")


def safe_environment(environ: Dict[str, str]) -> Dict[str, str]:
    """Keep only the variables a server needs to run, and drop the rest."""
    return {
        name: value
        for name, value in environ.items()
        if name.upper() in _SAFE_ENV_NAMES or name.upper().startswith(_SAFE_ENV_PREFIXES)
    }


class MCPError(Exception):
    """Base exception for all MCP client errors."""
    pass


class MCPToolError(MCPError):
    """Raised when an MCP tool reports a failure (`isError` in its result)."""
    pass


class MCPConnectionError(MCPError):
    """Raised when the MCP server cannot be started or stops responding."""
    pass


class MCPTimeoutError(MCPError):
    """Raised when the MCP server does not answer in time."""
    pass


class MCPClient:
    """Minimal MCP client talking to a local server over stdio (JSON-RPC 2.0).

    Messages are newline-delimited JSON written to the server's stdin and read
    from its stdout.
    """

    def __init__(self, command: str, args: Optional[List[str]] = None,
                 env: Optional[Dict[str, str]] = None, timeout: float = 10.0):
        self.command = command
        self.args = args or []
        self.env = env
        self.timeout = timeout
        self._process: Optional[subprocess.Popen] = None
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self._next_id = 0

    def __enter__(self) -> "MCPClient":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def start(self) -> None:
        """Launch the server and perform the MCP initialization handshake."""
        env = {**safe_environment(os.environ), **(self.env or {})}
        try:
            self._process = subprocess.Popen(
                [self.command, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                env=env,
            )
        except OSError as e:
            raise MCPConnectionError(f"Failed to start MCP server '{self.command}': {e}") from e

        threading.Thread(target=self._read_stdout, daemon=True).start()

        try:
            self._request("initialize", {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": CLIENT_INFO,
            })
            self._notify("notifications/initialized")
        except MCPError:
            self.close()
            raise

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return the tools exposed by the server (follows pagination)."""
        tools: List[Dict[str, Any]] = []
        cursor = None
        while True:
            params = {"cursor": cursor} if cursor else {}
            result = self._request("tools/list", params)
            tools.extend(result.get("tools", []))
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call a tool. Returns the raw result: {"content": [...], "isError": bool}."""
        return self._request("tools/call", {"name": name, "arguments": arguments or {}})

    @staticmethod
    def result_text(result: Dict[str, Any]) -> str:
        """Join the text parts of a tool result."""
        return "\n".join(
            part.get("text", "") for part in result.get("content", []) if part.get("type") == "text"
        )

    def close(self) -> None:
        """Stop the server subprocess. Safe to call several times."""
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stdout:
            process.stdout.close()

    def _read_stdout(self) -> None:
        process = self._process
        try:
            for line in process.stdout:
                self._lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            self._lines.put(None)  # server stopped

    def _send(self, message: Dict[str, Any]) -> None:
        if self._process is None or self._process.poll() is not None:
            raise MCPConnectionError("MCP server is not running")
        try:
            self._process.stdin.write(json.dumps(message) + "\n")
            self._process.stdin.flush()
        except (OSError, ValueError) as e:
            raise MCPConnectionError(f"Failed to send message to MCP server: {e}") from e

    def _notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        message: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            message["params"] = params
        self._send(message)

    def _request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})

        while True:
            try:
                line = self._lines.get(timeout=self.timeout)
            except queue.Empty:
                raise MCPTimeoutError(f"MCP server did not answer '{method}' within {self.timeout}s") from None
            if line is None:
                self._lines.put(None)  # keep the sentinel for later calls
                raise MCPConnectionError("MCP server stopped unexpectedly")

            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue  # not a JSON-RPC message (e.g. a log line)
            if not isinstance(message, dict):
                continue

            if "method" in message:
                self._answer_server_request(message)
                continue
            if message.get("id") != request_id:
                continue  # late answer to a request that already timed out

            if "error" in message:
                error = message["error"]
                raise MCPError(f"MCP error {error.get('code')}: {error.get('message')}")
            return message.get("result", {})

    def _answer_server_request(self, message: Dict[str, Any]) -> None:
        """Reply to requests sent by the server. Notifications need no answer."""
        if "id" not in message:
            return
        if message["method"] == "ping":
            reply = {"jsonrpc": "2.0", "id": message["id"], "result": {}}
        else:
            reply = {"jsonrpc": "2.0", "id": message["id"],
                     "error": {"code": -32601, "message": "Method not found"}}
        try:
            self._send(reply)
        except MCPConnectionError:
            pass
