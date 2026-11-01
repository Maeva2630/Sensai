import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# Resolved from the project root, like config.json and permissions.json, so starting
# SensAI from another folder neither loses the servers nor runs a file planted there.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "mcp_servers.json")
DEFAULT_TIMEOUT = 10.0
_SERVER_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class MCPServerConfig:
    """How to launch one MCP server."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    timeout: float = DEFAULT_TIMEOUT
    # True for servers that reach the internet; they start only when network access is enabled.
    network: bool = False
    # Words that make this server's tools worth offering for a request. Empty: always offered.
    keywords: List[str] = field(default_factory=list)
    # Folder the server runs in, so relative paths in `args` work from any working directory.
    cwd: str = PROJECT_ROOT


def load_mcp_config(path: str = DEFAULT_CONFIG_PATH) -> Tuple[List[MCPServerConfig], List[str]]:
    """Load the MCP servers declared in a JSON file.

    Expected format (same `mcpServers` layout as most MCP server documentation):

        {"mcpServers": {"name": {"command": "...", "args": ["..."], "env": {"K": "V"}, "timeout": 10,
                                 "network": false, "keywords": ["..."]}}}

    Returns the valid servers and a list of warnings. A missing file means no
    server; a malformed file or entry is reported and skipped, never raised.
    """
    if not os.path.exists(path):
        return [], []

    try:
        with open(path, encoding="utf-8") as config_file:
            data = json.load(config_file)
    except (OSError, json.JSONDecodeError) as e:
        return [], [f"Could not read MCP configuration '{path}': {e}"]

    entries = data.get("mcpServers") if isinstance(data, dict) else None
    if not isinstance(entries, dict):
        return [], [f"MCP configuration '{path}' must contain an 'mcpServers' object"]

    servers: List[MCPServerConfig] = []
    warnings: List[str] = []
    config_directory = os.path.dirname(os.path.abspath(path))
    for name, entry in entries.items():
        server, error = _parse_server(name, entry, config_directory)
        if error:
            warnings.append(f"Ignoring MCP server '{name}': {error}")
        else:
            servers.append(server)
    return servers, warnings


def _parse_server(name: str, entry: Any, config_directory: str = PROJECT_ROOT) -> Tuple[Any, str]:
    if not _SERVER_NAME.match(name):
        return None, "name may only contain letters, digits, '_' and '-'"
    if not isinstance(entry, dict):
        return None, "entry must be an object"

    command = entry.get("command")
    if not isinstance(command, str) or not command.strip():
        return None, "'command' must be a non-empty string"

    args = entry.get("args", [])
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        return None, "'args' must be a list of strings"

    env = entry.get("env", {})
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        return None, "'env' must be an object of strings"

    timeout = entry.get("timeout", DEFAULT_TIMEOUT)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        return None, "'timeout' must be a positive number"

    network = entry.get("network", False)
    if not isinstance(network, bool):
        return None, "'network' must be true or false"

    keywords = entry.get("keywords", [])
    if not isinstance(keywords, list) or not all(isinstance(word, str) and word.strip() for word in keywords):
        return None, "'keywords' must be a list of non-empty strings"

    return MCPServerConfig(
        name, command, args, env, float(timeout), network, [word.strip() for word in keywords], config_directory
    ), ""
