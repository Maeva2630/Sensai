"""Filesystem tools for the model. Every operation is checked by the permission layer first."""
import re
from pathlib import Path
from typing import Any, Dict, List

from tools.base import Tool, ToolError
from tools.permissions import PermissionChecker

# Below the registry's result limit, so the truncation marker added here is never cut off
MAX_READ_BYTES = 3500
MAX_WRITE_CHARS = 200_000
MAX_LISTED_ENTRIES = 200


class FilesystemTool(Tool):
    """Base of the filesystem tools: they share the permission checker and the routing rule."""

    # File contents and file names are data written by someone else, not instructions
    untrusted = True

    # Nouns and paths only: verbs such as "write" or "save" also appear in ordinary
    # teaching requests ("write a lesson plan") and made the model call file tools.
    _REQUEST_PATTERN = re.compile(
        r"\b(files?|folders?|directory|directories|export\w*|fichiers?|dossiers?|r[ée]pertoires?)\b"
        r"|\b[\w-]+\.(md|txt|csv|json)\b"
        # A folder written with its slash ("courses/"), but not a fraction like "3/4" or "et/ou"
        r"|\b[A-Za-z][\w-]{2,}/(?=\s|$|[.,;:!?])",
        re.IGNORECASE,
    )

    def __init__(self, checker: PermissionChecker):
        self._checker = checker

    def is_relevant_for(self, user_input: str) -> bool:
        """Offer file tools only when the request is about files, to keep the prompt short."""
        return bool(self._REQUEST_PATTERN.search(user_input))

    def _path(self, arguments: Dict[str, Any], mode: str) -> Path:
        """The real path of the `path` argument when the permission layer allows it."""
        return self._checker.require(arguments["path"], mode)


def _os_error(action: str, path: Any, error: OSError) -> ToolError:
    """A readable failure that names the path the model gave, not the machine's absolute path."""
    return ToolError(f"Could not {action} '{path}': {error.strerror or type(error).__name__}")


class ReadFileTool(FilesystemTool):
    name = "read_file"
    description = f"Read a text file (the first {MAX_READ_BYTES} bytes) from an allowed directory."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path of the file to read."}},
        "required": ["path"],
    }

    def execute(self, arguments: Dict[str, Any]) -> str:
        path = self._path(arguments, "read")
        if not path.exists():
            raise ToolError(f"File '{arguments['path']}' does not exist")
        if not path.is_file():
            raise ToolError(f"'{arguments['path']}' is not a file")
        try:
            with path.open("rb") as stream:
                data = stream.read(MAX_READ_BYTES + 1)
        except OSError as e:
            raise _os_error("read", arguments["path"], e) from e

        truncated = len(data) > MAX_READ_BYTES
        data = data[:MAX_READ_BYTES]
        if b"\0" in data:
            raise ToolError(f"'{arguments['path']}' is a binary file, not text")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as e:
            # A cut in the middle of a character at the limit is not an error
            if truncated and e.start >= len(data) - 3:
                text = data[:e.start].decode("utf-8")
            else:
                raise ToolError(f"'{arguments['path']}' is not a UTF-8 text file") from e
        if truncated:
            text += f"\n[file truncated: only the first {MAX_READ_BYTES} bytes are shown]"
        return text


class WriteFileTool(FilesystemTool):
    name = "write_file"
    description = (
        "Write a text file in an allowed directory. Missing folders are created. An existing file is kept unless overwrite is true. "
        "Do not use it to export a quiz: use save_last_quiz."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path of the file to write."},
            "content": {"type": "string", "description": "Text to write."},
            "overwrite": {"type": "boolean", "description": "Replace the file if it already exists (default false)."},
        },
        "required": ["path", "content"],
    }

    def confirmation_prompt(self, arguments: Dict[str, Any]) -> str | None:
        """Every write is shown to the user first: a document read earlier can contain
        instructions that push the model to overwrite a teacher's file."""
        content = str(arguments.get("content", ""))
        preview = " ".join(content.split())[:80]
        replace = " It REPLACES the existing file." if arguments.get("overwrite") else ""
        return (
            f"Write {len(content)} characters to '{arguments.get('path')}'?{replace} "
            f"Start of the text: \"{preview}\""
        )

    def execute(self, arguments: Dict[str, Any]) -> str:
        path = self._path(arguments, "write")
        content = arguments["content"]
        if len(content) > MAX_WRITE_CHARS:
            raise ToolError(f"Content is too long ({len(content)} characters, maximum {MAX_WRITE_CHARS})")
        if path.is_dir():
            raise ToolError(f"'{arguments['path']}' is a directory")
        if path.exists() and not arguments.get("overwrite", False):
            raise ToolError(f"File '{arguments['path']}' already exists. Use overwrite: true to replace it")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as e:
            raise _os_error("write", arguments["path"], e) from e
        return f"Wrote {len(content)} characters to '{arguments['path']}'"


class CreateDirectoryTool(FilesystemTool):
    name = "create_directory"
    description = "Create a directory (and its parents) in an allowed directory."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path of the directory to create."}},
        "required": ["path"],
    }

    def execute(self, arguments: Dict[str, Any]) -> str:
        path = self._path(arguments, "write")
        if path.exists() and not path.is_dir():
            raise ToolError(f"'{arguments['path']}' exists and is not a directory")
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise _os_error("create directory", arguments["path"], e) from e
        return f"Directory '{arguments['path']}' is ready"


class ListDirectoryTool(FilesystemTool):
    name = "list_directory"
    description = "List the files and folders of an allowed directory."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path of the directory to list."}},
        "required": ["path"],
    }

    def execute(self, arguments: Dict[str, Any]) -> str:
        path = self._path(arguments, "read")
        if not path.exists():
            raise ToolError(f"Directory '{arguments['path']}' does not exist")
        if not path.is_dir():
            raise ToolError(f"'{arguments['path']}' is not a directory")
        try:
            children = sorted(path.iterdir(), key=lambda child: (not child.is_dir(), child.name.lower()))
        except OSError as e:
            raise _os_error("list", arguments["path"], e) from e

        # Entries the agent could not open (deny rules, links leaving the folder) are not listed
        visible = [child for child in children if self._checker.check(child, "read").allowed]
        if not visible:
            return f"'{arguments['path']}' is empty"
        names: List[str] = [f"{child.name}/" if child.is_dir() else child.name for child in visible[:MAX_LISTED_ENTRIES]]
        if len(visible) > MAX_LISTED_ENTRIES:
            names.append(f"... and {len(visible) - MAX_LISTED_ENTRIES} more")
        return "\n".join(names)


def filesystem_tools(checker: PermissionChecker) -> List[FilesystemTool]:
    """The four filesystem tools sharing one permission checker."""
    return [ReadFileTool(checker), WriteFileTool(checker), CreateDirectoryTool(checker), ListDirectoryTool(checker)]
