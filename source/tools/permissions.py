"""File access permissions: which paths the agent may read or write.

Every filesystem tool must call `PermissionChecker.require` before touching a path.
Access is denied by default: a path is allowed only when it is inside a directory
listed in the permission file, with the requested mode, and matches no deny rule.

Example `permissions.json`:

    {
      "allowed": [
        {"path": "courses", "modes": ["read"]},
        {"path": "exports", "modes": ["read", "write"]}
      ],
      "deny": [".env", "*.pem"]
    }

Relative directories are relative to the folder holding the file, not to the
current directory.
"""
import fnmatch
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

from tools.base import ToolError

DEFAULT_PERMISSIONS_PATH = Path(__file__).resolve().parents[2] / "permissions.json"
VALID_MODES = ("read", "write")


class PermissionDeniedError(ToolError):
    """A path is not allowed for the requested access. The message is safe to show."""


@dataclass(frozen=True)
class AllowedDirectory:
    """A directory the agent may use, with the access modes granted inside it."""
    path: Path
    modes: Tuple[str, ...]


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    path: Optional[Path]
    reason: str


class PermissionChecker:
    """Decides whether a path may be read or written."""

    def __init__(self, allowed: Sequence[AllowedDirectory] = (), deny: Sequence[str] = (),
                 base_dir: Union[str, Path, None] = None):
        self._base_dir = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
        # Allowed directories are resolved too, so a symlinked directory is compared by its real location
        self._allowed = [AllowedDirectory(entry.path.resolve(), entry.modes) for entry in allowed]
        self._deny = [pattern.lower() for pattern in deny]

    def describe(self) -> List[str]:
        """One line per rule, for display."""
        lines = [f"{self._display(entry.path)}: {', '.join(entry.modes)}" for entry in self._allowed]
        lines.extend(f"deny: {pattern}" for pattern in self._deny)
        return lines

    def check(self, path: Any, mode: str) -> PermissionDecision:
        """Check one path for `mode` ("read" or "write"). Never raises."""
        if mode not in VALID_MODES:
            return PermissionDecision(False, None, f"unknown access mode '{mode}'")
        resolved = self._resolve(path)
        if resolved is None:
            return PermissionDecision(False, None, f"'{path}' is not a valid path")

        entry = self._closest_directory(resolved)
        if entry is None:
            return PermissionDecision(False, resolved, f"'{path}' is outside the allowed directories ({self._summary()})")
        pattern = self._denied_by(resolved.relative_to(entry.path))
        if pattern:
            return PermissionDecision(False, resolved, f"'{path}' matches the deny rule '{pattern}'")
        if mode not in entry.modes:
            return PermissionDecision(
                False, resolved,
                f"'{self._display(entry.path)}' does not allow {mode} access (allowed: {', '.join(entry.modes)})",
            )
        return PermissionDecision(True, resolved, "allowed")

    def require(self, path: Any, mode: str) -> Path:
        """Return the real path when access is allowed, otherwise raise PermissionDeniedError."""
        decision = self.check(path, mode)
        if not decision.allowed:
            raise PermissionDeniedError(f"Access denied: {decision.reason}")
        return decision.path

    def _resolve(self, path: Any) -> Optional[Path]:
        """Absolute real path: `~` expanded, `..` collapsed and symlinks followed."""
        if not isinstance(path, (str, os.PathLike)) or not str(path).strip() or "\0" in str(path):
            return None
        try:
            candidate = Path(path).expanduser()
            if not candidate.is_absolute():
                candidate = self._base_dir / candidate
            return candidate.resolve()
        except (OSError, RuntimeError, ValueError):
            return None  # for example a symlink loop or an unknown user after `~`

    def _closest_directory(self, resolved: Path) -> Optional[AllowedDirectory]:
        """The deepest allowed directory holding the path, so a nested rule overrides its parent."""
        holders = [entry for entry in self._allowed if resolved.is_relative_to(entry.path)]
        return max(holders, key=lambda entry: len(entry.path.parts), default=None)

    def _denied_by(self, inside_allowed: Path) -> Optional[str]:
        """Deny rule matching a name below the allowed directory (folders above it are not our business)."""
        names = [part.lower() for part in inside_allowed.parts]
        for pattern in self._deny:
            if any(fnmatch.fnmatchcase(name, pattern) for name in names):
                return pattern
        return None

    def _display(self, path: Path) -> str:
        try:
            return str(path.relative_to(self._base_dir))
        except ValueError:
            return str(path)

    def _summary(self) -> str:
        return "; ".join(f"{self._display(e.path)} ({', '.join(e.modes)})" for e in self._allowed) or "none"


def load_permissions(path: Union[str, Path] = DEFAULT_PERMISSIONS_PATH) -> Tuple[PermissionChecker, List[str]]:
    """Read a permission file. Returns the checker and warnings.

    A missing or unreadable file gives a checker that denies everything. An invalid
    entry is skipped with a warning, so a typo can only remove access, never add it.
    """
    config_path = Path(path)
    base_dir = config_path.resolve().parent
    if not config_path.exists():
        return PermissionChecker(base_dir=base_dir), []

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return PermissionChecker(base_dir=base_dir), [f"Could not read permissions '{config_path}': {e}. All file access is denied."]
    if not isinstance(data, dict):
        return PermissionChecker(base_dir=base_dir), [f"Permissions '{config_path}' must be a JSON object. All file access is denied."]

    warnings: List[str] = []
    allowed: List[AllowedDirectory] = []
    entries = data.get("allowed", [])
    for entry in entries if isinstance(entries, list) else []:
        directory, error = _parse_allowed(entry, base_dir)
        if error:
            warnings.append(f"Ignoring permission entry {entry!r}: {error}")
        else:
            allowed.append(directory)
    if not isinstance(entries, list):
        warnings.append("'allowed' must be a list. No directory is allowed.")

    deny = data.get("deny", [])
    if not isinstance(deny, list) or not all(isinstance(item, str) and item.strip() for item in deny):
        warnings.append("'deny' must be a list of non-empty strings. Deny rules were ignored: check the file.")
        deny = []
    return PermissionChecker(allowed, deny, base_dir), warnings


def _parse_allowed(entry: Any, base_dir: Path) -> Tuple[Optional[AllowedDirectory], str]:
    if not isinstance(entry, dict):
        return None, "entry must be an object"
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, "'path' must be a non-empty string"
    modes = entry.get("modes")
    if not isinstance(modes, list) or not modes or not all(mode in VALID_MODES for mode in modes):
        return None, f"'modes' must be a non-empty list of {' and '.join(VALID_MODES)}"

    directory = Path(raw_path).expanduser()
    if not directory.is_absolute():
        directory = base_dir / directory
    return AllowedDirectory(directory, tuple(dict.fromkeys(modes))), ""
