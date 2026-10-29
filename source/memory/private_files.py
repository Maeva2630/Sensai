"""Owner-only storage for files that can hold personal data (conversations, profile)."""

import json
import os
from pathlib import Path
from typing import Any

PRIVATE_FILE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700


def make_private_dir(path: Path) -> None:
    """Create a directory only the owner can open. Permission errors are ignored (e.g. Windows)."""
    path.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIR_MODE)
    try:
        os.chmod(path, PRIVATE_DIR_MODE)
    except OSError:
        pass


def write_private_json(path: Path, data: Any) -> None:
    """Write JSON to a file that only the owner can read, including when it already existed."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, PRIVATE_FILE_MODE)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    try:
        os.chmod(path, PRIVATE_FILE_MODE)
    except OSError:
        pass
