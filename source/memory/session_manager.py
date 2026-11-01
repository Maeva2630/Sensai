import json
import re
import uuid
from datetime import datetime
from pathlib import Path

from memory.artifact_manager import is_artifact_file
from memory.private_files import make_private_dir, write_private_json

_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class InvalidSessionIdError(ValueError):
    """Raised for a session identifier that could point outside the sessions folder."""


def _validated(session_id: str) -> str:
    if not _SESSION_ID.match(session_id):
        raise InvalidSessionIdError(
            "Invalid session id: use only letters, digits, '-' and '_' (64 characters at most)."
        )
    return session_id


class SessionManager:
    """Manage conversation sessions with JSON file storage."""

    def __init__(self, session_id: str | None = None, storage_dir: str = "sessions"):
        self.storage_dir = Path(storage_dir)
        self.session_id = _validated(session_id) if session_id else self._generate_id()
        self.messages: list[dict] = []

    def add_message(self, role: str, content: str) -> None:
        """Add a message turn to the current in-memory history."""
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        """Return the list of current conversation messages."""
        return self.messages

    def clear(self) -> None:
        """Clear messages in memory."""
        self.messages = []

    def save(self) -> str | None:
        """Save the session to a JSON file at sessions/<session_id>.json.

        An empty conversation is not written: there is nothing to keep, and a session
        erased with /forget must not come back as an empty file when the CLI exits.
        Returns the file path, or None when nothing was saved.
        """
        if not self.messages:
            return None
        make_private_dir(self.storage_dir)
        session_path = self.storage_dir / f"{_validated(self.session_id)}.json"

        data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "messages": self.messages,
        }

        write_private_json(session_path, data)

        print(f"Session saved at {session_path}") # can be removed 
        return str(session_path)

    def load(self, session_id: str | None = None) -> list[dict]:
        """Load messages from sessions/<session_id>.json."""
        target_id = _validated(session_id or self.session_id)
        session_path = self.storage_dir / f"{target_id}.json"

        if not session_path.exists():
            raise FileNotFoundError(f"Session {target_id} not found at {session_path}")

        with open(session_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.session_id = data.get("session_id", target_id)
        self.messages = data.get("messages", [])
        return self.messages

    def delete(self, session_id: str | None = None) -> bool:
        """Delete one saved session file. Returns whether a file was removed."""
        session_path = self.storage_dir / f"{_validated(session_id or self.session_id)}.json"
        if not session_path.is_file():
            return False
        session_path.unlink()
        return True

    def delete_all(self) -> int:
        """Delete every saved session file in the sessions folder and return how many.

        Artifact files stored in the same folder are not sessions: `ArtifactManager.delete_all`
        removes them.
        """
        removed = 0
        for session_path in self.saved_session_paths():
            session_path.unlink()
            removed += 1
        return removed

    def saved_session_paths(self) -> list[Path]:
        """Return the saved conversation files, without the artifact files kept next to them."""
        if not self.storage_dir.is_dir():
            return []
        return [
            path
            for path in self.storage_dir.glob("*.json")
            if path.is_file() and not is_artifact_file(path)
        ]

    def _generate_id(self) -> str:
        return str(uuid.uuid4())
