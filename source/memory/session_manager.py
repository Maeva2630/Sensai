import json
import uuid
from datetime import datetime
from pathlib import Path


class SessionManager:
    """Manage conversation sessions with JSON file storage."""

    def __init__(self, session_id: str | None = None, storage_dir: str = "sessions"):
        self.storage_dir = Path(storage_dir)
        self.session_id = session_id or self._generate_id()
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

    def save(self) -> str:
        """Save the session to a JSON file at sessions/<session_id>.json."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        session_path = self.storage_dir / f"{self.session_id}.json"

        data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "messages": self.messages,
        }

        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Session saved at {session_path}") # can be removed 
        return str(session_path)

    def load(self, session_id: str | None = None) -> list[dict]:
        """Load messages from sessions/<session_id>.json."""
        target_id = session_id or self.session_id
        session_path = self.storage_dir / f"{target_id}.json"

        if not session_path.exists():
            raise FileNotFoundError(f"Session {target_id} not found at {session_path}")

        with open(session_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.session_id = data.get("session_id", target_id)
        self.messages = data.get("messages", [])
        return self.messages

    def _generate_id(self) -> str:
        return str(uuid.uuid4())
