from datetime import datetime
import re
from typing import Any, Dict

from tools.base import Tool


class GetTimeTool(Tool):
    name = "get_time"
    description = "Get the current local date and time."
    parameters = {"type": "object", "properties": {}}

    _REQUEST_PATTERN = re.compile(
        r"\b(time|date|clock|heure|horloge|quelle heure|quel jour|jour sommes)\b",
        re.IGNORECASE,
    )

    def execute(self, arguments: Dict[str, Any]) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def is_relevant_for(self, user_input: str) -> bool:
        """Offer this narrow tool only for explicit date or time requests."""
        return bool(self._REQUEST_PATTERN.search(user_input))
