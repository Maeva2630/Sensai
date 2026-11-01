from datetime import datetime
import re
from typing import Any, Dict

from tools.base import Tool


class GetTimeTool(Tool):
    name = "get_time"
    description = "Get the current local date, weekday, time and time zone of this computer."
    parameters = {"type": "object", "properties": {}}

    # Words about the present moment that make the tool worth offering besides a direct
    # question. A date that belongs to a document ("la date du contrôle") is not enough:
    # offering the tool there made the model call it instead of reading the excerpts.
    _NOW_PATTERN = re.compile(
        r"\b(aujourd'hui|maintenant|actuellement|ce soir|demain|hier|today|tomorrow|yesterday|right now"
        r"|combien de (jours|temps) (avant|reste|jusqu)|how (many days|long) (until|before|left))\b",
        re.IGNORECASE,
    )
    # Only a question about the present moment is claimed. "Quelle est la date du
    # contrôle ?" mentions a date but asks about a document, so RAG must still run.
    _CURRENT_TIME_PATTERN = re.compile(
        r"(quelle\s+heure\s+(est[- ]il|il\s+est)|il\s+est\s+quelle\s+heure|l'heure\s+(actuelle|qu'il\s+est)"
        r"|quel\s+jour\s+(sommes[- ]nous|on\s+est|est[- ]on|est[- ]ce\s+aujourd'hui)|on\s+est\s+quel\s+jour"
        r"|date\s+(d'aujourd'hui|du\s+jour|actuelle)|(la\s+)?date\s+de\s+ce\s+jour"
        r"|what\s+time\s+is\s+it|what's\s+the\s+time|current\s+(date|time)|today'?s\s+date"
        r"|what\s+day\s+is\s+(it|today)|what\s+is\s+the\s+date\s+today|time\s+is\s+it\s+now)",
        re.IGNORECASE,
    )

    def execute(self, arguments: Dict[str, Any]) -> str:
        # Weekday and time zone are spelled out so the model has nothing to guess:
        # without them small models invent a zone ("EST") or a day.
        now = datetime.now().astimezone()
        offset = now.strftime("%z")
        return (
            f"Local date and time on this computer: {now.strftime('%A %Y-%m-%d %H:%M:%S')} "
            f"(time zone {now.tzname()}, UTC{offset[:3]}:{offset[3:]})"
        )

    def is_relevant_for(self, user_input: str) -> bool:
        """Offer this narrow tool only for the current date or time, or a delay from now."""
        return self.claims_request(user_input) or bool(self._NOW_PATTERN.search(user_input))

    def claims_request(self, user_input: str) -> bool:
        """Only a question about the current date or time is answered without documents."""
        return bool(self._CURRENT_TIME_PATTERN.search(user_input))
