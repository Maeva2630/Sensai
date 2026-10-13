from datetime import datetime
from typing import Any, Dict

from tools.base import Tool


class GetTimeTool(Tool):
    name = "get_time"
    description = "Get the current local date and time."
    parameters = {"type": "object", "properties": {}}

    def execute(self, arguments: Dict[str, Any]) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
