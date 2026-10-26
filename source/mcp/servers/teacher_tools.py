"""MCP server over stdio exposing classroom tools for the teacher scenario.

Run it standalone with `python3 source/mcp/servers/teacher_tools.py` and talk to it
with newline-delimited JSON-RPC messages. It only uses the standard library.

Tools:
- calculate_grade: weighted average of a list of marks
- class_statistics: mean, median, min, max and pass rate of a class
- start_timer / timer_status: a timer that never blocks the conversation
"""
import json
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Optional

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "teacher-tools", "version": "1.0"}

DEFAULT_MAX_MARK = 20.0
MAX_MARKS = 1000
MAX_TIMER_MINUTES = 480.0
MAX_TIMERS = 50

_MARKS_SCHEMA = {"type": "array", "items": {"type": "number"}, "minItems": 1}
_MAX_MARK_SCHEMA = {
    "type": "number",
    "description": f"Highest possible mark (default {DEFAULT_MAX_MARK:g}).",
}

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "calculate_grade",
        "description": "Compute a student's weighted average from their marks and coefficients.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "marks": {**_MARKS_SCHEMA, "description": "The student's marks."},
                "coefficients": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "One coefficient per mark. All equal to 1 when omitted.",
                },
                "max_mark": _MAX_MARK_SCHEMA,
            },
            "required": ["marks"],
        },
    },
    {
        "name": "class_statistics",
        "description": "Compute the mean, median, minimum, maximum and pass rate of a class.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "marks": {**_MARKS_SCHEMA, "description": "One mark per student."},
                "pass_mark": {
                    "type": "number",
                    "description": "Minimum mark to pass (default: half of max_mark).",
                },
                "max_mark": _MAX_MARK_SCHEMA,
            },
            "required": ["marks"],
        },
    },
    {
        "name": "start_timer",
        "description": "Start a countdown timer (for an exercise or an exam) and return its id immediately.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "minutes": {"type": "number", "description": "Duration in minutes."},
                "label": {"type": "string", "description": "What the timer is for."},
            },
            "required": ["minutes"],
        },
    },
    {
        "name": "timer_status",
        "description": "Return the remaining time of a timer started with start_timer.",
        "inputSchema": {
            "type": "object",
            "properties": {"timer_id": {"type": "integer", "description": "Id returned by start_timer."}},
            "required": ["timer_id"],
        },
    },
]


class ToolInputError(ValueError):
    """The arguments of a tool call are invalid; reported with `isError`."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(value) != float("inf")


def _number_list(arguments: Dict[str, Any], key: str, required: bool = True) -> Optional[List[float]]:
    value = arguments.get(key)
    if value is None:
        if required:
            raise ToolInputError(f"'{key}' is required")
        return None
    if not isinstance(value, list) or not value:
        raise ToolInputError(f"'{key}' must be a non-empty list of numbers")
    if len(value) > MAX_MARKS:
        raise ToolInputError(f"'{key}' accepts at most {MAX_MARKS} values")
    if not all(_is_number(item) for item in value):
        raise ToolInputError(f"'{key}' must only contain numbers")
    return [float(item) for item in value]


def _max_mark(arguments: Dict[str, Any]) -> float:
    value = arguments.get("max_mark", DEFAULT_MAX_MARK)
    if not _is_number(value) or value <= 0:
        raise ToolInputError("'max_mark' must be a positive number")
    return float(value)


def _checked_marks(arguments: Dict[str, Any], max_mark: float) -> List[float]:
    marks = _number_list(arguments, "marks")
    for mark in marks:
        if mark < 0 or mark > max_mark:
            raise ToolInputError(f"marks must be between 0 and {max_mark:g} (got {mark:g})")
    return marks


def _format(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def calculate_grade(arguments: Dict[str, Any]) -> str:
    max_mark = _max_mark(arguments)
    marks = _checked_marks(arguments, max_mark)
    coefficients = _number_list(arguments, "coefficients", required=False) or [1.0] * len(marks)
    if len(coefficients) != len(marks):
        raise ToolInputError("'coefficients' must have the same length as 'marks'")
    if any(coefficient < 0 for coefficient in coefficients):
        raise ToolInputError("coefficients must not be negative")
    total = sum(coefficients)
    if total == 0:
        raise ToolInputError("the sum of the coefficients must be greater than 0")

    average = sum(mark * coefficient for mark, coefficient in zip(marks, coefficients)) / total
    return f"Weighted average: {_format(average)}/{_format(max_mark)} ({len(marks)} marks)"


def class_statistics(arguments: Dict[str, Any]) -> str:
    max_mark = _max_mark(arguments)
    marks = _checked_marks(arguments, max_mark)
    pass_mark = arguments.get("pass_mark", max_mark / 2)
    if not _is_number(pass_mark) or not 0 <= pass_mark <= max_mark:
        raise ToolInputError(f"'pass_mark' must be a number between 0 and {max_mark:g}")

    passed = sum(1 for mark in marks if mark >= pass_mark)
    return (
        f"Students: {len(marks)}; mean: {_format(statistics.mean(marks))}/{_format(max_mark)}; "
        f"median: {_format(statistics.median(marks))}; min: {_format(min(marks))}; "
        f"max: {_format(max(marks))}; pass rate: {_format(100 * passed / len(marks))}% "
        f"(pass mark {_format(pass_mark)})"
    )


class TimerBoard:
    """Timers kept in memory. Starting one returns at once; the time is read on demand."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._timers: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1

    def start(self, arguments: Dict[str, Any]) -> str:
        minutes = arguments.get("minutes")
        if not _is_number(minutes) or not 0 < minutes <= MAX_TIMER_MINUTES:
            raise ToolInputError(f"'minutes' must be a number between 0 and {_format(MAX_TIMER_MINUTES)}")
        label = arguments.get("label", "")
        if not isinstance(label, str):
            raise ToolInputError("'label' must be a string")
        if len(self._timers) >= MAX_TIMERS:
            raise ToolInputError(f"too many timers (maximum {MAX_TIMERS})")

        timer_id = self._next_id
        self._next_id += 1
        self._timers[timer_id] = {"end": self._clock() + minutes * 60, "minutes": minutes, "label": label.strip()}
        return f"Timer {timer_id} started for {_format(minutes)} minutes{self._label(timer_id)}."

    def status(self, arguments: Dict[str, Any]) -> str:
        timer_id = arguments.get("timer_id")
        if isinstance(timer_id, bool) or not isinstance(timer_id, int):
            raise ToolInputError("'timer_id' must be an integer")
        if timer_id not in self._timers:
            raise ToolInputError(f"unknown timer {timer_id}")

        remaining = self._timers[timer_id]["end"] - self._clock()
        if remaining <= 0:
            return f"Timer {timer_id}{self._label(timer_id)} is finished."
        minutes, seconds = divmod(int(remaining), 60)
        return f"Timer {timer_id}{self._label(timer_id)}: {minutes} min {seconds:02d} s remaining."

    def _label(self, timer_id: int) -> str:
        label = self._timers[timer_id]["label"]
        return f" ({label})" if label else ""


class TeacherToolsServer:
    """Answers JSON-RPC requests. `handle` returns the response, or None for a notification."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        timers = TimerBoard(clock)
        self._handlers: Dict[str, Callable[[Dict[str, Any]], str]] = {
            "calculate_grade": calculate_grade,
            "class_statistics": class_statistics,
            "start_timer": timers.start,
            "timer_status": timers.status,
        }

    def handle(self, message: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(message, dict):
            return _error(None, -32600, "Invalid request")
        request_id = message.get("id")
        if request_id is None:
            return None  # notification such as notifications/initialized

        method = message.get("method")
        params = message.get("params")
        params = params if isinstance(params, dict) else {}

        if method == "initialize":
            return _result(request_id, {
                "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            })
        if method == "ping":
            return _result(request_id, {})
        if method == "tools/list":
            return _result(request_id, {"tools": TOOLS})
        if method == "tools/call":
            return self._call_tool(request_id, params)
        return _error(request_id, -32601, f"Method not found: {method}")

    def _call_tool(self, request_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        handler = self._handlers.get(name) if isinstance(name, str) else None
        if handler is None:
            return _error(request_id, -32602, f"Unknown tool: {name}")

        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            return _tool_result(request_id, "arguments must be an object", is_error=True)
        try:
            return _tool_result(request_id, handler(arguments))
        except ToolInputError as e:
            return _tool_result(request_id, f"Invalid arguments: {e}", is_error=True)


def _result(request_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _tool_result(request_id: Any, text: str, is_error: bool = False) -> Dict[str, Any]:
    return _result(request_id, {"content": [{"type": "text", "text": text}], "isError": is_error})


def main() -> None:
    server = TeacherToolsServer()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            response = _error(None, -32700, "Parse error")
        else:
            response = server.handle(message)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
