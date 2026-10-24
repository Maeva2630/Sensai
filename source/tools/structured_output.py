"""Structured JSON answers from the model: schema-constrained generation, validation and retry."""

import json
from typing import Any, Callable, Dict, List, Optional

from utilities.ollama_client import OllamaHTTPClient

DEFAULT_MAX_ATTEMPTS = 3
MAX_QUIZ_QUESTIONS = 20
DEFAULT_QUIZ_QUESTIONS = 5

QUIZ_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "questions": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_QUIZ_QUESTIONS,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "minLength": 1},
                    "choices": {"type": "array", "minItems": 2, "maxItems": 6, "items": {"type": "string"}},
                    "answer_index": {"type": "integer", "minimum": 0},
                    "explanation": {"type": "string"},
                },
                "required": ["question", "choices", "answer_index"],
            },
        },
    },
    "required": ["title", "questions"],
}


class StructuredOutputError(Exception):
    """The model could not produce a valid structured answer."""


def _matches_type(name: str, value: Any) -> bool:
    if name == "string":
        return isinstance(value, str)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "array":
        return isinstance(value, list)
    if name == "object":
        return isinstance(value, dict)
    if name == "null":
        return value is None
    return True  # unknown type name: no constraint


def validate_json(schema: Any, value: Any, path: str = "$") -> Optional[str]:
    """Check a value against a JSON schema. Return an error message, or None if valid.

    Supports the keywords the app needs: type, enum, properties, required,
    additionalProperties (false), items, minItems, maxItems, minimum, maximum and
    minLength. Nested objects and arrays are checked recursively.
    """
    if not isinstance(schema, dict):
        return None

    declared = schema.get("type")
    if declared is not None:
        names = declared if isinstance(declared, list) else [declared]
        if not any(_matches_type(name, value) for name in names):
            return f"{path} must be of type {' or '.join(str(n) for n in names)}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path} must be one of {schema['enum']}"

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                return f"{path} is missing required property '{name}'"
        for name, item in value.items():
            if name in properties:
                error = validate_json(properties[name], item, f"{path}.{name}")
                if error:
                    return error
            elif schema.get("additionalProperties") is False:
                return f"{path} has unexpected property '{name}'"
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            return f"{path} must have at least {schema['minItems']} items"
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            return f"{path} must have at most {schema['maxItems']} items"
        for index, item in enumerate(value):
            error = validate_json(schema.get("items"), item, f"{path}[{index}]")
            if error:
                return error
    elif isinstance(value, str):
        if len(value.strip()) < schema.get("minLength", 0):
            return f"{path} must not be empty"
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path} must be at least {schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{path} must be at most {schema['maximum']}"
    return None


def load_schema_file(path: str) -> Dict[str, Any]:
    """Read a JSON schema file. Raises StructuredOutputError with a readable message."""
    try:
        with open(path, encoding="utf-8") as schema_file:
            schema = json.load(schema_file)
    except OSError as e:
        raise StructuredOutputError(f"Could not read schema file '{path}': {e.strerror or e}") from e
    except json.JSONDecodeError as e:
        raise StructuredOutputError(f"Schema file '{path}' is not valid JSON: {e.msg}") from e
    if not isinstance(schema, dict):
        raise StructuredOutputError(f"Schema file '{path}' must contain a JSON object")
    return schema


def _answer_text(response: Any) -> str:
    try:
        content = response.json()["message"]["content"]
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise StructuredOutputError("Ollama returned an unreadable response.") from e
    if not isinstance(content, str):
        raise StructuredOutputError("Ollama returned an unreadable response.")
    return content


def generate_structured(
    client: OllamaHTTPClient,
    messages: List[Dict[str, Any]],
    schema: Dict[str, Any],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    extra_check: Optional[Callable[[Any], Optional[str]]] = None,
) -> Any:
    """Ask the model for JSON matching `schema` and return the parsed value.

    The schema is sent to Ollama, which constrains the output. The answer is still
    validated here: when it is invalid, the error is sent back to the model and the
    request is retried, up to `max_attempts` times. Ollama errors are not caught.
    """
    conversation = list(messages)
    last_error = "no attempt made"
    for _ in range(max(1, max_attempts)):
        text = _answer_text(client.complete_message(conversation, output_format=schema))
        try:
            value = json.loads(text)
        except json.JSONDecodeError as e:
            last_error = f"the answer is not valid JSON ({e.msg})"
        else:
            last_error = validate_json(schema, value) or (extra_check(value) if extra_check else None) or ""
            if not last_error:
                return value
        conversation = conversation + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": (
                f"Your answer is not valid: {last_error}. "
                "Answer again with only the corrected JSON, matching the schema."
            )},
        ]
    raise StructuredOutputError(f"The model did not produce a valid answer after {max(1, max_attempts)} attempts: {last_error}")


def check_quiz(quiz: Dict[str, Any]) -> Optional[str]:
    """Checks the JSON schema cannot express: every answer must point to a real choice."""
    for number, question in enumerate(quiz["questions"], start=1):
        if question["answer_index"] >= len(question["choices"]):
            return f"question {number}: answer_index {question['answer_index']} is not one of its {len(question['choices'])} choices"
    return None


def quiz_messages(topic: str, question_count: int, excerpts: Optional[str] = None) -> List[Dict[str, Any]]:
    """Messages asking for a quiz, grounded in course excerpts when they are given."""
    system = (
        "You are a teaching assistant who writes multiple-choice quizzes. "
        "Write in the language of the topic. Each question has one correct choice; "
        "answer_index is the 0-based position of the correct choice in choices."
    )
    if excerpts:
        system += (
            "\n\nBase every question only on the course excerpts below. "
            "Do not use outside knowledge.\n\nCourse excerpts:\n" + excerpts
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Write a quiz of {question_count} questions about: {topic}"},
    ]


def format_quiz(quiz: Dict[str, Any]) -> str:
    """Readable quiz followed by its answer key."""
    lines = [quiz["title"], ""]
    for number, question in enumerate(quiz["questions"], start=1):
        lines.append(f"{number}. {question['question']}")
        for letter, choice in zip("ABCDEF", question["choices"]):
            lines.append(f"   {letter}) {choice}")
        lines.append("")
    lines.append("Answer key:")
    for number, question in enumerate(quiz["questions"], start=1):
        letter = "ABCDEF"[question["answer_index"]]
        explanation = f" - {question['explanation']}" if question.get("explanation") else ""
        lines.append(f"{number}. {letter}{explanation}")
    return "\n".join(lines)
