"""CLI commands owned by the Tools feature."""

import re
from dataclasses import dataclass

from cli.session import CliCommandContext, CliSession, CommandDefinition
from rag.embeddings import EmbeddingError
from rag.storage import VectorStoreError
from tools.base import ToolError
from tools.quiz_export import save_quiz
from tools.registry import format_tool_listing
from tools.structured_output import (
    DEFAULT_QUIZ_QUESTIONS,
    MAX_QUIZ_QUESTIONS,
    QUIZ_SCHEMA,
    StructuredOutputError,
    check_quiz,
    format_quiz,
    generate_structured,
    quiz_messages,
)
from utilities.ollama_client import OllamaError

MAX_EXCERPT_CHARACTERS = 4000
QUIZ_OPTIONS = (
    f"[--questions <1-{MAX_QUIZ_QUESTIONS}>] [--language <name>] "
    "[--save <path> [--no-answers] [--overwrite]] <topic>"
)
QUIZ_USAGE = f"Usage: /quiz {QUIZ_OPTIONS}"
DEFAULT_QUIZ_LANGUAGE = "English"

# A word, or text in double quotes (for a path with spaces). An apostrophe is an ordinary character.
_TOKEN = re.compile(r'"([^"]*)"|(\S+)')
_OPTION = re.compile(r"--[A-Za-z]")


class QuizUsageError(ValueError):
    """The `/quiz` command line is invalid. The message says what is wrong."""


def _dispatch_tools(
    _user_input: str,
    _session: CliSession,
    context: CliCommandContext,
) -> None:
    """List the tools available to the model, native and MCP."""
    tools = context.registry.list_tools() if context.registry is not None else []
    print(format_tool_listing(tools))


def _dispatch_permissions(
    _user_input: str,
    _session: CliSession,
    context: CliCommandContext,
) -> None:
    """Show which directories the agent may read or write."""
    lines = context.permission_checker.describe() if context.permission_checker is not None else []
    print("File access rules:" if lines else "No file access is allowed.")
    for line in lines:
        print(f"  {line}")


@dataclass(frozen=True)
class QuizRequest:
    topic: str
    question_count: int = DEFAULT_QUIZ_QUESTIONS
    save_path: str | None = None
    answers: bool = True
    overwrite: bool = False
    language: str | None = None


def _parse_quiz_arguments(user_input: str) -> QuizRequest:
    """Read `/quiz` and its options. Raise QuizUsageError when the input is invalid."""
    if user_input.count('"') % 2:
        raise QuizUsageError("a double quote is not closed")
    tokens = [quoted if quoted or not word else word for quoted, word in _TOKEN.findall(user_input)][1:]

    values: dict[str, str] = {}
    flags: set[str] = set()
    topic_words: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in ("--questions", "--save", "--language"):
            if token in values:
                raise QuizUsageError(f"{token} is given twice")
            if index + 1 >= len(tokens) or _OPTION.match(tokens[index + 1]):
                raise QuizUsageError(f"{token} needs a value")
            values[token] = tokens[index + 1]
            index += 2
        elif token in ("--no-answers", "--overwrite"):
            flags.add(token)
            index += 1
        elif _OPTION.match(token):
            raise QuizUsageError(f"unknown option {token}")
        else:
            topic_words.append(token)
            index += 1

    question_count = DEFAULT_QUIZ_QUESTIONS
    if "--questions" in values:
        value = values["--questions"]
        if not value.isdigit() or not 1 <= int(value) <= MAX_QUIZ_QUESTIONS:
            raise QuizUsageError(f"--questions must be a number from 1 to {MAX_QUIZ_QUESTIONS}")
        question_count = int(value)
    topic = " ".join(topic_words).strip()
    if not topic:
        raise QuizUsageError("a topic is required")
    if "--save" not in values and flags:
        raise QuizUsageError(f"{' and '.join(sorted(flags))} only make sense with --save")
    return QuizRequest(
        topic, question_count, values.get("--save"), "--no-answers" not in flags,
        "--overwrite" in flags, values.get("--language"),
    )


def _quiz_language(request: QuizRequest, context: CliCommandContext) -> str:
    """The language asked with --language, otherwise the preferred language of the profile."""
    if request.language:
        return request.language
    profile = context.profile_manager.data if context.profile_manager is not None else {}
    language = profile.get("preferred_language") if isinstance(profile, dict) else None
    return language.strip() if isinstance(language, str) and language.strip() else DEFAULT_QUIZ_LANGUAGE


def _course_excerpts(topic: str, session: CliSession, context: CliCommandContext) -> str | None:
    """Excerpts of the indexed documents about the topic, or None when RAG finds nothing."""
    provider = context.rag_context_provider
    if session.metadata_filter is None:
        prepared = provider.prepare(topic)
    else:
        prepared = provider.prepare(topic, session.metadata_filter)
    if not prepared.results:
        return None
    parts = [f"[{result.source_filename}]\n{result.text}" for result in prepared.results]
    return "\n\n".join(parts)[:MAX_EXCERPT_CHARACTERS]


def _dispatch_quiz(
    user_input: str,
    session: CliSession,
    context: CliCommandContext,
) -> None:
    """Generate a validated multiple-choice quiz, from the indexed course when RAG is on."""
    try:
        request = _parse_quiz_arguments(user_input)
    except QuizUsageError as error:
        print(f"Invalid /quiz command: {error}.")
        print(QUIZ_USAGE)
        return

    try:
        excerpts = None
        if session.rag_enabled:
            excerpts = _course_excerpts(request.topic, session, context)
            if excerpts is None:
                print("No indexed course excerpt matches this topic. Index a document with /ingest or turn RAG off.")
                return
        quiz = generate_structured(
            context.chat_client,
            quiz_messages(request.topic, request.question_count, excerpts, _quiz_language(request, context)),
            QUIZ_SCHEMA,
            extra_check=lambda quiz: check_quiz(quiz, request.question_count),
        )
    except (StructuredOutputError, OllamaError, EmbeddingError, VectorStoreError) as error:
        print(f"Quiz generation failed: {error}")
        return

    text = format_quiz(quiz)
    print(text)
    if context.quiz_store is not None:
        context.quiz_store.quiz = quiz
    _remember_quiz(user_input, text, context)
    if request.save_path is not None:
        _save_quiz(quiz, request, context)


def _remember_quiz(user_input: str, quiz_text: str, context: CliCommandContext) -> None:
    """Keep the quiz in the conversation, so the model can answer questions about it later."""
    for role, content in (("user", user_input), ("assistant", quiz_text)):
        context.messages.append({"role": role, "content": content})
        context.conversation_session.add_message(role, content)


def _save_quiz(quiz: dict, request: QuizRequest, context: CliCommandContext) -> None:
    """Export the quiz. A failure is reported; the quiz stays displayed and remembered."""
    try:
        save_quiz(quiz, request.save_path, context.permission_checker, request.answers, request.overwrite)
    except ToolError as error:
        print(f"Quiz not saved: {error}")
        return
    print(f"Quiz saved to '{request.save_path}'" + ("" if request.answers else " (without the answer key)."))


TOOL_COMMANDS: tuple[CommandDefinition, ...] = (
    CommandDefinition(
        name="/tools",
        usage="/tools",
        description="List the tools available to the model.",
        feature="Tools",
        handler=_dispatch_tools,
    ),
    CommandDefinition(
        name="/permissions",
        usage="/permissions",
        description="Show the directories the agent may read or write.",
        feature="Tools",
        handler=_dispatch_permissions,
    ),
    CommandDefinition(
        name="/quiz",
        usage=f"/quiz {QUIZ_OPTIONS}",
        description="Generate a multiple-choice quiz (from the indexed course when RAG is on) and optionally save it.",
        feature="Tools",
        handler=_dispatch_quiz,
    ),
)
