"""CLI commands owned by the Tools feature."""

import shlex

from cli.session import CliCommandContext, CliSession, CommandDefinition
from rag.embeddings import EmbeddingError
from rag.storage import VectorStoreError
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
QUIZ_USAGE = f"Usage: /quiz [--questions <1-{MAX_QUIZ_QUESTIONS}>] <topic>"


def _dispatch_tools(
    _user_input: str,
    _session: CliSession,
    context: CliCommandContext,
) -> None:
    """List the tools available to the model, native and MCP."""
    tools = context.registry.list_tools() if context.registry is not None else []
    print(format_tool_listing(tools))


def _parse_quiz_arguments(user_input: str) -> tuple[int, str] | None:
    """Return (question count, topic) from `/quiz [--questions N] <topic>`, or None if invalid."""
    try:
        tokens = shlex.split(user_input)[1:]
    except ValueError:
        return None
    question_count = DEFAULT_QUIZ_QUESTIONS
    if tokens[:1] == ["--questions"]:
        if len(tokens) < 2 or not tokens[1].isdigit() or not 1 <= int(tokens[1]) <= MAX_QUIZ_QUESTIONS:
            return None
        question_count = int(tokens[1])
        tokens = tokens[2:]
    topic = " ".join(tokens).strip()
    return (question_count, topic) if topic else None


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
    parsed = _parse_quiz_arguments(user_input)
    if parsed is None:
        print(QUIZ_USAGE)
        return
    question_count, topic = parsed

    try:
        excerpts = None
        if session.rag_enabled:
            excerpts = _course_excerpts(topic, session, context)
            if excerpts is None:
                print("No indexed course excerpt matches this topic. Index a document with /ingest or turn RAG off.")
                return
        quiz = generate_structured(
            context.chat_client,
            quiz_messages(topic, question_count, excerpts),
            QUIZ_SCHEMA,
            extra_check=check_quiz,
        )
    except (StructuredOutputError, OllamaError, EmbeddingError, VectorStoreError) as error:
        print(f"Quiz generation failed: {error}")
        return
    print(format_quiz(quiz))


TOOL_COMMANDS: tuple[CommandDefinition, ...] = (
    CommandDefinition(
        name="/tools",
        usage="/tools",
        description="List the tools available to the model.",
        feature="Tools",
        handler=_dispatch_tools,
    ),
    CommandDefinition(
        name="/quiz",
        usage=f"/quiz [--questions <1-{MAX_QUIZ_QUESTIONS}>] <topic>",
        description="Generate a validated multiple-choice quiz, from the indexed course when RAG is on.",
        feature="Tools",
        handler=_dispatch_quiz,
    ),
)
