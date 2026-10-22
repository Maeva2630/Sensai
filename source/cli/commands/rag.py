"""Terminal commands owned by the RAG feature."""

import shlex

from cli.session import CliCommandContext, CliSession, CommandDefinition
from rag.context import ArchiveIndexingResult, RAGIndexer
from rag.embeddings import EmbeddingError
from rag.ingestion import DocumentIngestionError
from rag.storage import MetadataFilter, VectorStoreError


MULTIPLE_INGEST_FLAGS = frozenset({"-m", "--multiple"})


def _handle_ingest_command(user_input: str, rag_indexer: RAGIndexer) -> None:
    """Index one document or multiple quoted paths without sending them to the model."""
    parts = user_input.split(maxsplit=1)
    if len(parts) != 2 or parts[0] != "/ingest" or not parts[1].strip():
        _print_ingest_usage()
        return

    arguments = parts[1].strip()
    try:
        argument_tokens = shlex.split(arguments)
    except ValueError as error:
        print(f"Invalid /ingest arguments: {error}")
        return

    if argument_tokens and argument_tokens[0] in MULTIPLE_INGEST_FLAGS:
        document_paths = argument_tokens[1:]
        if not document_paths:
            _print_ingest_usage()
            return
        successful_paths = sum(_index_document(path, rag_indexer) for path in document_paths)
        print(f"Multiple ingestion complete: {successful_paths}/{len(document_paths)} paths indexed.")
        return

    _index_document(arguments, rag_indexer)


def _print_ingest_usage() -> None:
    print("Usage: /ingest <path-to-document>")
    print("       /ingest --multiple <path> [<path> ...]")
    print("       Quote paths that contain spaces when using --multiple.")


def _index_document(document_path: str, rag_indexer: RAGIndexer) -> bool:
    """Index one path and return whether it completed successfully."""
    try:
        result = rag_indexer.index_document(document_path)
    except (DocumentIngestionError, EmbeddingError, VectorStoreError) as error:
        print(f"Document indexing failed: {error}")
        return False

    if isinstance(result, ArchiveIndexingResult):
        _print_archive_indexing_summary(result)
        return True

    print(
        f"Document indexed: {result.document.source_filename} "
        f"({result.chunk_count} chunks; {result.inserted_chunk_count} new chunks stored)."
    )
    return True


def _print_archive_indexing_summary(result: ArchiveIndexingResult) -> None:
    """Display successful, skipped, and failed archive entries separately."""
    archive = result.archive
    print(
        f"Archive indexed: {archive.archive_filename} "
        f"({len(result.indexed_documents)} files indexed; {result.chunk_count} chunks; "
        f"{result.inserted_chunk_count} new chunks stored)."
    )
    if result.indexed_documents:
        indexed_files = ", ".join(
            document_result.document.archive_entry_path or document_result.document.source_filename
            for document_result in result.indexed_documents
        )
        print(f"Indexed files: {indexed_files}")
    if archive.skipped_entries:
        print(f"Skipped unsupported files: {', '.join(archive.skipped_entries)}")
    for failure in archive.failed_entries:
        print(f"Failed to extract {failure.entry_path}: {failure.reason}")


def _handle_rag_command(user_input: str, rag_enabled: bool) -> bool:
    """Enable, disable, or report the status of document-grounded chat mode."""
    command, _, option = user_input.partition(" ")
    normalized_option = option.strip().lower()
    if command != "/rag" or normalized_option not in {"on", "off", "status"}:
        print("Usage: /rag <on|off|status>")
        return rag_enabled
    if normalized_option == "on":
        print("RAG mode enabled. Responses will use indexed documents only.")
        return True
    if normalized_option == "off":
        print("RAG mode disabled. The normal chatbot behavior is restored.")
        return False
    print(f"RAG mode is {'enabled' if rag_enabled else 'disabled'}.")
    return rag_enabled


def _handle_rag_filter_command(
    user_input: str,
    current_filter: MetadataFilter | None,
) -> MetadataFilter | None:
    """Set or clear the metadata restriction used by subsequent RAG questions."""
    _, _, arguments = user_input.partition("/rag filter")
    option, _, value = arguments.strip().partition(" ")
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if option == "clear" and not value.strip():
        print("RAG metadata filter cleared.")
        return None
    if option == "name" and values:
        metadata_filter = MetadataFilter(source_filenames=values)
        print(f"RAG filter set to source name: {', '.join(values)}")
        return metadata_filter
    if option == "category" and values:
        allowed_categories = {"document", "code", "configuration", "markup"}
        normalized_values = tuple(item.lower() for item in values)
        invalid_values = sorted(set(normalized_values) - allowed_categories)
        if not invalid_values:
            metadata_filter = MetadataFilter(source_types=normalized_values)
            print(f"RAG filter set to category: {', '.join(normalized_values)}")
            return metadata_filter
        print(f"Unknown category: {', '.join(invalid_values)}")
        return current_filter
    print("Usage: /rag filter <name <source>[,<source>...]|category <type>[,<type>...]|clear>")
    return current_filter


def _handle_sources_command(user_input: str, show_sources: bool) -> bool:
    """Enable, disable, or report CLI-generated source attribution lines."""
    command, _, option = user_input.partition(" ")
    normalized_option = option.strip().lower()
    if command != "/sources" or normalized_option not in {"on", "off", "status"}:
        print("Usage: /sources <on|off|status>")
        return show_sources
    if normalized_option == "on":
        print("Source display enabled.")
        return True
    if normalized_option == "off":
        print("Source display disabled.")
        return False
    print(f"Source display is {'enabled' if show_sources else 'disabled'}.")
    return show_sources


def _dispatch_rag_filter(
    user_input: str,
    session: CliSession,
    _context: CliCommandContext,
) -> None:
    session.metadata_filter = _handle_rag_filter_command(user_input, session.metadata_filter)


def _dispatch_rag(
    user_input: str,
    session: CliSession,
    context: CliCommandContext,
) -> None:
    if not context.feature_config.rag_enabled:
        print("RAG is disabled in config.json.")
        return
    session.rag_enabled = _handle_rag_command(user_input, session.rag_enabled)


def _dispatch_ingest(
    user_input: str,
    _session: CliSession,
    context: CliCommandContext,
) -> None:
    _handle_ingest_command(user_input, context.rag_indexer)


def _dispatch_sources(
    user_input: str,
    session: CliSession,
    _context: CliCommandContext,
) -> None:
    session.show_sources = _handle_sources_command(user_input, session.show_sources)


# Specific command names must be listed before generic ones.
RAG_COMMANDS: tuple[CommandDefinition, ...] = (
    CommandDefinition(
        name="/rag filter",
        usage="/rag filter <name <source>[,<source>...]|category <type>[,<type>...]|clear>",
        description="Restrict RAG retrieval to selected source names or categories.",
        feature="RAG",
        handler=_dispatch_rag_filter,
    ),
    CommandDefinition(
        name="/rag",
        usage="/rag <on|off|status>",
        description="Enable, disable, or show the status of RAG mode.",
        feature="RAG",
        handler=_dispatch_rag,
    ),
    CommandDefinition(
        name="/ingest",
        usage="/ingest [--multiple] <path> [<path> ...]",
        description="Index one document, or multiple quoted paths with --multiple.",
        feature="RAG",
        handler=_dispatch_ingest,
    ),
    CommandDefinition(
        name="/sources",
        usage="/sources <on|off|status>",
        description="Show, hide, or display the status of RAG source attributions.",
        feature="RAG",
        handler=_dispatch_sources,
    ),
)
