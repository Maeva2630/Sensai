"""Local SQLite storage and cosine-similarity retrieval for RAG embeddings."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from .embeddings import EmbeddedChunk, QueryEmbedding


DEFAULT_TOP_K = 3


class VectorStoreError(Exception):
    """Base exception for local vector-store failures."""


class InvalidVectorError(VectorStoreError):
    """Raised when a vector cannot be safely stored or compared."""


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk with the score that explains its ranking."""

    document_id: str
    chunk_id: str
    source_filename: str
    chunk_index: int
    text: str
    similarity_score: float
    language: str | None = None
    source_type: str = "document"
    start_line: int = 1
    end_line: int = 1


@dataclass(frozen=True)
class MetadataFilter:
    """Optional source metadata restrictions applied before similarity ranking."""

    source_filenames: tuple[str, ...] = ()
    source_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(not filename.strip() for filename in self.source_filenames):
            raise ValueError("Source filename filters cannot be empty.")
        if any(not source_type.strip() for source_type in self.source_types):
            raise ValueError("Source type filters cannot be empty.")


class LocalVectorStore:
    """Persist embeddings locally and retrieve them through cosine similarity.

    SQLite is part of Python's standard library. A connection is opened per
    operation so the store can be reopened in a later application session.
    """

    def __init__(self, database_path: str | Path = "data/sensai_rag.sqlite3") -> None:
        self.database_path = Path(database_path)
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize_database()
        except OSError as error:
            raise VectorStoreError(
                f"Unable to prepare the local RAG database at '{self.database_path}'."
            ) from error
        except sqlite3.DatabaseError as error:
            raise VectorStoreError(
                f"The local RAG database at '{self.database_path}' is corrupted or unreadable."
            ) from error

    def store_chunks(self, embedded_chunks: Iterable[EmbeddedChunk]) -> int:
        """Persist new chunks and return how many rows were inserted.

        All records are validated before the transaction starts. If validation or
        insertion fails, SQLite rolls back the transaction and keeps the index
        in its previous consistent state.
        """
        records = list(embedded_chunks)
        self._validate_records(records)
        if not records:
            return 0

        with self._connect() as connection:
            cursor = connection.cursor()
            inserted_count = 0
            for record in records:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO chunks (
                        chunk_id, document_id, source_filename, chunk_index,
                        start_char, end_char, start_line, end_line, language, source_type, text, embedding_json,
                        embedding_dimension, embedding_model
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.chunk.chunk_id,
                        record.chunk.document_id,
                        record.chunk.source_filename,
                        record.chunk.chunk_index,
                        record.chunk.start_char,
                        record.chunk.end_char,
                        record.chunk.start_line,
                        record.chunk.end_line,
                        record.chunk.language,
                        record.chunk.source_type,
                        record.chunk.text,
                        json.dumps(record.vector),
                        len(record.vector),
                        record.model,
                    ),
                )
                inserted_count += cursor.rowcount
        return inserted_count

    def search(
        self,
        query: QueryEmbedding,
        top_k: int = DEFAULT_TOP_K,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        """Return the top ``k`` chunks with matching model and vector dimension."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        query_vector = self._validate_vector(query.vector)

        conditions = ["embedding_model = ?", "embedding_dimension = ?"]
        parameters: list[object] = [query.model, len(query_vector)]
        if metadata_filter and metadata_filter.source_filenames:
            placeholders = ", ".join("?" for _ in metadata_filter.source_filenames)
            conditions.append(f"source_filename IN ({placeholders})")
            parameters.extend(metadata_filter.source_filenames)
        if metadata_filter and metadata_filter.source_types:
            placeholders = ", ".join("?" for _ in metadata_filter.source_types)
            conditions.append(f"source_type IN ({placeholders})")
            parameters.extend(metadata_filter.source_types)

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT document_id, chunk_id, source_filename, chunk_index, start_line, end_line,
                       language, source_type, text, embedding_json
                FROM chunks
                WHERE """
                + " AND ".join(conditions),
                parameters,
            ).fetchall()

        results = [
            SearchResult(
                document_id=row["document_id"],
                chunk_id=row["chunk_id"],
                source_filename=row["source_filename"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                similarity_score=self._cosine_similarity(query_vector, self._decode_vector(row["embedding_json"])),
                language=row["language"],
                source_type=row["source_type"] or "document",
                start_line=row["start_line"] or 1,
                end_line=row["end_line"] or 1,
            )
            for row in rows
        ]
        return sorted(results, key=lambda result: (-result.similarity_score, result.chunk_id))[:top_k]

    def count_chunks(self) -> int:
        """Return the number of stored chunks, primarily for diagnostics and tests."""
        with self._connect() as connection:
            return connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def clear(self) -> int:
        """Delete every indexed chunk while preserving the local database structure."""
        try:
            with self._connect() as connection:
                cursor = connection.execute("DELETE FROM chunks")
                return cursor.rowcount
        except sqlite3.Error as error:
            raise VectorStoreError("Unable to clear the local RAG index.") from error

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        source_filename TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        start_char INTEGER NOT NULL,
                        end_char INTEGER NOT NULL,
                        start_line INTEGER NOT NULL DEFAULT 1,
                        end_line INTEGER NOT NULL DEFAULT 1,
                        language TEXT,
                        source_type TEXT NOT NULL DEFAULT 'document',
                        text TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    embedding_model TEXT NOT NULL
                )
                """
            )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(chunks)").fetchall()
            }
            for column, definition in (
                ("start_line", "INTEGER NOT NULL DEFAULT 1"),
                ("end_line", "INTEGER NOT NULL DEFAULT 1"),
                ("language", "TEXT"),
                ("source_type", "TEXT NOT NULL DEFAULT 'document'"),
            ):
                if column not in columns:
                    connection.execute(f"ALTER TABLE chunks ADD COLUMN {column} {definition}")
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding_lookup
                ON chunks (embedding_model, embedding_dimension)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(self.database_path)
        except sqlite3.Error as error:
            raise VectorStoreError(
                f"Unable to open the local RAG database at '{self.database_path}'."
            ) from error
        connection.row_factory = sqlite3.Row
        return connection

    @classmethod
    def _validate_records(cls, records: Sequence[EmbeddedChunk]) -> None:
        seen_chunk_ids: set[str] = set()
        for record in records:
            if not record.chunk.chunk_id or not record.chunk.text.strip():
                raise VectorStoreError("Stored chunks must have an identifier and non-empty text.")
            if record.chunk.chunk_id in seen_chunk_ids:
                raise VectorStoreError(f"Duplicate chunk id in one indexing operation: {record.chunk.chunk_id}")
            seen_chunk_ids.add(record.chunk.chunk_id)
            cls._validate_vector(record.vector)
            if not record.model.strip():
                raise VectorStoreError("An embedding model name is required for stored chunks.")

    @staticmethod
    def _validate_vector(vector: Sequence[float]) -> Tuple[float, ...]:
        if not vector:
            raise InvalidVectorError("An embedding vector cannot be empty.")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in vector):
            raise InvalidVectorError("Embedding vectors must contain only numeric values.")
        normalized = tuple(float(value) for value in vector)
        if not all(math.isfinite(value) for value in normalized):
            raise InvalidVectorError("Embedding vectors must contain only finite values.")
        return normalized

    @classmethod
    def _decode_vector(cls, serialized_vector: str) -> Tuple[float, ...]:
        try:
            return cls._validate_vector(json.loads(serialized_vector))
        except (json.JSONDecodeError, TypeError) as error:
            raise VectorStoreError("A stored embedding vector is corrupted.") from error

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise InvalidVectorError("Embedding vectors must have the same dimension to be compared.")
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
