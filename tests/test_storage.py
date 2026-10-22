import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from rag.chunking import DocumentChunk
from rag.embeddings import EmbeddedChunk, QueryEmbedding
from rag.storage import InvalidVectorError, LocalVectorStore, MetadataFilter, VectorStoreError


def embedded_chunk(
    chunk_id: str,
    vector: tuple[float, ...],
    text: str = "Document passage",
    *,
    language: str | None = None,
    start_line: int = 1,
    end_line: int = 1,
    source_type: str = "document",
    source_filename: str = "guide.md",
) -> EmbeddedChunk:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id="document-1",
        source_filename=source_filename,
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        text=text,
        language=language,
        source_type=source_type,
        start_line=start_line,
        end_line=end_line,
    )
    return EmbeddedChunk(chunk=chunk, vector=vector, model="test-embed")


class TestLocalVectorStore(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "rag.sqlite3"
        self.store = LocalVectorStore(self.database_path)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_chunks_remain_available_after_reopening_database(self):
        self.store.store_chunks(
            [
                embedded_chunk(
                    "chunk-1",
                    (1.0, 0.0),
                    "Persistent knowledge",
                    language="python",
                    start_line=4,
                    end_line=8,
                )
            ]
        )

        reopened_store = LocalVectorStore(self.database_path)
        results = reopened_store.search(QueryEmbedding("Question", (1.0, 0.0), "test-embed"))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "chunk-1")
        self.assertEqual(results[0].source_filename, "guide.md")
        self.assertEqual(results[0].text, "Persistent knowledge")
        self.assertEqual(results[0].language, "python")
        self.assertEqual((results[0].start_line, results[0].end_line), (4, 8))

    def test_search_orders_results_by_cosine_similarity(self):
        self.store.store_chunks(
            [
                embedded_chunk("best", (1.0, 0.0), "Best match"),
                embedded_chunk("middle", (0.8, 0.6), "Middle match"),
                embedded_chunk("last", (-1.0, 0.0), "Opposite meaning"),
            ]
        )

        results = self.store.search(QueryEmbedding("Question", (1.0, 0.0), "test-embed"), top_k=2)

        self.assertEqual([result.chunk_id for result in results], ["best", "middle"])
        self.assertGreater(results[0].similarity_score, results[1].similarity_score)

    def test_empty_knowledge_base_returns_no_results(self):
        results = self.store.search(QueryEmbedding("Question", (1.0, 0.0), "test-embed"))

        self.assertEqual(results, [])

    def test_zero_vector_has_a_safe_zero_similarity_score(self):
        self.store.store_chunks([embedded_chunk("zero", (0.0, 0.0), "No direction")])

        results = self.store.search(QueryEmbedding("Question", (1.0, 0.0), "test-embed"))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].similarity_score, 0.0)

    def test_duplicate_indexing_does_not_create_duplicate_rows(self):
        record = embedded_chunk("chunk-1", (1.0, 0.0))

        self.assertEqual(self.store.store_chunks([record]), 1)
        self.assertEqual(self.store.store_chunks([record]), 0)
        self.assertEqual(self.store.count_chunks(), 1)

    def test_invalid_record_does_not_partially_write_other_records(self):
        valid_record = embedded_chunk("valid", (1.0, 0.0))
        invalid_record = embedded_chunk("invalid", (float("nan"), 0.0))

        with self.assertRaises(InvalidVectorError):
            self.store.store_chunks([valid_record, invalid_record])

        self.assertEqual(self.store.count_chunks(), 0)

    def test_rejects_invalid_top_k(self):
        with self.assertRaises(ValueError):
            self.store.search(QueryEmbedding("Question", (1.0, 0.0), "test-embed"), top_k=0)

    def test_search_filters_chunks_by_source_filename_or_category(self):
        self.store.store_chunks(
            [
                embedded_chunk("guide", (1.0, 0.0), "Guide", source_type="document"),
                embedded_chunk(
                    "code",
                    (1.0, 0.0),
                    "Code",
                    source_type="code",
                    source_filename="main.py",
                ),
            ]
        )
        query = QueryEmbedding("Question", (1.0, 0.0), "test-embed")

        by_name = self.store.search(query, metadata_filter=MetadataFilter(source_filenames=("guide.md",)))
        by_category = self.store.search(query, metadata_filter=MetadataFilter(source_types=("code",)))
        no_match = self.store.search(query, metadata_filter=MetadataFilter(source_types=("markup",)))

        self.assertEqual([result.chunk_id for result in by_name], ["guide"])
        self.assertEqual([result.chunk_id for result in by_category], ["code"])
        self.assertEqual(no_match, [])

    def test_existing_database_is_migrated_with_line_metadata_columns(self):
        legacy_database_path = Path(self.temporary_directory.name) / "legacy.sqlite3"
        with sqlite3.connect(legacy_database_path) as connection:
            connection.execute(
                """
                CREATE TABLE chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    source_filename TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    embedding_model TEXT NOT NULL
                )
                """
            )

        LocalVectorStore(legacy_database_path)

        with sqlite3.connect(legacy_database_path) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(chunks)")}

        self.assertTrue({"language", "start_line", "end_line"}.issubset(columns))

    def test_corrupted_database_fails_with_a_clear_error(self):
        corrupted_database_path = Path(self.temporary_directory.name) / "corrupted.sqlite3"
        corrupted_database_path.write_bytes(b"not a SQLite database")

        with self.assertRaisesRegex(VectorStoreError, "corrupted or unreadable"):
            LocalVectorStore(corrupted_database_path)


if __name__ == "__main__":
    unittest.main()
