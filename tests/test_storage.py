import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from rag.chunking import DocumentChunk
from rag.embeddings import EmbeddedChunk, QueryEmbedding
from rag.storage import InvalidVectorError, LocalVectorStore


def embedded_chunk(chunk_id: str, vector: tuple[float, ...], text: str = "Document passage") -> EmbeddedChunk:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id="document-1",
        source_filename="guide.md",
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        text=text,
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
        self.store.store_chunks([embedded_chunk("chunk-1", (1.0, 0.0), "Persistent knowledge")])

        reopened_store = LocalVectorStore(self.database_path)
        results = reopened_store.search(QueryEmbedding("Question", (1.0, 0.0), "test-embed"))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "chunk-1")
        self.assertEqual(results[0].source_filename, "guide.md")
        self.assertEqual(results[0].text, "Persistent knowledge")

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


if __name__ == "__main__":
    unittest.main()
