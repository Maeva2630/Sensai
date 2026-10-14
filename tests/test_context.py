import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from rag.context import RAGContextConfig, RAGContextProvider, RAGIndexer, format_source_attribution
from rag.embeddings import QueryEmbedding
from rag.ingestion import IngestedDocument
from rag.storage import SearchResult


class TestRAGContextProvider(unittest.TestCase):
    def test_source_attribution_groups_chunks_by_source(self):
        attribution = format_source_attribution(
            [
                SearchResult("doc-1", "chunk-1", "guide.md", 0, "First excerpt", 0.95),
                SearchResult("doc-1", "chunk-2", "guide.md", 1, "Second excerpt", 0.90),
                SearchResult("doc-2", "chunk-3", "notes.md", 0, "Third excerpt", 0.80),
            ]
        )

        self.assertEqual(
            attribution,
            "Sources: guide.md (chunks: chunk-1, chunk-2); notes.md (chunks: chunk-3)",
        )

    def test_source_attribution_is_empty_without_retrieved_chunks(self):
        self.assertEqual(format_source_attribution([]), "")

    def test_includes_retrieved_sources_and_respects_character_budget(self):
        embedding_generator = MagicMock()
        embedding_generator.embed_query.return_value = QueryEmbedding("Question", (1.0, 0.0), "embed-model")
        vector_store = MagicMock()
        vector_store.count_chunks.return_value = 1
        vector_store.search.return_value = [
            SearchResult("doc-1", "chunk-1", "guide.md", 0, "A useful excerpt about RAG.", 0.95),
            SearchResult("doc-2", "chunk-2", "notes.md", 1, "Another useful excerpt.", 0.80),
        ]
        provider = RAGContextProvider(
            embedding_generator,
            vector_store,
            RAGContextConfig(top_k=2, max_context_characters=600),
        )

        prepared = provider.prepare("Question")

        self.assertIn("guide.md", prepared.system_message)
        self.assertIn("A useful excerpt about RAG.", prepared.system_message)
        self.assertLessEqual(len(prepared.system_message), 600)
        embedding_generator.embed_query.assert_called_once_with("Question")
        vector_store.search.assert_called_once_with(embedding_generator.embed_query.return_value, top_k=2)

    def test_empty_knowledge_base_does_not_call_embedding_model(self):
        embedding_generator = MagicMock()
        vector_store = MagicMock()
        vector_store.count_chunks.return_value = 0
        provider = RAGContextProvider(embedding_generator, vector_store)

        prepared = provider.prepare("Question")

        self.assertEqual(prepared.results, [])
        self.assertIn("No relevant indexed excerpts", prepared.system_message)
        embedding_generator.embed_query.assert_not_called()


class TestRAGIndexer(unittest.TestCase):
    def test_indexes_document_through_every_rag_stage(self):
        document = IngestedDocument("doc-1", "guide.md", "/tmp/guide.md", "Useful text", 11)
        ingestor = MagicMock()
        ingestor.ingest.return_value = document
        chunker = MagicMock()
        chunks = [MagicMock(), MagicMock()]
        chunker.chunk.return_value = chunks
        embedding_generator = MagicMock()
        embedded_chunks = [MagicMock(), MagicMock()]
        embedding_generator.embed_chunks.return_value = embedded_chunks
        vector_store = MagicMock()
        vector_store.store_chunks.return_value = 2
        indexer = RAGIndexer(ingestor, chunker, embedding_generator, vector_store)

        result = indexer.index_document("guide.md")

        self.assertEqual(result.document, document)
        self.assertEqual(result.chunk_count, 2)
        self.assertEqual(result.inserted_chunk_count, 2)
        ingestor.ingest.assert_called_once_with("guide.md")
        chunker.chunk.assert_called_once_with(document)
        embedding_generator.embed_chunks.assert_called_once_with(chunks)
        vector_store.store_chunks.assert_called_once_with(embedded_chunks)


if __name__ == "__main__":
    unittest.main()
