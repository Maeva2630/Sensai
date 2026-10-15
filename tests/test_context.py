import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from zipfile import ZipFile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from rag.context import (
    ArchiveIndexingResult,
    RAGContextConfig,
    RAGContextProvider,
    RAGIndexer,
    format_source_attribution,
)
from rag.chunking import DocumentChunker
from rag.embeddings import EmbeddedChunk, QueryEmbedding
from rag.ingestion import ArchiveIngestionResult, DocumentIngestor, IngestedDocument
from rag.storage import LocalVectorStore, SearchResult
from rag.storage import MetadataFilter


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

    def test_metadata_filter_is_forwarded_to_vector_search(self):
        embedding_generator = MagicMock()
        embedding_generator.embed_query.return_value = QueryEmbedding("Question", (1.0, 0.0), "embed-model")
        vector_store = MagicMock()
        vector_store.count_chunks.return_value = 1
        vector_store.search.return_value = []
        metadata_filter = MetadataFilter(source_types=("code",))
        provider = RAGContextProvider(embedding_generator, vector_store)

        prepared = provider.prepare("Question", metadata_filter)

        self.assertEqual(prepared.results, [])
        self.assertIn("No relevant indexed excerpts", prepared.system_message)
        vector_store.search.assert_called_once_with(
            embedding_generator.embed_query.return_value,
            top_k=3,
            metadata_filter=metadata_filter,
        )


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

    def test_indexes_each_valid_archive_document_through_every_rag_stage(self):
        first = IngestedDocument("doc-1", "project.zip::guide.md", "archive::guide.md", "Guide", 5)
        second = IngestedDocument("doc-2", "project.zip::main.py", "archive::main.py", "Code", 4)
        archive = ArchiveIngestionResult(
            "project.zip",
            "/tmp/project.zip",
            [first, second],
            ["image.png"],
            [],
        )
        ingestor = MagicMock()
        ingestor.ingest_archive.return_value = archive
        chunker = MagicMock()
        chunker.chunk.side_effect = [[MagicMock()], [MagicMock()]]
        embedding_generator = MagicMock()
        embedding_generator.embed_chunks.side_effect = [[MagicMock()], [MagicMock()]]
        vector_store = MagicMock()
        vector_store.store_chunks.side_effect = [1, 1]
        indexer = RAGIndexer(ingestor, chunker, embedding_generator, vector_store)

        result = indexer.index_document("project.zip")

        self.assertIsInstance(result, ArchiveIndexingResult)
        self.assertEqual(result.archive, archive)
        self.assertEqual(len(result.indexed_documents), 2)
        self.assertEqual(result.chunk_count, 2)
        self.assertEqual(result.inserted_chunk_count, 2)
        ingestor.ingest_archive.assert_called_once_with("project.zip")
        self.assertEqual(chunker.chunk.call_args_list[0].args, (first,))
        self.assertEqual(chunker.chunk.call_args_list[1].args, (second,))

    def test_duplicate_archive_entries_do_not_create_duplicate_stored_chunks(self):
        class FakeEmbeddingGenerator:
            def embed_chunks(self, chunks):
                return [EmbeddedChunk(chunk, (1.0, 0.0), "test-embedding") for chunk in chunks]

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            archive_path = temporary_path / "duplicates.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("docs/one.txt", "Duplicate content")
                archive.writestr("docs/two.txt", "Duplicate content")

            vector_store = LocalVectorStore(temporary_path / "rag.sqlite3")
            indexer = RAGIndexer(
                DocumentIngestor(),
                DocumentChunker(),
                FakeEmbeddingGenerator(),
                vector_store,
            )

            result = indexer.index_document(archive_path)

            self.assertIsInstance(result, ArchiveIndexingResult)
            self.assertEqual(result.inserted_chunk_count, 1)
            self.assertEqual(vector_store.count_chunks(), 1)


if __name__ == "__main__":
    unittest.main()
