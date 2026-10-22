"""End-to-end RAG flow tests using mocked Ollama responses only."""

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from cli.cli import CliDependencies, cli
from rag.context import RAGContextProvider, RAGIndexer
from rag.embeddings import EmbeddingConfig, EmbeddingGenerator
from rag.storage import LocalVectorStore


class TestRAGIntegration(unittest.TestCase):
    @patch("cli.cli.OllamaHTTPClient")
    def test_indexed_excerpt_is_injected_and_attributed_without_ollama(self, mock_chat_client_class):
        """Exercise ingestion through chat with fake embedding and chat services."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            document_path = workspace / "storage.md"
            document_path.write_text(
                "Embeddings are stored locally in a SQLite database.", encoding="utf-8"
            )

            embedding_client = MagicMock()
            embedding_client.get_embedding.return_value = {"embeddings": [[1.0, 0.0]]}
            embedding_generator = EmbeddingGenerator(
                EmbeddingConfig(model="test-embed"), client=embedding_client
            )
            vector_store = LocalVectorStore(workspace / "rag.sqlite3")
            indexer = RAGIndexer(
                embedding_generator=embedding_generator,
                vector_store=vector_store,
            )
            indexing_result = indexer.index_document(document_path)
            context_provider = RAGContextProvider(embedding_generator, vector_store)

            chat_client = MagicMock()
            chat_client.stream_message.return_value = [
                b'{"message": {"content": "Embeddings are stored in SQLite."}}'
            ]
            mock_chat_client_class.return_value = chat_client
            captured_output = io.StringIO()

            with patch("builtins.input", side_effect=["/rag on", "Where are embeddings stored?", "exit"]), patch(
                "sys.stdout", captured_output
            ):
                cli(
                    dependencies=CliDependencies(
                        rag_indexer=indexer,
                        rag_context_provider=context_provider,
                    ),
                )

        self.assertEqual(indexing_result.inserted_chunk_count, 1)
        self.assertEqual(embedding_client.get_embedding.call_count, 2)
        request_messages = chat_client.stream_message.call_args.args[0]
        self.assertEqual(request_messages[1]["role"], "system")
        self.assertIn("Source: storage.md", request_messages[1]["content"])
        self.assertIn("Embeddings are stored locally in a SQLite database.", request_messages[1]["content"])
        self.assertEqual(request_messages[-1], {"role": "user", "content": "Where are embeddings stored?"})
        self.assertIn("Assistant: Embeddings are stored in SQLite.", captured_output.getvalue())
        self.assertIn("Sources: storage.md", captured_output.getvalue())


if __name__ == "__main__":
    unittest.main()
