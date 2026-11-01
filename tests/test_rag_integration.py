"""End-to-end RAG flow tests using mocked Ollama responses only."""

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli.cli import CliDependencies, cli
from mcp.manager import MCPManager
from rag.context import RAGContextProvider, RAGIndexer
from support import use_temporary_working_directory_for_module
from rag.embeddings import EmbeddingConfig, EmbeddingGenerator
from rag.storage import LocalVectorStore


def setUpModule():
    # The CLI would start the servers declared in the repository's mcp_servers.json.
    patcher = patch("cli.cli.MCPManager.from_config_file", return_value=MCPManager([]))
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)
    # cli() saves its session under the working directory.
    use_temporary_working_directory_for_module()


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
        # SensAI's rules are a system message; the document text travels in the user's
        # turn, framed as data, next to the question.
        self.assertEqual(request_messages[1]["role"], "system")
        self.assertNotIn("Embeddings are stored locally in a SQLite database.", request_messages[1]["content"])
        question = request_messages[-1]
        self.assertEqual(question["role"], "user")
        self.assertIn("Source: storage.md", question["content"])
        self.assertIn("Embeddings are stored locally in a SQLite database.", question["content"])
        self.assertTrue(question["content"].startswith("<<<DOCUMENT EXCERPTS>>>"))
        self.assertIn("Question: Where are embeddings stored?", question["content"])
        self.assertIn("Reminder from SensAI", question["content"])
        self.assertIn("Assistant: Embeddings are stored in SQLite.", captured_output.getvalue())
        self.assertIn("Sources: storage.md", captured_output.getvalue())


if __name__ == "__main__":
    unittest.main()
