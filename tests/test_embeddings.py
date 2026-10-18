import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from rag.chunking import DocumentChunk
from rag.embeddings import (
    EmbeddedChunk,
    EmbeddingConfig,
    EmbeddingConnectionError,
    EmbeddingDimensionError,
    EmbeddingGenerator,
    EmbeddingInputError,
    EmbeddingModelError,
    EmbeddingResponseError,
)
from utilities.ollama_client import OllamaConnectionError, OllamaModelNotFoundError


def make_chunk(index: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"chunk-{index}",
        document_id="document-1",
        source_filename="notes.md",
        chunk_index=index,
        start_char=0,
        end_char=len(text),
        text=text,
    )


class TestEmbeddingGenerator(unittest.TestCase):
    def test_generates_one_embedding_for_each_non_empty_chunk(self):
        client = MagicMock()
        client.get_embedding.side_effect = [
            {"embeddings": [[0.1, 0.2, 0.3]]},
            {"embeddings": [[0.4, 0.5, 0.6]]},
        ]
        generator = EmbeddingGenerator(client=client)

        embedded = generator.embed_chunks([make_chunk(0, "First passage"), make_chunk(1, "Second passage")])

        self.assertEqual(len(embedded), 2)
        self.assertTrue(all(isinstance(item, EmbeddedChunk) for item in embedded))
        self.assertEqual(embedded[0].vector, (0.1, 0.2, 0.3))
        self.assertEqual(embedded[1].chunk.chunk_id, "chunk-1")
        client.get_embedding.assert_any_call("First passage")
        client.get_embedding.assert_any_call("Second passage")

    def test_generates_embedding_for_query(self):
        client = MagicMock()
        client.get_embedding.return_value = {"embedding": [0.1, 0.2]}
        generator = EmbeddingGenerator(config=EmbeddingConfig(model="custom-embed"), client=client)

        result = generator.embed_query("How does document retrieval work?")

        self.assertEqual(result.vector, (0.1, 0.2))
        self.assertEqual(result.model, "custom-embed")

    def test_skips_empty_chunks(self):
        client = MagicMock()
        embedded = EmbeddingGenerator(client=client).embed_chunks([make_chunk(0, " \n ")])

        self.assertEqual(embedded, [])
        client.get_embedding.assert_not_called()

    def test_rejects_empty_query(self):
        with self.assertRaises(EmbeddingInputError):
            EmbeddingGenerator(client=MagicMock()).embed_query(" \t")

    def test_rejects_invalid_ollama_response(self):
        client = MagicMock()
        client.get_embedding.return_value = {"embeddings": [["not-a-number"]]}

        with self.assertRaises(EmbeddingResponseError):
            EmbeddingGenerator(client=client).embed_chunks([make_chunk(0, "Passage")])

    def test_rejects_inconsistent_dimensions_without_mutating_chunks(self):
        client = MagicMock()
        client.get_embedding.side_effect = [
            {"embeddings": [[0.1, 0.2, 0.3]]},
            {"embeddings": [[0.1, 0.2]]},
        ]
        chunks = [make_chunk(0, "First passage"), make_chunk(1, "Second passage")]

        with self.assertRaises(EmbeddingDimensionError):
            EmbeddingGenerator(client=client).embed_chunks(chunks)

        self.assertEqual(chunks[0].text, "First passage")
        self.assertEqual(chunks[1].text, "Second passage")

    def test_converts_connection_error_to_a_clear_rag_error(self):
        client = MagicMock()
        client.get_embedding.side_effect = OllamaConnectionError("Connection refused")

        with self.assertRaises(EmbeddingConnectionError) as context:
            EmbeddingGenerator(client=client).embed_query("Question")

        self.assertIn("ollama serve", str(context.exception))

    def test_converts_missing_model_error_to_a_clear_rag_error(self):
        client = MagicMock()
        client.get_embedding.side_effect = OllamaModelNotFoundError("Model not found")

        with self.assertRaises(EmbeddingModelError) as context:
            EmbeddingGenerator(config=EmbeddingConfig(model="missing-model"), client=client).embed_query("Question")

        self.assertIn("ollama pull missing-model", str(context.exception))

    def test_configures_the_client_model_when_no_client_is_supplied(self):
        generator = EmbeddingGenerator(config=EmbeddingConfig(model="my-embedding-model"))

        self.assertEqual(generator.client.model, "my-embedding-model")


if __name__ == "__main__":
    unittest.main()
