import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from rag.chunking import ChunkingConfig, DocumentChunker
from rag.ingestion import IngestedDocument


def make_document(
    text: str,
    *,
    source_type: str = "document",
    language: str | None = None,
) -> IngestedDocument:
    return IngestedDocument(
        document_id="document-id",
        source_filename="guide.md",
        source_path=str(Path("/tmp/guide.md")),
        text=text,
        character_count=len(text),
        source_type=source_type,
        language=language,
    )


class TestDocumentChunker(unittest.TestCase):
    def setUp(self):
        self.config = ChunkingConfig(chunk_size=70, chunk_overlap=20)
        self.chunker = DocumentChunker(self.config)

    def test_splits_a_long_document_into_multiple_chunks_with_metadata(self):
        document = make_document(
            "SensAI uses local models. It can ingest documents for retrieval. "
            "RAG retrieves the most relevant passages."
        )

        chunks = self.chunker.chunk(document)

        self.assertGreater(len(chunks), 1)
        for index, chunk in enumerate(chunks):
            self.assertEqual(chunk.document_id, document.document_id)
            self.assertEqual(chunk.source_filename, "guide.md")
            self.assertEqual(chunk.chunk_index, index)
            self.assertEqual(chunk.text, document.text[chunk.start_char:chunk.end_char])
            self.assertTrue(chunk.text.strip())
            self.assertEqual(len(chunk.chunk_id), 64)

    def test_chunks_end_at_sentence_boundaries_when_sentences_fit(self):
        document = make_document(
            "First complete sentence. Second complete sentence. "
            "Third complete sentence. Fourth complete sentence."
        )

        chunks = self.chunker.chunk(document)

        self.assertTrue(all(chunk.text.endswith((".", "!", "?")) for chunk in chunks))

    def test_adjacent_chunks_share_readable_overlap(self):
        document = make_document(
            "First sentence is here. Second sentence is here. "
            "Third sentence is here. Fourth sentence is here."
        )

        chunks = self.chunker.chunk(document)

        self.assertGreater(len(chunks), 1)
        for previous, current in zip(chunks, chunks[1:]):
            self.assertLess(current.start_char, previous.end_char)
            overlap = document.text[current.start_char:previous.end_char]
            self.assertTrue(overlap.strip())

    def test_does_not_create_empty_chunks_for_whitespace_only_text(self):
        chunks = self.chunker.chunk(make_document(" \n\t "))

        self.assertEqual(chunks, [])

    def test_splits_oversized_sentences_on_whitespace(self):
        text = "word " * 40
        chunks = DocumentChunker(ChunkingConfig(chunk_size=30, chunk_overlap=0)).chunk(make_document(text))

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 30 for chunk in chunks))
        self.assertTrue(all(not chunk.text.startswith(" ") for chunk in chunks))
        self.assertTrue(all(not chunk.text.endswith(" ") for chunk in chunks))

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            ChunkingConfig(chunk_size=100, chunk_overlap=100)

    def test_python_keeps_small_top_level_functions_and_classes_together(self):
        document = make_document(
            "class Greeter:\n"
            "    def hello(self):\n"
            "        return 'hello'\n"
            "\n"
            "def add(left, right):\n"
            "    return left + right\n",
            source_type="code",
            language="python",
        )

        chunks = DocumentChunker(ChunkingConfig(chunk_size=90, chunk_overlap=20)).chunk(document)

        self.assertTrue(any("class Greeter:" in chunk.text and "return 'hello'" in chunk.text for chunk in chunks))
        self.assertTrue(any("def add(left, right):" in chunk.text and "return left + right" in chunk.text for chunk in chunks))
        self.assertTrue(all(chunk.language == "python" for chunk in chunks))
        self.assertEqual((chunks[0].start_line, chunks[0].end_line), (1, 3))

    def test_cpp_xml_and_configuration_use_readable_line_boundaries(self):
        cpp_document = make_document(
            "int add(int left, int right) {\n"
            "  return left + right;\n"
            "}\n"
            "\n"
            "int subtract(int left, int right) {\n"
            "  return left - right;\n"
            "}\n",
            source_type="code",
            language="c++",
        )
        xml_document = make_document(
            "<svg>\n<rect width=\"10\" />\n<circle r=\"5\" />\n</svg>",
            source_type="markup",
            language="svg",
        )
        config_document = make_document(
            "[database]\nhost = localhost\n\n[cache]\nenabled = true",
            source_type="configuration",
            language="ini",
        )
        chunker = DocumentChunker(ChunkingConfig(chunk_size=55, chunk_overlap=10))

        cpp_chunks = chunker.chunk(cpp_document)
        xml_chunks = chunker.chunk(xml_document)
        config_chunks = chunker.chunk(config_document)

        self.assertTrue(any("int add" in chunk.text and "return left + right" in chunk.text for chunk in cpp_chunks))
        self.assertTrue(any("<rect" in chunk.text for chunk in xml_chunks))
        self.assertTrue(any("[database]" in chunk.text and "host = localhost" in chunk.text for chunk in config_chunks))
        self.assertTrue(all(chunk.start_line <= chunk.end_line for chunk in cpp_chunks + xml_chunks + config_chunks))

    def test_long_function_respects_chunk_size_and_keeps_line_metadata(self):
        document = make_document(
            "def long_function():\n" + "    value = value + 1\n" * 20,
            source_type="code",
            language="python",
        )

        chunks = DocumentChunker(ChunkingConfig(chunk_size=80, chunk_overlap=20)).chunk(document)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 80 for chunk in chunks))
        self.assertTrue(all(chunk.start_line <= chunk.end_line for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
