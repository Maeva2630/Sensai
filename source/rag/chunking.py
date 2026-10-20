"""Readable, overlapping document chunks for the RAG pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from typing import List

from .ingestion import IngestedDocument


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration used to create bounded, overlapping text chunks."""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")


@dataclass(frozen=True)
class DocumentChunk:
    """A source-aware text passage ready for embedding and retrieval."""

    chunk_id: str
    document_id: str
    source_filename: str
    chunk_index: int
    start_char: int
    end_char: int
    text: str
    language: str | None = None
    source_type: str = "document"
    start_line: int = 1
    end_line: int = 1


@dataclass(frozen=True)
class _TextUnit:
    """An internal sentence-sized unit with its original character boundaries."""

    start_char: int
    end_char: int


class DocumentChunker:
    """Split extracted documents at paragraph and sentence boundaries.

    Paragraphs and sentences are preferred to fixed-width slicing so chunks stay
    readable. A very long sentence is split on whitespace only when necessary
    to honour the configured maximum size.
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, document: IngestedDocument) -> List[DocumentChunk]:
        """Create non-empty overlapping chunks from a normalized document."""
        if not document.text.strip():
            return []

        units = self._structured_units(document) or self._build_units(document.text)
        return self._chunk_with_units(document, units)

    def _chunk_with_units(
        self,
        document: IngestedDocument,
        units: List[_TextUnit],
    ) -> List[DocumentChunk]:
        """Build bounded, overlapping chunks from already selected boundaries."""
        chunks: List[DocumentChunk] = []
        chunk_start = units[0].start_char

        while chunk_start < len(document.text):
            chunk_end = self._find_chunk_end(units, chunk_start, len(document.text))
            chunk_index = len(chunks)
            chunks.append(
                DocumentChunk(
                    chunk_id=self._chunk_id(document.document_id, chunk_index, chunk_start, chunk_end),
                    document_id=document.document_id,
                    source_filename=document.source_filename,
                    chunk_index=chunk_index,
                    start_char=chunk_start,
                    end_char=chunk_end,
                    text=document.text[chunk_start:chunk_end],
                    language=document.language,
                    source_type=document.source_type,
                    start_line=self._line_number(document.text, chunk_start),
                    end_line=self._end_line_number(document.text, chunk_start, chunk_end),
                )
            )

            if chunk_end == len(document.text):
                break
            next_start = self._find_overlap_start(document.text, chunk_start, chunk_end)
            # A tiny chunk cannot overlap with itself while still moving forward.
            chunk_start = next_start if next_start > chunk_start else chunk_end

        return chunks

    def _structured_units(self, document: IngestedDocument) -> List[_TextUnit] | None:
        """Find conservative declaration boundaries for technical source files.

        These patterns intentionally do not parse or execute source code. They
        only improve chunk boundaries when a language has an obvious, readable
        line-level declaration. Unknown formats keep the generic chunker.
        """
        language = document.language
        if document.source_type not in {"code", "configuration", "markup"} or not language:
            return None

        starts: list[int] = [0]
        offset = 0
        for line in document.text.splitlines(keepends=True):
            if offset and self._is_structural_start(line, language):
                starts.append(offset)
            offset += len(line)

        if len(starts) == 1:
            return None
        starts.append(len(document.text))
        return [
            _TextUnit(start, end)
            for start, end in zip(starts, starts[1:])
            if document.text[start:end].strip()
        ]

    @staticmethod
    def _is_structural_start(line: str, language: str) -> bool:
        """Return whether a line is a conservative boundary for its language."""
        if language == "python":
            return bool(re.match(r"^(?:async\s+def|def|class)\s+", line))
        if language in {"c", "c++", "c/c++ header", "java", "go", "rust", "javascript", "typescript"}:
            if line[:1].isspace():
                return False
            return bool(
                re.match(r"^(?:class|struct|namespace|interface|enum)\b", line)
                or re.match(r"^[\w:<>,~*&\s]+\s+\w+\s*\([^;]*\)\s*(?:const\s*)?\{", line)
            )
        if language in {"xml", "svg", "html"}:
            return bool(re.match(r"^\s*<(?![/?!])", line))
        if language in {"ini", "toml"}:
            return bool(re.match(r"^\s*\[[^\]]+\]", line))
        if language in {"yaml", "json"}:
            return bool(re.match(r"^(?:[A-Za-z_][\w-]*|\"[^\"]+\")\s*:", line))
        if language in {"css", "sql", "shell"}:
            return bool(re.match(r"^\S.*(?:\{|;)$", line))
        return False

    def _find_chunk_end(self, units: List[_TextUnit], start_char: int, text_length: int) -> int:
        """Prefer the latest natural boundary that fits inside the chunk budget."""
        limit = min(start_char + self.config.chunk_size, text_length)
        natural_ends = [unit.end_char for unit in units if start_char < unit.end_char <= limit]
        if natural_ends:
            return natural_ends[-1]

        return self._last_whitespace_before_limit(units, start_char, limit, text_length)

    @staticmethod
    def _last_whitespace_before_limit(
        units: List[_TextUnit], start_char: int, limit: int, text_length: int
    ) -> int:
        candidate_ends = [unit.end_char for unit in units if start_char < unit.end_char <= limit]
        if candidate_ends:
            return candidate_ends[-1]
        return limit if limit < text_length else text_length

    def _build_units(self, text: str) -> List[_TextUnit]:
        units: List[_TextUnit] = []
        for paragraph in self._paragraph_units(text):
            for sentence in self._sentence_units(text, paragraph):
                units.extend(self._split_overlong_unit(text, sentence))
        return units

    @staticmethod
    def _paragraph_units(text: str) -> List[_TextUnit]:
        """Return non-empty paragraphs, separated by one or more blank lines."""
        paragraphs: List[_TextUnit] = []
        for match in re.finditer(r"(?:^|\n\s*\n)(.*?)(?=\n\s*\n|\Z)", text, re.DOTALL):
            paragraph = match.group(1)
            leading_whitespace = len(paragraph) - len(paragraph.lstrip())
            trailing_whitespace = len(paragraph) - len(paragraph.rstrip())
            start_char = match.start(1) + leading_whitespace
            end_char = match.end(1) - trailing_whitespace
            if start_char < end_char:
                paragraphs.append(_TextUnit(start_char, end_char))
        return paragraphs

    @staticmethod
    def _sentence_units(text: str, paragraph: _TextUnit) -> List[_TextUnit]:
        """Split a paragraph after sentence punctuation when a boundary exists."""
        paragraph_text = text[paragraph.start_char:paragraph.end_char]
        sentences: List[_TextUnit] = []
        for match in re.finditer(r".+?(?:[.!?]+(?=\s|\Z)|\Z)", paragraph_text, re.DOTALL):
            sentence = match.group(0)
            leading_whitespace = len(sentence) - len(sentence.lstrip())
            trailing_whitespace = len(sentence) - len(sentence.rstrip())
            start_char = paragraph.start_char + match.start() + leading_whitespace
            end_char = paragraph.start_char + match.end() - trailing_whitespace
            if start_char < end_char:
                sentences.append(_TextUnit(start_char, end_char))
        return sentences

    def _split_overlong_unit(self, text: str, unit: _TextUnit) -> List[_TextUnit]:
        """Split an oversized sentence at whitespace while preserving whole words."""
        if unit.end_char - unit.start_char <= self.config.chunk_size:
            return [unit]

        pieces: List[_TextUnit] = []
        start_char = unit.start_char
        while unit.end_char - start_char > self.config.chunk_size:
            limit = start_char + self.config.chunk_size
            split_at = text.rfind(" ", start_char + 1, limit + 1)
            if split_at <= start_char:
                split_at = text.find(" ", limit, unit.end_char)
                if split_at == -1:
                    pieces.append(_TextUnit(start_char, unit.end_char))
                    return pieces

            pieces.append(_TextUnit(start_char, split_at))
            start_char = split_at
            while start_char < unit.end_char and text[start_char].isspace():
                start_char += 1

        pieces.append(_TextUnit(start_char, unit.end_char))
        return pieces

    def _find_overlap_start(self, text: str, current_start: int, end_char: int) -> int:
        """Find a word boundary that retains roughly ``chunk_overlap`` characters."""
        if self.config.chunk_overlap == 0:
            return self._skip_whitespace(text, end_char)

        target = max(current_start + 1, end_char - self.config.chunk_overlap)
        whitespace_boundaries = list(re.finditer(r"\s+", text[current_start + 1:target + 1]))
        if whitespace_boundaries:
            return current_start + 1 + whitespace_boundaries[-1].end()
        return target

    @staticmethod
    def _skip_whitespace(text: str, start_char: int) -> int:
        while start_char < len(text) and text[start_char].isspace():
            start_char += 1
        return start_char

    @staticmethod
    def _line_number(text: str, character_offset: int) -> int:
        """Convert a zero-based character offset into a one-based line number."""
        return text.count("\n", 0, character_offset) + 1

    @classmethod
    def _end_line_number(cls, text: str, start_char: int, end_char: int) -> int:
        """Return the line containing the final non-whitespace character."""
        content_end = end_char
        while content_end > start_char and text[content_end - 1].isspace():
            content_end -= 1
        return cls._line_number(text, max(start_char, content_end - 1))

    @staticmethod
    def _chunk_id(document_id: str, chunk_index: int, start_char: int, end_char: int) -> str:
        identity = f"{document_id}:{chunk_index}:{start_char}:{end_char}"
        return sha256(identity.encode("utf-8")).hexdigest()
