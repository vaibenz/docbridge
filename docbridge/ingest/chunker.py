"""Sentence aware chunking with overlap.

Chunks are built from whole sentences up to a word budget, and each new chunk
repeats the last sentence of the previous one so an answer that straddles a
boundary is still retrievable from a single chunk.
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def chunk_text(text: str, max_words: int = 90, overlap_sentences: int = 1) -> list[str]:
    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    words = 0
    for sentence in sentences:
        n = len(sentence.split())
        if current and words + n > max_words:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            words = sum(len(s.split()) for s in current)
        current.append(sentence)
        words += n
    if current:
        tail = " ".join(current)
        if not chunks or tail != chunks[-1]:
            chunks.append(tail)
    return chunks
