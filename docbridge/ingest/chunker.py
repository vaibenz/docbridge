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


def _cap_sentence(sentence: str, max_words: int) -> list[str]:
    """A "sentence" longer than the whole budget (a PDF table flattened to text, a list with no full stops)
    is cut into consecutive runs of max_words words, so it cannot become one oversized chunk."""
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def chunk_text(text: str, max_words: int = 90, overlap_sentences: int = 1) -> list[str]:
    """No chunk is longer than max_words plus the overlapped sentence carried over from the one before."""
    sentences = [piece for s in split_sentences(text) for piece in _cap_sentence(s, max_words)]
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
