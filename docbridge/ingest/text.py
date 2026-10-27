"""Plain text and Markdown ingestion.

Sections start at Markdown headings (or at the top of the file), and each
section is chunked on sentence boundaries like PDF pages are. Chunks are cited
as `file#s2` for the second section.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import Chunk
from .chunker import chunk_text

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")


def split_sections(text: str) -> list[str]:
    sections: list[list[str]] = [[]]
    for line in text.splitlines():
        m = _HEADING.match(line)
        if m and any(l.strip() for l in sections[-1]):
            sections.append([])
        # A heading becomes its own sentence so it stays attached to its section.
        sections[-1].append(m.group(1).strip().rstrip(".") + "." if m else line)
    return [re.sub(r"\s+", " ", " ".join(s)).strip() for s in sections if any(l.strip() for l in s)]


def load_text(path: Path, groups: tuple[str, ...], title: str = "", max_words: int = 90) -> list[Chunk]:
    chunks: list[Chunk] = []
    for s, section in enumerate(split_sections(path.read_text(encoding="utf-8")), start=1):
        for i, text in enumerate(chunk_text(section, max_words=max_words)):
            chunks.append(Chunk(source=path.name, location=f"s{s}", index=i, text=text,
                                title=title or path.stem, groups=groups, kind="text"))
    return chunks
