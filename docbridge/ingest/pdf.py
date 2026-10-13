"""Unstructured ingestion: PDF pages to chunks."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from ..models import Chunk
from .chunker import chunk_text


def extract_pages(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]


def strip_boilerplate(pages: list[str]) -> list[str]:
    """Removes lines that repeat on most pages, such as running headers and footers."""
    if len(pages) < 2:
        return pages
    counts = Counter(line.strip() for page in pages for line in set(page.splitlines()) if line.strip())
    repeated = {line for line, c in counts.items() if c >= max(2, len(pages) * 0.6)}
    return ["\n".join(l for l in page.splitlines() if l.strip() not in repeated) for page in pages]


def _join_lines(page: str) -> str:
    """Joins wrapped lines into prose while keeping short heading lines as their own sentence."""
    out: list[str] = []
    for line in page.splitlines():
        line = line.strip()
        if not line:
            continue
        is_heading = len(line.split()) <= 6 and not line.endswith((".", ",", ";", ":"))
        out.append(line + "." if is_heading else line)
    text = " ".join(out)
    return re.sub(r"\s+", " ", text)


def load_pdf(path: Path, groups: tuple[str, ...], title: str = "", max_words: int = 90) -> list[Chunk]:
    pages = strip_boilerplate(extract_pages(path))
    chunks: list[Chunk] = []
    for page_no, page in enumerate(pages, start=1):
        for i, text in enumerate(chunk_text(_join_lines(page), max_words=max_words)):
            chunks.append(Chunk(source=path.name, location=f"p{page_no}", index=i, text=text,
                                title=title or path.stem, groups=groups, kind="text"))
    return chunks
