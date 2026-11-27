"""Plain text and Markdown ingestion.

Sections start at Markdown headings (or at the top of the file), and each
section is chunked on sentence boundaries like PDF pages are. Chunks are cited
as `file#s2` for the second section. A heading's optional closing run of `#`s
("## Leave ##") is dropped, as Markdown does. A `#` line inside a fenced code block
(a shell comment, say) is code, not a heading.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import Chunk
from .chunker import chunk_text

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
# The optional closing run of #s ("## Leave ##"), which is markup, not part of the heading text.
_CLOSING = re.compile(r"(^|\s+)#+\s*$")


def _heading_sentence(text: str) -> str:
    text = _CLOSING.sub("", text.strip()).strip().rstrip(".")
    return text + "." if text else ""


def split_sections(text: str) -> list[str]:
    sections: list[list[str]] = [[]]
    fence = ""  # the open fence ("```" or "~~~~", say), empty outside a code block
    for line in text.splitlines():
        f = _FENCE.match(line)
        if f and not fence:
            fence = f.group(1)
        elif f and f.group(1)[0] == fence[0] and len(f.group(1)) >= len(fence) and not line.strip()[len(f.group(1)):]:
            fence = ""  # a closing fence: same character, at least as long, nothing after it
        m = None if fence or f else _HEADING.match(line)
        if m and any(l.strip() for l in sections[-1]):
            sections.append([])
        # A heading becomes its own sentence so it stays attached to its section.
        # "### system ###" must reach the screen as "system.", like "### system", not as "system ###.".
        sections[-1].append(_heading_sentence(m.group(1)) if m else line)
    return [re.sub(r"\s+", " ", " ".join(s)).strip() for s in sections if any(l.strip() for l in s)]


def load_text(path: Path, groups: tuple[str, ...], title: str = "", max_words: int = 90) -> list[Chunk]:
    chunks: list[Chunk] = []
    # utf-8-sig drops the byte order mark a Windows editor writes first; kept, it hides the first heading.
    for s, section in enumerate(split_sections(path.read_text(encoding="utf-8-sig")), start=1):
        for i, text in enumerate(chunk_text(section, max_words=max_words)):
            chunks.append(Chunk(source=path.name, location=f"s{s}", index=i, text=text,
                                title=title or path.stem, groups=groups, kind="text"))
    return chunks
