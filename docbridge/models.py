from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of text with the metadata needed to cite and filter it."""

    source: str  # file name, for example employee_handbook.pdf
    location: str  # "p2" for a PDF page, "row:Tuas" for a table row
    index: int  # position within the location
    text: str
    title: str = ""
    groups: tuple[str, ...] = ("all",)
    kind: str = "text"  # "text" or "table"

    @property
    def chunk_id(self) -> str:
        return f"{self.source}#{self.location}#c{self.index}"

    @property
    def ref(self) -> str:
        """Citation reference at page or row level."""
        return f"{self.source}#{self.location}"

    @property
    def content_hash(self) -> str:
        # Everything that ends up in the stored row or the embedded text ("title\ntext"),
        # so a change to any of it, a manifest rename included, re-embeds the chunk.
        return hashlib.sha256(
            f"{self.chunk_id}|{self.title}|{self.kind}|{self.text}|{','.join(self.groups)}".encode()).hexdigest()[:16]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    meta: dict = field(default_factory=dict)
