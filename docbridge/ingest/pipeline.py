"""Discovers sources in a folder, parses them, embeds new or changed chunks and upserts them.

Access control comes from corpus/manifest.json, which lists the groups allowed
to read each source. Sources missing from the manifest default to the most
restrictive setting (no groups), so nothing leaks by accident.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..embed import Embedder
from ..models import Chunk
from ..store.base import VectorStore
from .pdf import load_pdf
from .tables import load_csv
from .text import load_text
from ..security import injection_score


@dataclass
class IngestReport:
    sources: int
    chunks: int
    embedded: int
    skipped_unchanged: int
    removed: int
    quarantined: list[dict] = field(default_factory=list)


def load_manifest(folder: Path) -> dict:
    path = folder / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


def parse_folder(folder: Path) -> list[Chunk]:
    manifest = load_manifest(folder)
    chunks: list[Chunk] = []
    for path in sorted(folder.iterdir()):
        spec = manifest.get(path.name, {})
        groups = tuple(spec.get("groups", ()))
        title = spec.get("title", "")
        if path.suffix.lower() == ".pdf":
            chunks.extend(load_pdf(path, groups, title))
        elif path.suffix.lower() == ".csv":
            chunks.extend(load_csv(path, groups, title))
        elif path.suffix.lower() in (".md", ".txt"):
            chunks.extend(load_text(path, groups, title))
    return chunks


def screen(chunks: list[Chunk]) -> tuple[list[Chunk], list[dict]]:
    """Splits chunks into those safe to index and those that look like prompt injection.
    Quarantined chunks are not indexed; the report lists them for a person to review."""
    keep, held = [], []
    for c in chunks:
        verdict = injection_score(c.text)
        if verdict.suspicious:
            held.append({"chunk_id": c.chunk_id, "reasons": list(verdict.reasons), "text": c.text[:200]})
        else:
            keep.append(c)
    return keep, held


def ingest_folder(folder: str | Path, embedder: Embedder, store: VectorStore, batch_size: int = 64,
                  screening: bool = True) -> IngestReport:
    folder = Path(folder)
    chunks = parse_folder(folder)
    quarantined: list[dict] = []
    if screening:
        chunks, quarantined = screen(chunks)
    existing = store.content_hashes()
    todo = [c for c in chunks if existing.get(c.chunk_id) != c.content_hash]
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        vectors = embedder.embed_documents([f"{c.title}\n{c.text}" for c in batch])
        store.upsert(batch, vectors)
    live_ids = {c.chunk_id for c in chunks}
    stale = [cid for cid in existing if cid not in live_ids]
    store.delete(stale)
    return IngestReport(
        sources=len({c.source for c in chunks}),
        chunks=len(chunks),
        embedded=len(todo),
        skipped_unchanged=len(chunks) - len(todo),
        removed=len(stale),
        quarantined=quarantined,
    )
