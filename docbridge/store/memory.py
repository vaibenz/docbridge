"""In memory vector store with exact cosine search. Used for tests and small demos."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..models import Chunk, Hit


class MemoryStore:
    def __init__(self):
        self._chunks: dict[str, Chunk] = {}
        self._vecs: dict[str, np.ndarray] = {}

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        for chunk, vec in zip(chunks, vectors):
            self._chunks[chunk.chunk_id] = chunk
            self._vecs[chunk.chunk_id] = np.asarray(vec, dtype=np.float32)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        for cid in chunk_ids:
            self._chunks.pop(cid, None)
            self._vecs.pop(cid, None)

    def search(self, vector: np.ndarray, k: int, groups: Sequence[str]) -> list[Hit]:
        allowed = set(groups)
        ids = [cid for cid, c in self._chunks.items() if allowed & set(c.groups)]
        if not ids:
            return []
        matrix = np.stack([self._vecs[cid] for cid in ids])
        scores = matrix @ np.asarray(vector, dtype=np.float32)
        order = np.argsort(-scores)[:k]
        return [Hit(self._chunks[ids[i]], float(scores[i])) for i in order]

    def content_hashes(self) -> dict[str, str]:
        return {cid: c.content_hash for cid, c in self._chunks.items()}

    def count(self) -> int:
        return len(self._chunks)
