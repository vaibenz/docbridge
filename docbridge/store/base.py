from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from ..models import Chunk, Hit


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...

    def delete(self, chunk_ids: Sequence[str]) -> None: ...

    def search(self, vector: np.ndarray, k: int, groups: Sequence[str]) -> list[Hit]:
        """Top k chunks by cosine similarity among chunks shared with any of `groups`."""
        ...

    def content_hashes(self) -> dict[str, str]: ...

    def count(self) -> int: ...

    def all_chunks(self) -> list[Chunk]:
        """Every stored chunk, used to build the lexical index."""
        ...
