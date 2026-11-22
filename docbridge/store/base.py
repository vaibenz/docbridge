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


def check_upsert(chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
    """One vector per chunk. zip() would quietly drop the chunks an embedder returned no vector for:
    ingestion would report them embedded while neither dense nor BM25 search could ever find them."""
    if len(vectors) != len(chunks):
        raise ValueError(f"{len(vectors)} vectors for {len(chunks)} chunks")
