"""Retrievers: dense, lexical (BM25), and a hybrid of the two.

Dense search finds passages that mean the same thing in different words
("holiday" for "annual leave"). Lexical search finds exact terms that an
embedding blurs: a policy number, a site name, a rare word. Neither wins
everywhere, so the hybrid runs both and merges the rankings with reciprocal
rank fusion (RRF):

    score(d) = sum over retrievers of 1 / (k + rank of d in that retriever)

RRF uses ranks only, so it needs no calibration between BM25 scores (unbounded)
and cosine similarities (between -1 and 1), which is why it is the usual default.

Every retriever takes the asker's groups and only ever scores chunks shared with
one of them, so access control holds on every path, not just the vector store.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Protocol, Sequence

import numpy as np

from .embed import Embedder, normalize_tokens
from .models import Chunk, Hit
from .store.base import VectorStore


class Retriever(Protocol):
    def search(self, query: str, k: int, groups: Sequence[str]) -> list[Hit]: ...


class DenseRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store, self.embedder = store, embedder

    def search(self, query: str, k: int, groups: Sequence[str]) -> list[Hit]:
        hits = self.store.search(self.embedder.embed_query(query), k, groups)
        for h in hits:
            h.meta["dense_score"] = h.score
        return hits


class BM25Index:
    """Okapi BM25 over chunks, held in memory as an inverted index.

    Postings are numpy arrays, so a query adds each term's contribution to a
    score vector in one vectorised step instead of looping over documents.
    Chunks can be added and removed by id, so incremental ingestion keeps the
    index in step with the vector store.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self._chunks: dict[str, Chunk] = {}
        self._dirty = True

    @classmethod
    def from_chunks(cls, chunks: Iterable[Chunk], **kw) -> "BM25Index":
        index = cls(**kw)
        index.add(chunks)
        return index

    def add(self, chunks: Iterable[Chunk]) -> None:
        for c in chunks:
            self._chunks[c.chunk_id] = c
        self._dirty = True

    def remove(self, chunk_ids: Iterable[str]) -> None:
        for cid in chunk_ids:
            self._chunks.pop(cid, None)
        self._dirty = True

    def __len__(self) -> int:
        return len(self._chunks)

    def _build(self) -> None:
        self._ids = list(self._chunks)
        n = len(self._ids)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = np.zeros(n, dtype=np.float32)
        group_rows: dict[str, list[int]] = defaultdict(list)
        for i, cid in enumerate(self._ids):
            chunk = self._chunks[cid]
            toks = normalize_tokens(f"{chunk.title} {chunk.text}")
            lengths[i] = len(toks)
            counts: dict[str, int] = defaultdict(int)
            for t in toks:
                counts[t] += 1
            for t, c in counts.items():
                postings[t].append((i, c))
            for g in chunk.groups:
                group_rows[g].append(i)
        avg = float(lengths.mean()) if n else 1.0
        norm = self.k1 * (1 - self.b + self.b * lengths / max(avg, 1e-9))
        self._postings = {}
        for t, plist in postings.items():
            rows = np.fromiter((p[0] for p in plist), dtype=np.int64, count=len(plist))
            tf = np.fromiter((p[1] for p in plist), dtype=np.float32, count=len(plist))
            idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            # Precompute the full per posting weight: idf * tf * (k1 + 1) / (tf + norm)
            self._postings[t] = (rows, (idf * tf * (self.k1 + 1) / (tf + norm[rows])).astype(np.float32))
        self._group_rows = {g: np.asarray(r, dtype=np.int64) for g, r in group_rows.items()}
        self._n = n
        self._dirty = False

    def _allowed(self, groups: Sequence[str]) -> np.ndarray:
        mask = np.zeros(self._n, dtype=bool)
        for g in groups:
            rows = self._group_rows.get(g)
            if rows is not None:
                mask[rows] = True
        return mask

    def search(self, query: str, k: int, groups: Sequence[str]) -> list[Hit]:
        if self._dirty:
            self._build()
        if self._n == 0:
            return []
        scores = np.zeros(self._n, dtype=np.float32)
        for t in dict.fromkeys(normalize_tokens(query)):
            entry = self._postings.get(t)
            if entry is not None:
                np.add.at(scores, entry[0], entry[1])
        scores[~self._allowed(groups)] = 0.0
        candidates = np.flatnonzero(scores > 0)
        if candidates.size == 0:
            return []
        if candidates.size > k:
            candidates = candidates[np.argpartition(-scores[candidates], k - 1)[:k]]
        order = candidates[np.argsort(-scores[candidates], kind="stable")]
        return [Hit(self._chunks[self._ids[i]], float(scores[i]), {"bm25_score": float(scores[i])}) for i in order]


def rrf_fuse(rankings: Sequence[Sequence[Hit]], k: int, rrf_k: int = 60,
             weights: Sequence[float] | None = None) -> list[Hit]:
    """Reciprocal rank fusion of several ranked lists of hits."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[str, float] = defaultdict(float)
    best: dict[str, Hit] = {}
    for ranking, w in zip(rankings, weights):
        for rank, hit in enumerate(ranking, start=1):
            cid = hit.chunk.chunk_id
            fused[cid] += w / (rrf_k + rank)
            if cid in best:
                best[cid].meta.update(hit.meta)
            else:
                best[cid] = Hit(hit.chunk, 0.0, dict(hit.meta))
    order = sorted(fused, key=lambda cid: -fused[cid])[:k]
    out = []
    for cid in order:
        h = best[cid]
        h.score = fused[cid]
        h.meta["rrf_score"] = fused[cid]
        out.append(h)
    return out


class HybridRetriever:
    """Runs every retriever to `depth`, then fuses the rankings with RRF."""

    def __init__(self, retrievers: Sequence[Retriever], depth: int = 50, rrf_k: int = 60,
                 weights: Sequence[float] | None = None):
        self.retrievers, self.depth, self.rrf_k, self.weights = list(retrievers), depth, rrf_k, weights

    def search(self, query: str, k: int, groups: Sequence[str]) -> list[Hit]:
        rankings = [r.search(query, max(self.depth, k), groups) for r in self.retrievers]
        return rrf_fuse(rankings, k, self.rrf_k, self.weights)


class LLMReranker:
    """Second stage: an LLM scores each of the top candidates for relevance, then
    results are reordered by that score (ties keep the first stage order).
    Only used with a real model; the offline stack has no reranker."""

    def __init__(self, first_stage: Retriever, llm, candidates: int = 20):
        self.first_stage, self.llm, self.candidates = first_stage, llm, candidates

    def search(self, query: str, k: int, groups: Sequence[str]) -> list[Hit]:
        hits = self.first_stage.search(query, max(self.candidates, k), groups)
        scored = self.llm.score_passages(query, [h.chunk.text for h in hits])
        for h, s in zip(hits, scored):
            h.meta["rerank_score"] = s
        order = sorted(range(len(hits)), key=lambda i: (-scored[i], i))
        return [hits[i] for i in order[:k]]
