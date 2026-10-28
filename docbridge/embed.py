"""Embedding backends.

GeminiEmbedder is the production path. HashingEmbedder is a dependency free,
deterministic fallback (feature hashed unigrams and bigrams) used in tests and
keyless demos, so the whole pipeline runs anywhere.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol

import numpy as np

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how", "i", "if",
    "in", "is", "it", "its", "my", "of", "on", "or", "the", "to", "we", "what", "when", "which", "with", "you",
    "your", "our", "me", "this", "that", "there", "many", "much", "long", "after", "before", "any", "per",
}


def normalize_tokens(text: str) -> list[str]:
    out = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in _STOP:
            continue
        # Light suffix stripping so "reporting", "reported" and "reports" meet "report".
        if len(t) > 5 and t.endswith("ing"):
            t = t[:-3]
        elif len(t) > 4 and t.endswith("ed"):
            t = t[:-2]
        elif len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _l2(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class HashingEmbedder:
    def __init__(self, dim: int = 768):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        toks = normalize_tokens(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        counts: dict[str, int] = {}
        for f in feats:
            counts[f] = counts.get(f, 0) + 1
        for f, c in counts.items():
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += (1 if (h >> 63) & 1 else -1) * (1 + math.log(c))
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _l2(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2(self._vec(text)[None, :])[0]


class WordLlamaEmbedder:
    """A small learned embedding model that runs on the CPU with no network.

    WordLlama (256 dimensions) pools token embeddings distilled from a large
    language model. It is far weaker than Gemini embeddings, but it is real
    semantics ("vacation" lands near "annual leave"), its weights ship inside the
    pip package, and it embeds thousands of passages a second. That makes it the
    offline default and the dense half of the hybrid in tests and benchmarks.
    """

    def __init__(self, dim: int = 256):
        from pathlib import Path

        import wordllama
        from wordllama import WordLlama

        # The weights and tokenizer are inside the package; point the loader at them so it
        # never tries to download anything.
        self.model = WordLlama.load(cache_dir=Path(wordllama.__file__).parent, disable_download=True, trunc_dim=dim)
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _l2(np.asarray(self.model.embed(texts, norm=True), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class GeminiEmbedder:
    """Gemini embeddings with retrieval task types. Needs GOOGLE_API_KEY or Vertex AI settings.

    Paced to stay inside the free tier (DOCBRIDGE_GEMINI_EMBED_PER_MIN texts a
    minute, 90 by default), retries 429s for as long as the server asks, and with
    DOCBRIDGE_EMBED_CACHE set keeps every vector on disk, so a long run that hits a
    daily cap resumes where it stopped.
    """

    def __init__(self, model: str = "gemini-embedding-001", dim: int = 768, batch_size: int = 50,
                 cache_path: str | None = None, per_minute: int | None = None):
        from google import genai

        from .quota import EmbeddingCache, Pacer

        self.client = genai.Client()
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.pacer = Pacer(per_minute or int(os.environ.get("DOCBRIDGE_GEMINI_EMBED_PER_MIN", 90)))
        path = cache_path or os.environ.get("DOCBRIDGE_EMBED_CACHE")
        self.cache = EmbeddingCache(path) if path else None

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        from google.genai import types

        from .quota import EmbeddingCache, with_backoff

        keys = [EmbeddingCache.key(self.model, self.dim, task_type, t) for t in texts]
        found = self.cache.get_many(list(dict.fromkeys(keys))) if self.cache else {}
        todo = [i for i, k in enumerate(keys) if k not in found]
        for start in range(0, len(todo), self.batch_size):
            idx = todo[start:start + self.batch_size]
            self.pacer.acquire(len(idx))  # every text in a batch counts against the per minute quota
            resp = with_backoff(lambda: self.client.models.embed_content(
                model=self.model,
                contents=[texts[i] for i in idx],
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim),
            ))
            new = {keys[i]: np.asarray(e.values, dtype=np.float32) for i, e in zip(idx, resp.embeddings)}
            found.update(new)
            if self.cache:
                self.cache.put_many(new)
            if len(todo) > self.batch_size:
                print(f"  embedded {min(start + self.batch_size, len(todo))}/{len(todo)}", flush=True)
        # Truncated Gemini embeddings are not unit length, so normalise for cosine search.
        return _l2(np.stack([found[k] for k in keys]).astype(np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_QUERY")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def has_gemini_credentials() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY")) or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true")


def default_embedder() -> Embedder:
    """DOCBRIDGE_EMBEDDER picks gemini, wordllama or hashing; otherwise Gemini when
    credentials exist, else WordLlama when installed, else the hashing stand in."""
    choice = os.environ.get("DOCBRIDGE_EMBEDDER")
    if choice == "hashing":
        return HashingEmbedder()
    if choice == "gemini" or (choice is None and has_gemini_credentials()):
        return GeminiEmbedder()
    try:
        return WordLlamaEmbedder()
    except ImportError:
        if choice == "wordllama":
            raise
        return HashingEmbedder()
