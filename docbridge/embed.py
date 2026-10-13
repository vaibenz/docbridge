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


class GeminiEmbedder:
    """Gemini embeddings with retrieval task types. Needs GOOGLE_API_KEY or Vertex AI settings."""

    def __init__(self, model: str = "gemini-embedding-001", dim: int = 768, batch_size: int = 100):
        from google import genai

        self.client = genai.Client()
        self.model = model
        self.dim = dim
        self.batch_size = batch_size

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        from google.genai import types

        vectors = []
        for start in range(0, len(texts), self.batch_size):
            resp = self.client.models.embed_content(
                model=self.model,
                contents=texts[start:start + self.batch_size],
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim),
            )
            vectors.extend(e.values for e in resp.embeddings)
        # Truncated Gemini embeddings are not unit length, so normalise for cosine search.
        return _l2(np.asarray(vectors, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def has_gemini_credentials() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY")) or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true")


def default_embedder() -> Embedder:
    choice = os.environ.get("DOCBRIDGE_EMBEDDER")
    if choice == "hashing" or (choice is None and not has_gemini_credentials()):
        return HashingEmbedder()
    return GeminiEmbedder()
