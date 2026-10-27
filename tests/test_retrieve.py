import math

import numpy as np
import pytest

from docbridge.embed import normalize_tokens
from docbridge.models import Chunk, Hit
from docbridge.retrieve import BM25Index, HybridRetriever, rrf_fuse

DOCS = [
    Chunk("a.pdf", "p1", 0, "Annual leave is 18 days per year for full time staff."),
    Chunk("b.pdf", "p1", 0, "Sick leave is 14 days. Hospitalisation leave is 60 days."),
    Chunk("c.pdf", "p1", 0, "The merit budget for the salary review is 4.5 percent.", groups=("hr",)),
    Chunk("d.pdf", "p1", 0, "Laptops must lock after 5 minutes and use disk encryption."),
]


def reference_bm25(query, docs, k1=1.2, b=0.75):
    """BM25 written the slow, obvious way, to check the vectorised index against."""
    toks = [normalize_tokens(f"{d.title} {d.text}") for d in docs]
    avg = sum(map(len, toks)) / len(toks)
    out = []
    for t_doc in toks:
        s = 0.0
        for term in dict.fromkeys(normalize_tokens(query)):
            df = sum(1 for t in toks if term in t)
            if not df:
                continue
            idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
            tf = t_doc.count(term)
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(t_doc) / avg))
        out.append(s)
    return out


@pytest.mark.parametrize("query", ["annual leave days", "leave", "salary merit budget", "laptop encryption lock"])
def test_bm25_matches_reference(query):
    index = BM25Index.from_chunks(DOCS)
    expected = reference_bm25(query, DOCS)
    got = {h.chunk.source: h.score for h in index.search(query, 10, ["all", "hr"])}
    for doc, score in zip(DOCS, expected):
        assert got.get(doc.source, 0.0) == pytest.approx(score, rel=1e-5)


def test_bm25_respects_groups_and_updates():
    index = BM25Index.from_chunks(DOCS)
    assert index.search("merit budget", 5, ["all"]) == []
    assert index.search("merit budget", 5, ["hr"])[0].chunk.source == "c.pdf"
    index.remove([DOCS[0].chunk_id])
    assert all(h.chunk.source != "a.pdf" for h in index.search("annual leave", 5, ["all"]))
    index.add([DOCS[0]])
    assert index.search("annual leave", 1, ["all"])[0].chunk.source == "a.pdf"


def _hits(*names):
    return [Hit(Chunk(n, "p1", 0, n), 0.0) for n in names]


def test_rrf_arithmetic():
    fused = rrf_fuse([_hits("x", "y", "z"), _hits("y", "w")], k=4, rrf_k=60)
    scores = {h.chunk.source: h.score for h in fused}
    assert scores["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["x"] == pytest.approx(1 / 61)
    assert [h.chunk.source for h in fused][:2] == ["y", "x"]  # agreement beats one first place
    weighted = rrf_fuse([_hits("x"), _hits("w")], k=2, weights=[1.0, 2.0])
    assert weighted[0].chunk.source == "w"


class _Fixed:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, k, groups):
        return [h for h in self.hits if set(h.chunk.groups) & set(groups)][:k]


def test_hybrid_never_returns_what_a_retriever_filtered():
    secret = Hit(Chunk("s.pdf", "p1", 0, "secret", groups=("hr",)), 1.0)
    public = Hit(Chunk("p.pdf", "p1", 0, "public"), 0.5)
    hybrid = HybridRetriever([_Fixed([secret, public]), _Fixed([public])])
    assert [h.chunk.source for h in hybrid.search("q", 5, ["all"])] == ["p.pdf"]


def test_wordllama_captures_paraphrase():
    pytest.importorskip("wordllama")
    from docbridge.embed import WordLlamaEmbedder

    emb = WordLlamaEmbedder()
    docs = emb.embed_documents([DOCS[0].text, DOCS[3].text])
    q = emb.embed_query("How many vacation days do employees get?")
    assert float(docs[0] @ q) > float(docs[1] @ q)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-5)
