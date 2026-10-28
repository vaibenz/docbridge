"""Retrieval benchmark on SQuAD 2.0 (dev): human written questions over Wikipedia.

    python bench/squad.py                      # offline retrievers
    python bench/squad.py --gemini             # Gemini too, on a sample of questions (needs GOOGLE_API_KEY)

The dev set has 35 articles split into 1,204 paragraphs. Each paragraph becomes
one chunk, and each of the 5,928 answerable questions has exactly one gold
chunk: the paragraph it was written about. That gives a retrieval benchmark
labelled by people, not by me.

One caveat matters for reading the numbers: SQuAD annotators wrote questions
while looking at the paragraph, so questions reuse its words more than real
users do. That flatters lexical search. The documents' own question set
(eval/questions.jsonl) and real usage are where paraphrase matters more.

Access control is checked on the same data: about a third of the articles are
restricted to a group the asker is not in, every retriever runs with the asker's
groups, and any restricted paragraph in any result list counts as a leak.

The dataset (CC BY-SA 4.0, Rajpurkar et al. 2018) is downloaded on first use
and cached in bench/.cache, not committed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docbridge.embed import HashingEmbedder  # noqa: E402
from docbridge.models import Chunk  # noqa: E402
from docbridge.retrieve import BM25Index, DenseRetriever, HybridRetriever  # noqa: E402
from docbridge.store.memory import MemoryStore  # noqa: E402

URL = "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v2.0.json"
CACHE = Path(__file__).resolve().parent / ".cache" / "squad-dev-v2.0.json"
KS = (1, 5, 20)


def load_squad():
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, CACHE)
    data = json.loads(CACHE.read_text())["data"]
    chunks, questions = [], []
    for a, article in enumerate(data):
        # Every third article is restricted to "team"; the benchmark asks as a user without it.
        groups = ("team",) if a % 3 == 2 else ("all",)
        for p, para in enumerate(article["paragraphs"]):
            chunk = Chunk(source=article["title"], location=f"p{p}", index=0, text=para["context"], groups=groups)
            chunks.append(chunk)
            for qa in para["qas"]:
                if not qa["is_impossible"]:
                    questions.append((qa["question"], chunk.chunk_id, groups == ("all",), a))
    return chunks, questions


def run(name, retriever, questions, groups, restricted_ids, depth=20):
    ranks, latencies, leaks = [], [], 0
    for q, gold, *_ in questions:
        t = time.perf_counter()
        hits = retriever.search(q, depth, groups)
        latencies.append((time.perf_counter() - t) * 1000)
        ids = [h.chunk.chunk_id for h in hits]
        leaks += sum(1 for i in ids if i in restricted_ids)
        ranks.append(ids.index(gold) + 1 if gold in ids else None)
    n = len(ranks)
    row = {"retriever": name}
    for k in KS:
        row[f"recall@{k}"] = round(sum(1 for r in ranks if r and r <= k) / n, 3)
    row["mrr@20"] = round(sum(1 / r for r in ranks if r) / n, 3)
    lat = sorted(latencies)
    row["p50 ms"] = round(statistics.median(lat), 2)
    row["p99 ms"] = round(lat[int(0.99 * (len(lat) - 1))], 2)
    row["leaks"] = leaks
    return row


def tune_weights(lexical, dense_retriever, tune, groups, restricted):
    """Grid search the dense weight and RRF k on the tuning articles, by MRR."""
    best = None
    for rrf_k in (10, 60):
        for w in (0.25, 0.5, 0.75, 1.0, 1.5):
            row = run("tune", HybridRetriever([lexical, dense_retriever], weights=[1.0, w], rrf_k=rrf_k),
                      tune, groups, restricted)
            if best is None or row["mrr@20"] > best[0]:
                best = (row["mrr@20"], w, rrf_k)
    print(f"tuned on {len(tune)} questions: dense weight {best[1]}, rrf k {best[2]} (mrr {best[0]})")
    return best[1], best[2]


class PrecomputedQueries:
    """Dense retrieval with query vectors embedded up front in batches, so a rate
    limited API is called a few dozen times instead of once per question."""

    def __init__(self, inner, vectors: dict):
        self.inner, self.vectors = inner, vectors

    def search(self, query, k, groups):
        hits = self.inner.store.search(self.vectors[query], k, groups)
        for h in hits:
            h.meta["dense_score"] = h.score
        return hits


def sample(items, n):
    """Every (len/n)th item: deterministic and spread across articles."""
    if n <= 0 or n >= len(items):
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def dense(chunks, embedder):
    store = MemoryStore()
    t = time.perf_counter()
    store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
    return DenseRetriever(store, embedder), time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemini", action="store_true", help="also run Gemini embeddings, on a sample of questions")
    ap.add_argument("--gemini-test", type=int, default=400, help="test questions in the Gemini sample")
    ap.add_argument("--gemini-tune", type=int, default=200, help="tuning questions in the Gemini sample")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results-squad.md")
    args = ap.parse_args()

    chunks, questions = load_squad()
    visible = [q for q in questions if q[2]]  # gold paragraph readable by the asker
    # Fusion weights are tuned on the first half of the articles and reported on the
    # second half, so the hybrid's numbers are not fitted to the questions they are measured on.
    tune = [q for q in visible if q[3] < 18]
    visible = [q for q in visible if q[3] >= 18]
    if args.limit:
        visible = visible[:args.limit]
    if args.gemini:
        # Gemini's free tier embeds about 100 texts a minute, so every row of this run uses the
        # same deterministic sample of questions, which keeps the rows comparable with each other.
        tune, visible = sample(tune, args.gemini_tune), sample(visible, args.gemini_test)
        if args.out.name == "results-squad.md":
            args.out = args.out.with_name("results-squad-gemini.md")
    restricted = {c.chunk_id for c in chunks if "all" not in c.groups}
    groups = ["all"]
    print(f"{len(chunks)} paragraphs ({len(restricted)} restricted), {len(tune)} tuning and {len(visible)} test "
          f"questions whose paragraph the asker may read")

    bm25 = BM25Index.from_chunks(chunks)
    systems = [("BM25", bm25)]
    hashing, _ = dense(chunks, HashingEmbedder())
    systems.append(("hashing (lexical stand in)", hashing))
    try:
        from docbridge.embed import WordLlamaEmbedder

        wl, secs = dense(chunks, WordLlamaEmbedder())
        print(f"WordLlama embedded {len(chunks)} paragraphs in {secs:.2f} s")
        systems += [("WordLlama dense", wl), ("hybrid: BM25 + WordLlama, plain RRF", HybridRetriever([bm25, wl]))]
        w, rrf_k = tune_weights(bm25, wl, tune, groups, restricted)
        systems.append((f"hybrid: BM25 + WordLlama, tuned (dense weight {w}, k {rrf_k})",
                        HybridRetriever([bm25, wl], weights=[1.0, w], rrf_k=rrf_k)))
    except ImportError:
        print("wordllama not installed, skipping")
    if args.gemini:
        from docbridge.embed import GeminiEmbedder

        print(f"Gemini: embedding {len(chunks)} paragraphs and {len(tune) + len(visible)} questions, about "
              f"{(len(chunks) + len(tune) + len(visible)) / 90:.0f} minutes on the free tier. Vectors are cached "
              f"in bench/.cache, so if it stops, rerun and it carries on.")
        embedder = GeminiEmbedder(cache_path=str(CACHE.parent / "embeddings.sqlite"))
        store_retriever, secs = dense(chunks, embedder)
        qs = [q[0] for q in tune + visible]
        qvec = dict(zip(qs, embedder.embed_queries(qs)))
        gem = PrecomputedQueries(store_retriever, qvec)
        print(f"Gemini embeddings ready in {secs:.0f} s")
        systems += [("Gemini dense", gem), ("hybrid: BM25 + Gemini, plain RRF", HybridRetriever([bm25, gem]))]
        w, rrf_k = tune_weights(bm25, gem, tune, groups, restricted)
        systems.append((f"hybrid: BM25 + Gemini, tuned (dense weight {w}, k {rrf_k})",
                        HybridRetriever([bm25, gem], weights=[1.0, w], rrf_k=rrf_k)))

    rows = []
    for name, r in systems:
        row = run(name, r, visible, groups, restricted)
        rows.append(row)
        print(row)

    cols = list(rows[0])
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    title = "Retrieval on SQuAD 2.0 dev" + (", with Gemini, on a sample of questions" if args.gemini else "")
    header = (f"# {title}\n\n{len(chunks)} paragraphs, {len(restricted)} of them restricted. "
              f"Fusion weights tuned on {len(tune)} questions from articles 1 to 18; every row below is measured on "
              f"the {len(visible)} answerable questions from articles 19 to 35, asked by a user who may read the gold "
              f"paragraph but not the restricted ones.\n\n")
    args.out.write_text(header + "\n".join(md) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
