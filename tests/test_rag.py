import os

import numpy as np
import pytest

from docbridge.evaluate import evaluate, groundedness, load_items, sentence_supported
from docbridge.llm import IDK
from docbridge.models import Chunk, Hit

from .conftest import ROOT


def test_store_filters_by_group(store, embedder):
    q = embedder.embed_query("merit budget salary review")
    assert all("hr" not in h.chunk.groups for h in store.search(q, 10, ["all"]))
    assert store.search(q, 1, ["hr", "all"])[0].chunk.source == "compensation_guidelines.pdf"


def test_direct_hit_answers_without_rewrite(pipeline):
    r = pipeline.ask("What is the nightly hotel cap in Europe?")
    assert r.rewrites == 0
    assert "350" in r.answer
    assert r.citations == ["travel_and_expense_policy.pdf#p1"]
    assert r.steps[-1] == "generate"


def test_unanswerable_question_rewrites_then_declines(pipeline):
    r = pipeline.ask("Does Harbourline offer a gym membership subsidy?")
    assert r.answer == IDK and r.citations == []
    assert r.rewrites == 2
    assert [s.split(":")[0] for s in r.steps] == [
        "retrieve", "grade", "rewrite", "retrieve", "grade", "rewrite", "retrieve", "grade", "fallback"]


def test_restricted_content_needs_the_group(pipeline):
    denied = pipeline.ask("What is the merit budget for this salary review?", ["all"])
    allowed = pipeline.ask("What is the merit budget for this salary review?", ["hr"])
    assert denied.answer == IDK
    assert all("hr" not in h.chunk.groups for h in denied.retrieved)
    assert "4.5 percent" in allowed.answer


def test_table_rows_are_retrievable(pipeline):
    r = pipeline.ask("What is the order cutoff at the Changi Airfreight site?")
    assert "4pm" in r.answer
    assert r.citations == ["warehouse_sla.csv#row:Changi Airfreight"]


def _hit(text):
    return Hit(Chunk("doc.pdf", "p1", 0, text), 1.0)


def test_groundedness_scoring():
    hits = [_hit("Passwords must be at least 14 characters long.")]
    assert groundedness("Passwords must be at least 14 characters long. [1]", hits) == 1.0
    # an uncited sentence and a fabricated one both count against the answer
    assert groundedness("Passwords must be 14 characters long. [1] They expire every week. [1]", hits) == 0.5
    assert groundedness("Passwords must be at least 14 characters long.", hits) == 0.0
    assert sentence_supported("14 characters minimum", ["Passwords must be at least 14 characters long."])


def test_eval_harness_thresholds(pipeline):
    report = evaluate(pipeline, load_items(ROOT / "eval" / "questions.jsonl"))
    s = report["summary"]
    assert s["questions"] == 26
    assert s["recall@5"] >= 0.9
    assert s["loop_recall@5"] >= 0.9
    assert s["answer_accuracy"] >= 0.9
    assert s["refusal_accuracy"] == 1.0
    assert s["access_leaks"] == 0


@pytest.mark.skipif(not os.environ.get("DOCBRIDGE_DATABASE_URL"), reason="needs Postgres with pgvector")
def test_pgvector_store_round_trip():
    from docbridge.store.pgvector import PgVectorStore

    store = PgVectorStore(os.environ["DOCBRIDGE_DATABASE_URL"], dim=3, table="docbridge_test")
    store.conn.execute("TRUNCATE docbridge_test")
    chunks = [Chunk("a.pdf", "p1", 0, "public", groups=("all",)), Chunk("b.pdf", "p1", 0, "secret", groups=("hr",))]
    vecs = np.array([[1, 0, 0], [0.9, 0.1, 0]], dtype=np.float32)
    store.upsert(chunks, vecs)
    store.upsert(chunks[:1], vecs[:1])  # upsert is idempotent
    assert store.count() == 2
    assert [h.chunk.source for h in store.search(np.array([1, 0, 0], dtype=np.float32), 5, ["all"])] == ["a.pdf"]
    assert len(store.search(np.array([1, 0, 0], dtype=np.float32), 5, ["all", "hr"])) == 2
    assert set(store.content_hashes()) == {c.chunk_id for c in chunks}
    assert sorted(c.source for c in store.all_chunks()) == ["a.pdf", "b.pdf"]
    store.delete([chunks[1].chunk_id])
    assert store.count() == 1
    store.conn.execute("DROP TABLE docbridge_test")
