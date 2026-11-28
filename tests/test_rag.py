import os

import numpy as np
import pytest

from docbridge.evaluate import evaluate, groundedness, load_items, sentence_supported, to_markdown
from docbridge.graph import RagPipeline
from docbridge.llm import IDK, OfflineLLM
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


class _Fixed:
    """A retriever and LLM that return what they are given, to drive the graph directly."""

    def __init__(self, hits, answer):
        self.hits, self.answer = hits, answer

    def search(self, query, k, groups):
        return list(self.hits)

    def grade(self, question, text):
        return True

    def rewrite(self, question, attempt):
        return question

    def generate(self, question, hits):
        return self.answer


def test_two_chunks_of_one_page_are_cited_once():
    hits = [Hit(Chunk("doc.pdf", "p1", 0, "First part."), 1.0), Hit(Chunk("doc.pdf", "p1", 1, "Second part."), 0.9),
            Hit(Chunk("doc.pdf", "p2", 0, "Other page."), 0.8)]
    fixed = _Fixed(hits, "First part. [2] Second part. [1] Other page. [3] Again. [1] Out of range. [9]")
    r = RagPipeline(None, None, fixed, retriever=fixed, guard=False).ask("q")
    assert r.citations == ["doc.pdf#p1", "doc.pdf#p2"]


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


def test_markdown_report_keeps_a_multi_line_answer_on_one_table_row():
    # Model answers are often several lines (a list of steps). A raw newline ended the table row early, so
    # the rest of the answer, and every row after it, rendered as loose text under a broken table.
    item = {"id": "q1", "groups": ["all"], "first_pass_rank": 1, "rewrites": 0, "correct": True,
            "grounded": 1.0, "answer": "Two steps:\n1. File the claim [1].\r\n2. Attach | the receipt."}
    md = to_markdown({"summary": {"questions": 1}, "items": [item]})
    rows = [line for line in md.splitlines() if line.startswith("| q1 ")]
    assert len(rows) == 1
    assert rows[0].endswith("| Two steps: 1. File the claim [1]. 2. Attach / the receipt. |")
    assert md.splitlines()[-1] == rows[0]


@pytest.mark.parametrize("k", [0, -1, 2.5, True])
def test_pipeline_refuses_a_k_below_one(store, embedder, k):
    # k=-1 used to reach the store as a slice, which returned every allowed chunk but the last.
    with pytest.raises(ValueError, match="k must be"):
        RagPipeline(store, embedder, OfflineLLM(), k=k)


def test_pipeline_refuses_a_negative_max_rewrites(store, embedder):
    with pytest.raises(ValueError, match="max_rewrites"):
        RagPipeline(store, embedder, OfflineLLM(), max_rewrites=-1)


@pytest.mark.parametrize("k", ["0", "-3", "five"])
def test_cli_eval_refuses_a_bad_k_before_ingesting(k, capsys):
    from docbridge.cli import main

    with pytest.raises(SystemExit) as e:
        main(["eval", "eval/questions.jsonl", "-k", k])
    assert e.value.code == 2
    assert "-k" in capsys.readouterr().err


def test_cli_groups_ignore_spaces_and_empty_entries():
    from docbridge.cli import parse_groups

    assert parse_groups("all, hr") == ["all", "hr"]
    assert parse_groups(" hr ,,") == ["hr"]
    assert parse_groups("") == []


def test_cli_ask_with_a_spaced_group_list_reaches_the_group(capsys):
    from docbridge.cli import main

    main(["ask", "What is the merit budget for this salary review?", "--groups", "all, hr"])
    out = capsys.readouterr().out
    assert "4.5 percent" in out and "compensation_guidelines.pdf" in out


@pytest.mark.parametrize("var, value", [("DOCBRIDGE_EMBEDDER", "hashng"), ("DOCBRIDGE_EMBEDDER", "openai"),
                                        ("DOCBRIDGE_LLM", "ofline"), ("DOCBRIDGE_LLM", "gpt")])
def test_an_unknown_backend_is_refused_not_replaced(monkeypatch, var, value):
    from docbridge.embed import default_embedder
    from docbridge.llm import default_llm

    monkeypatch.setenv(var, value)
    with pytest.raises(ValueError, match=var):
        default_embedder() if var == "DOCBRIDGE_EMBEDDER" else default_llm()


def test_backend_names_ignore_case_and_an_empty_value_means_unset(monkeypatch):
    from docbridge.embed import HashingEmbedder, default_embedder
    from docbridge.llm import default_llm

    monkeypatch.setenv("DOCBRIDGE_EMBEDDER", " Hashing ")
    monkeypatch.setenv("DOCBRIDGE_LLM", "OFFLINE")
    assert isinstance(default_embedder(), HashingEmbedder)
    assert isinstance(default_llm(), OfflineLLM)
    monkeypatch.setenv("DOCBRIDGE_LLM", "")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    assert isinstance(default_llm(), OfflineLLM)
