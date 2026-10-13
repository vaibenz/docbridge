"""Evaluation harness for retrieval and answers.

Each labelled question lists the gold references (page or table row) that
contain its answer, the groups the asker belongs to, and strings the answer
must contain. Questions with no gold references are unanswerable, either
because the information does not exist or because it is restricted to a group
the asker is not in; for those the correct behaviour is to decline.

Metrics
* recall@k        share of answerable questions with a gold reference in the top k chunks
* mrr             mean reciprocal rank of the first gold chunk in the top k
* loop recall     the same after the grade and rewrite loop (chunks passed to the generator)
* answer accuracy share of answerable questions whose answer contains every expected string
* groundedness    share of answer sentences supported by the chunks they cite
* refusal accuracy share of unanswerable questions the system declined
* access leaks    retrieved chunks outside the asker's groups (must be 0)
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from .embed import normalize_tokens
from .graph import PUBLIC_GROUP, RagPipeline
from .ingest.chunker import split_sentences
from .models import Hit

CITATION = re.compile(r"\s*\[(\d+)\]")


@dataclass
class EvalItem:
    id: str
    question: str
    groups: list[str]
    gold: list[str]
    expect: list[str]
    restricted: bool = False

    @property
    def answerable(self) -> bool:
        return bool(self.gold)


def load_items(path: str | Path) -> list[EvalItem]:
    return [EvalItem(**json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]


def first_gold_rank(hits: Sequence[Hit], gold: Sequence[str]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if hit.chunk.ref in gold:
            return rank
    return None


def sentence_supported(sentence: str, sources: Sequence[str], threshold: float = 0.6) -> bool:
    """A sentence is supported if most of its content words appear together in one cited chunk."""
    toks = set(normalize_tokens(sentence))
    if not toks:
        return True
    return any(len(toks & set(normalize_tokens(src))) / len(toks) >= threshold for src in sources)


def groundedness(answer: str, relevant: Sequence[Hit]) -> float | None:
    """Fraction of answer sentences supported by the chunks they cite. None for declined answers."""
    # Move trailing citations inside the sentence: "long. [1] Next" becomes "long [1]. Next"
    normalised = re.sub(r"([.!?])\s*((?:\[\d+\]\s*)+)", lambda m: f" {m.group(2).strip()}{m.group(1)} ", answer)
    sentences = [s for s in split_sentences(normalised) if s.strip()]
    if not sentences or not relevant:
        return None
    supported = 0
    for s in sentences:
        cited = [int(n) for n in CITATION.findall(s) if 0 < int(n) <= len(relevant)]
        sources = [relevant[n - 1].chunk.text for n in cited] or [h.chunk.text for h in relevant]
        if cited and sentence_supported(CITATION.sub("", s), sources):
            supported += 1
    return supported / len(sentences)


@dataclass
class ItemResult:
    id: str
    question: str
    groups: list[str]
    answerable: bool
    first_pass_rank: int | None
    loop_found_gold: bool
    answer: str
    answered: bool
    correct: bool | None
    grounded: float | None
    rewrites: int
    leaks: int
    latency_ms: float
    citations: list[str] = field(default_factory=list)


def evaluate(pipeline: RagPipeline, items: Sequence[EvalItem]) -> dict:
    results: list[ItemResult] = []
    for item in items:
        first_pass = pipeline.retrieve_only(item.question, item.groups)
        out = pipeline.ask(item.question, item.groups)
        allowed = set(item.groups) | {PUBLIC_GROUP}
        leaks = sum(1 for h in first_pass + out.retrieved if not (set(h.chunk.groups) & allowed))
        correct = None
        if item.answerable:
            correct = out.answered and all(e.lower() in out.answer.lower() for e in item.expect)
        results.append(ItemResult(
            id=item.id, question=item.question, groups=item.groups, answerable=item.answerable,
            first_pass_rank=first_gold_rank(first_pass, item.gold) if item.answerable else None,
            loop_found_gold=bool(first_gold_rank(out.relevant, item.gold)) if item.answerable else False,
            answer=out.answer, answered=out.answered, correct=correct,
            grounded=groundedness(out.answer, out.relevant) if out.answered else None,
            rewrites=out.rewrites, leaks=leaks, latency_ms=out.latency_ms, citations=out.citations,
        ))

    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    grounded = [r.grounded for r in results if r.grounded is not None]
    k = pipeline.k
    summary = {
        "questions": len(results),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        f"recall@{k}": _ratio(sum(r.first_pass_rank is not None for r in answerable), len(answerable)),
        "mrr": round(statistics.mean(1 / r.first_pass_rank if r.first_pass_rank else 0 for r in answerable), 3) if answerable else None,
        f"loop_recall@{k}": _ratio(sum(r.loop_found_gold for r in answerable), len(answerable)),
        "answer_accuracy": _ratio(sum(bool(r.correct) for r in answerable), len(answerable)),
        "groundedness": round(statistics.mean(grounded), 3) if grounded else None,
        "refusal_accuracy": _ratio(sum(not r.answered for r in unanswerable), len(unanswerable)),
        "access_leaks": sum(r.leaks for r in results),
        "questions_rewritten": sum(1 for r in results if r.rewrites),
        "latency_ms_p50": round(statistics.median(r.latency_ms for r in results), 1),
    }
    return {"summary": summary, "items": [asdict(r) for r in results]}


def _ratio(n: int, d: int) -> float | None:
    return round(n / d, 3) if d else None


def to_markdown(report: dict, k: int = 5) -> str:
    s = report["summary"]
    lines = ["# DocBridge evaluation", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {key} | {value} |" for key, value in s.items()]
    lines += ["", "| id | groups | first pass rank | rewrites | correct | grounded | answer |", "|---|---|---|---|---|---|---|"]
    for r in report["items"]:
        answer = r["answer"].replace("|", "/")
        if len(answer) > 90:
            answer = answer[:87] + "..."
        lines.append(f"| {r['id']} | {','.join(r['groups'])} | {r['first_pass_rank'] or '-'} | {r['rewrites']} | "
                     f"{'-' if r['correct'] is None else ('yes' if r['correct'] else 'no')} | "
                     f"{'-' if r['grounded'] is None else round(r['grounded'], 2)} | {answer} |")
    return "\n".join(lines) + "\n"
