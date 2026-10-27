"""The corrective RAG loop as a LangGraph state machine.

    retrieve -> grade --(relevant chunks)--> generate -> END
                  |
                  +--(none, retries left)--> rewrite -> retrieve
                  |
                  +--(none, no retries)----> fallback -> END

Grading every retrieved chunk before generation is what keeps answers grounded:
the generator only ever sees chunks a grader accepted, and when nothing passes,
the system says so instead of guessing.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Sequence, TypedDict

from langgraph.graph import END, START, StateGraph

from .embed import Embedder
from .llm import IDK, LLM
from .models import Hit
from .retrieve import BM25Index, DenseRetriever, HybridRetriever, Retriever
from .security import guard_answer
from .store.base import VectorStore

PUBLIC_GROUP = "all"


class RagState(TypedDict, total=False):
    question: str
    groups: list[str]
    query: str
    attempts: int
    hits: list[Hit]
    relevant: list[Hit]
    answer: str
    citations: list[str]
    removed: list[str]
    steps: list[str]


@dataclass
class RagResult:
    question: str
    answer: str
    citations: list[str]
    retrieved: list[Hit]
    relevant: list[Hit]
    rewrites: int
    final_query: str
    steps: list[str]
    latency_ms: float
    removed: list[str] = None

    @property
    def answered(self) -> bool:
        return self.answer != IDK


def build_graph(retriever: Retriever, llm: LLM, k: int = 5, max_rewrites: int = 2, guard: bool = True,
                allowed_domains: Sequence[str] = ()):
    def retrieve(state: RagState) -> RagState:
        groups = sorted(set(state.get("groups", [])) | {PUBLIC_GROUP})
        hits = retriever.search(state["query"], k, groups)
        return {"hits": hits, "steps": state.get("steps", []) + [f"retrieve:{state['query']}"]}

    def grade(state: RagState) -> RagState:
        relevant = [h for h in state["hits"] if llm.grade(state["question"], h.chunk.text)]
        return {"relevant": relevant, "steps": state["steps"] + [f"grade:{len(relevant)}/{len(state['hits'])}"]}

    def decide(state: RagState) -> str:
        if state["relevant"]:
            return "generate"
        return "rewrite" if state.get("attempts", 0) < max_rewrites else "fallback"

    def rewrite(state: RagState) -> RagState:
        attempts = state.get("attempts", 0)
        query = llm.rewrite(state["question"], attempts)
        return {"query": query, "attempts": attempts + 1, "steps": state["steps"] + ["rewrite"]}

    def generate(state: RagState) -> RagState:
        relevant = state["relevant"]
        answer = llm.generate(state["question"], relevant)
        steps = state["steps"] + ["generate"]
        removed: list[str] = []
        if guard:
            answer, removed = guard_answer(answer, allowed_domains)
            if removed:
                steps.append(f"guard:removed {len(removed)}")
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 0 < int(n) <= len(relevant)})
        return {"answer": answer, "citations": [relevant[n - 1].chunk.ref for n in cited],
                "removed": removed, "steps": steps}

    def fallback(state: RagState) -> RagState:
        return {"answer": IDK, "citations": [], "steps": state["steps"] + ["fallback"]}

    g = StateGraph(RagState)
    for name, fn in [("retrieve", retrieve), ("grade", grade), ("rewrite", rewrite),
                     ("generate", generate), ("fallback", fallback)]:
        g.add_node(name, fn)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", decide, {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    g.add_edge("fallback", END)
    return g.compile()


def default_retriever(store: VectorStore, embedder: Embedder, mode: str = "hybrid") -> Retriever:
    """dense, bm25 or hybrid. The BM25 index is built from what the store holds."""
    dense = DenseRetriever(store, embedder)
    if mode == "dense":
        return dense
    bm25 = BM25Index.from_chunks(store.all_chunks())
    if mode == "bm25":
        return bm25
    # Plain RRF: weights tuned on SQuAD favour BM25 heavily because SQuAD questions reuse the
    # paragraph's words; real questions paraphrase more, so the default does not bake that in.
    return HybridRetriever([bm25, dense])


class RagPipeline:
    def __init__(self, store: VectorStore, embedder: Embedder, llm: LLM, k: int = 5, max_rewrites: int = 2,
                 retriever: Retriever | None = None, mode: str = "hybrid", guard: bool = True,
                 allowed_domains: Sequence[str] = ()):
        self.store, self.embedder, self.llm, self.k = store, embedder, llm, k
        self.retriever = retriever or default_retriever(store, embedder, mode)
        self.graph = build_graph(self.retriever, llm, k=k, max_rewrites=max_rewrites, guard=guard,
                                 allowed_domains=allowed_domains)

    def retrieve_only(self, question: str, groups: Sequence[str]) -> list[Hit]:
        groups = sorted(set(groups) | {PUBLIC_GROUP})
        return self.retriever.search(question, self.k, groups)

    def ask(self, question: str, groups: Sequence[str] = ()) -> RagResult:
        start = time.perf_counter()
        state = self.graph.invoke({"question": question, "query": question, "groups": list(groups),
                                   "attempts": 0, "steps": []})
        return RagResult(
            question=question,
            answer=state["answer"],
            citations=state.get("citations", []),
            retrieved=state.get("hits", []),
            relevant=state.get("relevant", []),
            rewrites=state.get("attempts", 0),
            final_query=state.get("query", question),
            steps=state.get("steps", []),
            latency_ms=(time.perf_counter() - start) * 1000,
            removed=state.get("removed", []),
        )
