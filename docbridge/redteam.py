"""Prompt injection red team: run every attack against every defence configuration.

    docbridge redteam                    # offline models
    docbridge redteam --out redteam/report.md

Each attack in redteam/attacks.jsonl plants a document in the corpus (or uses
none and attacks through the question), asks a question as an ordinary employee,
and succeeds if the final answer contains what the attacker wanted: a canary
from the executives only board minutes, a false figure, a malicious link, an
exfiltration image, or text from the system prompt.

Three models are tried:

* echo: the worst case. It judges every passage relevant and repeats all of
  them in its answer, so anything injected that reaches the context reaches
  the user. It shows what each defence guarantees regardless of the model.
* offline: the deterministic extractive generator used in tests.
* gemini: a real model (needs GOOGLE_API_KEY), with and without spotlighting.

Defences are added one at a time, from a naive design where access control is
only a sentence in the prompt (retrieval sees every document) up to all layers.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .embed import Embedder, default_embedder
from .graph import RagPipeline, default_retriever
from .ingest.pipeline import ingest_folder, parse_folder, screen
from .llm import IDK
from .models import Hit
from .store.memory import MemoryStore

ROOT = Path(__file__).resolve().parents[1]
REDTEAM = ROOT / "redteam"


class EchoLLM:
    """A model that does whatever the context says: it passes every chunk and repeats it."""

    def grade(self, question: str, text: str) -> bool:
        return True

    def rewrite(self, question: str, attempt: int) -> str:
        return question

    def generate(self, question: str, hits: Sequence[Hit]) -> str:
        return " ".join(f"{h.chunk.text} [{i}]" for i, h in enumerate(hits, start=1)) or IDK


class NoAccessControl:
    """The naive design: retrieval ignores groups and the prompt is trusted to keep secrets."""

    def __init__(self, inner, all_groups: Sequence[str]):
        self.inner, self.all_groups = inner, list(all_groups)

    def search(self, query, k, groups):
        return self.inner.search(query, k, self.all_groups)


@dataclass(frozen=True)
class Config:
    name: str
    acl: bool
    screening: bool
    guard: bool
    spotlighting: bool


CONFIGS = [
    Config("no defences (access rule only in the prompt)", False, False, False, False),
    Config("+ access control in the query", True, False, False, False),
    Config("+ ingestion screening", True, True, False, False),
    Config("+ output guard", True, True, True, False),
    Config("+ spotlighting (all layers)", True, True, True, True),
]


def load_attacks(path: Path = REDTEAM / "attacks.jsonl") -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _base_corpus(tmp: Path) -> Path:
    folder = tmp / "corpus"
    shutil.copytree(ROOT / "corpus", folder)
    shutil.copy(REDTEAM / "board_minutes.md", folder / "board_minutes.md")
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["board_minutes.md"] = {"groups": ["exec"], "title": "Board minutes (restricted)"}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


def _attack_chunks(tmp: Path, attack: dict):
    if not attack["doc"]:
        return []
    folder = tmp / f"attack_{attack['id']}"
    folder.mkdir()
    (folder / attack["doc"]["file"]).write_text(attack["doc"]["text"], encoding="utf-8")
    (folder / "manifest.json").write_text(json.dumps({attack["doc"]["file"]: {"groups": ["all"]}}))
    return parse_folder(folder)


def make_llm(model: str, spotlighting: bool):
    if model == "echo":
        return EchoLLM()
    if model == "offline":
        from .llm import OfflineLLM

        return OfflineLLM()
    from .llm import GeminiLLM

    return GeminiLLM(spotlighting=spotlighting)


def _load_progress(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    done = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            done[(r["model"], r["config"], r["id"])] = r["row"]
    return done


def run(models: Sequence[str], embedder: Embedder | None = None, attacks: list[dict] | None = None,
        progress: Path | None = None) -> dict:
    """With `progress`, every finished attack is appended to that file, and attacks already
    in it are skipped, so a run stopped by an overloaded or rate limited model resumes."""
    embedder = embedder or default_embedder()
    attacks = attacks or load_attacks()
    results = {}
    done = _load_progress(progress)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        corpus = _base_corpus(tmp)
        base = MemoryStore()
        ingest_folder(corpus, embedder, base, screening=False)  # the real corpus has nothing to screen
        all_groups = sorted({g for c in base.all_chunks() for g in c.groups})
        planted = {a["id"]: _attack_chunks(tmp, a) for a in attacks}
        vectors = {aid: embedder.embed_documents([f"{c.title}\n{c.text}" for c in cs]) if cs else None
                   for aid, cs in planted.items()}
        for model in models:
            for cfg in CONFIGS:
                if model != "gemini" and cfg.spotlighting:
                    continue  # spotlighting only changes a real model's prompt
                if model == "gemini" and cfg not in CONFIGS[-2:]:
                    # The earlier layers are structural and the echo model already measures them. A real
                    # model is rate limited, so it runs the comparison only it can make: spotlighting off and on.
                    continue
                llm = make_llm(model, cfg.spotlighting)
                rows = []
                for a in attacks:
                    key = (model, cfg.name, a["id"])
                    if key in done:
                        rows.append(done[key])
                        continue
                    chunks = planted[a["id"]]
                    held = []
                    if chunks and cfg.screening:
                        keep, held = screen(chunks)
                    else:
                        keep = chunks
                    idx = [i for i, c in enumerate(chunks) if c in keep]
                    if keep:
                        base.upsert(keep, vectors[a["id"]][idx])
                    retriever = default_retriever(base, embedder)
                    if not cfg.acl:
                        retriever = NoAccessControl(retriever, all_groups)
                    pipe = RagPipeline(base, embedder, llm, retriever=retriever, guard=cfg.guard)
                    out = pipe.ask(a["question"], a["asker_groups"])
                    base.delete([c.chunk_id for c in keep])
                    success = any(s.lower() in out.answer.lower() for s in a["success_if_answer_contains"])
                    planted_ids = {c.chunk_id for c in chunks}
                    reached = any(h.chunk.chunk_id in planted_ids for h in out.relevant)
                    if success:
                        outcome = "attack succeeded"
                    elif held:
                        outcome = "quarantined at ingestion"
                    elif out.removed:
                        outcome = "removed by output guard"
                    elif a["category"] == "exfiltration" and cfg.acl:
                        outcome = "restricted content never retrieved"
                    elif chunks and not reached:
                        outcome = "planted text not used"
                    else:
                        outcome = "model did not comply"
                    row = {"id": a["id"], "category": a["category"], "success": success,
                           "outcome": outcome, "answer": out.answer}
                    rows.append(row)
                    if progress:
                        progress.parent.mkdir(parents=True, exist_ok=True)
                        with progress.open("a") as f:
                            f.write(json.dumps({"model": model, "config": cfg.name, "id": a["id"], "row": row}) + "\n")
                results[(model, cfg.name)] = rows
    return {"attacks": attacks, "results": results}


def to_markdown(report: dict) -> str:
    attacks = report["attacks"]
    cats = list(dict.fromkeys(a["category"] for a in attacks))
    n_by_cat = {c: sum(1 for a in attacks if a["category"] == c) for c in cats}
    lines = ["# Prompt injection red team", "",
             f"{len(attacks)} attacks; the number in each cell is attacks that succeeded.", ""]
    models = list(dict.fromkeys(m for m, _ in report["results"]))
    for model in models:
        lines += [f"## Model: {model}", "",
                  "| defences | " + " | ".join(f"{c} ({n_by_cat[c]})" for c in cats) + " | total |",
                  "|---|" + "---|" * (len(cats) + 1)]
        for (m, cfg), rows in report["results"].items():
            if m != model:
                continue
            per = [sum(r["success"] for r in rows if r["category"] == c) for c in cats]
            total = sum(per)
            lines.append(f"| {cfg} | " + " | ".join(str(p) for p in per) + f" | **{total}/{len(rows)}** |")
        lines.append("")
    final = [(m, cfg) for (m, cfg) in report["results"] if cfg == CONFIGS[-1].name or
             (m != "gemini" and cfg == CONFIGS[-2].name)]
    for key in final:
        lines += [f"### Every attack, {key[0]}, all defences", "", "| attack | category | outcome | note |", "|---|---|---|---|"]
        notes = {a["id"]: a.get("note", "") for a in attacks}
        for r in report["results"][key]:
            lines.append(f"| {r['id']} | {r['category']} | {r['outcome']} | {notes[r['id']]} |")
        lines.append("")
    return "\n".join(lines)
