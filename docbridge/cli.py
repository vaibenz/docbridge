"""Command line entry point.

    docbridge ingest corpus/
    docbridge ask "How much holiday do I get?" --groups all
    docbridge eval eval/questions.jsonl --out eval/report.md
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .embed import default_embedder
from .evaluate import evaluate, load_items, to_markdown
from .graph import RagPipeline
from .ingest.pipeline import ingest_folder
from .llm import default_llm
from .store import default_store

DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / "corpus"


def build_pipeline(corpus: Path | None = None, k: int = 5, mode: str = "hybrid") -> RagPipeline:
    embedder = default_embedder()
    store = default_store(dim=embedder.dim)
    # The in memory store starts empty in every process, so ingest on startup.
    # With pgvector this is incremental: unchanged chunks are skipped by content hash.
    ingest_folder(corpus or DEFAULT_CORPUS, embedder, store)
    return RagPipeline(store, embedder, default_llm(), k=k, mode=mode)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="docbridge")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="parse, embed and store a folder of PDFs and CSVs")
    p_ingest.add_argument("folder", type=Path)

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--groups", default="all", help="comma separated groups the asker belongs to")
    p_ask.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p_ask.add_argument("--retriever", choices=["hybrid", "dense", "bm25"], default="hybrid")

    p_eval = sub.add_parser("eval", help="run the evaluation set")
    p_eval.add_argument("questions", type=Path)
    p_eval.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p_eval.add_argument("--out", type=Path)
    p_eval.add_argument("-k", type=int, default=5)
    p_eval.add_argument("--retriever", choices=["hybrid", "dense", "bm25"], default="hybrid")

    p_red = sub.add_parser("redteam", help="run the prompt injection red team suite")
    p_red.add_argument("--models", default="echo,offline", help="comma separated: echo, offline, gemini")
    p_red.add_argument("--out", type=Path)
    p_red.add_argument("--fresh", action="store_true", help="ignore saved progress and run every attack again")

    args = parser.parse_args(argv)
    # Gemini embeddings are cached on disk, so repeated runs and runs that hit a quota resume cheaply.
    os.environ.setdefault("DOCBRIDGE_EMBED_CACHE", str(Path.home() / ".cache" / "docbridge" / "embeddings.sqlite"))
    if args.cmd == "ingest":
        embedder = default_embedder()
        report = ingest_folder(args.folder, embedder, default_store(dim=embedder.dim))
        print(json.dumps(report.__dict__, indent=2))
    elif args.cmd == "ask":
        result = build_pipeline(args.corpus, mode=args.retriever).ask(args.question, args.groups.split(","))
        print(result.answer)
        for ref in result.citations:
            print(f"  source: {ref}")
        print(f"  steps: {' > '.join(result.steps)}")
    elif args.cmd == "redteam":
        from . import redteam

        # Gemini runs are slow and can be interrupted by quotas or an overloaded model, so finished
        # attacks are saved and a rerun picks up where the last one stopped.
        progress = Path.home() / ".cache" / "docbridge" / f"redteam-{args.models.replace(',', '-')}.jsonl"
        if args.fresh and progress.exists():
            progress.unlink()
        uses_gemini = "gemini" in args.models.split(",")
        report = redteam.run(args.models.split(","), progress=progress if uses_gemini else None)
        md = redteam.to_markdown(report)
        print(md)
        if args.out:
            args.out.write_text(md)
    elif args.cmd == "eval":
        pipeline = build_pipeline(args.corpus, k=args.k, mode=args.retriever)
        report = evaluate(pipeline, load_items(args.questions))
        print(json.dumps(report["summary"], indent=2))
        if args.out:
            args.out.write_text(to_markdown(report, args.k))
            args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
            print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
