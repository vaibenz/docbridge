"""HTTP service: sign in, ask, get a cited answer drawn only from what you may read.

    DOCBRIDGE_AUTH=dev uvicorn docbridge.api:app --reload      # local, no sign in
    uvicorn docbridge.api:app                                   # OIDC, see auth.py

Endpoints
* GET  /            the web app
* GET  /healthz     liveness and the number of indexed chunks
* GET  /config      what the web app needs to start sign in (public values only)
* GET  /me          who the server thinks you are, and your groups
* POST /ask         {"question": "..."} -> answer, citations, steps, latency
* POST /admin/ingest  re-ingest the corpus folder (admin group only); returns the
                      report, including chunks quarantined as possible prompt injection

Every /ask writes one JSON audit line to stdout: who asked, their groups, the
question, which sources were cited, what the output guard removed, and timing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .auth import AuthError, Principal, RateLimiter, TokenVerifier
from .embed import default_embedder
from .graph import RagPipeline
from .ingest.pipeline import ingest_folder
from .llm import default_llm
from .store import default_store

log = logging.getLogger("docbridge")
audit = logging.getLogger("docbridge.audit")
STATIC = Path(__file__).resolve().parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class Service:
    """Holds the pipeline and swaps in a new one after re-ingestion, so requests in
    flight keep using the old index until the new one is ready."""

    def __init__(self, corpus: Path, allowed_domains: list[str]):
        self.corpus, self.allowed_domains = corpus, allowed_domains
        self.embedder = default_embedder()
        self.store = default_store(dim=self.embedder.dim)
        self.llm = default_llm()
        self.lock = threading.Lock()
        self.last_report = None
        self.reload()

    def reload(self):
        with self.lock:
            report = ingest_folder(self.corpus, self.embedder, self.store)
            self.pipeline = RagPipeline(self.store, self.embedder, self.llm, allowed_domains=self.allowed_domains)
            self.last_report = report
        if report.quarantined:
            log.warning("quarantined %d chunks as possible prompt injection: %s", len(report.quarantined),
                        ", ".join(q["chunk_id"] for q in report.quarantined))
        return report


def create_app(corpus: Path | None = None, verifier: TokenVerifier | None = None,
               limiter: RateLimiter | None = None) -> FastAPI:
    corpus = corpus or Path(os.environ.get("DOCBRIDGE_CORPUS", Path(__file__).resolve().parents[1] / "corpus"))
    dev = os.environ.get("DOCBRIDGE_AUTH") == "dev"
    if verifier is None and not dev:
        verifier = TokenVerifier.from_env()
    limiter = limiter or RateLimiter(float(os.environ.get("DOCBRIDGE_RATE_PER_MIN", 30)))
    allowed = [d for d in os.environ.get("DOCBRIDGE_ALLOWED_LINK_DOMAINS", "").split(",") if d]
    service = Service(corpus, allowed)
    app = FastAPI(title="DocBridge", docs_url=None, redoc_url=None)
    app.state.service = service

    def principal(authorization: str | None = Header(default=None),
                  x_dev_user: str | None = Header(default=None),
                  x_dev_groups: str | None = Header(default=None)) -> Principal:
        if verifier is None:  # dev mode
            log.warning("DOCBRIDGE_AUTH=dev: trusting X-Dev-User and X-Dev-Groups")
            groups = {g.strip() for g in (x_dev_groups or "").split(",") if g.strip()} | {"all"}
            return Principal(x_dev_user or "dev", None, tuple(sorted(groups)))
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "sign in required", headers={"WWW-Authenticate": "Bearer"})
        try:
            return verifier.verify(authorization.split(" ", 1)[1].strip())
        except AuthError as e:
            raise HTTPException(401, f"invalid token: {e}", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "chunks": service.store.count()}

    @app.get("/config")
    def config():
        return {"auth": "dev" if dev else "oidc",
                "google_client_id": os.environ.get("DOCBRIDGE_GOOGLE_CLIENT_ID", "")}

    @app.get("/me")
    def me(p: Principal = Depends(principal)):
        return {"subject": p.subject, "email": p.email, "groups": list(p.groups)}

    @app.post("/ask")
    def ask(body: AskRequest, request: Request, p: Principal = Depends(principal)):
        if not limiter.allow(p.subject):
            raise HTTPException(429, "too many questions, slow down", headers={"Retry-After": "10"})
        result = service.pipeline.ask(body.question, p.groups)
        audit.info(json.dumps({
            "event": "ask", "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "subject": hashlib.sha256(p.subject.encode()).hexdigest()[:16], "groups": list(p.groups),
            "question": body.question, "answered": result.answered, "citations": result.citations,
            "guard_removed": len(result.removed or []), "rewrites": result.rewrites,
            "latency_ms": round(result.latency_ms, 1)}))
        return {"answer": result.answer, "answered": result.answered, "citations": result.citations,
                "steps": result.steps, "latency_ms": round(result.latency_ms, 1)}

    @app.post("/admin/ingest")
    def reingest(p: Principal = Depends(principal)):
        if "admin" not in p.groups:
            raise HTTPException(403, "admins only")
        return service.reload().__dict__

    return app


def _configure_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    audit.addHandler(handler)
    audit.setLevel(logging.INFO)
    audit.propagate = False
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def __getattr__(name):
    # `uvicorn docbridge.api:app` builds the app on first access, so importing this
    # module (in tests, for example) does not ingest a corpus or need OIDC settings.
    if name == "app":
        _configure_logging()
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
