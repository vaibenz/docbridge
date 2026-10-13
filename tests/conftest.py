from pathlib import Path

import pytest

from docbridge.embed import HashingEmbedder
from docbridge.graph import RagPipeline
from docbridge.ingest.pipeline import ingest_folder
from docbridge.llm import OfflineLLM
from docbridge.store import MemoryStore

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"


@pytest.fixture(scope="session")
def embedder():
    return HashingEmbedder()


@pytest.fixture
def store(embedder):
    s = MemoryStore()
    ingest_folder(CORPUS, embedder, s)
    return s


@pytest.fixture
def pipeline(store, embedder):
    return RagPipeline(store, embedder, OfflineLLM(), k=5)
