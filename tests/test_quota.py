import types

import numpy as np
import pytest

from docbridge.quota import DailyQuotaExhausted, EmbeddingCache, Pacer, with_backoff


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def test_pacer_waits_for_the_window_to_roll():
    clock = FakeClock()
    p = Pacer(100, clock=clock, sleep=clock.sleep)
    p.acquire(60)
    p.acquire(40)
    assert clock.t == 0
    p.acquire(10)  # the window is full until the first batch is 60 s old
    assert clock.t == pytest.approx(60)


class ApiError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def test_backoff_honours_retry_delay_then_succeeds():
    calls, slept = [], []

    def call():
        calls.append(1)
        if len(calls) < 3:
            raise ApiError(429, "Quota exceeded for metric PerMinute. Please retry in 7.5s.")
        return "ok"

    assert with_backoff(call, sleep=slept.append) == "ok"
    assert slept == [8.5, 8.5]


def test_backoff_stops_at_a_daily_cap_and_does_not_retry_client_errors():
    with pytest.raises(DailyQuotaExhausted):
        with_backoff(lambda: (_ for _ in ()).throw(ApiError(429, "quotaId: EmbedContentRequestsPerDayPerUser")),
                     sleep=lambda s: None)
    calls = []

    def bad_request():
        calls.append(1)
        raise ApiError(400, "invalid argument")

    with pytest.raises(ApiError):
        with_backoff(bad_request, sleep=lambda s: None)
    assert len(calls) == 1


def test_cache_round_trip(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    k = EmbeddingCache.key("m", 3, "RETRIEVAL_QUERY", "hello")
    cache.put_many({k: np.array([1, 2, 3], dtype=np.float32)})
    again = EmbeddingCache(tmp_path / "e.sqlite").get_many([k, "missing"])
    assert list(again) == [k] and again[k].tolist() == [1, 2, 3]


def test_gemini_embedder_uses_the_cache_and_survives_a_429(tmp_path, monkeypatch):
    pytest.importorskip("google.genai")
    from google import genai

    batches, failed = [], []

    class Models:
        def embed_content(self, model, contents, config):
            batches.append(list(contents))
            if not failed:
                failed.append(1)
                raise ApiError(429, "PerMinute. Please retry in 0s.")
            return types.SimpleNamespace(embeddings=[types.SimpleNamespace(values=[float(len(t)), 1.0, 0.0])
                                                     for t in contents])

    monkeypatch.setattr(genai, "Client", lambda: types.SimpleNamespace(models=Models()))
    monkeypatch.setattr("docbridge.quota.time.sleep", lambda s: None)
    from docbridge.embed import GeminiEmbedder

    emb = GeminiEmbedder(dim=3, batch_size=2, cache_path=str(tmp_path / "c.sqlite"))
    first = emb.embed_documents(["a", "bb", "ccc"])
    assert batches == [["a", "bb"], ["a", "bb"], ["ccc"]]  # the 429'd batch was retried
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
    batches.clear()
    again = GeminiEmbedder(dim=3, batch_size=2, cache_path=str(tmp_path / "c.sqlite")).embed_documents(["ccc", "a", "dddd"])
    assert batches == [["dddd"]]  # only the new text goes to the API
    assert np.allclose(again[0], first[2]) and np.allclose(again[1], first[0])
