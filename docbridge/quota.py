"""Living within API quotas: pacing, backoff on 429, and a resumable embedding cache.

Gemini's free tier allows about 100 embedded texts a minute (each text in a
batch counts) and a few requests a minute for generation, plus daily caps. A
benchmark that embeds thousands of texts has to pace itself, wait out a 429 for
as long as the server says, and stop cleanly at a daily cap with its progress
saved, so the next run carries on instead of starting again.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from collections import deque
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np

T = TypeVar("T")


class DailyQuotaExhausted(RuntimeError):
    pass


class Pacer:
    """Allows at most `per_minute` units in any 60 second window (a unit is a request or a text)."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.per_minute, self.clock, self.sleep = per_minute, clock, sleep
        self._log: deque[tuple[float, int]] = deque()

    def acquire(self, units: int = 1) -> None:
        units = min(units, self.per_minute)
        while True:
            now = self.clock()
            while self._log and now - self._log[0][0] >= 60:
                self._log.popleft()
            used = sum(u for _, u in self._log)
            if used + units <= self.per_minute:
                self._log.append((now, units))
                return
            self.sleep(max(0.05, 60 - (now - self._log[0][0])))


def _retry_delay(err: Exception) -> float | None:
    m = re.search(r"retry in ([\d.]+)s|'retryDelay': '(\d+)s'", str(err))
    if not m:
        return None
    return float(m.group(1) or m.group(2))


def with_backoff(call: Callable[[], T], attempts: int = 8, sleep: Callable[[float], None] | None = None) -> T:
    """Runs `call`, waiting and retrying on 429 and 5xx. A daily quota is not worth waiting for."""
    sleep = sleep or time.sleep
    delay = 2.0
    for attempt in range(attempts):
        try:
            return call()
        except Exception as e:  # google.genai.errors.APIError, without importing it here
            code = getattr(e, "code", None) or getattr(e, "status_code", None)
            if code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
            if code == 429 and re.search(r"PerDay|per day", str(e)):
                raise DailyQuotaExhausted(
                    "Gemini's daily free quota is used up. Progress is cached: run the same command "
                    "tomorrow and it carries on where it stopped.") from e
            wait = _retry_delay(e) or delay
            print(f"  rate limited ({code}), waiting {wait:.0f} s", flush=True)
            sleep(wait + 1)
            delay = min(delay * 2, 60)
    raise AssertionError("unreachable")


class EmbeddingCache:
    """Vectors on disk, keyed by model, dimension, task and text."""

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB NOT NULL)")

    @staticmethod
    def key(model: str, dim: int, task: str, text: str) -> str:
        return hashlib.sha256(f"{model}|{dim}|{task}|{text}".encode()).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        out = {}
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            rows = self.db.execute(f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(part))})", part)
            out.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})
        return out

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        self.db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)",
                            [(k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in items.items()])
        self.db.commit()
