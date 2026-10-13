"""PostgreSQL + pgvector store.

* HNSW index on cosine distance for approximate nearest neighbour search.
* A text[] `groups` column with a GIN index for document level access control.
  The filter runs inside the query, so restricted chunks never leave the database.
* Iterative index scans (pgvector 0.8+) keep returning rows until k rows pass
  the access filter, instead of filtering a fixed candidate list down to fewer than k.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from ..models import Chunk, Hit

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS {table} (
    chunk_id     text PRIMARY KEY,
    source       text NOT NULL,
    location     text NOT NULL,
    idx          integer NOT NULL,
    title        text NOT NULL DEFAULT '',
    kind         text NOT NULL DEFAULT 'text',
    body         text NOT NULL,
    groups       text[] NOT NULL,
    content_hash text NOT NULL,
    embedding    vector({dim}) NOT NULL,
    updated_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw ON {table} USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS {table}_groups_gin ON {table} USING gin (groups);
"""


class PgVectorStore:
    def __init__(self, dsn: str, dim: int = 768, table: str = "docbridge_chunks"):
        if not table.replace("_", "").isalnum():
            raise ValueError("table name must be alphanumeric with underscores")
        self.table = table
        self.dim = dim
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.conn.execute(SCHEMA.format(table=table, dim=dim))
        try:
            self.conn.execute("SET hnsw.iterative_scan = relaxed_order")
        except psycopg.Error:
            pass  # pgvector older than 0.8: still correct, may return fewer than k rows when filtering heavily

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        rows = [
            (c.chunk_id, c.source, c.location, c.index, c.title, c.kind, c.text, list(c.groups),
             c.content_hash, np.asarray(v, dtype=np.float32))
            for c, v in zip(chunks, vectors)
        ]
        with self.conn.cursor() as cur:
            cur.executemany(
                f"""INSERT INTO {self.table}
                    (chunk_id, source, location, idx, title, kind, body, groups, content_hash, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                      body = EXCLUDED.body, groups = EXCLUDED.groups, title = EXCLUDED.title,
                      content_hash = EXCLUDED.content_hash, embedding = EXCLUDED.embedding, updated_at = now()""",
                rows,
            )

    def delete(self, chunk_ids: Sequence[str]) -> None:
        if chunk_ids:
            self.conn.execute(f"DELETE FROM {self.table} WHERE chunk_id = ANY(%s)", (list(chunk_ids),))

    def search(self, vector: np.ndarray, k: int, groups: Sequence[str]) -> list[Hit]:
        vec = np.asarray(vector, dtype=np.float32)
        rows = self.conn.execute(
            f"""SELECT source, location, idx, title, kind, body, groups, 1 - (embedding <=> %s) AS score
                FROM {self.table}
                WHERE groups && %s::text[]
                ORDER BY embedding <=> %s
                LIMIT %s""",
            (vec, list(groups), vec, k),
        ).fetchall()
        return [
            Hit(Chunk(source=r[0], location=r[1], index=r[2], title=r[3], kind=r[4], text=r[5], groups=tuple(r[6])),
                float(r[7]))
            for r in rows
        ]

    def content_hashes(self) -> dict[str, str]:
        return dict(self.conn.execute(f"SELECT chunk_id, content_hash FROM {self.table}").fetchall())

    def count(self) -> int:
        return self.conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]

    def close(self) -> None:
        self.conn.close()
