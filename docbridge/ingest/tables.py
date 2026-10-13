"""Structured ingestion: table rows to chunks.

Each row becomes a self describing sentence ("site: Tuas; region: Singapore; ...")
prefixed with the table title, so it embeds well and can be cited by row.
Rows can come from a CSV file or from a BigQuery query.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from ..models import Chunk


def rows_to_chunks(rows: Iterable[dict], source: str, title: str, groups: tuple[str, ...]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, row in enumerate(rows):
        cells = [(k.strip(), str(v).strip()) for k, v in row.items() if k and str(v).strip()]
        if not cells:
            continue
        key = cells[0][1]
        body = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in cells)
        chunks.append(Chunk(source=source, location=f"row:{key}", index=0, text=f"{title}. {body}.",
                            title=title, groups=groups, kind="table"))
    return chunks


def load_csv(path: Path, groups: tuple[str, ...], title: str = "") -> list[Chunk]:
    with path.open(newline="", encoding="utf-8") as f:
        return rows_to_chunks(csv.DictReader(f), path.name, title or path.stem, groups)


def load_bigquery(query: str, source: str, title: str, groups: tuple[str, ...]) -> list[Chunk]:
    """Loads rows from BigQuery. Needs `pip install google-cloud-bigquery` and GCP credentials."""
    from google.cloud import bigquery

    rows = bigquery.Client().query(query).result()
    return rows_to_chunks((dict(r.items()) for r in rows), source, title, groups)
