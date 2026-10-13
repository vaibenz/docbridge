import json
import shutil

from docbridge.ingest.chunker import chunk_text, split_sentences
from docbridge.ingest.pdf import load_pdf, strip_boilerplate
from docbridge.ingest.pipeline import ingest_folder, parse_folder
from docbridge.ingest.tables import load_csv
from docbridge.store import MemoryStore

from .conftest import CORPUS


def test_split_sentences_keeps_decimals():
    assert split_sentences("The budget is 4.5 percent. Bonus is one month.") == [
        "The budget is 4.5 percent.", "Bonus is one month."]


def test_chunks_respect_budget_and_overlap():
    text = " ".join(f"Sentence number {i} has exactly seven words." for i in range(30))
    chunks = chunk_text(text, max_words=40, overlap_sentences=1)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 47 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert split_sentences(a)[-1] == split_sentences(b)[0]  # one sentence of overlap


def test_boilerplate_lines_removed():
    pages = ["Header\nPage one body.", "Header\nPage two body.", "Header\nPage three body."]
    assert strip_boilerplate(pages) == ["Page one body.", "Page two body.", "Page three body."]


def test_pdf_chunks_have_page_locations_and_no_running_header():
    chunks = load_pdf(CORPUS / "employee_handbook.pdf", ("all",), "Employee Handbook")
    assert {c.location for c in chunks} == {"p1", "p2", "p3"}
    assert not any("Harbourline Logistics Employee Handbook" in c.text for c in chunks)
    assert any("18 days of annual leave" in c.text for c in chunks if c.location == "p2")


def test_csv_rows_become_self_describing_chunks():
    chunks = load_csv(CORPUS / "warehouse_sla.csv", ("all",), "Warehouse service levels")
    assert len(chunks) == 5
    tuas = chunks[0]
    assert tuas.location == "row:Tuas" and tuas.kind == "table"
    assert "dock to stock hours: 24" in tuas.text


def test_manifest_sets_groups():
    chunks = parse_folder(CORPUS)
    groups = {c.source: c.groups for c in chunks}
    assert groups["compensation_guidelines.pdf"] == ("hr",)
    assert groups["incident_runbook.pdf"] == ("engineering",)
    assert groups["employee_handbook.pdf"] == ("all",)


def test_sources_missing_from_manifest_are_locked_down(tmp_path):
    shutil.copy(CORPUS / "warehouse_sla.csv", tmp_path / "unlisted.csv")
    (tmp_path / "manifest.json").write_text(json.dumps({}))
    chunks = parse_folder(tmp_path)
    assert chunks and all(c.groups == () for c in chunks)


def test_ingest_is_incremental(tmp_path, embedder):
    shutil.copytree(CORPUS, tmp_path / "c")
    store = MemoryStore()
    first = ingest_folder(tmp_path / "c", embedder, store)
    assert first.embedded == first.chunks and first.skipped_unchanged == 0
    second = ingest_folder(tmp_path / "c", embedder, store)
    assert second.embedded == 0 and second.skipped_unchanged == first.chunks

    (tmp_path / "c" / "salary_bands.csv").unlink()
    third = ingest_folder(tmp_path / "c", embedder, store)
    assert third.removed == 4 and store.count() == first.chunks - 4
