import json
import shutil

import pytest

from docbridge.ingest.chunker import chunk_text, split_sentences
from docbridge.ingest.pdf import load_pdf, strip_boilerplate
from docbridge.ingest.pipeline import ingest_folder, parse_folder
from docbridge.ingest.tables import load_csv
from docbridge.ingest.text import load_text
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


def test_a_sentence_longer_than_the_budget_is_split():
    # Text with no sentence breaks, like a flattened PDF table, used to come back as one huge chunk.
    run_on = " ".join(f"cell{i}" for i in range(500))
    chunks = chunk_text(f"Intro sentence here. {run_on}. Closing sentence.", max_words=40, overlap_sentences=1)
    assert len(chunks) > 10
    assert all(len(c.split()) <= 80 for c in chunks)  # the budget plus at most one carried over piece
    words = {w.rstrip(".") for w in " ".join(chunks).split()}
    assert all(f"cell{i}" in words for i in range(500))  # nothing dropped


@pytest.mark.parametrize("kw", [{"max_words": 0}, {"max_words": -5}, {"overlap_sentences": -1}])
def test_a_budget_below_one_or_a_negative_overlap_is_refused(kw):
    with pytest.raises(ValueError):
        chunk_text("One two three. Four five six. Seven eight nine.", **kw)


def test_no_overlap_repeats_nothing():
    chunks = chunk_text("One two three. Four five six. Seven eight nine.", max_words=3, overlap_sentences=0)
    assert chunks == ["One two three.", "Four five six.", "Seven eight nine."]


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


def test_csv_rows_with_the_same_key_are_all_kept(tmp_path, embedder):
    # The first column repeats (one row per quarter), so it cannot be the whole chunk id on its own.
    (tmp_path / "throughput.csv").write_text(
        "site,quarter,pallets\nTuas,Q1,1200\nTuas,Q2,1350\nJurong,Q1,900\nTuas,Q3,1410\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"throughput.csv": {"groups": ["all"]}}))
    chunks = load_csv(tmp_path / "throughput.csv", ("all",), "Throughput")
    assert len({c.chunk_id for c in chunks}) == 4
    assert [c.ref for c in chunks] == ["throughput.csv#row:Tuas"] * 2 + ["throughput.csv#row:Jurong",
                                                                          "throughput.csv#row:Tuas"]
    store = MemoryStore()
    report = ingest_folder(tmp_path, embedder, store)
    assert report.chunks == report.embedded == 4
    texts = " ".join(c.text for c in store.all_chunks())
    assert all(f"pallets: {n}" in texts for n in (1200, 1350, 900, 1410))


def test_missing_cells_are_left_out_not_written_as_none(tmp_path):
    # csv.DictReader fills a short row with None, and BigQuery returns NULL as None. Writing
    # "cap: None" would hand the model a value the table never had.
    from docbridge.ingest.tables import rows_to_chunks

    (tmp_path / "caps.csv").write_text("site,region,cap\nTuas,SG\nJurong,SG,5\n", encoding="utf-8")
    tuas, jurong = load_csv(tmp_path / "caps.csv", ("all",), "Caps")
    assert "None" not in tuas.text and "cap" not in tuas.text
    assert "cap: 5" in jurong.text
    (row,) = rows_to_chunks([{"site": "Tuas", "cap": None}], "bq", "Caps", ("all",))
    assert row.text == "Caps. site: Tuas."


def test_csv_saved_with_a_byte_order_mark(tmp_path):
    # Excel's "CSV UTF-8" writes a byte order mark first. Read as plain utf-8 it sticks to the
    # first header, so every row's text and its key column start with "\ufeffsite".
    (tmp_path / "sites.csv").write_bytes("site,region\nTuas,Singapore\n".encode("utf-8-sig"))
    (row,) = load_csv(tmp_path / "sites.csv", ("all",), "Sites")
    assert "\ufeff" not in row.text
    assert row.text == "Sites. site: Tuas; region: Singapore."


def test_markdown_saved_with_a_byte_order_mark(tmp_path):
    # Notepad and many Windows editors save "UTF-8" with a byte order mark. Read as plain utf-8 the
    # mark sits in front of the first "#", so the first heading is not a heading: it stays in the
    # text as "\ufeff# Leave" and the next section's body runs into it.
    (tmp_path / "leave.md").write_bytes("# Leave\n\nStaff get 20 days.\n\n# Travel\n\nBook economy.\n"
                                        .encode("utf-8-sig"))
    chunks = load_text(tmp_path / "leave.md", ("all",), "Policy")
    assert [(c.location, c.text) for c in chunks] == [("s1", "Leave. Staff get 20 days."),
                                                      ("s2", "Travel. Book economy.")]


def test_closing_hashes_are_not_part_of_a_heading(tmp_path):
    # "## Leave ##" is the heading "Leave"; "C#" keeps its # because no space comes before it.
    (tmp_path / "notes.md").write_text("## Leave ##\n\nStaff get 20 days.\n\n# C# #\n\nUse .NET 8.\n")
    chunks = load_text(tmp_path / "notes.md", ("all",), "Notes")
    assert [c.text for c in chunks] == ["Leave. Staff get 20 days.", "C#. Use .NET 8."]


def test_hidden_files_in_the_corpus_folder_are_skipped(tmp_path):
    # Copying a folder from a Mac onto a share or into a zip leaves a "._name" file of resource fork
    # bytes beside each real one. "._handbook.pdf" is not a PDF, and pypdf failing on it stopped the
    # whole ingest; a hidden ".notes.md" is an editor's, not the corpus's.
    (tmp_path / "sites.csv").write_text("site,region\nTuas,Singapore\n", encoding="utf-8")
    (tmp_path / "._sites.csv").write_bytes(b"\x00\x05\x16\x07Mac OS X")
    (tmp_path / "._handbook.pdf").write_bytes(b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X")
    (tmp_path / ".draft.md").write_text("# Draft\n\nNot ready.\n", encoding="utf-8")
    (tmp_path / "archive.pdf").mkdir()  # a folder that happens to end in .pdf
    chunks = parse_folder(tmp_path)
    assert {c.source for c in chunks} == {"sites.csv"}


def test_manifest_sets_groups():
    chunks = parse_folder(CORPUS)
    groups = {c.source: c.groups for c in chunks}
    assert groups["compensation_guidelines.pdf"] == ("hr",)
    assert groups["incident_runbook.pdf"] == ("engineering",)
    assert groups["employee_handbook.pdf"] == ("all",)


def test_manifest_groups_written_as_one_string(tmp_path):
    # tuple("hr") is ("h", "r"): a manifest saying "groups": "hr" locked HR out of its own
    # documents, and "groups": "all" made a public file unreadable to everyone.
    import pytest

    (tmp_path / "a.md").write_text("Leave is 18 days a year.", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"a.md": {"groups": "hr"}}))
    assert {c.groups for c in parse_folder(tmp_path)} == {("hr",)}
    (tmp_path / "manifest.json").write_text(json.dumps({"a.md": {"groups": {"hr": True}}}))
    with pytest.raises(ValueError, match="a.md"):
        parse_folder(tmp_path)


def test_a_malformed_manifest_entry_is_refused_by_name(tmp_path):
    import pytest

    (tmp_path / "a.md").write_text("Leave is 18 days a year.", encoding="utf-8")
    for bad in ({"a.md": ["hr"]}, {"a.md": {"groups": ["hr"], "title": ["Handbook"]}}, [{"a.md": {}}]):
        (tmp_path / "manifest.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="manifest.json"):
            parse_folder(tmp_path)


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


def test_a_new_title_in_the_manifest_is_picked_up(tmp_path, embedder):
    # The title is embedded with the text and shown in citations, so renaming a
    # source in the manifest has to re-embed its chunks, not skip them as unchanged.
    (tmp_path / "notes.md").write_text("# Leave\nStaff get eighteen days of annual leave.\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"notes.md": {"groups": ["all"], "title": "Old Handbook"}}))
    store = MemoryStore()
    first = ingest_folder(tmp_path, embedder, store)

    manifest.write_text(json.dumps({"notes.md": {"groups": ["all"], "title": "Staff Handbook 2026"}}))
    second = ingest_folder(tmp_path, embedder, store)
    assert second.embedded == first.chunks and second.skipped_unchanged == 0
    assert {c.title for c in store.all_chunks()} == {"Staff Handbook 2026"}


def test_an_embedder_that_drops_a_vector_stops_ingest(tmp_path, embedder):
    # If the embedder returns fewer vectors than texts, the store used to zip them and skip the rest:
    # the report still counted every chunk as embedded, but the last one was in no index at all.
    (tmp_path / "sites.csv").write_text("site,region\nTuas,Singapore\nJurong,Singapore\n", encoding="utf-8")

    class Short:
        dim = embedder.dim

        def embed_documents(self, texts):
            return embedder.embed_documents(texts)[:-1]

    with pytest.raises(ValueError, match="1 vectors for 2 chunks"):
        ingest_folder(tmp_path, Short(), MemoryStore(), screening=False)
