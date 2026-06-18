"""Tests for the code-0 ("Code Biology" foundational papers) ingestion step.

The step is the seed-corpus analogue of extract_csv + download_pdfs: instead of
parsing the source PDF and downloading, it copies hand-curated foundational texts
into ``pdfs/`` and appends them to ``biological_codes.csv`` under the reserved
``Code Number 0``. The pure logic (manifest parse, corpus-consistent path naming,
row building, idempotent append planning) is tested here; the filesystem copy is
exercised by a tmp-dir round trip. No network, no DB.
"""

import csv
import os

import seed_code_biology as seed


# --- load_manifest ---------------------------------------------------------

def test_load_manifest_parses_rows(tmp_path):
    p = tmp_path / "seed.csv"
    p.write_text(
        "Source File,Paper Name,URL\n"
        "Intro.pdf,Barbieri (2014). Introduction to Code Biology.,https://doi.org/10.1007/x\n"
        "Book.pdf,Barbieri. The Organic Codes.,\n",
        encoding="utf-8",
    )
    rows = seed.load_manifest(str(p))
    assert rows[0] == {
        "source_file": "Intro.pdf",
        "paper_name": "Barbieri (2014). Introduction to Code Biology.",
        "url": "https://doi.org/10.1007/x",
    }
    assert rows[1]["url"] == ""  # missing URL tolerated (a book with no DOI)


# --- dest_path_for (corpus-consistent naming) ------------------------------

def test_dest_path_for_uses_doi_naming_when_url_present():
    entry = {"source_file": "Intro.pdf",
             "url": "https://doi.org/10.1016/j.biosystems.2017.10.005"}
    # identical naming to the rest of the corpus (download_pdfs.output_path_for)
    assert seed.dest_path_for(entry, output_dir="pdfs") == \
        os.path.join("pdfs", "10.1016_j.biosystems.2017.10.005.pdf")


def test_dest_path_for_slugs_source_filename_when_no_url():
    entry = {"source_file": "The Organic Codes - Barbieri.pdf", "url": ""}
    dest = seed.dest_path_for(entry, output_dir="pdfs")
    assert dest.startswith(os.path.join("pdfs", "The_Organic_Codes"))
    assert dest.endswith(".pdf")
    assert ".pdf.pdf" not in dest  # no doubled extension


# --- code_row --------------------------------------------------------------

def test_code_row_uses_reserved_code_zero():
    entry = {"paper_name": "Barbieri (2018). What Is Code Biology?", "url": "u"}
    assert seed.code_row(entry) == [
        seed.CODE_NUMBER, seed.CODE_NAME,
        "Barbieri (2018). What Is Code Biology?", "u"]
    assert seed.CODE_NUMBER == "0"
    assert seed.CODE_NAME == "Code Biology"


# --- rows_to_append (idempotency) ------------------------------------------

def test_rows_to_append_skips_already_present_code_zero_rows():
    manifest = [
        {"paper_name": "P1", "url": "u1", "source_file": "a.pdf"},
        {"paper_name": "P2", "url": "u2", "source_file": "b.pdf"},
    ]
    existing = [
        {"Code Number": "1", "Paper Name": "unrelated", "URL": "x"},
        {"Code Number": "0", "Paper Name": "P1", "URL": "u1"},  # already seeded
    ]
    rows = seed.rows_to_append(manifest, existing)
    assert rows == [["0", "Code Biology", "P2", "u2"]]  # only the new one


def test_rows_to_append_matches_on_paper_name_when_url_absent():
    manifest = [{"paper_name": "Book", "url": "", "source_file": "book.pdf"}]
    existing = [{"Code Number": "0", "Paper Name": "Book", "URL": ""}]
    assert seed.rows_to_append(manifest, existing) == []  # idempotent on name


def test_rows_to_append_all_new_when_corpus_has_no_code_zero():
    manifest = [{"paper_name": "P1", "url": "u1", "source_file": "a.pdf"}]
    existing = [{"Code Number": "1", "Paper Name": "x", "URL": "y"}]
    assert seed.rows_to_append(manifest, existing) == [["0", "Code Biology", "P1", "u1"]]


# --- seed() end-to-end on tmp dirs (copy + append, idempotent) -------------

def test_seed_copies_pdf_and_appends_row_then_is_idempotent(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    out = tmp_path / "pdfs"; out.mkdir()
    (src / "Intro.pdf").write_bytes(b"%PDF-1.4 fake")
    manifest = tmp_path / "seed.csv"
    manifest.write_text(
        "Source File,Paper Name,URL\n"
        "Intro.pdf,Barbieri (2014). Introduction.,https://doi.org/10.1007/x\n",
        encoding="utf-8",
    )
    codes = tmp_path / "biological_codes.csv"
    with open(codes, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Code Number", "Code Name", "Paper Name", "URL"])
        w.writerow(["1", "14-3-3 code", "Winter et al.", "https://doi.org/10.1/a"])

    n = seed.seed(str(manifest), str(codes), str(src), str(out))
    assert n == 1
    # PDF landed under the corpus-consistent DOI name
    assert (out / "10.1007_x.pdf").exists()
    # row appended, original rows preserved
    rows = list(csv.DictReader(open(codes, encoding="utf-8")))
    assert len(rows) == 2
    assert rows[-1]["Code Number"] == "0"
    assert rows[-1]["Code Name"] == "Code Biology"

    # re-running adds nothing (idempotent) and doesn't re-copy
    n2 = seed.seed(str(manifest), str(codes), str(src), str(out))
    assert n2 == 0
    assert len(list(csv.DictReader(open(codes, encoding="utf-8")))) == 2
