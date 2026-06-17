"""Tests for make_ranking_report.py — offline, no live DB/GPU.

The generator reads the chunk embedding scores and the domain-general graded
verdicts **straight from the DB** (system of record), joins the canonical citation
list (biological_codes.csv), and writes one self-contained, sortable HTML page.

The DB read is exercised here by faking ``db.run_with_reconnect`` (it returns the
raw rows the two queries would), so the whole suite stays offline.
"""

import json
import os
import re

import pytest

import make_ranking_report as mr


def _write_codes(tmp_path, rows):
    p = tmp_path / "codes.csv"
    body = "Code Number,Code Name,Paper Name,URL\n" + "\n".join(rows) + "\n"
    p.write_text(body, encoding="utf-8")
    return str(p)


# --- aggregation helpers ---------------------------------------------------

def test_aggregate_mean_median_min():
    vals = [0.1, 0.4, 0.3]
    assert mr.aggregate(vals, "mean") == sum(vals) / 3
    assert mr.aggregate(vals, "median") == 0.3
    assert mr.aggregate(vals, "min") == 0.1


def test_aggregate_ignores_none_but_min_is_weakest():
    assert mr.aggregate([0.2, None, 0.4], "median") == pytest.approx(0.3)
    assert mr.aggregate([0.2, None, 0.4], "min") == 0.2
    assert mr.aggregate([None, None, None], "mean") is None


def test_verdict_ordinal_matches_judge():
    assert mr.verdict_ordinal("not_met") == 0.0
    assert mr.verdict_ordinal("unclear") == 0.5
    assert mr.verdict_ordinal("met") == 1.0
    assert mr.verdict_ordinal(None) is None  # absent verdict is not coerced to 0.5


# --- DB assembly (pure pivot over the rows the two queries return) ---------

def test_assemble_pivots_scores_and_verdicts():
    score_rows = [
        (62, "pdfs/x.pdf", "two_worlds", 0.081),
        (62, "pdfs/x.pdf", "adaptors", 0.093),
        (62, "pdfs/x.pdf", "arbitrariness", 0.026),
        (99, "pdfs/y.pdf", "two_worlds", -0.03),  # no verdict for this paper
    ]
    verdict_rows = [
        ("pdfs/x.pdf", "two_worlds", "met", 1.0, 0.95),
        ("pdfs/x.pdf", "adaptors", "not_met", 0.0, 1.0),
        ("pdfs/x.pdf", "arbitrariness", "unclear", 0.5, 0.8),
    ]
    recs = mr._assemble(score_rows, verdict_rows)
    by_code = {r["code"]: r for r in recs}
    x = by_code[62]
    assert x["pdf_path"] == "pdfs/x.pdf"
    assert x["e"]["two_worlds"] == 0.081
    assert x["verdict"]["two_worlds"] == "met"
    assert x["graded"]["two_worlds"] == 1.0
    assert x["conf"]["arbitrariness"] == 0.8
    # paper with embedding but no new verdict -> verdict fields stay None
    y = by_code[99]
    assert y["e"]["two_worlds"] == -0.03
    assert y["verdict"]["two_worlds"] is None
    assert y["graded"]["two_worlds"] is None


def test_load_papers_from_db_uses_reconnect(monkeypatch):
    score_rows = [(62, "pdfs/x.pdf", "two_worlds", 0.081)]
    verdict_rows = [("pdfs/x.pdf", "two_worlds", "met", 1.0, 0.95)]

    ct_rows = [("pdfs/x.pdf", 0, 18, 0.9)]
    cen_rows = [(18, "Histone")]

    class FakeCursor:
        def __init__(self):
            self.q = None

        def execute(self, sql, *a):
            self.q = sql

        def fetchall(self):
            if "FROM verdicts" in self.q:
                return verdict_rows
            if "FROM chunk_topics" in self.q:
                return ct_rows
            if "FROM topic_centroids" in self.q:
                return cen_rows
            return score_rows

    class FakeConn:
        def cursor(self):
            return FakeCursor()

    monkeypatch.setattr(mr.db, "run_with_reconnect", lambda work, *a, **k: work(FakeConn()))
    recs = mr.load_papers_from_db()
    assert len(recs) == 1
    assert recs[0]["verdict"]["two_worlds"] == "met"
    assert recs[0]["topics"] == ["Histone"]
    assert recs[0]["dominant_topic"] == "Histone"


# --- topic coverage --------------------------------------------------------

def test_paper_topics_dominant_first_then_by_chunk_count():
    # topic 18 has the single strongest chunk (0.95) -> dominant by max-pool affinity;
    # topic 11 covers the most chunks (3) -> first of the remainder.
    chunks = [(0, 11, 0.9), (1, 11, 0.8), (2, 11, 0.7), (3, 18, 0.95), (4, 7, 0.4)]
    labels = {11: "Neural", 18: "Histone", 7: "Regulatory"}
    assert mr.paper_topics(chunks, labels) == ["Histone", "Neural", "Regulatory"]


def test_paper_topics_empty_and_missing_label():
    assert mr.paper_topics([], {}) == []
    # an unlabelled topic id degrades to a "topic N" placeholder, still listed
    assert mr.paper_topics([(0, 42, 0.5)], {}) == ["topic 42"]


def test_attach_topics_sets_list_and_dominant():
    recs = [
        {"code": 62, "pdf_path": "pdfs/x.pdf"},
        {"code": 99, "pdf_path": "pdfs/notopic.pdf"},  # no chunk topics
    ]
    chunk_topics = {"pdfs/x.pdf": [(0, 11, 0.6), (1, 18, 0.9)]}
    centroids = {11: {"label": "Neural"}, 18: {"label": "Histone"}}
    mr.attach_topics(recs, chunk_topics, centroids)
    by_code = {r["code"]: r for r in recs}
    assert by_code[62]["dominant_topic"] == "Histone"  # max affinity
    assert by_code[62]["topics"] == ["Histone", "Neural"]
    # paper with no chunk topics degrades to empty list / blank dominant
    assert by_code[99]["topics"] == []
    assert by_code[99]["dominant_topic"] == ""


# --- citation join ---------------------------------------------------------

def test_attach_citations_by_basename_and_degrades(tmp_path):
    recs = [
        {"code": 62, "pdf_path": "pdfs/10.1234_x.pdf",
         "e": {c: 0.0 for c in mr.CRITERIA},
         "verdict": {c: None for c in mr.CRITERIA},
         "graded": {c: None for c in mr.CRITERIA},
         "conf": {c: None for c in mr.CRITERIA}},
        {"code": 99, "pdf_path": "pdfs/unmatched.pdf",
         "e": {c: 0.0 for c in mr.CRITERIA},
         "verdict": {c: None for c in mr.CRITERIA},
         "graded": {c: None for c in mr.CRITERIA},
         "conf": {c: None for c in mr.CRITERIA}},
    ]
    codes = _write_codes(tmp_path, [
        '62,Genetic code,"Crick, F. (1968).",https://doi.org/10.1234/x',
    ])
    mr.attach_citations(recs, codes)
    by_code = {r["code"]: r for r in recs}
    assert by_code[62]["code_name"] == "Genetic code"
    assert by_code[62]["url"] == "https://doi.org/10.1234/x"
    # unmatched paper still present, blank citation fields
    assert by_code[99]["code_name"] == ""
    assert by_code[99]["paper_name"] == ""


# --- HTML emission ---------------------------------------------------------

def _sample_papers(tmp_path):
    recs = mr._assemble(
        [(62, "pdfs/10.1234_x.pdf", c, 0.05) for c in mr.CRITERIA],
        [("pdfs/10.1234_x.pdf", "two_worlds", "met", 1.0, 0.95)],
    )
    codes = _write_codes(tmp_path, [
        '62,Genetic code,"Crick.",https://doi.org/10.1234/x'])
    return mr.attach_citations(recs, codes)


def test_build_html_is_self_contained_with_inlined_data(tmp_path):
    papers = _sample_papers(tmp_path)
    html = mr.build_html(papers)
    assert "<table" in html
    m = re.search(r"const PAPERS\s*=\s*(\[.*?\]);", html, re.DOTALL)
    assert m, "PAPERS array not found"
    data = json.loads(m.group(1))
    assert len(data) == len(papers) == 1
    # fully offline: no external script/style sources
    assert not re.search(r'(src|href)\s*=\s*["\']https?://', html)
    # the two axes are the chunk embedding ("similarity to genetic code", internal key
    # 'pages') and the "LLM verdicts" (internal key 'verdicts'); metrics kept; the retired
    # full/abstract axes are gone from the selector
    for token in ('data-v="pages"', 'data-v="verdicts"', "similarity to genetic code",
                  "LLM verdicts", "mean", "median", "min"):
        assert token in html
    assert 'data-v="full"' not in html
    assert 'data-v="abstract"' not in html
    assert 'data-v="chunk"' not in html
    # the topic-coverage column is present (JS-rendered header) and sortable
    assert 'th("topics","Topics"' in html
    assert "topicsHTML(p)" in html


def test_default_axis_is_verdicts_and_metric_is_min(tmp_path):
    html = mr.build_html(_sample_papers(tmp_path))
    # JS state defaults
    assert 'let source = "verdicts", metric = "min";' in html
    # the segmented controls mark the matching buttons active ("on")
    assert '<button data-v="verdicts" class="on">LLM verdicts</button>' in html
    assert 'data-v="min" class="on"' in html
    # the previous defaults are no longer pre-selected
    assert '<button data-v="pages" class="on">' not in html
    assert '<button data-v="median" class="on">' not in html


def test_main_writes_html_file_from_db(tmp_path, monkeypatch):
    score_rows = [(62, "pdfs/10.1234_x.pdf", c, 0.05) for c in mr.CRITERIA]
    verdict_rows = [("pdfs/10.1234_x.pdf", "two_worlds", "met", 1.0, 0.95)]

    ct_rows = [("pdfs/10.1234_x.pdf", 0, 18, 0.9)]
    cen_rows = [(18, "Histone")]

    class FakeCursor:
        def execute(self, sql, *a):
            self.q = sql

        def fetchall(self):
            if "FROM verdicts" in self.q:
                return verdict_rows
            if "FROM chunk_topics" in self.q:
                return ct_rows
            if "FROM topic_centroids" in self.q:
                return cen_rows
            return score_rows

    class FakeConn:
        def cursor(self):
            return FakeCursor()

    monkeypatch.setattr(mr.db, "run_with_reconnect", lambda work, *a, **k: work(FakeConn()))
    codes = _write_codes(tmp_path, [
        '62,Genetic code,"Crick.",https://doi.org/10.1234/x'])
    out = str(tmp_path / "out.html")
    mr.main(["--codes", codes, "--out", out])
    assert os.path.exists(out)
    html = open(out, encoding="utf-8").read()
    assert "<table" in html
    assert "Histone" in html  # topic coverage surfaced in the inlined data
