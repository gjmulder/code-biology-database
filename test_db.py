"""Offline tests for db.py pure transforms (no live MySQL needed)."""

import db


SAMPLE_OUT = {
    "model": "harrier",
    "dim": 5376,
    "methods": ["full", "abstract", "chunk"],
    "chunk_size": 8192,
    "chunk_overlap": 4096,
    "scores": {
        "pdfs/a.pdf": {
            "full": {"two_worlds": 0.2, "adaptors": -0.1, "arbitrariness": 0.05},
            "abstract": {"two_worlds": 0.1, "adaptors": 0.0, "arbitrariness": 0.0},
            "chunk": {"two_worlds": 0.3, "adaptors": 0.1, "arbitrariness": 0.2},
        }
    },
    "controls": {"genetic_code_positive": {"two_worlds": 0.4, "adaptors": 0.4,
                                           "arbitrariness": 0.4}},
    "pole_separation": {"pos": {"two_worlds~adaptors": 0.2},
                        "neg": {"two_worlds~adaptors": 0.3}},
}

RECS = [{
    "pdf_path": "pdfs/a.pdf", "code_number": 233,
    "criteria": {
        "two_worlds": {"verdict": "met", "confidence": 0.9},
        "adaptors": {"verdict": "not_met", "confidence": 1.0},
        "arbitrariness": {"verdict": "unclear", "confidence": 0.8},
    },
}]


def test_scores_to_rows_grain_and_keying():
    rows = db.scores_to_rows(SAMPLE_OUT, RECS, run_ts="2026-06-14 00:00:00")
    # 1 paper x 3 methods x 3 criteria = 9 rows
    assert len(rows) == 9
    # leading column is the code id
    assert all(r[0] == 233 for r in rows)
    # tuple shape: (code, pid, method, criterion, e, verdict, confidence, model, ts)
    full_tw = next(r for r in rows if r[2] == "full" and r[3] == "two_worlds")
    assert full_tw[1] == "pdfs/a.pdf"
    assert full_tw[4] == 0.2
    assert full_tw[5] == "met" and full_tw[6] == 0.9
    assert full_tw[7] == "harrier"


def test_scores_to_rows_carries_each_methods_verdict_consistently():
    rows = db.scores_to_rows(SAMPLE_OUT, RECS, run_ts="t")
    # every row for adaptors carries the same not_met verdict regardless of method
    adaptor_rows = [r for r in rows if r[3] == "adaptors"]
    assert {r[5] for r in adaptor_rows} == {"not_met"}


def test_controls_and_poles_and_meta_rows():
    assert db.controls_to_rows(SAMPLE_OUT, "t") == [
        ("genetic_code_positive", "two_worlds", 0.4, "t"),
        ("genetic_code_positive", "adaptors", 0.4, "t"),
        ("genetic_code_positive", "arbitrariness", 0.4, "t"),
    ]
    poles = db.poles_to_rows(SAMPLE_OUT, "t")
    assert ("pos", "two_worlds~adaptors", 0.2, "t") in poles
    assert ("neg", "two_worlds~adaptors", 0.3, "t") in poles
    meta = dict((k, v) for k, v, *_ in db.meta_rows(SAMPLE_OUT, "t"))
    assert meta["model"] == "harrier"
    assert meta["chunk_size"] == "8192"
