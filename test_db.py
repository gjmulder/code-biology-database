"""Offline tests for db.py pure transforms (no live MySQL needed)."""

import numpy as np

import db


SAMPLE_OUT = {
    "model": "harrier",
    "dim": 3,
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
    "doc_vectors": {
        "pdfs/a.pdf": {
            "full": [0.1, 0.2, 0.3],
            "abstract": [0.4, 0.5, 0.6],
            "chunk": [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
        }
    },
    "pole_vectors": {
        "two_worlds": {"pos": [1.0, 0.0, 0.0], "neg": [0.0, 1.0, 0.0]},
        "adaptors": {"pos": [0.0, 0.0, 1.0], "neg": [1.0, 1.0, 0.0]},
        "arbitrariness": {"pos": [1.0, 1.0, 1.0], "neg": [-1.0, 0.0, 0.0]},
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


def test_pack_unpack_vec_roundtrips_as_float32():
    v = [0.1, -0.2, 0.3]
    blob = db.pack_vec(v)
    assert isinstance(blob, (bytes, bytearray))
    out = db.unpack_vec(blob)
    assert out.dtype == np.float32
    assert np.allclose(out, np.asarray(v, dtype=np.float32))


def test_doc_vectors_to_rows_grain_and_chunk_indexing():
    rows = db.doc_vectors_to_rows(SAMPLE_OUT, RECS, run_ts="t")
    # full(1) + abstract(1) + chunk(2) = 4 vector rows for the one paper
    assert len(rows) == 4
    # tuple shape: (code_number, pdf_path, method, chunk_idx, dim, vec_blob, run_ts)
    assert all(r[0] == 233 and r[1] == "pdfs/a.pdf" and r[4] == 3 for r in rows)
    # single-vector methods sit at chunk_idx 0
    full = next(r for r in rows if r[2] == "full")
    assert full[3] == 0
    assert np.allclose(db.unpack_vec(full[5]), [0.1, 0.2, 0.3])
    # chunk vectors are indexed 0..n-1 in order
    chunks = sorted((r for r in rows if r[2] == "chunk"), key=lambda r: r[3])
    assert [r[3] for r in chunks] == [0, 1]
    assert np.allclose(db.unpack_vec(chunks[1][5]), [1.0, 1.1, 1.2])


def test_pole_vectors_to_rows_grain():
    rows = db.pole_vectors_to_rows(SAMPLE_OUT, run_ts="t")
    # 3 criteria x 2 poles = 6 rows; shape (criterion, pole, dim, vec_blob, run_ts)
    assert len(rows) == 6
    pos_tw = next(r for r in rows if r[0] == "two_worlds" and r[1] == "pos")
    assert pos_tw[2] == 3
    assert np.allclose(db.unpack_vec(pos_tw[3]), [1.0, 0.0, 0.0])


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


def test_recompute_score_rows_carry_code_and_e_only():
    scores = {
        "pdfs/a.pdf": {
            "full": {"two_worlds": 0.5, "adaptors": -0.2},
            "chunk": {"two_worlds": 0.3, "adaptors": 0.1},
        }
    }
    codes = {"pdfs/a.pdf": 233}
    rows = db.recompute_score_rows(scores, codes, run_ts="t")
    # (code_number, pdf_path, method, criterion, e, run_ts) — for an e-only upsert
    assert (233, "pdfs/a.pdf", "full", "two_worlds", 0.5, "t") in rows
    assert (233, "pdfs/a.pdf", "chunk", "adaptors", 0.1, "t") in rows
    assert len(rows) == 4
    assert all(isinstance(r[4], float) for r in rows)


def test_within_rows_store_pole_width_under_within():
    rows = db.within_rows({"two_worlds": 0.77, "adaptors": 0.64}, run_ts="t")
    assert ("within", "two_worlds", 0.77, "t") in rows
    assert ("within", "adaptors", 0.64, "t") in rows
