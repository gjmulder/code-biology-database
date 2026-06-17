"""Offline tests for db.py pure transforms (no live MySQL needed)."""

import numpy as np
import pymysql
import pytest

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
    "control_vectors": {"genetic_code_positive": [0.1, 0.2, 0.3],
                        "deterministic_chemistry": [0.3, 0.2, 0.1]},
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
    # tuple shape: (code, pid, method, criterion, e, model, run_ts, run) — verdict/
    # confidence are no longer here (normalised into the run-agnostic `verdicts` table).
    full_tw = next(r for r in rows if r[2] == "full" and r[3] == "two_worlds")
    assert full_tw[1] == "pdfs/a.pdf"
    assert full_tw[4] == 0.2
    assert full_tw[5] == "harrier"            # embedding model
    assert full_tw[6] == "2026-06-14 00:00:00"  # run_ts
    assert full_tw[7] == "baseline"           # default run label


def test_scores_to_rows_run_label_overridable():
    rows = db.scores_to_rows(SAMPLE_OUT, RECS, run_ts="t", run="gte-qwen2")
    assert all(r[7] == "gte-qwen2" for r in rows)


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
    # tuple shape: (code_number, pdf_path, method, chunk_idx, dim, vec_blob, run_ts, run)
    assert all(r[0] == 233 and r[1] == "pdfs/a.pdf" and r[4] == 3 for r in rows)
    assert all(r[7] == "baseline" for r in rows)
    # single-vector methods sit at chunk_idx 0
    full = next(r for r in rows if r[2] == "full")
    assert full[3] == 0
    assert np.allclose(db.unpack_vec(full[5]), [0.1, 0.2, 0.3])
    # chunk vectors are indexed 0..n-1 in order
    chunks = sorted((r for r in rows if r[2] == "chunk"), key=lambda r: r[3])
    assert [r[3] for r in chunks] == [0, 1]
    assert np.allclose(db.unpack_vec(chunks[1][5]), [1.0, 1.1, 1.2])


def test_control_vectors_to_rows_grain():
    rows = db.control_vectors_to_rows(SAMPLE_OUT, run_ts="t")
    # 2 controls; tuple shape (name, dim, vec_blob, run_ts, run)
    assert len(rows) == 2
    gc = next(r for r in rows if r[0] == "genetic_code_positive")
    assert gc[1] == 3 and gc[3] == "t" and gc[4] == "baseline"
    assert np.allclose(db.unpack_vec(gc[2]), [0.1, 0.2, 0.3])


def test_control_vectors_to_rows_empty_when_absent():
    assert db.control_vectors_to_rows({"controls": {}}, run_ts="t") == []


def test_pole_vectors_to_rows_grain():
    rows = db.pole_vectors_to_rows(SAMPLE_OUT, run_ts="t")
    # 3 criteria x 2 poles = 6 rows; shape (criterion, pole, dim, vec_blob, run_ts, run)
    assert len(rows) == 6
    pos_tw = next(r for r in rows if r[0] == "two_worlds" and r[1] == "pos")
    assert pos_tw[2] == 3 and pos_tw[5] == "baseline"
    assert np.allclose(db.unpack_vec(pos_tw[3]), [1.0, 0.0, 0.0])


def test_controls_and_poles_and_meta_rows():
    assert db.controls_to_rows(SAMPLE_OUT, "t") == [
        ("genetic_code_positive", "two_worlds", 0.4, "t", "baseline"),
        ("genetic_code_positive", "adaptors", 0.4, "t", "baseline"),
        ("genetic_code_positive", "arbitrariness", 0.4, "t", "baseline"),
    ]
    poles = db.poles_to_rows(SAMPLE_OUT, "t")
    assert ("pos", "two_worlds~adaptors", 0.2, "t", "baseline") in poles
    assert ("neg", "two_worlds~adaptors", 0.3, "t", "baseline") in poles
    # meta_rows tuples are (k, v, run); the run column is absorbed by `*_`.
    meta = dict((k, v) for k, v, *_ in db.meta_rows(SAMPLE_OUT, "t"))
    assert meta["model"] == "harrier"
    assert meta["chunk_size"] == "8192"
    assert all(r[2] == "baseline" for r in db.meta_rows(SAMPLE_OUT, "t"))


def test_recompute_score_rows_carry_code_and_e_only():
    scores = {
        "pdfs/a.pdf": {
            "full": {"two_worlds": 0.5, "adaptors": -0.2},
            "chunk": {"two_worlds": 0.3, "adaptors": 0.1},
        }
    }
    codes = {"pdfs/a.pdf": 233}
    rows = db.recompute_score_rows(scores, codes, run_ts="t")
    # (code_number, pdf_path, method, criterion, e, run_ts, run) — for an e-only upsert
    assert (233, "pdfs/a.pdf", "full", "two_worlds", 0.5, "t", "baseline") in rows
    assert (233, "pdfs/a.pdf", "chunk", "adaptors", 0.1, "t", "baseline") in rows
    assert len(rows) == 4
    assert all(isinstance(r[4], float) for r in rows)


def test_within_rows_store_pole_width_under_within():
    rows = db.within_rows({"two_worlds": 0.77, "adaptors": 0.64}, run_ts="t")
    assert ("within", "two_worlds", 0.77, "t", "baseline") in rows
    assert ("within", "adaptors", 0.64, "t", "baseline") in rows


CENTROIDS_OUT = {
    "model": "harrier",
    "dim": 3,
    "run": "baseline",
    "centroids": [
        {"topic_id": 3, "label": "Genetic Code", "vec": [0.1, 0.2, 0.3]},
        {"topic_id": 18, "label": "Histone Code", "vec": [0.4, 0.5, 0.6]},
    ],
}


def test_topic_centroids_to_rows_grain():
    rows = db.topic_centroids_to_rows(CENTROIDS_OUT, run_ts="t")
    # one row per centroid; shape (topic_id, label, dim, vec_blob, run_ts, run)
    assert len(rows) == 2
    gec = next(r for r in rows if r[0] == 3)
    assert gec[1] == "Genetic Code"
    assert gec[2] == 3 and gec[4] == "t" and gec[5] == "baseline"
    assert np.allclose(db.unpack_vec(gec[3]), [0.1, 0.2, 0.3])


def test_topic_centroids_to_rows_run_overridable_and_empty():
    rows = db.topic_centroids_to_rows(CENTROIDS_OUT, run_ts="t", run="gte-qwen2")
    assert all(r[5] == "gte-qwen2" for r in rows)
    assert db.topic_centroids_to_rows({"centroids": []}, run_ts="t") == []


ASSIGNMENTS = {
    "pdfs/a.pdf": {"chunks": [(0, 3, 0.42), (1, 18, 0.31)],
                   "dominant": 3, "affinity": {3: 0.42, 18: 0.31}},
    "pdfs/b.pdf": {"chunks": [(0, 9, 0.27)],
                   "dominant": 9, "affinity": {9: 0.27}},
}
CODES = {"pdfs/a.pdf": 233, "pdfs/b.pdf": 7}


def test_chunk_topic_rows_grain():
    rows = db.chunk_topic_rows(ASSIGNMENTS, CODES, run_ts="t")
    # one row per chunk: 2 + 1 = 3; shape
    # (code_number, pdf_path, method, chunk_idx, topic_id, sim, run_ts, run)
    assert len(rows) == 3
    a0 = next(r for r in rows if r[1] == "pdfs/a.pdf" and r[3] == 0)
    assert a0[0] == 233 and a0[2] == "chunk" and a0[4] == 3
    assert a0[5] == 0.42 and a0[6] == "t" and a0[7] == "baseline"
    b0 = next(r for r in rows if r[1] == "pdfs/b.pdf")
    assert b0[0] == 7 and b0[4] == 9 and b0[3] == 0


def test_chunk_topic_rows_method_and_run_overridable():
    rows = db.chunk_topic_rows(ASSIGNMENTS, CODES, run_ts="t",
                               method="full", run="gte-qwen2")
    assert all(r[2] == "full" and r[7] == "gte-qwen2" for r in rows)
    assert db.chunk_topic_rows({}, {}, run_ts="t") == []


def test_verdict_update_rows_one_row_per_paper_criterion():
    # judged records (criteria_judge shape): verdict/confidence per criterion.
    records = [{
        "code_number": "428", "pdf_path": "pdfs/x.pdf",
        "criteria": {
            "two_worlds": {"verdict": "met", "confidence": 0.95},
            "adaptors": {"verdict": "not_met", "confidence": 1.0},
            "arbitrariness": {"verdict": "unclear", "confidence": 0.7},
        },
    }]
    rows = db.verdict_update_rows(records, run_ts="t")
    # repointed at the run-agnostic `verdicts` table: one row per (paper, criterion),
    # tuple `(code_number, pdf_path, criterion, verdict, confidence, graded, model, run_ts)`.
    # No method fan-out, no run, no embedding model. The judge model isn't carried on the
    # record so `model` is None; the old categorical path carries no graded score so it is
    # None; code_number is coerced to int.
    # tuple `(code_number, pdf_path, criterion, verdict, confidence, graded, model,
    # prompt_hash, run_ts)`; prompt_hash is None when the record carries none.
    assert (428, "pdfs/x.pdf", "two_worlds", "met", 0.95, None, None, None, "t") in rows
    assert (428, "pdfs/x.pdf", "adaptors", "not_met", 1.0, None, None, None, "t") in rows
    assert (428, "pdfs/x.pdf", "arbitrariness", "unclear", 0.7, None, None, None, "t") in rows
    assert len(rows) == 3


def test_verdict_update_rows_carries_graded_when_present():
    # the graded judge path attaches a per-paper graded_max alongside the derived verdict
    records = [{
        "code_number": "9", "pdf_path": "pdfs/g.pdf",
        "criteria": {"two_worlds": {"verdict": "met", "confidence": 0.66, "graded": 0.5}},
    }]
    rows = db.verdict_update_rows(records, run_ts="t")
    assert rows == [(9, "pdfs/g.pdf", "two_worlds", "met", 0.66, 0.5, None, None, "t")]


def test_verdict_update_rows_skips_criteria_without_a_verdict():
    records = [{
        "code_number": "21", "pdf_path": "pdfs/y.pdf",
        "criteria": {
            "two_worlds": {"verdict": "met", "confidence": 0.9},
            "adaptors": {},  # no verdict emitted -> not written (don't clobber)
        },
    }]
    rows = db.verdict_update_rows(records, run_ts="t")
    assert rows == [(21, "pdfs/y.pdf", "two_worlds", "met", 0.9, None, None, None, "t")]


def test_chunk_verdict_rows_grain_and_keying():
    # flat per-chunk graded records -> one chunk_verdicts row each, tuple
    # (code_number, pdf_path, criterion, chunk_idx, agreement, confidence, evidence_quote,
    #  model, run_ts); code_number coerced to int, model defaults None.
    records = [
        {"code_number": "5", "pdf_path": "pdfs/a.pdf", "criterion": "two_worlds",
         "chunk_idx": 0, "agreement": 0.5, "confidence": 0.66, "evidence_quote": "q0"},
        {"code_number": "5", "pdf_path": "pdfs/a.pdf", "criterion": "two_worlds",
         "chunk_idx": 1, "agreement": -0.5, "confidence": 0.33, "evidence_quote": ""},
    ]
    rows = db.chunk_verdict_rows(records, run_ts="t")
    assert (5, "pdfs/a.pdf", "two_worlds", 0, 0.5, 0.66, "q0", None, None, "t") in rows
    assert (5, "pdfs/a.pdf", "two_worlds", 1, -0.5, 0.33, "", None, None, "t") in rows
    assert len(rows) == 2


def test_chunk_verdict_rows_carries_model():
    records = [{"code_number": "1", "pdf_path": "p", "criterion": "adaptors",
                "chunk_idx": 0, "agreement": 1.0, "confidence": 1.0,
                "evidence_quote": "x"}]
    rows = db.chunk_verdict_rows(records, run_ts="t", model="gemma-4-31b")
    assert rows[0] == (1, "p", "adaptors", 0, 1.0, 1.0, "x", "gemma-4-31b", None, "t")


def test_verdict_update_rows_carries_model():
    # the judge model is now part of the verdicts PK, so it must thread through the row
    # builder (None preserved here; update_verdicts coerces the live default).
    records = [{"code_number": "9", "pdf_path": "p", "criteria": {
        "two_worlds": {"verdict": "met", "confidence": 1.0}}}]
    rows = db.verdict_update_rows(records, run_ts="t", model="deepseek/deepseek-v4-pro")
    assert rows == [(9, "p", "two_worlds", "met", 1.0, None,
                     "deepseek/deepseek-v4-pro", None, "t")]


def test_verdict_tables_key_on_judge_model():
    # model is part of the PK of both verdict tables so multiple judges coexist
    # non-destructively (mirrors the embedding side's `run` column).
    ddl = "\n".join(db.DDL)
    assert "PRIMARY KEY (code_number, pdf_path, criterion, model)" in ddl
    assert "PRIMARY KEY (code_number, pdf_path, criterion, chunk_idx, model)" in ddl


def test_verdict_update_rows_carries_prompt_hash():
    # the graded path stamps the criterion's prompt-version hash onto each verdict
    records = [{"code_number": "9", "pdf_path": "p", "criteria": {
        "two_worlds": {"verdict": "met", "confidence": 1.0, "graded": 0.5,
                       "prompt_hash": "abc123"}}}]
    rows = db.verdict_update_rows(records, run_ts="t")
    assert rows == [(9, "p", "two_worlds", "met", 1.0, 0.5, None, "abc123", "t")]


def test_chunk_verdict_rows_carries_prompt_hash():
    records = [{"code_number": "1", "pdf_path": "p", "criterion": "adaptors",
                "chunk_idx": 0, "agreement": 1.0, "confidence": 1.0,
                "evidence_quote": "x", "prompt_hash": "deadbeef"}]
    rows = db.chunk_verdict_rows(records, run_ts="t", model="m")
    assert rows[0] == (1, "p", "adaptors", 0, 1.0, 1.0, "x", "m", "deadbeef", "t")


def test_prompt_registry_rows_grain():
    # the prompt registry stores each prompt version's full template text once, keyed by hash
    entries = [{"prompt_hash": "h1", "criterion": "two_worlds", "prompt_text": "TEMPLATE"}]
    rows = db.prompt_registry_rows(entries, run_ts="t")
    assert rows == [("h1", "two_worlds", "TEMPLATE", "t")]


# --- graceful reconnect on transient "server gone away" / "lost connection" ----
#
# The persist path (register_prompts -> store_chunk_verdicts -> update_verdicts) is a
# sequence of *idempotent* units (guarded init_schema + upserts), so a transient MySQL
# drop mid-write can be recovered by reconnecting and re-running the unit from the top
# rather than aborting and losing the persist progress.

class _FakeConn:
    """Records that it was closed; close() must never raise on an already-dead conn."""
    _next_id = 0

    def __init__(self):
        self.id = _FakeConn._next_id
        _FakeConn._next_id += 1
        self.closed = False

    def close(self):
        self.closed = True


def _conn_factory():
    """Returns (connect_fn, conns) — connect_fn hands out fresh recorded fake conns."""
    conns = []

    def connect_fn(env=None):
        c = _FakeConn()
        conns.append(c)
        return c

    return connect_fn, conns


def test_run_with_reconnect_retries_transient_then_succeeds():
    connect_fn, conns = _conn_factory()
    slept = []
    calls = {"n": 0}

    def work(conn):
        calls["n"] += 1
        if calls["n"] < 3:                      # fail the first two attempts
            raise pymysql.err.OperationalError(2013, "Lost connection during query")
        return "persisted"

    out = db.run_with_reconnect(work, connect_fn=connect_fn, retries=3,
                                backoff=0.0, sleep=slept.append)
    assert out == "persisted"
    assert calls["n"] == 3
    assert len(conns) == 3                       # a fresh connection per attempt
    assert all(c.closed for c in conns)          # every connection closed, even the dead ones
    assert len(slept) == 2                       # backoff between the two failures


def test_run_with_reconnect_reraises_nontransient_immediately():
    connect_fn, conns = _conn_factory()

    def work(conn):
        raise pymysql.err.OperationalError(1064, "You have an error in your SQL syntax")

    with pytest.raises(pymysql.err.OperationalError):
        db.run_with_reconnect(work, connect_fn=connect_fn, retries=3,
                              backoff=0.0, sleep=lambda s: None)
    assert len(conns) == 1                        # no reconnect on a non-transient error
    assert conns[0].closed


def test_run_with_reconnect_gives_up_after_retries():
    connect_fn, conns = _conn_factory()
    calls = {"n": 0}

    def work(conn):
        calls["n"] += 1
        raise pymysql.err.OperationalError(2006, "MySQL server has gone away")

    with pytest.raises(pymysql.err.OperationalError):
        db.run_with_reconnect(work, connect_fn=connect_fn, retries=2,
                              backoff=0.0, sleep=lambda s: None)
    assert calls["n"] == 3                        # initial attempt + 2 retries
    assert len(conns) == 3
    assert all(c.closed for c in conns)
