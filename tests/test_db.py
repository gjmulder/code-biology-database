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
    # raw_agreement defaults to the live agreement, coverage to None, grounding_failed to 0
    rows = db.chunk_verdict_rows(records, run_ts="t")
    assert (5, "pdfs/a.pdf", "two_worlds", 0, 0.5, 0.66, "q0", None, None,
            0.5, None, 0, "t") in rows
    assert (5, "pdfs/a.pdf", "two_worlds", 1, -0.5, 0.33, "", None, None,
            -0.5, None, 0, "t") in rows
    assert len(rows) == 2


def test_chunk_verdict_rows_carries_model():
    records = [{"code_number": "1", "pdf_path": "p", "criterion": "adaptors",
                "chunk_idx": 0, "agreement": 1.0, "confidence": 1.0,
                "evidence_quote": "x"}]
    rows = db.chunk_verdict_rows(records, run_ts="t", model="gemma-4-31b")
    assert rows[0] == (1, "p", "adaptors", 0, 1.0, 1.0, "x", "gemma-4-31b", None,
                       1.0, None, 0, "t")


def test_chunk_verdict_rows_carries_pre_gate_snapshot():
    # a gated cell: live agreement neutralised, but the pre-gate value + coverage are stored
    records = [{"code_number": "1", "pdf_path": "p", "criterion": "adaptors",
                "chunk_idx": 0, "agreement": 0.0, "confidence": 0.66, "evidence_quote": "x",
                "raw_agreement": 1.0, "coverage": 0.4, "grounding_failed": True}]
    rows = db.chunk_verdict_rows(records, run_ts="t", model="m")
    assert rows[0] == (1, "p", "adaptors", 0, 0.0, 0.66, "x", "m", None,
                       1.0, 0.4, 1, "t")


def test_chunk_verdicts_ddl_has_pre_gate_columns():
    ddl = "\n".join(db.DDL)
    for col in ("raw_agreement", "coverage", "grounding_failed"):
        assert col in ddl


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
    assert rows[0] == (1, "p", "adaptors", 0, 1.0, 1.0, "x", "m", "deadbeef",
                       1.0, None, 0, "t")


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


# --- gold_labels (Phase 5): the Barbieri-anchored gold reference set ---------

GOLD_RECORDS = [
    {"code_number": "3", "pdf_path": "pdfs/g.pdf", "polarity": "pos", "tier": "2",
     "source": "db", "criterion": "all", "evidence": "Genetic code · topic 3"},
    {"code_number": "29", "pdf_path": "pdfs/n.pdf", "polarity": "neg", "tier": "soft",
     "source": "implicit", "criterion": "all", "evidence": "non-molecular: topic 11"},
]


def test_gold_rows_grain_and_keying():
    # one row per gold_set.csv record, tuple
    # (code_number, pdf_path, polarity, criterion, tier, source, evidence, run_ts);
    # code_number coerced to int.
    rows = db.gold_rows(GOLD_RECORDS, run_ts="t")
    assert (3, "pdfs/g.pdf", "pos", "all", "2", "db", "Genetic code · topic 3", "t") in rows
    assert (29, "pdfs/n.pdf", "neg", "all", "soft", "implicit",
            "non-molecular: topic 11", "t") in rows
    assert len(rows) == 2


def test_gold_rows_criterion_defaults_to_all():
    # a record with no/blank criterion lands as the run-agnostic 'all' default
    rows = db.gold_rows([{"code_number": 1, "pdf_path": "p", "polarity": "pos",
                          "tier": "1", "source": "code0", "criterion": "",
                          "evidence": "x"}], run_ts="t")
    assert rows[0] == (1, "p", "pos", "all", "1", "code0", "x", "t")


def test_gold_labels_ddl_run_and_judge_agnostic():
    ddl = "\n".join(db.DDL)
    assert "CREATE TABLE IF NOT EXISTS gold_labels" in ddl
    # ground truth: keyed on (code, paper, polarity, criterion) — NO run, NO judge model
    assert "PRIMARY KEY (code_number, pdf_path, polarity, criterion)" in ddl
    # locate the gold_labels DDL block and assert it carries no run/model column
    block = ddl.split("CREATE TABLE IF NOT EXISTS gold_labels", 1)[1].split("ENGINE", 1)[0]
    assert " run " not in block and "model" not in block


# --- load_env: .env parsing + process-env override (offline) ----------------

def test_load_env_parses_file_and_skips_comments(tmp_path, monkeypatch):
    # ensure no stray process env leaks into the parse
    for k in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASS"):
        monkeypatch.delenv(k, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n"
        "DB_HOST = asushimu \n"
        "DB_PORT=3306\n"
        "DB_PASS=p@ss=with=equals\n"   # value may contain '=' — only the first splits
        "\n"
        "   # indented comment\n"
    )
    env = db.load_env(str(env_file))
    assert env["DB_HOST"] == "asushimu"          # whitespace trimmed both sides
    assert env["DB_PORT"] == "3306"
    assert env["DB_PASS"] == "p@ss=with=equals"   # split only on the first '='
    assert "a comment" not in " ".join(env)


def test_load_env_process_env_overrides_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("DB_HOST=fromfile\n")
    monkeypatch.setenv("DB_HOST", "fromenv")
    assert db.load_env(str(env_file))["DB_HOST"] == "fromenv"


def test_load_env_missing_file_is_empty_without_process_env(tmp_path, monkeypatch):
    for k in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASS"):
        monkeypatch.delenv(k, raising=False)
    assert db.load_env(str(tmp_path / "nope.env")) == {}


# --- migrate_runs: the idempotent, guarded schema migration -----------------
#
# CLAUDE.md §3/§7.8 lean on migrate_runs being a no-op on an already-migrated DB (the
# dump is the rollback path, but the migration must not fire twice). A programmable cursor
# answers the information_schema probes from an in-memory schema and records every mutating
# statement, so the guards are tested offline without a live MySQL.

class _SchemaCursor:
    """Answers _has_column / _pk_columns from an in-memory schema; records mutations.

    ``columns[table]`` = set of column names; ``pks[table]`` = ordered PK column list.
    Every non-SELECT statement is appended to ``executed`` (the migration's effect)."""

    def __init__(self, columns, pks):
        self.columns = columns
        self.pks = pks
        self.executed = []
        self._pending = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=()):
        low = " ".join(sql.split()).lower()
        if "information_schema.columns" in low:
            table, col = params
            self._pending = [(1,)] if col in self.columns.get(table, set()) else []
        elif "information_schema.key_column_usage" in low:
            (table,) = params
            self._pending = [(c,) for c in self.pks.get(table, [])]
        else:
            self.executed.append((" ".join(sql.split()), tuple(params)))
            self._pending = []

    def fetchone(self):
        return self._pending[0] if self._pending else None

    def fetchall(self):
        return list(self._pending)


class _CursorConn:
    """Hands out a single pre-built cursor; records commit()."""

    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def close(self):
        pass


def _migrated_schema():
    """An already-run-scoped, judge-keyed schema — every guard should see its column/PK."""
    run_tables = list(db._OLD_PKS)
    columns = {t: set(db._OLD_PKS[t]) | {"run"} for t in run_tables}
    columns["verdicts"] = {"code_number", "pdf_path", "criterion", "verdict",
                           "confidence", "graded", "model", "prompt_hash"}
    columns["chunk_verdicts"] = {"code_number", "pdf_path", "criterion", "chunk_idx",
                                 "model", "prompt_hash", "raw_agreement", "coverage",
                                 "grounding_failed"}
    # embedding_scores no longer carries the legacy verdict/confidence columns
    pks = {t: ["run"] + db._OLD_PKS[t] for t in run_tables}
    pks["verdicts"] = ["code_number", "pdf_path", "criterion", "model"]
    pks["chunk_verdicts"] = ["code_number", "pdf_path", "criterion", "chunk_idx", "model"]
    return columns, pks


def test_migrate_runs_is_noop_on_already_migrated_db():
    cur = _SchemaCursor(*_migrated_schema())
    conn = _CursorConn(cur)
    db.migrate_runs(conn)
    assert cur.executed == []          # not a single ALTER/INSERT/UPDATE fired
    assert conn.committed              # still commits the (empty) transaction


def test_migrate_runs_adds_run_and_rebuilds_pks_on_legacy_db():
    # legacy: no `run` anywhere, embedding_scores still has verdict/confidence,
    # verdicts lacks graded/prompt_hash and is not judge-keyed.
    columns = {t: set(db._OLD_PKS[t]) for t in db._OLD_PKS}
    columns["embedding_scores"] |= {"verdict", "confidence"}
    columns["verdicts"] = {"code_number", "pdf_path", "criterion", "verdict", "confidence"}
    columns["chunk_verdicts"] = {"code_number", "pdf_path", "criterion", "chunk_idx"}
    pks = {t: list(db._OLD_PKS[t]) for t in db._OLD_PKS}
    pks["verdicts"] = ["code_number", "pdf_path", "criterion"]
    pks["chunk_verdicts"] = ["code_number", "pdf_path", "criterion", "chunk_idx"]
    cur = _SchemaCursor(columns, pks)
    db.migrate_runs(_CursorConn(cur))
    joined = "\n".join(sql for sql, _ in cur.executed)
    params_seen = [p for _, ps in cur.executed for p in ps]
    # every embedding-side table gains `run` and a rebuilt PK with run leading
    for table in db._OLD_PKS:
        assert f"ALTER TABLE {table} ADD COLUMN run" in joined
        assert f"ADD PRIMARY KEY (run, {', '.join(db._OLD_PKS[table])})" in joined
    # legacy verdict columns back-filled into `verdicts`, then dropped
    assert "INSERT INTO verdicts" in joined
    assert "DROP COLUMN verdict, DROP COLUMN confidence" in joined
    # graded + prompt provenance added; judge `model` promoted into both verdict PKs
    assert "ADD COLUMN graded DOUBLE" in joined
    assert "ADD PRIMARY KEY (code_number, pdf_path, criterion, model)" in joined
    assert ("ADD PRIMARY KEY (code_number, pdf_path, criterion, chunk_idx, model)"
            in joined)
    # legacy NULL/blank judge tags back-filled to the Gemma corpus before the PK widens
    assert "UPDATE verdicts SET model=%s WHERE model IS NULL OR model=''" in joined
    assert db.BACKFILL_JUDGE_MODEL in params_seen


# --- fetch_report: assemble the report payload + verdict last-wins join ------

class _QueuedCursor:
    """Returns canned result-sets in execute() order — for the read assemblers."""

    def __init__(self, resultsets):
        self._rs = list(resultsets)
        self._cur = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=()):
        self._cur = self._rs.pop(0) if self._rs else []

    def fetchall(self):
        return list(self._cur)

    def fetchone(self):
        return self._cur[0] if self._cur else None


def test_fetch_report_assembles_papers_scores_and_verdicts():
    # query order in fetch_report: scores, controls, pole_separation, run_meta, verdicts
    resultsets = [
        [  # embedding_scores: code, pid, method, criterion, e
            (3, "pdfs/a.pdf", "chunk", "two_worlds", 0.4),
            (3, "pdfs/a.pdf", "chunk", "adaptors", 0.1),
        ],
        [("AGREE_genetic", "two_worlds", 0.9)],       # control_scores
        [("within", "two_worlds", 0.6)],              # pole_separation
        [("run_ts", "2026-06-19 00:00:00")],          # run_meta
        [  # verdicts: code, pid, criterion, verdict, confidence — ORDER BY run_ts
            (3, "pdfs/a.pdf", "two_worlds", "unclear", 0.5),
            (3, "pdfs/a.pdf", "two_worlds", "met", 0.8),   # later row wins (last-wins)
            (3, "pdfs/a.pdf", "adaptors", "not_met", 0.7),
        ],
    ]
    conn = _CursorConn(_QueuedCursor(resultsets))
    rep = db.fetch_report(conn, run="baseline")
    assert rep["order"] == ["pdfs/a.pdf"]
    paper = rep["papers"]["pdfs/a.pdf"]
    assert paper["code"] == 3
    assert paper["scores"]["chunk"] == {"two_worlds": 0.4, "adaptors": 0.1}
    # last verdict row per (code,pid,crit) wins the join
    assert paper["verdict"]["two_worlds"] == "met"
    assert paper["confidence"]["two_worlds"] == 0.8
    assert paper["verdict"]["adaptors"] == "not_met"
    assert rep["controls"] == {"AGREE_genetic": {"two_worlds": 0.9}}
    assert rep["pole_separation"]["within"]["two_worlds"] == 0.6
    assert rep["meta"]["run_ts"] == "2026-06-19 00:00:00"


def test_fetch_gold_keys_on_code_pid_criterion():
    rows = [
        (3, "pdfs/g.pdf", "pos", "all", "2", "db", "Genetic code"),
        (29, "pdfs/n.pdf", "neg", "all", "soft", "implicit", "non-molecular"),
    ]
    conn = _CursorConn(_QueuedCursor([rows]))
    gold = db.fetch_gold(conn)
    assert gold[(3, "pdfs/g.pdf", "all")] == {
        "polarity": "pos", "tier": "2", "source": "db", "evidence": "Genetic code"}
    assert gold[(29, "pdfs/n.pdf", "all")]["polarity"] == "neg"


def test_fetch_chunk_topics_groups_by_pid_in_chunk_order():
    rows = [
        ("pdfs/a.pdf", 0, 3, 0.51),
        ("pdfs/a.pdf", 1, 3, 0.62),
        ("pdfs/b.pdf", 0, 11, 0.40),
    ]
    conn = _CursorConn(_QueuedCursor([rows]))
    out = db.fetch_chunk_topics(conn)
    assert out["pdfs/a.pdf"] == [(0, 3, 0.51), (1, 3, 0.62)]
    assert out["pdfs/b.pdf"] == [(0, 11, 0.40)]


def test_fetch_chunk_verdicts_groups_by_paper_criterion():
    rows = [
        ("pdfs/a.pdf", "two_worlds", 0, 0.5, 0.8, "quote one"),
        ("pdfs/a.pdf", "two_worlds", 1, 0.0, 0.2, ""),
    ]
    conn = _CursorConn(_QueuedCursor([rows]))
    out = db.fetch_chunk_verdicts(conn, judge="deepseek/deepseek-v4-pro")
    assert out[("pdfs/a.pdf", "two_worlds")] == [
        (0, 0.5, 0.8, "quote one"), (1, 0.0, 0.2, "")]


def test_fetch_report_scopes_verdicts_to_judge_when_given():
    # with judge=, the verdict SELECT must filter WHERE model=%s (not newest-wins)
    cur = _QueuedCursor([[], [], [], [], []])
    captured = {}
    orig = cur.execute

    def spy(sql, params=()):
        if "FROM verdicts" in sql:
            captured["sql"] = " ".join(sql.split())
            captured["params"] = tuple(params)
        return orig(sql, params)

    cur.execute = spy
    db.fetch_report(_CursorConn(cur), run="baseline", judge="deepseek/deepseek-v4-pro")
    assert "WHERE model=%s" in captured["sql"]
    assert captured["params"] == ("deepseek/deepseek-v4-pro",)


# --- store_* write paths: the system-of-record upserts (recording cursor) ---
#
# Thin wrappers over the (separately tested) *_rows transforms, but they own the upsert SQL
# and the judge/run keying, so a recording cursor confirms the right statement fires and the
# written-row count is returned — offline, no live MySQL.

class _RecCursor:
    """Records execute()/executemany() statements + rows; no DB."""

    def __init__(self):
        self.many = []   # [(sql, rows), ...]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=()):
        pass

    def executemany(self, sql, rows):
        self.many.append((" ".join(sql.split()), list(rows)))


def _rec_conn(monkeypatch):
    """A recording conn with init_schema stubbed (no DDL against a live server)."""
    monkeypatch.setattr(db, "init_schema", lambda conn: None)
    cur = _RecCursor()
    return _CursorConn(cur), cur


def test_update_verdicts_upserts_and_returns_count(monkeypatch):
    conn, cur = _rec_conn(monkeypatch)
    records = [{"code_number": 3, "pdf_path": "a.pdf",
                "criteria": {"two_worlds": {"verdict": "met", "confidence": 0.8,
                                            "graded": 1.0, "prompt_hash": "h"}}}]
    n = db.update_verdicts(conn, records, model="deepseek/deepseek-v4-pro")
    assert n == 1 and conn.committed
    sql, rows = cur.many[0]
    assert "INSERT INTO verdicts" in sql and "ON DUPLICATE KEY UPDATE" in sql
    assert rows[0][0] == 3 and rows[0][6] == "deepseek/deepseek-v4-pro"  # code, judge model


def test_store_gold_upserts_gold_labels(monkeypatch):
    conn, cur = _rec_conn(monkeypatch)
    n = db.store_gold(conn, GOLD_RECORDS)
    assert n == 2 and conn.committed
    sql, rows = cur.many[0]
    assert "INSERT INTO gold_labels" in sql
    assert len(rows) == 2


def test_delete_score_rows_noop_on_empty(monkeypatch):
    conn, cur = _rec_conn(monkeypatch)
    db.delete_score_rows(conn, [], run="baseline")
    assert cur.many == [] and not conn.committed   # nothing executed, nothing committed


def test_delete_score_rows_deletes_each_pid(monkeypatch):
    conn, cur = _rec_conn(monkeypatch)
    db.delete_score_rows(conn, ["a.pdf", "b.pdf"], run="baseline")
    sql, rows = cur.many[0]
    assert "DELETE FROM embedding_scores WHERE run=%s AND pdf_path=%s" in sql
    assert rows == [("baseline", "a.pdf"), ("baseline", "b.pdf")]
    assert conn.committed
