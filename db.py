"""MySQL persistence for the embedding analysis — the system of record.

All embedding output lives here (not JSON), indexed on the **code id**: the main
table ``embedding_scores`` has ``code_number`` as the leading column of its primary
key, one row per (code, paper, method, criterion). Diagnostics (pole separation,
control checks) and run metadata get their own small tables so the whole result set
is queryable.

Connection params come from ``.env`` (gitignored): ``DB_HOST DB_PORT DB_NAME
DB_USER DB_PASS``. The pure ``*_rows`` transforms are unit-tested offline; the SQL
runs against the live server on asushimu.
"""

import os
import pathlib

import numpy as np
import pymysql

CRITERIA = ["two_worlds", "adaptors", "arbitrariness"]

# Every embedding-side table is **run-scoped**: a ``run`` column (``baseline`` for the
# harrier vectors, e.g. ``gte-qwen2`` for a model swap) is the leading PK column so two
# models' results coexist and are directly comparable. The LLM verdicts are NOT here —
# they are produced by the judge, not the embedding model, so they are normalised into
# the run-agnostic ``verdicts`` table below and shared across runs via a join.
DDL = [
    """
    CREATE TABLE IF NOT EXISTS embedding_scores (
        run          VARCHAR(64)  NOT NULL DEFAULT 'baseline',
        code_number  INT          NOT NULL,
        pdf_path     VARCHAR(255) NOT NULL,
        method       VARCHAR(16)  NOT NULL,   -- full | abstract | chunk
        criterion    VARCHAR(32)  NOT NULL,   -- two_worlds | adaptors | arbitrariness
        e            DOUBLE       NOT NULL,
        model        VARCHAR(128),            -- the EMBEDDING model
        run_ts       DATETIME,
        PRIMARY KEY (run, code_number, pdf_path, method, criterion)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # Run-agnostic LLM verdicts (the shared ground-truth axis): one row per
    # (paper, criterion), independent of the embedding run. ``model`` is the JUDGE model.
    """
    CREATE TABLE IF NOT EXISTS verdicts (
        code_number INT          NOT NULL,
        pdf_path    VARCHAR(255) NOT NULL,
        criterion   VARCHAR(32)  NOT NULL,
        verdict     VARCHAR(16),
        confidence  DOUBLE,
        model       VARCHAR(128),
        run_ts      DATETIME,
        PRIMARY KEY (code_number, pdf_path, criterion)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS control_scores (
        run        VARCHAR(64) NOT NULL DEFAULT 'baseline',
        name       VARCHAR(64) NOT NULL,
        criterion  VARCHAR(32) NOT NULL,
        e          DOUBLE      NOT NULL,
        run_ts     DATETIME,
        PRIMARY KEY (run, name, criterion)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS pole_separation (
        run     VARCHAR(64) NOT NULL DEFAULT 'baseline',
        pole    VARCHAR(8)  NOT NULL,   -- pos | neg
        pair    VARCHAR(64) NOT NULL,
        cosine  DOUBLE      NOT NULL,
        run_ts  DATETIME,
        PRIMARY KEY (run, pole, pair)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS run_meta (
        run  VARCHAR(64)  NOT NULL DEFAULT 'baseline',
        k    VARCHAR(32)  NOT NULL,
        v    VARCHAR(255),
        PRIMARY KEY (run, k)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # Raw document embeddings kept so the contrast math (centring, whitening, axis
    # projection, orthogonalization) can run **offline** from the DB without a GPU
    # re-embed. full/abstract are one vector (chunk_idx 0); chunk keeps every window.
    """
    CREATE TABLE IF NOT EXISTS doc_vectors (
        run          VARCHAR(64)  NOT NULL DEFAULT 'baseline',
        code_number  INT          NOT NULL,
        pdf_path     VARCHAR(255) NOT NULL,
        method       VARCHAR(16)  NOT NULL,   -- full | abstract | chunk
        chunk_idx    INT          NOT NULL,   -- 0 for full/abstract; 0..n for chunk
        dim          INT          NOT NULL,
        vec          LONGBLOB     NOT NULL,   -- float32 little-endian bytes
        run_ts       DATETIME,
        PRIMARY KEY (run, code_number, pdf_path, method, chunk_idx)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # Raw control-document vectors, kept for the same reason as doc_vectors: so the
    # offline recompute can score the controls through the **same** corpus geometry as the
    # papers (centred/whitened/decongested), instead of leaving the stale pre-lever
    # double-cosine numbers in the report. One representative vector per control.
    """
    CREATE TABLE IF NOT EXISTS control_vectors (
        run     VARCHAR(64) NOT NULL DEFAULT 'baseline',
        name    VARCHAR(64) NOT NULL,
        dim     INT         NOT NULL,
        vec     LONGBLOB    NOT NULL,   -- float32 little-endian bytes
        run_ts  DATETIME,
        PRIMARY KEY (run, name)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS pole_vectors (
        run        VARCHAR(64) NOT NULL DEFAULT 'baseline',
        criterion  VARCHAR(32) NOT NULL,
        pole       VARCHAR(8)  NOT NULL,   -- pos | neg
        dim        INT         NOT NULL,
        vec        LONGBLOB    NOT NULL,
        run_ts     DATETIME,
        PRIMARY KEY (run, criterion, pole)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # The 24 scientometric topic centroids (Paredes & Prinz 2025), embedded with the
    # SAME model/precision/run as the corpus chunks so a paper chunk can be assigned to
    # its nearest topic (centred nearest-centroid). One raw vector per topic, per run.
    """
    CREATE TABLE IF NOT EXISTS topic_centroids (
        run       VARCHAR(64)  NOT NULL DEFAULT 'baseline',
        topic_id  INT          NOT NULL,
        label     VARCHAR(128),
        dim       INT          NOT NULL,
        vec       LONGBLOB     NOT NULL,   -- float32 little-endian bytes
        run_ts    DATETIME,
        PRIMARY KEY (run, topic_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # Per-chunk nearest-topic assignment (assign_topics.py): each chunk's argmax topic
    # and its cosine in the centred space. Lets the report stratify ρ by a paper's
    # (max-pooled) dominant topic without re-deriving it. Run/method-keyed.
    """
    CREATE TABLE IF NOT EXISTS chunk_topics (
        run         VARCHAR(64) NOT NULL DEFAULT 'baseline',
        code_number INT         NOT NULL,
        pdf_path    VARCHAR(512) NOT NULL,
        method      VARCHAR(16) NOT NULL,
        chunk_idx   INT         NOT NULL,
        topic_id    INT         NOT NULL,
        sim         DOUBLE      NOT NULL,
        run_ts      DATETIME,
        PRIMARY KEY (run, code_number, pdf_path, method, chunk_idx)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]


def pack_vec(v):
    """Pack a float sequence into compact float32 little-endian bytes for BLOB storage."""
    return np.asarray(v, dtype="<f4").tobytes()


def unpack_vec(blob):
    """Inverse of :func:`pack_vec` — a 1-D float32 ``np.ndarray``."""
    return np.frombuffer(blob, dtype="<f4")


def load_env(path=".env"):
    p = pathlib.Path(path)
    env = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    # process env overrides the file (handy for the remote writer)
    for k in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASS"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def connect(env=None):
    env = env or load_env()
    return pymysql.connect(
        host=env["DB_HOST"], port=int(env.get("DB_PORT", 3306)),
        user=env["DB_USER"], password=env["DB_PASS"], database=env["DB_NAME"],
        charset="utf8mb4", autocommit=False,
    )


# Pre-``run``-column primary keys, used by the live migration to rebuild each PK with
# ``run`` prepended. Order matters: it is the exact old PK column order.
_OLD_PKS = {
    "embedding_scores": ["code_number", "pdf_path", "method", "criterion"],
    "control_scores": ["name", "criterion"],
    "pole_separation": ["pole", "pair"],
    "run_meta": ["k"],
    "doc_vectors": ["code_number", "pdf_path", "method", "chunk_idx"],
    "control_vectors": ["name"],
    "pole_vectors": ["criterion", "pole"],
}


def _has_column(c, table, column):
    c.execute("SELECT 1 FROM information_schema.columns "
              "WHERE table_schema=DATABASE() AND table_name=%s AND column_name=%s",
              (table, column))
    return c.fetchone() is not None


def migrate_runs(conn):
    """Idempotently migrate a pre-``run`` DB to the run-scoped schema (safe to re-run).

    For a fresh DB the DDL already includes ``run`` + the ``verdicts`` table, so every
    step here is a guarded no-op. For the existing harrier DB it: (1) adds a ``run``
    column (back-filled to ``baseline``) to each embedding-side table and rebuilds its PK
    with ``run`` leading; (2) creates ``verdicts`` and back-fills it from the soon-to-be
    dropped ``embedding_scores.verdict``/``confidence`` columns, then drops them. Every
    step is guarded on ``information_schema`` so a second run does nothing."""
    with conn.cursor() as c:
        for table, old_pk in _OLD_PKS.items():
            if _has_column(c, table, "run"):
                continue
            c.execute(f"ALTER TABLE {table} "
                      "ADD COLUMN run VARCHAR(64) NOT NULL DEFAULT 'baseline'")
            cols = ", ".join(["run"] + old_pk)
            c.execute(f"ALTER TABLE {table} DROP PRIMARY KEY, ADD PRIMARY KEY ({cols})")
        # verdicts table is created via DDL (init_schema runs first); back-fill from the
        # old embedding_scores verdict columns while they still exist, then drop them.
        if _has_column(c, "embedding_scores", "verdict"):
            c.execute(
                "INSERT INTO verdicts "
                "(code_number, pdf_path, criterion, verdict, confidence, model, run_ts) "
                "SELECT DISTINCT code_number, pdf_path, criterion, verdict, confidence, "
                "       NULL, run_ts "
                "FROM embedding_scores WHERE verdict IS NOT NULL "
                "ON DUPLICATE KEY UPDATE verdict=VALUES(verdict), "
                "confidence=VALUES(confidence), run_ts=VALUES(run_ts)")
            c.execute("ALTER TABLE embedding_scores "
                      "DROP COLUMN verdict, DROP COLUMN confidence")
    conn.commit()


def init_schema(conn):
    with conn.cursor() as c:
        for stmt in DDL:
            c.execute(stmt)
    conn.commit()
    migrate_runs(conn)


# --- pure transforms (unit-tested offline) ---------------------------------

def scores_to_rows(out, recs, run_ts, run="baseline"):
    """``out['scores'][pid][method][criterion] = e`` → row tuples
    ``(code_number, pdf_path, method, criterion, e, model, run_ts, run)``.

    The embedding axis only — verdicts/confidence are no longer written here (they live
    in the run-agnostic ``verdicts`` table). ``model`` is the embedding model."""
    by_pid = {r["pdf_path"]: r for r in recs}
    model = out.get("model")
    rows = []
    for pid, methods in out["scores"].items():
        r = by_pid.get(pid, {})
        code = r.get("code_number")
        for method, cdict in methods.items():
            for criterion, e in cdict.items():
                rows.append((code, pid, method, criterion, float(e),
                             model, run_ts, run))
    return rows


def controls_to_rows(out, run_ts, run="baseline"):
    return [(name, crit, float(e), run_ts, run)
            for name, cdict in out.get("controls", {}).items()
            for crit, e in cdict.items()]


def poles_to_rows(out, run_ts, run="baseline"):
    return [(pole, pair, float(cos), run_ts, run)
            for pole, pairs in out.get("pole_separation", {}).items()
            for pair, cos in pairs.items()]


def doc_vectors_to_rows(out, recs, run_ts, run="baseline"):
    """``out['doc_vectors'][pid][method] = vec | [vec, ...]`` → row tuples
    ``(code_number, pdf_path, method, chunk_idx, dim, vec_blob, run_ts, run)``.

    full/abstract carry a single vector (stored at ``chunk_idx`` 0); chunk carries a
    list of per-window vectors stored at ``chunk_idx`` 0..n-1 in order."""
    by_pid = {r["pdf_path"]: r for r in recs}
    rows = []
    for pid, methods in out.get("doc_vectors", {}).items():
        code = by_pid.get(pid, {}).get("code_number")
        for method, vecs in methods.items():
            # normalise to a list of vectors so full/abstract/chunk share one path
            seq = vecs if (vecs and isinstance(vecs[0], (list, tuple))) else [vecs]
            for idx, v in enumerate(seq):
                rows.append((code, pid, method, idx, len(v), pack_vec(v), run_ts, run))
    return rows


def control_vectors_to_rows(out, run_ts, run="baseline"):
    """``out['control_vectors'][name] = vec`` → ``(name, dim, vec_blob, run_ts, run)``
    rows.

    One representative vector per control, persisted so the offline recompute can score
    the controls with the corpus geometry. Empty/absent → no rows."""
    return [(name, len(v), pack_vec(v), run_ts, run)
            for name, v in out.get("control_vectors", {}).items()]


def pole_vectors_to_rows(out, run_ts, run="baseline"):
    """``out['pole_vectors'][criterion][pole] = vec`` → ``(criterion, pole, dim,
    vec_blob, run_ts, run)`` rows."""
    return [(crit, pole, len(v), pack_vec(v), run_ts, run)
            for crit, poles in out.get("pole_vectors", {}).items()
            for pole, v in poles.items()]


def topic_centroids_to_rows(out, run_ts, run="baseline"):
    """``out['centroids'] = [{topic_id, label, vec}, ...]`` → row tuples
    ``(topic_id, label, dim, vec_blob, run_ts, run)``.

    One row per scientometric topic centroid; the vector is packed to float32 bytes
    like every other raw-vector table. Empty/absent → no rows."""
    return [(c["topic_id"], c.get("label"), len(c["vec"]),
             pack_vec(c["vec"]), run_ts, run)
            for c in out.get("centroids", [])]


def chunk_topic_rows(assignments, codes, run_ts, method="chunk", run="baseline"):
    """``assignments[pid]['chunks'] = [(chunk_idx, topic_id, sim), ...]`` → row tuples
    ``(code_number, pdf_path, method, chunk_idx, topic_id, sim, run_ts, run)``.

    One row per chunk assignment (``assign_topics.build_assignments`` output);
    ``codes[pid] = code_number``. The paper's dominant topic is recoverable by max-pool
    over these rows, so only the per-chunk grain is stored. Empty → no rows."""
    return [(codes.get(pid), pid, method, idx, tid, float(sim), run_ts, run)
            for pid, rec in assignments.items()
            for idx, tid, sim in rec["chunks"]]


def recompute_score_rows(scores, codes, run_ts, run="baseline"):
    """Leverred-``e`` rows from the offline recompute → ``(code_number, pdf_path,
    method, criterion, e, run_ts, run)`` for an **e-only** upsert that preserves the
    existing model. ``scores[pid][method][crit] = e``; ``codes[pid] = code_number``."""
    return [(codes.get(pid), pid, method, crit, float(e), run_ts, run)
            for pid, methods in scores.items()
            for method, cdict in methods.items()
            for crit, e in cdict.items()]


def verdict_update_rows(records, run_ts=None, model=None):
    """Judged ``criteria_judge`` records → tuples for the run-agnostic ``verdicts`` table.

    Each tuple is ``(code_number, pdf_path, criterion, verdict, confidence, model,
    run_ts)`` — one row per (paper, criterion), with no embedding method or run (the
    verdict is shared by every embedding run via a join). ``model`` is the judge model
    (not carried on the record → ``None`` by default). Criteria with no ``verdict`` are
    skipped so a missing judgment never clobbers an existing one."""
    rows = []
    for r in records:
        code = int(r["code_number"])
        pid = r["pdf_path"]
        for crit, v in r.get("criteria", {}).items():
            verdict = v.get("verdict")
            if not verdict:
                continue
            rows.append((code, pid, crit, verdict, v.get("confidence"), model, run_ts))
    return rows


def update_verdicts(conn, records):
    """Upsert the LLM verdicts into the run-agnostic ``verdicts`` table.

    Returns the number of (paper, criterion) verdicts written. Run-agnostic: every
    embedding run shares these labels via a join on (code_number, pdf_path, criterion),
    so judging is never repeated per embedding model."""
    import datetime
    init_schema(conn)
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = verdict_update_rows(records, run_ts=run_ts)
    with conn.cursor() as c:
        c.executemany(
            "INSERT INTO verdicts "
            "(code_number,pdf_path,criterion,verdict,confidence,model,run_ts) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s) AS new "
            "ON DUPLICATE KEY UPDATE verdict=new.verdict, confidence=new.confidence, "
            "model=new.model, run_ts=new.run_ts",
            rows)
    conn.commit()
    return len(rows)


def within_rows(within, run_ts, run="baseline"):
    """Centred pole widths → ``pole_separation`` rows under the ``within`` pole, one per
    criterion (``pair`` = criterion name)."""
    return [("within", crit, float(cos), run_ts, run) for crit, cos in within.items()]


def meta_rows(out, run_ts, run="baseline"):
    return [(k, str(out.get(k)), run) for k in
            ("model", "dim", "use_4bit", "methods", "chunk_size", "chunk_overlap")
            if k in out] + [("run_ts", str(run_ts), run)]


# --- writes ----------------------------------------------------------------

def store(conn, out, recs, run_ts, run="baseline"):
    """Upsert the whole run (scores + controls + pole separation + meta) under ``run``.

    Writes the embedding axis only — verdicts are persisted separately, run-agnostically,
    via :func:`update_verdicts`."""
    init_schema(conn)
    with conn.cursor() as c:
        c.executemany(
            "INSERT INTO embedding_scores "
            "(code_number,pdf_path,method,criterion,e,model,run_ts,run) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) AS new "
            "ON DUPLICATE KEY UPDATE e=new.e, model=new.model, run_ts=new.run_ts",
            scores_to_rows(out, recs, run_ts, run))
        c.executemany(
            "INSERT INTO control_scores (name,criterion,e,run_ts,run) "
            "VALUES (%s,%s,%s,%s,%s) "
            "AS new ON DUPLICATE KEY UPDATE e=new.e, run_ts=new.run_ts",
            controls_to_rows(out, run_ts, run))
        c.executemany(
            "INSERT INTO pole_separation (pole,pair,cosine,run_ts,run) "
            "VALUES (%s,%s,%s,%s,%s) "
            "AS new ON DUPLICATE KEY UPDATE cosine=new.cosine, run_ts=new.run_ts",
            poles_to_rows(out, run_ts, run))
        c.executemany(
            "INSERT INTO run_meta (k,v,run) VALUES (%s,%s,%s) AS new "
            "ON DUPLICATE KEY UPDATE v=new.v",
            meta_rows(out, run_ts, run))
        dv = doc_vectors_to_rows(out, recs, run_ts, run)
        if dv:
            c.executemany(
                "INSERT INTO doc_vectors "
                "(code_number,pdf_path,method,chunk_idx,dim,vec,run_ts,run) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) AS new "
                "ON DUPLICATE KEY UPDATE dim=new.dim, vec=new.vec, run_ts=new.run_ts",
                dv)
        cv = control_vectors_to_rows(out, run_ts, run)
        if cv:
            c.executemany(
                "INSERT INTO control_vectors (name,dim,vec,run_ts,run) "
                "VALUES (%s,%s,%s,%s,%s) AS new "
                "ON DUPLICATE KEY UPDATE dim=new.dim, vec=new.vec, run_ts=new.run_ts",
                cv)
        pv = pole_vectors_to_rows(out, run_ts, run)
        if pv:
            c.executemany(
                "INSERT INTO pole_vectors (criterion,pole,dim,vec,run_ts,run) "
                "VALUES (%s,%s,%s,%s,%s,%s) AS new "
                "ON DUPLICATE KEY UPDATE dim=new.dim, vec=new.vec, run_ts=new.run_ts",
                pv)
    conn.commit()


# --- offline recompute (no GPU): read vectors, write leverred scores --------

def fetch_vectors(conn, run="baseline"):
    """Load persisted vectors for the offline recompute (for one embedding ``run``).

    Returns ``(doc_vecs, poles, codes)`` where
    ``doc_vecs[pid][method] = [vec, ...]`` (chunk windows ordered by ``chunk_idx``),
    ``poles[criterion] = {'pos': vec, 'neg': vec}``, and ``codes[pid] = code_number``."""
    with conn.cursor() as c:
        c.execute("SELECT code_number,pdf_path,method,chunk_idx,vec FROM doc_vectors "
                  "WHERE run=%s ORDER BY code_number,pdf_path,method,chunk_idx", (run,))
        doc_rows = c.fetchall()
        c.execute("SELECT criterion,pole,vec FROM pole_vectors WHERE run=%s", (run,))
        pole_rows = c.fetchall()

    doc_vecs, codes = {}, {}
    for code, pid, method, _idx, blob in doc_rows:
        codes[pid] = code
        doc_vecs.setdefault(pid, {}).setdefault(method, []).append(unpack_vec(blob))
    poles = {}
    for crit, pole, blob in pole_rows:
        poles.setdefault(crit, {})[pole] = unpack_vec(blob)
    return doc_vecs, poles, codes


def fetch_control_vectors(conn, run="baseline"):
    """Load persisted control vectors for one ``run`` → ``{name: np.ndarray}`` (empty if
    none stored).

    Returned for the offline recompute to score with the corpus geometry; an empty dict
    means the structural embed predates control-vector capture, so the report keeps the
    pre-lever control numbers and flags them."""
    with conn.cursor() as c:
        c.execute("SELECT name,vec FROM control_vectors WHERE run=%s", (run,))
        return {name: unpack_vec(blob) for name, blob in c.fetchall()}


def store_topic_centroids(conn, out, run="baseline"):
    """Upsert the embedded topic centroids under ``run`` (idempotent).

    ``out`` is the ``run_harrier_centroids`` transport payload. The ``run`` defaults to
    ``baseline`` so the centroids land in the same space as the harrier corpus chunks;
    pass ``out['run']`` through the caller to keep them aligned. Returns rows written."""
    import datetime
    init_schema(conn)
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = topic_centroids_to_rows(out, run_ts, run)
    with conn.cursor() as c:
        c.executemany(
            "INSERT INTO topic_centroids (topic_id,label,dim,vec,run_ts,run) "
            "VALUES (%s,%s,%s,%s,%s,%s) AS new "
            "ON DUPLICATE KEY UPDATE label=new.label, dim=new.dim, vec=new.vec, "
            "run_ts=new.run_ts",
            rows)
    conn.commit()
    return len(rows)


def fetch_topic_centroids(conn, run="baseline"):
    """Load topic centroids for one ``run`` → ``{topic_id: {'label': str,
    'vec': np.ndarray}}`` (empty if none stored)."""
    with conn.cursor() as c:
        c.execute("SELECT topic_id,label,vec FROM topic_centroids WHERE run=%s", (run,))
        return {tid: {"label": label, "vec": unpack_vec(blob)}
                for tid, label, blob in c.fetchall()}


def store_chunk_topics(conn, assignments, codes, method="chunk", run="baseline"):
    """Upsert per-chunk topic assignments under ``(run, method)`` (idempotent).

    ``assignments`` is the :func:`assign_topics.build_assignments` output; ``codes`` maps
    ``pdf_path`` → ``code_number``. Returns rows written."""
    import datetime
    init_schema(conn)
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = chunk_topic_rows(assignments, codes, run_ts, method=method, run=run)
    with conn.cursor() as c:
        c.executemany(
            "INSERT INTO chunk_topics "
            "(code_number,pdf_path,method,chunk_idx,topic_id,sim,run_ts,run) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) AS new "
            "ON DUPLICATE KEY UPDATE topic_id=new.topic_id, sim=new.sim, "
            "run_ts=new.run_ts",
            rows)
    conn.commit()
    return len(rows)


def fetch_chunk_topics(conn, run="baseline", method="chunk"):
    """Load chunk assignments for one ``(run, method)`` → ``{pdf_path:
    [(chunk_idx, topic_id, sim), ...]}`` ordered by ``chunk_idx`` (empty if none)."""
    with conn.cursor() as c:
        c.execute("SELECT pdf_path,chunk_idx,topic_id,sim FROM chunk_topics "
                  "WHERE run=%s AND method=%s ORDER BY pdf_path,chunk_idx",
                  (run, method))
        out = {}
        for pid, idx, tid, sim in c.fetchall():
            out.setdefault(pid, []).append((idx, tid, sim))
        return out


def delete_score_rows(conn, pids, run="baseline"):
    """Delete ``embedding_scores`` rows for the given ``pdf_path``s within one ``run``.

    Used to evict in-corpus self-reference documents (dropped from the recompute corpus)
    so they vanish from the regenerated report rather than lingering with stale pre-lever
    scores. No-op for an empty list."""
    if not pids:
        return
    with conn.cursor() as c:
        c.executemany("DELETE FROM embedding_scores WHERE run=%s AND pdf_path=%s",
                      [(run, p) for p in pids])
    conn.commit()


def apply_recompute(conn, scores, codes, within, params, run_ts, control_scores=None,
                    run="baseline"):
    """Persist an offline recompute under ``run``: upsert the leverred ``e`` (model
    untouched), the centred pole widths, the lever params into ``run_meta``, and — when
    ``control_scores`` is given — the leverred control ``e`` (so the report's control
    checks reflect the same geometry as the papers)."""
    init_schema(conn)
    with conn.cursor() as c:
        c.executemany(
            "INSERT INTO embedding_scores "
            "(code_number,pdf_path,method,criterion,e,run_ts,run) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s) AS new "
            "ON DUPLICATE KEY UPDATE e=new.e, run_ts=new.run_ts",
            recompute_score_rows(scores, codes, run_ts, run))
        c.executemany(
            "INSERT INTO pole_separation (pole,pair,cosine,run_ts,run) "
            "VALUES (%s,%s,%s,%s,%s) "
            "AS new ON DUPLICATE KEY UPDATE cosine=new.cosine, run_ts=new.run_ts",
            within_rows(within, run_ts, run))
        c.executemany(
            "INSERT INTO run_meta (k,v,run) VALUES (%s,%s,%s) AS new "
            "ON DUPLICATE KEY UPDATE v=new.v",
            [(k, str(v), run) for k, v in params.items()])
        if control_scores:
            c.executemany(
                "INSERT INTO control_scores (name,criterion,e,run_ts,run) "
                "VALUES (%s,%s,%s,%s,%s) AS new "
                "ON DUPLICATE KEY UPDATE e=new.e, run_ts=new.run_ts",
                [(name, crit, float(e), run_ts, run)
                 for name, cdict in control_scores.items()
                 for crit, e in cdict.items()])
    conn.commit()


# --- reads (for report generation) -----------------------------------------

def fetch_report(conn, run="baseline"):
    """Reassemble the report payload for one embedding ``run`` from the DB tables.

    The embedding columns are read ``WHERE run=%s``; the LLM verdicts are read from the
    run-agnostic ``verdicts`` table and attached per (code, pid, criterion), so every run
    shares the same ground-truth labels."""
    with conn.cursor() as c:
        c.execute("SELECT code_number,pdf_path,method,criterion,e "
                  "FROM embedding_scores WHERE run=%s ORDER BY code_number,pdf_path",
                  (run,))
        score_rows = c.fetchall()
        c.execute("SELECT name,criterion,e FROM control_scores WHERE run=%s", (run,))
        ctrl_rows = c.fetchall()
        c.execute("SELECT pole,pair,cosine FROM pole_separation WHERE run=%s", (run,))
        pole_rows = c.fetchall()
        c.execute("SELECT k,v FROM run_meta WHERE run=%s", (run,))
        meta = dict(c.fetchall())
        c.execute("SELECT code_number,pdf_path,criterion,verdict,confidence "
                  "FROM verdicts")
        verdict_rows = c.fetchall()

    # verdicts keyed by (code, pid, criterion) — shared across runs.
    vmap = {}
    for code, pid, crit, verdict, conf in verdict_rows:
        vmap[(code, pid, crit)] = (verdict, conf)

    papers = {}      # pid -> {"code": int, "scores": {method:{crit:e}},
                     #         "verdict": {crit:v}, "confidence": {crit:conf}}
    order = []
    for code, pid, method, crit, e in score_rows:
        p = papers.get(pid)
        if p is None:
            p = papers[pid] = {"code": code, "scores": {}, "verdict": {},
                               "confidence": {}}
            order.append(pid)
        p["scores"].setdefault(method, {})[crit] = e
        verdict, conf = vmap.get((code, pid, crit), (None, None))
        p["verdict"][crit] = verdict
        p["confidence"][crit] = conf

    controls = {}
    for name, crit, e in ctrl_rows:
        controls.setdefault(name, {})[crit] = e
    pole_sep = {}
    for pole, pair, cos in pole_rows:
        pole_sep.setdefault(pole, {})[pair] = cos

    return {"papers": papers, "order": order, "controls": controls,
            "pole_separation": pole_sep, "meta": meta}
