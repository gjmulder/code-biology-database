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

DDL = [
    """
    CREATE TABLE IF NOT EXISTS embedding_scores (
        code_number  INT          NOT NULL,
        pdf_path     VARCHAR(255) NOT NULL,
        method       VARCHAR(16)  NOT NULL,   -- full | abstract | chunk
        criterion    VARCHAR(32)  NOT NULL,   -- two_worlds | adaptors | arbitrariness
        e            DOUBLE       NOT NULL,
        verdict      VARCHAR(16),
        confidence   DOUBLE,
        model        VARCHAR(128),
        run_ts       DATETIME,
        PRIMARY KEY (code_number, pdf_path, method, criterion)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS control_scores (
        name       VARCHAR(64) NOT NULL,
        criterion  VARCHAR(32) NOT NULL,
        e          DOUBLE      NOT NULL,
        run_ts     DATETIME,
        PRIMARY KEY (name, criterion)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS pole_separation (
        pole    VARCHAR(8)  NOT NULL,   -- pos | neg
        pair    VARCHAR(64) NOT NULL,
        cosine  DOUBLE      NOT NULL,
        run_ts  DATETIME,
        PRIMARY KEY (pole, pair)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS run_meta (
        k  VARCHAR(32)  NOT NULL,
        v  VARCHAR(255),
        PRIMARY KEY (k)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # Raw document embeddings kept so the contrast math (centring, whitening, axis
    # projection, orthogonalization) can run **offline** from the DB without a GPU
    # re-embed. full/abstract are one vector (chunk_idx 0); chunk keeps every window.
    """
    CREATE TABLE IF NOT EXISTS doc_vectors (
        code_number  INT          NOT NULL,
        pdf_path     VARCHAR(255) NOT NULL,
        method       VARCHAR(16)  NOT NULL,   -- full | abstract | chunk
        chunk_idx    INT          NOT NULL,   -- 0 for full/abstract; 0..n for chunk
        dim          INT          NOT NULL,
        vec          LONGBLOB     NOT NULL,   -- float32 little-endian bytes
        run_ts       DATETIME,
        PRIMARY KEY (code_number, pdf_path, method, chunk_idx)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # Raw control-document vectors, kept for the same reason as doc_vectors: so the
    # offline recompute can score the controls through the **same** corpus geometry as the
    # papers (centred/whitened/decongested), instead of leaving the stale pre-lever
    # double-cosine numbers in the report. One representative vector per control.
    """
    CREATE TABLE IF NOT EXISTS control_vectors (
        name    VARCHAR(64) NOT NULL,
        dim     INT         NOT NULL,
        vec     LONGBLOB    NOT NULL,   -- float32 little-endian bytes
        run_ts  DATETIME,
        PRIMARY KEY (name)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS pole_vectors (
        criterion  VARCHAR(32) NOT NULL,
        pole       VARCHAR(8)  NOT NULL,   -- pos | neg
        dim        INT         NOT NULL,
        vec        LONGBLOB    NOT NULL,
        run_ts     DATETIME,
        PRIMARY KEY (criterion, pole)
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


def init_schema(conn):
    with conn.cursor() as c:
        for stmt in DDL:
            c.execute(stmt)
    conn.commit()


# --- pure transforms (unit-tested offline) ---------------------------------

def scores_to_rows(out, recs, run_ts):
    """``out['scores'][pid][method][criterion] = e`` + per-paper verdicts → row tuples
    ``(code_number, pdf_path, method, criterion, e, verdict, confidence, model, run_ts)``."""
    by_pid = {r["pdf_path"]: r for r in recs}
    model = out.get("model")
    rows = []
    for pid, methods in out["scores"].items():
        r = by_pid.get(pid, {})
        code = r.get("code_number")
        crit = r.get("criteria", {})
        for method, cdict in methods.items():
            for criterion, e in cdict.items():
                v = crit.get(criterion, {})
                rows.append((code, pid, method, criterion, float(e),
                             v.get("verdict"), v.get("confidence"), model, run_ts))
    return rows


def controls_to_rows(out, run_ts):
    return [(name, crit, float(e), run_ts)
            for name, cdict in out.get("controls", {}).items()
            for crit, e in cdict.items()]


def poles_to_rows(out, run_ts):
    return [(pole, pair, float(cos), run_ts)
            for pole, pairs in out.get("pole_separation", {}).items()
            for pair, cos in pairs.items()]


def doc_vectors_to_rows(out, recs, run_ts):
    """``out['doc_vectors'][pid][method] = vec | [vec, ...]`` → row tuples
    ``(code_number, pdf_path, method, chunk_idx, dim, vec_blob, run_ts)``.

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
                rows.append((code, pid, method, idx, len(v), pack_vec(v), run_ts))
    return rows


def control_vectors_to_rows(out, run_ts):
    """``out['control_vectors'][name] = vec`` → ``(name, dim, vec_blob, run_ts)`` rows.

    One representative vector per control, persisted so the offline recompute can score
    the controls with the corpus geometry. Empty/absent → no rows."""
    return [(name, len(v), pack_vec(v), run_ts)
            for name, v in out.get("control_vectors", {}).items()]


def pole_vectors_to_rows(out, run_ts):
    """``out['pole_vectors'][criterion][pole] = vec`` → ``(criterion, pole, dim,
    vec_blob, run_ts)`` rows."""
    return [(crit, pole, len(v), pack_vec(v), run_ts)
            for crit, poles in out.get("pole_vectors", {}).items()
            for pole, v in poles.items()]


def recompute_score_rows(scores, codes, run_ts):
    """Leverred-``e`` rows from the offline recompute → ``(code_number, pdf_path,
    method, criterion, e, run_ts)`` for an **e-only** upsert that preserves the existing
    verdict/confidence. ``scores[pid][method][crit] = e``; ``codes[pid] = code_number``."""
    return [(codes.get(pid), pid, method, crit, float(e), run_ts)
            for pid, methods in scores.items()
            for method, cdict in methods.items()
            for crit, e in cdict.items()]


def verdict_update_rows(records):
    """Judged ``criteria_judge`` records → tuples for a verdict-only UPDATE that
    preserves the embedding ``e`` (already persisted by the structural run).

    Each tuple is ``(verdict, confidence, code_number, pdf_path, criterion)`` matching
    ``SET verdict=%s, confidence=%s WHERE code_number=%s AND pdf_path=%s AND
    criterion=%s`` — method-agnostic, so the UPDATE writes the verdict onto every
    method row (full/abstract/chunk) for that paper×criterion. Criteria with no
    ``verdict`` are skipped so a missing judgment never clobbers an existing one."""
    rows = []
    for r in records:
        code = int(r["code_number"])
        pid = r["pdf_path"]
        for crit, v in r.get("criteria", {}).items():
            verdict = v.get("verdict")
            if not verdict:
                continue
            rows.append((verdict, v.get("confidence"), code, pid, crit))
    return rows


def update_verdicts(conn, records):
    """Upsert verdict/confidence onto existing ``embedding_scores`` rows (no e change).

    Returns the number of (paper, criterion) verdicts written. Used to backfill the
    LLM verdicts for the embedding corpus after judging, so ρ(e, verdict) is
    measurable from the DB without re-embedding."""
    rows = verdict_update_rows(records)
    with conn.cursor() as c:
        c.executemany(
            "UPDATE embedding_scores SET verdict=%s, confidence=%s "
            "WHERE code_number=%s AND pdf_path=%s AND criterion=%s",
            rows)
    conn.commit()
    return len(rows)


def within_rows(within, run_ts):
    """Centred pole widths → ``pole_separation`` rows under the ``within`` pole, one per
    criterion (``pair`` = criterion name)."""
    return [("within", crit, float(cos), run_ts) for crit, cos in within.items()]


def meta_rows(out, run_ts):
    return [(k, str(out.get(k))) for k in
            ("model", "dim", "use_4bit", "methods", "chunk_size", "chunk_overlap")
            if k in out] + [("run_ts", str(run_ts))]


# --- writes ----------------------------------------------------------------

def store(conn, out, recs, run_ts):
    """Upsert the whole run (scores + controls + pole separation + meta)."""
    init_schema(conn)
    with conn.cursor() as c:
        c.executemany(
            "INSERT INTO embedding_scores "
            "(code_number,pdf_path,method,criterion,e,verdict,confidence,model,run_ts) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) AS new "
            "ON DUPLICATE KEY UPDATE e=new.e, verdict=new.verdict, "
            "confidence=new.confidence, model=new.model, run_ts=new.run_ts",
            scores_to_rows(out, recs, run_ts))
        c.executemany(
            "INSERT INTO control_scores (name,criterion,e,run_ts) VALUES (%s,%s,%s,%s) "
            "AS new ON DUPLICATE KEY UPDATE e=new.e, run_ts=new.run_ts",
            controls_to_rows(out, run_ts))
        c.executemany(
            "INSERT INTO pole_separation (pole,pair,cosine,run_ts) VALUES (%s,%s,%s,%s) "
            "AS new ON DUPLICATE KEY UPDATE cosine=new.cosine, run_ts=new.run_ts",
            poles_to_rows(out, run_ts))
        c.executemany(
            "INSERT INTO run_meta (k,v) VALUES (%s,%s) AS new "
            "ON DUPLICATE KEY UPDATE v=new.v",
            meta_rows(out, run_ts))
        dv = doc_vectors_to_rows(out, recs, run_ts)
        if dv:
            c.executemany(
                "INSERT INTO doc_vectors "
                "(code_number,pdf_path,method,chunk_idx,dim,vec,run_ts) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s) AS new "
                "ON DUPLICATE KEY UPDATE dim=new.dim, vec=new.vec, run_ts=new.run_ts",
                dv)
        cv = control_vectors_to_rows(out, run_ts)
        if cv:
            c.executemany(
                "INSERT INTO control_vectors (name,dim,vec,run_ts) "
                "VALUES (%s,%s,%s,%s) AS new "
                "ON DUPLICATE KEY UPDATE dim=new.dim, vec=new.vec, run_ts=new.run_ts",
                cv)
        pv = pole_vectors_to_rows(out, run_ts)
        if pv:
            c.executemany(
                "INSERT INTO pole_vectors (criterion,pole,dim,vec,run_ts) "
                "VALUES (%s,%s,%s,%s,%s) AS new "
                "ON DUPLICATE KEY UPDATE dim=new.dim, vec=new.vec, run_ts=new.run_ts",
                pv)
    conn.commit()


# --- offline recompute (no GPU): read vectors, write leverred scores --------

def fetch_vectors(conn):
    """Load persisted vectors for the offline recompute.

    Returns ``(doc_vecs, poles, codes)`` where
    ``doc_vecs[pid][method] = [vec, ...]`` (chunk windows ordered by ``chunk_idx``),
    ``poles[criterion] = {'pos': vec, 'neg': vec}``, and ``codes[pid] = code_number``."""
    with conn.cursor() as c:
        c.execute("SELECT code_number,pdf_path,method,chunk_idx,vec FROM doc_vectors "
                  "ORDER BY code_number,pdf_path,method,chunk_idx")
        doc_rows = c.fetchall()
        c.execute("SELECT criterion,pole,vec FROM pole_vectors")
        pole_rows = c.fetchall()

    doc_vecs, codes = {}, {}
    for code, pid, method, _idx, blob in doc_rows:
        codes[pid] = code
        doc_vecs.setdefault(pid, {}).setdefault(method, []).append(unpack_vec(blob))
    poles = {}
    for crit, pole, blob in pole_rows:
        poles.setdefault(crit, {})[pole] = unpack_vec(blob)
    return doc_vecs, poles, codes


def fetch_control_vectors(conn):
    """Load persisted control vectors → ``{name: np.ndarray}`` (empty if none stored).

    Returned for the offline recompute to score with the corpus geometry; an empty dict
    means the structural embed predates control-vector capture, so the report keeps the
    pre-lever control numbers and flags them."""
    with conn.cursor() as c:
        c.execute("SELECT name,vec FROM control_vectors")
        return {name: unpack_vec(blob) for name, blob in c.fetchall()}


def delete_score_rows(conn, pids):
    """Delete ``embedding_scores`` rows for the given ``pdf_path``s.

    Used to evict in-corpus self-reference documents (dropped from the recompute corpus)
    so they vanish from the regenerated report rather than lingering with stale pre-lever
    scores. No-op for an empty list."""
    if not pids:
        return
    with conn.cursor() as c:
        c.executemany("DELETE FROM embedding_scores WHERE pdf_path=%s",
                      [(p,) for p in pids])
    conn.commit()


def apply_recompute(conn, scores, codes, within, params, run_ts, control_scores=None):
    """Persist an offline recompute: upsert the leverred ``e`` (verdict/confidence
    untouched), the centred pole widths, the lever params into ``run_meta``, and — when
    ``control_scores`` is given — the leverred control ``e`` (so the report's control
    checks reflect the same geometry as the papers)."""
    init_schema(conn)
    with conn.cursor() as c:
        c.executemany(
            "INSERT INTO embedding_scores "
            "(code_number,pdf_path,method,criterion,e,run_ts) "
            "VALUES (%s,%s,%s,%s,%s,%s) AS new "
            "ON DUPLICATE KEY UPDATE e=new.e, run_ts=new.run_ts",
            recompute_score_rows(scores, codes, run_ts))
        c.executemany(
            "INSERT INTO pole_separation (pole,pair,cosine,run_ts) VALUES (%s,%s,%s,%s) "
            "AS new ON DUPLICATE KEY UPDATE cosine=new.cosine, run_ts=new.run_ts",
            within_rows(within, run_ts))
        c.executemany(
            "INSERT INTO run_meta (k,v) VALUES (%s,%s) AS new "
            "ON DUPLICATE KEY UPDATE v=new.v",
            [(k, str(v)) for k, v in params.items()])
        if control_scores:
            c.executemany(
                "INSERT INTO control_scores (name,criterion,e,run_ts) "
                "VALUES (%s,%s,%s,%s) AS new "
                "ON DUPLICATE KEY UPDATE e=new.e, run_ts=new.run_ts",
                [(name, crit, float(e), run_ts)
                 for name, cdict in control_scores.items()
                 for crit, e in cdict.items()])
    conn.commit()


# --- reads (for report generation) -----------------------------------------

def fetch_report(conn):
    """Reassemble the report payload from the DB tables."""
    with conn.cursor() as c:
        c.execute("SELECT code_number,pdf_path,method,criterion,e,verdict,confidence "
                  "FROM embedding_scores ORDER BY code_number,pdf_path")
        score_rows = c.fetchall()
        c.execute("SELECT name,criterion,e FROM control_scores")
        ctrl_rows = c.fetchall()
        c.execute("SELECT pole,pair,cosine FROM pole_separation")
        pole_rows = c.fetchall()
        c.execute("SELECT k,v FROM run_meta")
        meta = dict(c.fetchall())

    papers = {}      # pid -> {"code": int, "scores": {method:{crit:e}},
                     #         "verdict": {crit:v}, "confidence": {crit:conf}}
    order = []
    for code, pid, method, crit, e, verdict, conf in score_rows:
        p = papers.get(pid)
        if p is None:
            p = papers[pid] = {"code": code, "scores": {}, "verdict": {},
                               "confidence": {}}
            order.append(pid)
        p["scores"].setdefault(method, {})[crit] = e
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
