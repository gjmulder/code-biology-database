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
]


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
