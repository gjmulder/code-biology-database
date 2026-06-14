"""Backfill LLM verdicts for the embedding corpus, so ρ(e, verdict) is measurable.

The structural embedding run persisted all 219 corpus papers to
``embedding_scores`` but only 10 carry an LLM verdict (the original sample) — the
other ~209 have ``verdict IS NULL``. This driver judges exactly those NULL-verdict
papers through the same pipeline as ``run_sample.py``:

  * criteria 1 & 2  → local Gemma-4-31B (32k batch server, no MTP)
  * criterion 3     → paid Nemotron via OpenRouter (full-paper, 1M ctx)

Verdicts are checkpointed to ``sample_verdicts.jsonl`` (APPEND — never deleted, so
the Nemotron spend survives a crash and the run is resumable), then upserted into
MySQL ``embedding_scores`` (verdict/confidence only; the embedding ``e`` is left
untouched). The labelled set then matches the embedded corpus.

  python3 -u judge_corpus.py [--limit N] [--workers 6]
"""

import argparse
import logging
import os

import run_sample  # reuse load_env

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("judge_corpus")


def null_verdict_pdf_paths(conn):
    """The distinct corpus papers whose verdict is still NULL (need judging)."""
    with conn.cursor() as c:
        c.execute("SELECT DISTINCT pdf_path FROM embedding_scores "
                  "WHERE verdict IS NULL")
        return {p for (p,) in c.fetchall()}


def select_papers(csv_path, pdf_dir, want_paths):
    """Full paper metadata (code_number, code_name, paper_name, url, pdf_path) for
    every wanted pdf_path that is present on disk, via criteria_judge.iter_papers."""
    import criteria_judge as cj
    by_path = {p["pdf_path"]: p for p in cj.iter_papers(csv_path, pdf_dir)}
    found, missing = [], []
    for path in want_paths:
        (found if path in by_path else missing).append(path)
    if missing:
        log.warning("%d NULL-verdict papers not found via iter_papers (skipped): %s",
                    len(missing), ", ".join(sorted(missing)[:5]) + (" ..." if len(missing) > 5 else ""))
    return [by_path[p] for p in found]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="judge at most N papers this run (0 = all NULL-verdict); "
                         "resumable, so a small smoke run then the rest is fine")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--csv", default="biological_codes.csv")
    ap.add_argument("--pdf-dir", default="pdfs")
    ap.add_argument("--checkpoint", default="sample_verdicts.jsonl")
    args = ap.parse_args()

    run_sample.load_env()
    import criteria_judge as cj
    import db

    conn = db.connect()
    try:
        want = null_verdict_pdf_paths(conn)
        log.info("MySQL: %d corpus papers with NULL verdict", len(want))
        papers = select_papers(args.csv, args.pdf_dir, want)
        papers.sort(key=lambda p: int(p["code_number"]))
        if args.limit:
            papers = papers[:args.limit]
        log.info("judging %d papers this run (workers=%d, checkpoint=%s APPEND)",
                 len(papers), args.workers, args.checkpoint)

        lc = cj.local_complete_factory()
        orc = cj.openrouter_complete_factory()

        def judge_fn(paper):
            return cj.judge_paper(paper, lc, orc)

        # run_batch is resumable: it loads `done` from the checkpoint and APPENDS new
        # records — the existing 10 verdicts (and any prior partial run) are preserved.
        cj.run_batch(papers, judge_fn, args.checkpoint, max_workers=args.workers)

        # persist every verdict in the checkpoint to MySQL (idempotent for the 10).
        import json
        records = [json.loads(l) for l in open(args.checkpoint, encoding="utf-8") if l.strip()]
        n = db.update_verdicts(conn, records)
        log.info("persisted %d (paper,criterion) verdicts to MySQL embedding_scores "
                 "from %d checkpoint records", n, len(records))

        with conn.cursor() as c:
            c.execute("SELECT COUNT(DISTINCT pdf_path) FROM embedding_scores "
                      "WHERE verdict IS NOT NULL")
            labelled = c.fetchone()[0]
        log.info("embedding_scores now has %d papers with a verdict", labelled)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
