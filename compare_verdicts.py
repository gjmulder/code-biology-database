"""Validate the graded judge pilot by comparison only — no gold set (plan, locked).

For the pilot papers (dominant topic in the top-N strata), this reports:

  1. the **old vs new categorical** distribution per criterion (met/unclear/not_met) — did
     the new derived categorical shift?
  2. the **graded** value-count distribution + spread per criterion — did real gradation
     materialise vs the old confidence clustering at 0.95-1.0 (CLAUDE.md §6)?
  3. pooled **ρ(value, e)** over the pilot papers for both axes — ρ(categorical_ordinal, e)
     and ρ(graded, e) — does the graded axis track the embedding at least as well?

The stats here are pure and unit-tested; ``main()`` wires them to MySQL and prints a table.

Old-vs-new on the *same* papers needs a snapshot, because the pilot upserts the new verdict
onto the same ``(code, pdf, criterion)`` key in the run-agnostic ``verdicts`` table. So the
workflow is: ``compare_verdicts.py --snapshot old.json`` *before* the pilot run, then
``compare_verdicts.py --old old.json`` *after* — the new axis is read independently from the
never-overwritten ``chunk_verdicts`` table.
"""

import argparse
import json
import logging
import math

import criteria_judge as cj
from criteria_judge import verdict_ordinal
from embed_independent import spearman

logger = logging.getLogger(__name__)

CRITERIA = cj.ALL_CRITERIA
METHODS = ["full", "abstract", "chunk"]
VERDICTS = ("met", "unclear", "not_met")
GRADED_LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)


# --- pure stats ------------------------------------------------------------

def categorical_distribution(records):
    """``[(criterion, verdict), ...]`` → ``{criterion: {met,unclear,not_met: count}}``.

    Unknown/None verdicts are ignored; every criterion seen gets a full three-key bucket."""
    dist = {}
    for crit, verdict in records:
        bucket = dist.setdefault(crit, {v: 0 for v in VERDICTS})
        if verdict in bucket:
            bucket[verdict] += 1
    return dist


def _nearest_level(x):
    return min(GRADED_LEVELS, key=lambda lvl: abs(lvl - float(x)))


def graded_distribution(records):
    """``[(criterion, graded), ...]`` → ``{criterion: {level: count}}`` over the five graded
    levels. Graded values are discrete (max-pool of discrete chunk scores) but float noise is
    snapped to the nearest level."""
    dist = {}
    for crit, value in records:
        bucket = dist.setdefault(crit, {lvl: 0 for lvl in GRADED_LEVELS})
        bucket[_nearest_level(value)] += 1
    return dist


def graded_spread(records):
    """``[(criterion, graded), ...]`` → ``{criterion: {n,mean,std,min,max}}`` — the headline
    "did gradation materialise" numbers (a non-trivial std is the signal)."""
    grouped = {}
    for crit, value in records:
        grouped.setdefault(crit, []).append(float(value))
    spread = {}
    for crit, vals in grouped.items():
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        spread[crit] = {"n": n, "mean": mean, "std": math.sqrt(var),
                        "min": min(vals), "max": max(vals)}
    return spread


def pooled_spearman(papers, order, methods, criteria, value_of):
    """Pooled ρ(value, e) over ``order`` for each ``criterion × method``.

    One helper serves both axes: pass ``value_of(p, crit)`` returning the graded score, or
    the verdict ordinal, or ``None`` to drop a paper. A paper is included only when both its
    ``e`` (``p['scores'][method][crit]``) and its value are present. A cell is ``None`` when
    fewer than two distinct values survive (no rank variation — the report's n/a rule)."""
    out = {}
    for crit in criteria:
        out[crit] = {}
        for m in methods:
            e_vals, v_vals = [], []
            for pid in order:
                p = papers[pid]
                e = p.get("scores", {}).get(m, {}).get(crit)
                v = value_of(p, crit)
                if e is not None and v is not None:
                    e_vals.append(e)
                    v_vals.append(v)
            out[crit][m] = (spearman(e_vals, v_vals)
                            if len(set(v_vals)) > 1 else None)
    return out


# --- DB glue + table (exercised manually) ----------------------------------

def aggregate_chunk_verdicts(chunk_verdicts):
    """``db.fetch_chunk_verdicts`` output → ``{(pdf_path, criterion): (graded_max,
    categorical)}`` via :func:`criteria_judge.aggregate_graded`."""
    out = {}
    for (pid, crit), rows in chunk_verdicts.items():
        scores = [{"agreement": agr, "confidence": conf}
                  for _idx, agr, conf, _quote in rows]
        graded_max, _mean, _conf, categorical = cj.aggregate_graded(scores)
        out[(pid, crit)] = (graded_max, categorical)
    return out


def _fmt(x):
    return "n/a" if x is None else f"{x:+.3f}"


def build_report(papers, order, old_verdict, methods=METHODS, criteria=CRITERIA):
    """Assemble the printable comparison text from prepared ``papers`` (each carrying
    ``scores``, new ``verdict``, ``graded``) plus an ``old_verdict`` map
    ``{(pid, crit): verdict}``. Returns a string."""
    old_recs = [(c, old_verdict.get((pid, c))) for pid in order for c in criteria]
    new_recs = [(c, papers[pid]["verdict"].get(c)) for pid in order for c in criteria]
    graded_recs = [(c, papers[pid]["graded"].get(c)) for pid in order for c in criteria
                   if papers[pid]["graded"].get(c) is not None]

    old_dist = categorical_distribution(old_recs)
    new_dist = categorical_distribution(new_recs)
    g_dist = graded_distribution(graded_recs)
    g_spread = graded_spread(graded_recs)

    rho_cat = pooled_spearman(
        papers, order, methods, criteria,
        value_of=lambda p, c: verdict_ordinal(p["verdict"].get(c))
        if p["verdict"].get(c) in VERDICTS else None)
    rho_graded = pooled_spearman(
        papers, order, methods, criteria,
        value_of=lambda p, c: p["graded"].get(c))

    lines = [f"# Judge pilot comparison — {len(order)} pilot papers\n"]

    lines.append("## Categorical distribution (old → new) per criterion\n")
    for c in criteria:
        o = old_dist.get(c, {v: 0 for v in VERDICTS})
        n = new_dist.get(c, {v: 0 for v in VERDICTS})
        cells = ", ".join(f"{v}: {o[v]}→{n[v]}" for v in VERDICTS)
        lines.append(f"- {c}: {cells}")
    lines.append("")

    lines.append("## Graded value distribution (new axis) per criterion\n")
    for c in criteria:
        d = g_dist.get(c, {lvl: 0 for lvl in GRADED_LEVELS})
        s = g_spread.get(c, {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0})
        counts = " ".join(f"{lvl:+.1f}:{d[lvl]}" for lvl in GRADED_LEVELS)
        lines.append(f"- {c}: [{counts}]  (n={s['n']} mean={s['mean']:+.3f} "
                     f"std={s['std']:.3f} min={s['min']:+.1f} max={s['max']:+.1f})")
    lines.append("")

    lines.append("## Pooled ρ over pilot papers (categorical / graded vs e)\n")
    lines.append("| criterion | " + " | ".join(
        f"{m} cat / grad" for m in methods) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in methods) + " |")
    for c in criteria:
        cells = [f"{_fmt(rho_cat[c][m])} / {_fmt(rho_graded[c][m])}" for m in methods]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _snapshot(conn, pids, path):
    """Dump current ``verdicts`` for ``pids`` to JSON (run *before* the pilot overwrites)."""
    with conn.cursor() as c:
        c.execute("SELECT pdf_path,criterion,verdict FROM verdicts")
        rows = [{"pdf_path": pid, "criterion": crit, "verdict": v}
                for pid, crit, v in c.fetchall() if pid in pids]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
    logger.info("snapshotted %d verdict rows for %d pilot papers → %s",
                len(rows), len(pids), path)
    return rows


def _load_snapshot(path):
    with open(path, encoding="utf-8") as fh:
        return {(r["pdf_path"], r["criterion"]): r["verdict"] for r in json.load(fh)}


def main(argv=None):
    import db
    import judge_pilot as jp
    from assign_topics import paper_dominant_topic

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="baseline")
    ap.add_argument("--method", default="chunk", help="chunk_topics method for topic strata")
    ap.add_argument("--top", type=int, default=jp.DEFAULT_TOP)
    ap.add_argument("--snapshot", help="dump current verdicts for pilot papers, then exit")
    ap.add_argument("--old", help="snapshot JSON for the OLD categorical distribution")
    ap.add_argument("--out", help="write the report here as well as stdout")
    ap.add_argument("--judge", default=None,
                    help="scope verdicts/chunk_verdicts to one judge model "
                         "(e.g. deepseek/deepseek-v4-pro); default = newest judge per key")
    args = ap.parse_args(argv)

    conn = db.connect()
    try:
        chunk_topics = db.fetch_chunk_topics(conn, args.run, args.method)
        pilot = jp.select_pilot_papers(chunk_topics, args.top)   # {pid: dominant}
        pids = set(pilot)
        logger.info("pilot: %d papers in top-%d topics", len(pids), args.top)

        if args.snapshot:
            _snapshot(conn, pids, args.snapshot)
            return

        payload = db.fetch_report(conn, args.run, judge=args.judge)
        papers, order = payload["papers"], payload["order"]
        order = [pid for pid in order if pid in pids]

        agg = aggregate_chunk_verdicts(db.fetch_chunk_verdicts(conn, judge=args.judge))
        for pid in order:
            p = papers[pid]
            p["graded"] = {}
            p["verdict"] = {}
            for c in CRITERIA:
                graded_max, categorical = agg.get((pid, c), (None, None))
                p["graded"][c] = graded_max
                p["verdict"][c] = categorical    # NEW derived categorical

        old_verdict = _load_snapshot(args.old) if args.old else {
            (pid, c): payload["papers"][pid]["verdict"].get(c)
            for pid in order for c in CRITERIA}
        if not args.old:
            logger.warning("no --old snapshot; OLD column reads live verdicts "
                           "(== NEW once the pilot has overwritten them)")

        report = build_report(papers, order, old_verdict)
        print(report)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(report)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
