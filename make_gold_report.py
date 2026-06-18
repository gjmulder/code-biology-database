"""Phase 6 — gold-set validation report (CLAUDE.md §8 / plan).

The project's binding constraint is **label quality**: both measurement axes (embedding `e`,
LLM graded verdicts) are synthetic and, until now, validated only against *each other* (ρ).
This report adjudicates them against the Barbieri-anchored **gold set** (`gold_labels`,
run-/judge-agnostic ground truth), per criterion and split by tier:

- **Embedding axis** — does gold+ outrank gold−? ``auc`` (rank-statistic P[random gold+ e >
  random gold− e]) and Spearman ρ(e, gold polarity).
- **Judge axis** — categorical verdict vs gold polarity: precision / recall / confusion, with
  ``met`` as the positive prediction.

Pure metric + join logic is unit-tested offline; the DB read (``db.fetch_report`` +
``db.fetch_gold``) and markdown emission run in :func:`main`.
"""

import argparse
import logging
import math

from embed_score import CRITERIA
from embed_independent import spearman

logger = logging.getLogger(__name__)

GOLD_REPORT_PATH = "gold_report.md"
POSITIVE_VERDICT = "met"
LABELLED = ("met", "not_met", "unclear")


# --- rank / classification statistics (pure) -------------------------------

def auc(pos, neg):
    """P[a random gold+ score outranks a random gold− score], ties counting half.

    The Mann-Whitney U statistic normalised to [0, 1]: 1.0 = perfect separation (every gold+
    above every gold−), 0.5 = no separation, 0.0 = perfectly reversed. ``nan`` if either side
    is empty (no pair to rank)."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def confusion(preds, golds):
    """Confusion matrix + precision/recall/F1 for boolean predictions vs boolean truth.

    ``preds[i]`` is "predicted positive" (verdict == ``met``); ``golds[i]`` is "actually
    positive" (gold polarity == ``pos``). precision/recall/F1 are ``nan`` when their
    denominator is zero (no predicted-positive / no actual-positive)."""
    tp = sum(1 for p, g in zip(preds, golds) if p and g)
    fp = sum(1 for p, g in zip(preds, golds) if p and not g)
    fn = sum(1 for p, g in zip(preds, golds) if not p and g)
    tn = sum(1 for p, g in zip(preds, golds) if not p and not g)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if not math.isnan(precision) and not math.isnan(recall) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision,
            "recall": recall, "f1": f1, "n": len(preds)}


# --- join: gold polarity × embedding e × categorical verdict ---------------

def _gold_rows(papers, gold, method):
    """Inner-join gold labels to the report payload → one row per gold paper.

    Each row: ``{pid, polarity, tier, pos(bool), e:{crit:e}, verdict:{crit:v}}``. Gold rows
    whose paper is absent from the embedding run (``papers``) are dropped (unembedded). Gold
    is keyed ``(code, pid, criterion)`` with criterion ``'all'`` — one polarity per paper."""
    rows = []
    for (code, pid, _gcrit), g in gold.items():
        p = papers.get(pid)
        if p is None:
            continue
        scores = p.get("scores", {}).get(method, {})
        rows.append({
            "pid": pid,
            "polarity": g["polarity"],
            "tier": g.get("tier"),
            "pos": g["polarity"] == "pos",
            "e": {c: scores.get(c) for c in CRITERIA},
            "verdict": {c: p.get("verdict", {}).get(c) for c in CRITERIA},
        })
    return rows


def gold_validation(papers, gold, method="chunk", criteria=CRITERIA):
    """Per-criterion embedding-axis (AUC, ρ) and judge-axis (confusion) metrics over the gold
    set, plus a per-tier count breakdown. ``papers`` is :func:`db.fetch_report`'s ``papers``;
    ``gold`` is :func:`db.fetch_gold`."""
    rows = _gold_rows(papers, gold, method)
    out = {}
    for c in criteria:
        pos_e = [r["e"][c] for r in rows if r["pos"] and r["e"][c] is not None]
        neg_e = [r["e"][c] for r in rows if not r["pos"] and r["e"][c] is not None]
        e_vals = [r["e"][c] for r in rows if r["e"][c] is not None]
        pol_vals = [1.0 if r["pos"] else 0.0 for r in rows if r["e"][c] is not None]
        rho = (spearman(e_vals, pol_vals)
               if len(set(pol_vals)) > 1 and len(e_vals) > 1 else float("nan"))

        judged = [r for r in rows if r["verdict"][c] in LABELLED]
        preds = [r["verdict"][c] == POSITIVE_VERDICT for r in judged]
        golds = [r["pos"] for r in judged]

        tiers = {}
        for r in rows:
            t = r["tier"]
            b = tiers.setdefault(t, {"n": 0, "n_pos": 0, "n_neg": 0})
            b["n"] += 1
            b["n_pos" if r["pos"] else "n_neg"] += 1

        out[c] = {
            "embedding": {"auc": auc(pos_e, neg_e), "rho": rho,
                          "n_pos": len(pos_e), "n_neg": len(neg_e)},
            "judge": confusion(preds, golds),
            "tiers": tiers,
        }
    return out


# --- markdown -------------------------------------------------------------

def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def format_report(res, judge, method="chunk", criteria=CRITERIA):
    """Render :func:`gold_validation` output as markdown."""
    md = [f"# Gold-set validation (Phase 6) — judge `{judge}`, embedding method `{method}`\n",
          "Adjudicates the two synthetic axes against the Barbieri-anchored gold set "
          "(`gold_labels`). Embedding axis: does gold+ outrank gold− (AUC) and track polarity "
          "(ρ)? Judge axis: categorical verdict (`met` = positive) vs gold polarity.\n",
          "## Embedding axis — gold+ vs gold−\n",
          "| criterion | AUC | ρ(e, gold) | n+ | n− |",
          "|---|---|---|---|---|"]
    for c in criteria:
        e = res[c]["embedding"]
        md.append(f"| {c} | {_f(e['auc'])} | {_f(e['rho'])} | {e['n_pos']} | {e['n_neg']} |")
    md.append("\n## Judge axis — verdict vs gold polarity\n")
    md.append("| criterion | precision | recall | F1 | TP | FP | FN | TN | n |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for c in criteria:
        j = res[c]["judge"]
        md.append(f"| {c} | {_f(j['precision'])} | {_f(j['recall'])} | {_f(j['f1'])} | "
                  f"{j['tp']} | {j['fp']} | {j['fn']} | {j['tn']} | {j['n']} |")
    md.append("\n## Tier breakdown (gold paper counts)\n")
    md.append("| criterion | tier | n | n+ | n− |")
    md.append("|---|---|---|---|---|")
    for c in criteria:
        for tier, b in sorted(res[c]["tiers"].items(), key=lambda kv: str(kv[0])):
            md.append(f"| {c} | {tier} | {b['n']} | {b['n_pos']} | {b['n_neg']} |")
    return "\n".join(md) + "\n"


# --- driver (DB I/O) ------------------------------------------------------

def main(argv=None):
    import db
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="baseline", help="embedding run to read e from")
    ap.add_argument("--judge", default="deepseek/deepseek-v4-pro",
                    help="judge model whose verdicts to validate against gold")
    ap.add_argument("--method", default="chunk", help="embedding granularity (working: chunk)")
    ap.add_argument("--out", default=GOLD_REPORT_PATH)
    args = ap.parse_args(argv)

    conn = db.connect(db.load_env())
    try:
        payload = db.fetch_report(conn, run=args.run, judge=args.judge)
        gold = db.fetch_gold(conn)
    finally:
        conn.close()

    res = gold_validation(payload["papers"], gold, method=args.method)
    md = format_report(res, judge=args.judge, method=args.method)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info("wrote %s (%d gold papers joined)", args.out,
                sum(1 for k in gold if k[1] in payload["papers"]))
    print(md)


if __name__ == "__main__":
    main()
