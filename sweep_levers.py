"""Offline corpus-scale parameter sweep over the two embedding levers.

The four space-level levers ship with defaults (`whiten-k=0`, `shared-strength=0.5`)
that were calibrated on a *20-paper smoke test* — `embed_score.py` itself flags "k>=1
hurt at small sample; revisit at corpus scale". The corpus is now 219 papers but the
revisit never happened. This sweep does it **for free**: it rescores every persisted
paper from the stored `doc_vectors`/`pole_vectors` across a `(whiten-k, shared-strength)`
grid and reports Spearman ρ(e, verdict) per method × criterion — **no GPU, no spend, no
DB writes**. The winning cell is then applied the normal way:

    python3 embed_independent.py --recompute --whiten-k K --shared-strength S

`sweep` is a pure function over plain dicts so it is unit-tested offline with a fake
corpus; only `main` touches MySQL (read-only, to load the vectors + verdicts).
"""

import argparse
import logging

import embed_score as es
from criteria_judge import verdict_ordinal
from embed_independent import spearman

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("sweep")

VALID_VERDICTS = ("met", "not_met", "unclear")

# Grid defaults: k spans 'off' through an aggressive top-PC strip; strength spans no
# orthogonalization through full removal. Both are CLI-overridable.
DEFAULT_KS = [0, 1, 2, 4, 8, 16]
DEFAULT_STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0]


def methods_of(doc_vecs):
    """The chunking methods present in the corpus (full/abstract/chunk), in METHOD order
    where known, else discovery order."""
    seen = []
    for methods in doc_vecs.values():
        for m in methods:
            if m not in seen:
                seen.append(m)
    pref = ["full", "abstract", "chunk"]
    return [m for m in pref if m in seen] + [m for m in seen if m not in pref]


def rho_or_none(e_vals, ordinals):
    """Spearman ρ, or ``None`` when it is undefined (fewer than two points, or no rank
    variation in the verdict ordinals — mirrors the report's ``n/a``)."""
    if len(e_vals) < 2 or len(set(ordinals)) < 2:
        return None
    return spearman(e_vals, ordinals)


def _rho_table(scores, verdicts, methods):
    """ρ per method × criterion for one already-computed score set."""
    rho = {m: {} for m in methods}
    for m in methods:
        for c in es.CRITERIA:
            e_vals, ordinals = [], []
            for pid, crit_verdicts in verdicts.items():
                v = crit_verdicts.get(c)
                e = scores.get(pid, {}).get(m, {}).get(c)
                if e is not None and v in VALID_VERDICTS:
                    e_vals.append(e)
                    ordinals.append(verdict_ordinal(v))
            rho[m][c] = rho_or_none(e_vals, ordinals)
    return rho


def sweep(doc_vecs, poles, verdicts, ks, strengths, methods=None):
    """Rescore the corpus across the ``(k, strength)`` grid and tabulate ρ.

    Pure: ``doc_vecs[pid][method] = [vec, ...]``, ``poles[crit] = {'pos','neg'}``,
    ``verdicts[pid][crit] = 'met'|'not_met'|'unclear'|...``. Returns a list of cells
    ``{'k', 'strength', 'rho': {method: {criterion: float|None}}, 'within': {crit: w}}``.
    No GPU, no I/O. ``within`` (centred pole width) is geometry-only and therefore
    identical across the grid; it is recomputed per cell for transparency."""
    methods = methods or methods_of(doc_vecs)
    results = []
    for k in ks:
        for strength in strengths:
            scores, within = es.recompute(doc_vecs, poles, k=k, strength=strength)
            results.append({"k": k, "strength": strength,
                            "rho": _rho_table(scores, verdicts, methods),
                            "within": within})
    return results


def best_per_criterion(results, method):
    """For one method, the grid cell maximising ρ for each criterion → ``{crit:
    {'k','strength','rho'}}``. Criteria whose ρ is ``None`` in every cell are omitted."""
    best = {}
    for c in es.CRITERIA:
        cells = [(cell["rho"][method][c], cell) for cell in results
                 if cell["rho"][method].get(c) is not None]
        if not cells:
            continue
        rho, cell = max(cells, key=lambda t: t[0])
        best[c] = {"k": cell["k"], "strength": cell["strength"], "rho": rho}
    return best


# --- reporting --------------------------------------------------------------

def _fmt(x):
    return " n/a " if x is None else f"{x:+.3f}"


def format_report(results, methods):
    """Human-readable ρ grid per method + the argmax cell per criterion per method."""
    lines = []
    for m in methods:
        lines.append(f"\n=== method: {m} — Spearman ρ(e, verdict) ===")
        header = ["  k", "strength"] + list(es.CRITERIA)
        lines.append("  ".join(f"{h:>13}" for h in header))
        for cell in results:
            row = [f"{cell['k']:>3}", f"{cell['strength']:>8.2f}"]
            row += [f"{_fmt(cell['rho'][m][c]):>13}" for c in es.CRITERIA]
            lines.append("  ".join(f"{v:>13}" for v in row))
        best = best_per_criterion(results, m)
        lines.append(f"  best per criterion ({m}):")
        for c in es.CRITERIA:
            if c in best:
                b = best[c]
                lines.append(f"    {c:<14} ρ={b['rho']:+.3f} "
                             f"@ k={b['k']}, strength={b['strength']}")
            else:
                lines.append(f"    {c:<14} ρ=n/a (no verdict variation)")
    within = results[0]["within"] if results else {}
    if within:
        lines.append("\n=== pole width `within` (centred cos pos↔neg; constant over grid) ===")
        lines.append("  " + ", ".join(f"{c}={within[c]:+.3f}" for c in es.CRITERIA
                                      if c in within))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS,
                    help="whiten-k grid (top-PCs removed; 0 = whitening off)")
    ap.add_argument("--strengths", type=float, nargs="+", default=DEFAULT_STRENGTHS,
                    help="shared-strength grid (axis orthogonalization, [0,1])")
    args = ap.parse_args()

    import db
    conn = db.connect()
    try:
        doc_vecs, poles, _codes = db.fetch_vectors(conn)
        if not doc_vecs or not poles:
            log.warning("no persisted vectors in MySQL — run the GPU embed first")
            return
        # Match the live recompute corpus exactly: drop in-corpus self-references so the
        # sweep ρ is directly comparable to CLAUDE.md §5 / report.md.
        doc_vecs, dropped = es.drop_self_references(doc_vecs)
        if dropped:
            log.info("dropped %d in-corpus self-reference doc(s): %s", len(dropped), dropped)
        payload = db.fetch_report(conn)
    finally:
        conn.close()

    verdicts = {pid: p["verdict"] for pid, p in payload["papers"].items()}
    methods = methods_of(doc_vecs)
    labelled = sum(1 for v in verdicts.values()
                   if any(x in VALID_VERDICTS for x in v.values()))
    log.info("sweeping %d papers (%d labelled) over k=%s x strength=%s",
             len(doc_vecs), labelled, args.ks, args.strengths)

    results = sweep(doc_vecs, poles, verdicts, args.ks, args.strengths, methods)
    print(format_report(results, methods))
    print("\nApply a winner with:")
    print("  python3 embed_independent.py --recompute --whiten-k K --shared-strength S")


if __name__ == "__main__":
    main()
