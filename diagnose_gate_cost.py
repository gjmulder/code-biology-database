"""Free read-only diagnostic: how many positives did the strict-verbatim grounding gate cost?

Phase 4.5 step 3 (CLAUDE.md §9, label-quality fix). The retired gate zeroed any positive
whose ``evidence_quote`` was not an exact ``_norm_ws`` substring of its chunk; the new gate
grounds fuzzily (``criteria_judge.grounding_detail``, τ/L). Pre-gate agreement of cells the
strict gate already zeroed is unrecoverable from storage, so we bound the cost instead:

  among stored ``chunk_verdicts`` cells with ``agreement==0.0`` **and** a non-empty quote,
  count those the strict gate would reject (``not strict``) but the fuzzy gate admits
  (``fuzzy``) — an **upper bound** on the positives the strict gate cost.

No GPU and no spend: it re-walks the persisted chunks with the CPU-only harrier tokenizer
(``chunk_text.reproduce_chunks``) and re-grounds offline. The actual recovery still needs a
re-judge (Phase 6); this only sizes the prize before paying for it.
"""

import argparse
import collections

import criteria_judge as cj


def strict_grounded(quote, chunk):
    """The retired gate: ``_norm_ws(quote)`` is a verbatim substring of ``_norm_ws(chunk)``."""
    q = cj._norm_ws(quote)
    return bool(q) and q in cj._norm_ws(chunk)


def classify_cell(quote, chunk):
    """Grade one zeroed-with-quote cell under strict vs fuzzy grounding.

    ``newly_passes`` is the cell of interest: rejected by the strict gate, admitted by the
    fuzzy gate — i.e. a positive the strict gate would have zeroed but the fuzzy one keeps."""
    strict = strict_grounded(quote, chunk)
    fuzzy, coverage, longest = cj.grounding_detail(quote, chunk)
    return {
        "strict": strict,
        "fuzzy": fuzzy,
        "coverage": coverage,
        "longest": longest,
        "newly_passes": fuzzy and not strict,
    }


def tally(pairs):
    """Tally ``classify_cell`` over an iterable of ``(quote, chunk)`` pairs."""
    t = collections.Counter()
    for quote, chunk in pairs:
        t["candidates"] += 1
        c = classify_cell(quote, chunk)
        if c["strict"]:
            t["strict_grounded"] += 1
        if c["fuzzy"]:
            t["fuzzy_grounded"] += 1
        if c["newly_passes"]:
            t["newly_passes"] += 1
    return dict(t)


# --- driver (DB + tokenizer I/O; exercised manually, not in the offline suite) -------------

DEFAULT_JUDGES = ("deepseek/deepseek-v4-pro", "gemma-4-31b")


def _candidate_rows(conn, judges):
    """Stored cells eligible to have been zeroed by the strict gate: ``agreement==0.0`` with a
    non-empty quote, for the given judges. Returns ``[(model, pdf_path, criterion, idx, quote)]``."""
    placeholders = ",".join(["%s"] * len(judges))
    sql = ("SELECT model,pdf_path,criterion,chunk_idx,evidence_quote FROM chunk_verdicts "
           "WHERE agreement=0.0 AND evidence_quote IS NOT NULL AND evidence_quote<>'' "
           f"AND model IN ({placeholders}) ORDER BY pdf_path,chunk_idx")
    with conn.cursor() as c:
        c.execute(sql, tuple(judges))
        return list(c.fetchall())


def run(conn, tokenizer, judges=DEFAULT_JUDGES, pdf_text_mod=None):
    """Re-walk persisted chunks and tally strict-gate cost per judge and per criterion.

    ``tokenizer`` is the CPU-only harrier tokenizer; chunks are reproduced once per paper and
    indexed by ``chunk_idx`` to align with the persisted ``doc_vectors``/``chunk_verdicts``."""
    import chunk_text
    if pdf_text_mod is None:
        import pdf_text as pdf_text_mod

    rows = _candidate_rows(conn, judges)
    by_paper = collections.defaultdict(list)
    for model, pid, crit, idx, quote in rows:
        by_paper[pid].append((model, crit, idx, quote))

    per_judge = collections.defaultdict(collections.Counter)
    per_crit = collections.defaultdict(collections.Counter)
    missing_chunks = 0
    for pid, cells in sorted(by_paper.items()):
        try:
            full_text = pdf_text_mod.extract_text(pid)
        except Exception as exc:  # noqa: BLE001 — a missing/broken PDF must not abort the sweep
            print(f"  ! skip {pid}: {exc}")
            continue
        chunks = dict(chunk_text.reproduce_chunks(full_text, tokenizer))
        for model, crit, idx, quote in cells:
            chunk = chunks.get(idx)
            if chunk is None:
                missing_chunks += 1
                continue
            c = classify_cell(quote, chunk)
            for bucket, key in ((per_judge, model), (per_crit, crit)):
                bucket[key]["candidates"] += 1
                if c["strict"]:
                    bucket[key]["strict"] += 1
                if c["newly_passes"]:
                    bucket[key]["newly"] += 1
    return per_judge, per_crit, missing_chunks


def _print_table(title, table):
    print(f"\n{title}")
    print(f"  {'key':<40} {'cand':>6} {'strict':>7} {'newly':>6}")
    tot = collections.Counter()
    for key, c in sorted(table.items()):
        print(f"  {key:<40} {c['candidates']:>6} {c['strict']:>7} {c['newly']:>6}")
        tot.update(c)
    print(f"  {'TOTAL':<40} {tot['candidates']:>6} {tot['strict']:>7} {tot['newly']:>6}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", default="./harrier_tokenizer",
                    help="CPU-only harrier tokenizer path (no model weights / GPU)")
    ap.add_argument("--judge", action="append", dest="judges",
                    help="judge model tag (repeatable); default the two primary judges")
    args = ap.parse_args(argv)
    judges = tuple(args.judges) if args.judges else DEFAULT_JUDGES

    import db
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    conn = db.connect(db.load_env())
    try:
        per_judge, per_crit, missing = run(conn, tokenizer, judges=judges)
    finally:
        conn.close()

    print("Strict-gate cost — upper bound on positives the strict-verbatim gate zeroed")
    print(f"(judges: {', '.join(judges)}; tau={cj.GROUNDING_TAU}, "
          f"min_block={cj.GROUNDING_MIN_BLOCK})")
    _print_table("Per judge", per_judge)
    _print_table("Per criterion", per_crit)
    if missing:
        print(f"\n  ({missing} cells skipped — chunk_idx not reproducible from current text)")


if __name__ == "__main__":
    main()
