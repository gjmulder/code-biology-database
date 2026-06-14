"""Driver (this host): independent embedding analysis vs the prior verdict run.

For each of the 10 preselected papers this attaches an **independent** embedding
axis to the EXISTING LLM verdicts (verdicts are never modified — plan decision 0).
Each paper is scored three ways as separate documents, to test which granularity
best tracks the verdict:

  * ``full``     — whole (budget-capped) paper            (one document)
  * ``abstract`` — abstract section only                  (one document)
  * ``chunk``    — 8192-token windows @50% overlap, max-pooled

**All embedding output is persisted to MySQL** (``db.py``), keyed on the code id —
not to JSON. The GPU host returns a transient ``embed_out.json`` purely as transport;
the driver loads it into MySQL and deletes it. ``report.md`` is then generated *from
the database*: a per-paper, per-criterion table carrying the criteria_judge verdict +
its confidence alongside the three embedding columns (``e_full`` / ``e_abstract`` /
``e_chunk``), plus Spearman ρ(e, verdict_ordinal) per method×criterion (the headline:
which chunking actually tracks the verdict), the pole-separation diagnostic, and the
control checks.

Stages (default: all): build input, ship+run on the GPU host, load to MySQL, report.
Use --report-only to regenerate report.md from the current DB contents.
"""

import argparse
import csv
import datetime
import json
import logging
import os
import random
import subprocess

import numpy as np

import db
import embed_score as es_mod
import pdf_text
from criteria_judge import verdict_ordinal
from download_pdfs import output_path_for, read_rows
from embed_score import CRITERIA, load_prototypes

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("embed")

METHODS = ["full", "abstract", "chunk"]


def load_verdicts(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def build_pool(csv_path, pdf_dir):
    """Every downloaded paper as ``{code_number, pdf_path}`` — one per PDF actually on
    disk, deduped by path (the first code that cites a shared DOI wins). This is the
    pool the ``--extra-sample`` draw is taken from."""
    pool, seen = [], set()
    for row in read_rows(csv_path):
        path = output_path_for(row, pdf_dir)
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        pool.append({"code_number": row["code_number"], "pdf_path": path})
    return pool


def sample_extra_papers(pool, existing_pids, n, seed=0):
    """Deterministically draw up to ``n`` papers from ``pool`` whose ``pdf_path`` is not
    already in ``existing_pids`` (the verdict set).

    The draw is seeded so a run is reproducible. The returned recs carry only
    ``code_number`` + ``pdf_path`` — **no verdict criteria** — so they widen the embedded
    corpus (a richer basis for centring/whitening μ and a broader on-axis spread) without
    entering the ρ(e, verdict) calculation, which keys on the labelled papers only."""
    existing = set(existing_pids)
    candidates = [p for p in pool if p["pdf_path"] not in existing]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[: max(n, 0)]


def build_input(recs, prototypes, controls, char_budget):
    papers = {}
    for r in recs:
        pid = r["pdf_path"]
        text = pdf_text.extract_text(pid)
        full = text[:char_budget]
        abstract = pdf_text.extract_abstract(text)
        papers[pid] = {"full": full, "abstract": abstract}
        log.info("  %s: full %d chars (capped %d), abstract %d chars",
                 os.path.basename(pid), len(text), char_budget, len(abstract))
    return {"papers": papers, "prototypes": prototypes, "controls": controls}


def build_controls_input(prototypes, controls):
    """Controls-only embed input: prototypes + controls, **no papers** — so the GPU run
    captures the control vectors in one model load without re-embedding the corpus."""
    return {"papers": {}, "prototypes": prototypes, "controls": controls}


def run_remote(host, remote_dir, in_path, out_path, model, use_4bit, cuda_devices,
               max_seq, controls_only=False):
    """scp the runner + input to the GPU host, run it pinned to the 3090 Ti, fetch out.

    The fetched JSON is a transient transport artifact — the caller loads it into
    MySQL and deletes it; MySQL is the system of record."""
    subprocess.run(["ssh", host, f"mkdir -p {remote_dir}"], check=True)
    for f in ("embed_score.py", "run_harrier_embed.py", in_path):
        subprocess.run(["scp", "-q", f, f"{host}:{remote_dir}/"], check=True)
    flag = "" if use_4bit else "--no-4bit"
    # 27B/4-bit on a 24 GB 3090 Ti can't do a 32k-token forward (activations OOM);
    # cap the single-pass methods (full/abstract) at --max-seq and reduce allocator
    # fragmentation. The chunk method stays at 8k windows (proven to fit).
    env = (f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={cuda_devices} "
           f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    co = "--controls-only" if controls_only else ""
    cmd = (f"cd {remote_dir} && {env} python3 run_harrier_embed.py --model {model} "
           f"--in {os.path.basename(in_path)} --out embed_out.json "
           f"--max-seq {max_seq} {flag} {co}")
    log.info("remote: %s", cmd)
    subprocess.run(["ssh", host, cmd], check=True)
    subprocess.run(["scp", "-q", f"{host}:{remote_dir}/embed_out.json", out_path],
                   check=True)
    # remove the remote transient too — MySQL is the store, not JSON
    subprocess.run(["ssh", host, f"rm -f {remote_dir}/embed_out.json"], check=False)


# --- average-rank Spearman (no scipy dependency) ---------------------------

def _rankdata(a):
    a = np.asarray(a, dtype=float)
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    s = a[order]
    while i < len(a):
        j = i
        while j < len(a) and s[j] == s[i]:
            j += 1
        avg = (i + 1 + j) / 2.0  # mean of ranks i+1..j
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def spearman(x, y):
    rx, ry = _rankdata(x), _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def _fmt(x):
    return "" if x is None else f"{x:+.3f}"


def report_from_db(conn, md_path, csv_path, run="baseline"):
    """Generate report.md + a flat CSV entirely from the MySQL tables, for one
    embedding ``run`` (baseline=harrier, gte-qwen2=the model swap)."""
    payload = db.fetch_report(conn, run)
    papers, order = payload["papers"], payload["order"]
    meta = payload["meta"]
    methods = [m for m in METHODS
               if any(m in papers[p]["scores"] for p in order)] or METHODS

    md = ["# Independent Embedding Analysis vs LLM Verdicts",
          "",
          f"Sample: {len(order)} preselected papers. Model: `{meta.get('model','?')}` "
          f"({meta.get('dim','?')}-dim, 4-bit={meta.get('use_4bit','?')}). "
          f"Source of record: MySQL `embedding_scores` (keyed on code id).",
          "",
          "`e = cos(paper, POS_prototype) − cos(paper, NEG_prototype)` — higher means the "
          "text reads as *arguing* the criterion. This embedding axis is **independent**: "
          "it is reported beside the LLM verdict and never overrides it (plan decision 0). "
          "Each paper is scored three ways as separate documents — **full** text, "
          "**abstract** only, and **chunk** (8192-token windows @50% overlap, max-pooled) — "
          f"to test which granularity best tracks the verdict (chunk size "
          f"{meta.get('chunk_size','?')}, overlap {meta.get('chunk_overlap','?')}).",
          ""]
    if meta.get("scoring") == "leverred-axis":
        md += ["**Scoring:** offline recompute from persisted vectors with all four "
               "space-level levers — centre (μ), whiten (top-`k` PCs, "
               f"`k={meta.get('whiten_k','?')}`), unit difference-axis projection, and "
               "shared-register orthogonalization "
               f"(`strength={meta.get('shared_strength','?')}`). `e` here is the "
               "decongested axis projection, not the raw double-cosine.", ""]

    # Spearman accumulators: cols[method][crit] = (e_vals, ordinal_vals)
    cols = {m: {c: ([], []) for c in CRITERIA} for m in methods}
    csv_rows = []

    md.append("## Per-paper verdicts (criteria_judge) + embedding columns\n")
    for c in CRITERIA:
        md.append(f"### Criterion: `{c}`\n")
        headers = ["code", "paper", "verdict", "conf", "e_full", "e_abstract", "e_chunk"]
        trows = []
        for pid in order:
            p = papers[pid]
            v = p["verdict"].get(c, "?")
            conf = p["confidence"].get(c)
            es = {m: p["scores"].get(m, {}).get(c) for m in methods}
            paper = os.path.basename(pid)
            paper = (paper[:40] + "…") if len(paper) > 41 else paper
            trows.append([p["code"], paper, v,
                          "" if conf is None else f"{conf:.2f}",
                          _fmt(es.get("full")), _fmt(es.get("abstract")),
                          _fmt(es.get("chunk"))])
            for m in methods:
                e = es.get(m)
                if e is not None and v in ("met", "not_met", "unclear"):
                    cols[m][c][0].append(e)
                    cols[m][c][1].append(verdict_ordinal(v))
            if c == CRITERIA[0]:
                csv_rows.append({"code_number": p["code"], "pdf_path": os.path.basename(pid)})
        md.append(_md_table(headers, trows))
        md.append("")

    # flat CSV (one row per paper, all criteria x methods)
    for row in csv_rows:
        pid = next(pp for pp in order if os.path.basename(pp) == row["pdf_path"])
        p = papers[pid]
        for c in CRITERIA:
            row[f"{c}_verdict"] = p["verdict"].get(c)
            row[f"{c}_confidence"] = p["confidence"].get(c)
            for m in methods:
                row[f"{c}_e_{m}"] = p["scores"].get(m, {}).get(c)

    md.append("## Which granularity tracks the verdict? Spearman ρ(e, verdict_ordinal)\n")
    md.append("Higher ρ = the embedding ranks papers in the same order the LLM does. "
              "ρ is `n/a` when all verdicts for a criterion are identical (no rank "
              "variation).\n")
    sp_headers = ["criterion"] + methods
    sp_rows = []
    for c in CRITERIA:
        line = [c]
        for m in methods:
            e_vals, o_vals = cols[m][c]
            line.append(f"{spearman(e_vals, o_vals):+.3f}" if len(set(o_vals)) > 1
                        else "n/a")
        sp_rows.append(line)
    md.append(_md_table(sp_headers, sp_rows))
    md.append("")

    sep = payload["pole_separation"]
    md.append("## Pole separation (pairwise cosine; high neg-neg ≈ muddied poles)\n")
    for pole in ("pos", "neg"):
        pairs = sep.get(pole, {})
        md.append(f"- **{pole}**: " +
                  ", ".join(f"`{k}`={v:+.2f}" for k, v in pairs.items()))
    md.append("")

    within = sep.get("within", {})
    if within:
        md.append("## Pole width `within` (centred cosine pos↔neg per criterion)\n")
        md.append("Bounds the dynamic range of `e`: ≈+1 means the poles overlap and "
                  "magnitudes are compressed (trust ranks, not magnitudes); toward 0/−1 "
                  "the poles are well separated and magnitudes carry signal.\n")
        md.append(_md_table(["criterion", "within"],
                            [[c, f"{within[c]:+.3f}"] for c in CRITERIA if c in within]))
        md.append("")

    ctrl = payload["controls"]
    if ctrl:
        md.append("## Control checks\n")
        leverred = meta.get("scoring") == "leverred-axis"
        controls_leverred = meta.get("controls_scoring") == "leverred-axis"
        if leverred and not controls_leverred:
            md.append("> ⚠️ These control scores are the **prior contrastive** values — "
                      "control document vectors are not persisted, so the offline "
                      "recompute cannot re-score them with the levers. Re-embed with "
                      "control-vector capture to refresh them.\n")
        elif controls_leverred:
            md.append("> Scored through the same corpus geometry (μ / whitening / "
                      "decongested axes) as the papers — directly comparable to the "
                      "per-paper `e`.\n")
        md.append("genetic-code control should read high on all three; "
                  "deterministic-chemistry should read low on `arbitrariness`.\n")
        crows = [[name] + [_fmt(sc.get(c)) for c in CRITERIA]
                 for name, sc in ctrl.items()]
        md.append(_md_table(["control"] + CRITERIA, crows))
        md.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)

    print("\n=== Spearman ρ(e, verdict_ordinal) — which granularity tracks the verdict ===")
    print(_md_table(sp_headers, sp_rows))
    print(f"\nwrote {md_path} and {csv_path} ({len(order)} papers) from MySQL")


def recompute_from_db(conn, md_path, csv_path, k, strength, run="baseline"):
    """Offline rescore (no GPU): load persisted vectors, apply all four levers, write the
    leverred ``e`` + centred pole widths back to MySQL, regenerate the report — all scoped
    to one embedding ``run``."""
    db.init_schema(conn)
    doc_vecs, poles, codes = db.fetch_vectors(conn, run)
    if not doc_vecs or not poles:
        log.warning("no persisted vectors in MySQL for run=%s — run the GPU embed first; "
                    "nothing to recompute", run)
        return
    # Drop in-corpus Code Biology self-references (conference/society pages mirrored under
    # codebiology.org): the poles are mined from that same corpus, so they read maximally
    # in-register and leak to the top of every criterion. Evict their stale score rows too.
    doc_vecs, dropped = es_mod.drop_self_references(doc_vecs)
    if dropped:
        db.delete_score_rows(conn, dropped, run)
        log.info("dropped %d in-corpus self-reference doc(s) from the corpus: %s",
                 len(dropped), dropped)
    scores, within = es_mod.recompute(doc_vecs, poles, k=k, strength=strength)
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Score the controls through the SAME corpus geometry as the papers (leverred), if the
    # structural embed captured their raw vectors; otherwise the report keeps the pre-lever
    # contrastive numbers and flags them.
    control_vecs = db.fetch_control_vectors(conn, run)
    control_scores = None
    params = {"whiten_k": k, "shared_strength": strength, "scoring": "leverred-axis"}
    if control_vecs:
        control_scores = es_mod.score_controls(control_vecs, doc_vecs, poles,
                                               k=k, strength=strength)
        params["controls_scoring"] = "leverred-axis"
        log.info("rescored %d control(s) with the corpus geometry (leverred)",
                 len(control_scores))

    db.apply_recompute(conn, scores, codes, within, params, run_ts,
                       control_scores=control_scores, run=run)
    n_methods = len(next(iter(scores.values())))
    log.info("recomputed %d papers x %d methods (k=%d, strength=%.2f) from vectors; "
             "no GPU. within=%s", len(scores), n_methods, k, strength,
             {c: round(v, 3) for c, v in within.items()})
    report_from_db(conn, md_path, csv_path, run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", default="sample_verdicts.jsonl")
    ap.add_argument("--prototypes", default="prototypes.json")
    ap.add_argument("--host", default="asushimu")
    ap.add_argument("--remote-dir", default="/data/vllm/harrier_run")
    ap.add_argument("--model", default="/data/vllm/harrier-oss-v1-27b")
    ap.add_argument("--cuda-devices", default="2",
                    help="CUDA_VISIBLE_DEVICES on the GPU host (PCI order); 2 = 3090 Ti")
    ap.add_argument("--in", dest="in_path", default="embed_in.json")
    ap.add_argument("--run", default="baseline",
                    help="embedding-run label scoping every DB read/write (rows coexist "
                         "non-destructively): 'baseline' = harrier, 'gte-qwen2' = the model "
                         "swap. Verdicts are run-agnostic (shared `verdicts` table).")
    ap.add_argument("--csv", default="embedding_scores.csv")
    ap.add_argument("--md", default="report.md")
    ap.add_argument("--char-budget", type=int, default=120000)
    ap.add_argument("--extra-sample", type=int, default=0,
                    help="randomly draw this many UNLABELLED papers from the downloaded "
                         "corpus (in addition to the verdict set) to widen the embedded "
                         "basis for centring/whitening; they don't enter ρ(e, verdict)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for --extra-sample (reproducible draw)")
    ap.add_argument("--codes-csv", default="biological_codes.csv",
                    help="source for the --extra-sample pool (code -> URL -> pdf path)")
    ap.add_argument("--pdf-dir", default="pdfs",
                    help="directory of downloaded PDFs for the --extra-sample pool")
    ap.add_argument("--max-seq", type=int, default=16384,
                    help="token cap for full/abstract single-pass embeds; 32k OOMs "
                         "27B/4-bit on a 24 GB card")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="skip build+remote; regenerate report.md from current DB")
    ap.add_argument("--recompute", action="store_true",
                    help="offline (no GPU): rescore from persisted vectors with the "
                         "space-level levers, write leverred e + pole widths, re-report")
    ap.add_argument("--controls-only", action="store_true",
                    help="one cheap GPU run embedding ONLY the control texts (capture "
                         "control_vectors), upsert them, then recompute so the controls "
                         "are scored leverred; the 219 persisted papers are untouched")
    ap.add_argument("--whiten-k", type=int, default=es_mod.DEFAULT_WHITEN_K,
                    help="top-k principal components removed in whitening (0 = off; the "
                         "20-paper smoke test found k>=1 hurt — raise only on a big corpus)")
    ap.add_argument("--shared-strength", type=float, default=es_mod.DEFAULT_SHARED_STRENGTH,
                    help="how strongly each criterion axis is orthogonalized against the "
                         "shared register direction, in [0,1]; 1.0 over-corrected on the "
                         "smoke test, so the default is partial")
    args = ap.parse_args()

    if args.recompute:
        conn = db.connect()
        try:
            recompute_from_db(conn, args.md, args.csv, args.whiten_k, args.shared_strength,
                              run=args.run)
        finally:
            conn.close()
        return

    if args.controls_only:
        proto_raw = json.load(open(args.prototypes, encoding="utf-8"))
        prototypes = load_prototypes(args.prototypes)
        controls = proto_raw.get("_controls", {})
        log.info("controls-only GPU run: embedding %d control text(s); papers untouched",
                 len(controls))
        payload = build_controls_input(prototypes, controls)
        with open(args.in_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        out_tmp = "embed_out.json"
        run_remote(args.host, args.remote_dir, args.in_path, out_tmp, args.model,
                   use_4bit=not args.no_4bit, cuda_devices=args.cuda_devices,
                   max_seq=args.max_seq, controls_only=True)
        with open(out_tmp, encoding="utf-8") as f:
            out = json.load(f)
        run_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = db.connect()
        try:
            db.store(conn, out, [], run_ts, run=args.run)  # control_vectors only (no papers)
            log.info("stored %d control vector(s) to MySQL",
                     len(out.get("control_vectors", {})))
            # now rescore the controls through the corpus geometry + regenerate the report
            recompute_from_db(conn, args.md, args.csv, args.whiten_k, args.shared_strength,
                              run=args.run)
        finally:
            conn.close()
        os.remove(out_tmp)
        return

    if args.report_only:
        conn = db.connect()
        try:
            report_from_db(conn, args.md, args.csv, run=args.run)
        finally:
            conn.close()
        return

    recs = load_verdicts(args.verdicts)
    log.info("loaded %d prior verdicts", len(recs))

    if args.extra_sample > 0:
        pool = build_pool(args.codes_csv, args.pdf_dir)
        extra = sample_extra_papers(pool, {r["pdf_path"] for r in recs},
                                    args.extra_sample, args.seed)
        recs += extra
        log.info("drew %d extra unlabelled papers from a pool of %d (seed=%d); "
                 "%d papers total to embed", len(extra), len(pool), args.seed, len(recs))

    proto_raw = json.load(open(args.prototypes, encoding="utf-8"))
    prototypes = load_prototypes(args.prototypes)
    controls = proto_raw.get("_controls", {})

    log.info("extracting paper text + building %s", args.in_path)
    payload = build_input(recs, prototypes, controls, args.char_budget)
    with open(args.in_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    out_tmp = "embed_out.json"   # transient transport only — deleted after DB load
    run_remote(args.host, args.remote_dir, args.in_path, out_tmp,
               args.model, use_4bit=not args.no_4bit, cuda_devices=args.cuda_devices,
               max_seq=args.max_seq)
    with open(out_tmp, encoding="utf-8") as f:
        out = json.load(f)

    run_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = db.connect()
    try:
        db.store(conn, out, recs, run_ts, run=args.run)
        log.info("stored %d papers x %d methods to MySQL (run=%s)",
                 len(out["scores"]), len(out.get("methods", METHODS)), args.run)
        report_from_db(conn, args.md, args.csv, run=args.run)
    finally:
        conn.close()
    os.remove(out_tmp)   # MySQL is the store, not JSON


if __name__ == "__main__":
    main()
