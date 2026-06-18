"""Build the Barbieri-anchored gold reference set (plan: stateless-leaping-fiddle).

Phase 1 — **embedding-driven molecular selection**. Gold *positives* are defined by
molecular code membership; rather than hand-curate which of the 435 codes / 24 topics are
"molecular", we let the embedding space decide. The anchor is the **genetic-code centroid**
— the mean of the ``Genetic code*`` papers' vectors, Barbieri's canonical molecular
exemplar (§1) — projected through the *identical* μ-centred / whitened §4 scorer the
criterion scores and topic assignment use (``embed_score.build_scorer``). Molecular-ness is
then cosine to that anchor:

    anchor              = unit mean of the genetic-code papers' projected chunk vectors
    molecularness(d)    = cos(project(d), anchor)             # max-pooled over chunks
    molecularness(topic)= cos(project(centroid), anchor)

This produces an **auditable ranking** of every code and of the 24 topics (``select``
writes ``gold/molecular_ranking.csv`` + ``gold/topic_ranking.csv``); the molecular cut is
confirmed with the user before Phase 2 consumes it. The four borderline topics
(Morphological, Pathological, Olfactory, Synthetic) are expected to fall on the
non-molecular side — a testable prediction, not a hand decision.

A second, **artificial-code anchor** (``computer_code_positive``) provides a contrast pole: a
clean exemplar of the §9.1 *broadened* criteria (source↔execution bridged by the interpreter,
arbitrary symbol→operation mapping) that is explicitly **not** an organic code under Barbieri's
strict §1 definition. The corpus holds no computer-code papers, so unlike the molecular anchor it
is seeded from the authored ``_controls`` exemplar embedded once
(``embed_independent --controls-only``) and projected into the *same* centred space. ``select``
then writes ``gold/artificial_contrast.csv`` (molecular vs artificial cosine per code) — a
diagnostic only; the artificial anchor is kept **out** of the molecular gold-positive pool.

Offline: reads persisted ``doc_vectors`` + ``topic_centroids`` (one embedding ``run``) and
``biological_codes.csv``; no GPU, no spend. The selection run is gated on the completed
``baseline`` embed (Phase 0).
"""

import argparse
import csv as _csv
import logging
import os
import re

import numpy as np

import embed_score as es

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("build_gold_set")

CSV_PATH = "biological_codes.csv"
GOLD_DIR = "gold"
# "Genetic code", its variants A–D and "Mitochondrial genetic code" — the molecular anchor.
GENETIC_RE = re.compile(r"genetic\s+code", re.I)


# --- code metadata ---------------------------------------------------------

def load_code_names(csv_path=CSV_PATH):
    """``biological_codes.csv`` → ``{code_number: code_name}`` (first name per number)."""
    names = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            try:
                cn = int(row["Code Number"])
            except (KeyError, ValueError, TypeError):
                continue
            names.setdefault(cn, (row.get("Code Name") or "").strip())
    return names


def anchor_pids(codes, code_names, pattern=GENETIC_RE):
    """pdf_paths whose code name matches the genetic-code ``pattern``.

    ``codes`` maps ``pdf_path → code_number`` (``db.fetch_vectors``); ``code_names`` maps
    ``code_number → name``. These papers seed the molecular anchor centroid."""
    return {pid for pid, cn in codes.items()
            if pattern.search(code_names.get(cn, ""))}


# --- molecular anchor & scoring (centred §4 space) -------------------------

def _paper_vecs(methods, method="chunk"):
    """A paper's vectors for ``method``, falling back to ``full`` then any method."""
    return methods.get(method) or methods.get("full") or next(iter(methods.values()))


# The control exemplar(s) that seed the artificial-code anchor (prototypes.json `_controls`).
# Unlike the molecular anchor — averaged over real in-corpus genetic-code papers — the corpus
# holds no computer-code papers, so this anchor is seeded from an authored exemplar passage
# embedded once (embed_independent --controls-only → control_vectors). It is a *contrast* anchor:
# a clean artificial code under the §9.1 broadened criteria, but NOT an organic code under
# Barbieri's strict §1 definition, so it is kept out of the molecular gold-positive pool.
ARTIFICIAL_SEED_KEYS = ("computer_code_positive",)


def _anchor_from_reps(reps, what):
    """Unit mean of a list of projected representative vectors (the anchor direction)."""
    if not reps:
        raise ValueError(f"no {what} found to seed the anchor")
    return es._l2(np.mean(reps, axis=0))


def molecular_anchor(doc_vecs, poles, anchor_ids, method="chunk",
                     k=es.DEFAULT_WHITEN_K, strength=es.DEFAULT_SHARED_STRENGTH):
    """``(project, anchor)`` — the genetic-code centroid in the centred §4 space.

    ``project`` is the shared μ-centred / top-``k``-whitened / unit scorer built from the
    **paper** corpus (so anchor, papers and centroids live in the same space the criterion
    scores and topic assignment use). ``anchor`` is the unit mean of the genetic-code
    papers' projected chunk vectors. Raises if no anchor paper is present."""
    project, _axes, _within = es.build_scorer(doc_vecs, poles, k, strength)
    reps = []
    for pid in anchor_ids:
        methods = doc_vecs.get(pid)
        if not methods:
            continue
        vecs = _paper_vecs(methods, method)
        reps.append(np.mean([project(np.asarray(v, dtype=np.float64)) for v in vecs],
                            axis=0))
    return project, _anchor_from_reps(reps, "genetic-code anchor papers in doc_vecs")


def artificial_anchor(project, control_vecs, seed_keys=ARTIFICIAL_SEED_KEYS):
    """The artificial-code anchor in the *same* centred space as ``project``.

    Seeded from the authored ``_controls`` exemplar(s) named in ``seed_keys`` (default
    ``computer_code_positive``), each embedded as one ``control_vectors`` row and projected
    through the shared scorer. Unit mean of the present seeds. Raises if no named seed is in
    ``control_vecs`` (run ``embed_independent --controls-only`` first). Reuse the same
    ``project`` returned by :func:`molecular_anchor` so both anchors share one geometry."""
    reps = [project(np.asarray(control_vecs[k], dtype=np.float64))
            for k in seed_keys if k in control_vecs]
    return _anchor_from_reps(reps, f"artificial seed control vectors {list(seed_keys)}")


def paper_molecularness(project, anchor, vecs):
    """Max-pool cosine of a paper's chunk windows to ``anchor`` (matches §4 max-pool):
    one strongly-molecular window makes the paper molecular."""
    return max(float(project(np.asarray(v, dtype=np.float64)) @ anchor) for v in vecs)


def rank_papers(doc_vecs, project, anchor, method="chunk"):
    """``{pdf_path: molecularness}`` over every paper (max-pooled chunk cosine)."""
    return {pid: paper_molecularness(project, anchor, _paper_vecs(methods, method))
            for pid, methods in doc_vecs.items()}


def rank_codes(doc_vecs, codes, code_names, project, anchor, method="chunk"):
    """Codes ranked by mean paper molecular-ness, most molecular first.

    Returns ``[(code_number, code_name, n_papers, mean_mol, max_mol), ...]``."""
    papermol = rank_papers(doc_vecs, project, anchor, method)
    by_code = {}
    for pid, m in papermol.items():
        by_code.setdefault(codes.get(pid), []).append(m)
    rows = [(cn, code_names.get(cn, ""), len(ms), float(np.mean(ms)), float(np.max(ms)))
            for cn, ms in by_code.items()]
    rows.sort(key=lambda r: -r[3])
    return rows


def rank_codes_contrast(doc_vecs, codes, code_names, project, mol_anchor, art_anchor,
                        method="chunk"):
    """Codes scored against BOTH anchors, ranked most-artificial-leaning first.

    Returns ``[(code_number, code_name, n_papers, mean_mol, mean_art, mean_diff), ...]`` where
    ``mean_diff = mean_art - mean_mol``; a positive diff means the code's papers sit closer to the
    artificial (computer-code) anchor than the molecular one. Diagnostic contrast only — it does
    not assign gold labels (the artificial anchor is excluded from the molecular gold pool)."""
    mol = rank_papers(doc_vecs, project, mol_anchor, method)
    art = rank_papers(doc_vecs, project, art_anchor, method)
    by_code = {}
    for pid in doc_vecs:
        by_code.setdefault(codes.get(pid), []).append((mol[pid], art[pid]))
    rows = []
    for cn, pairs in by_code.items():
        mm = float(np.mean([m for m, _ in pairs]))
        ma = float(np.mean([a for _, a in pairs]))
        rows.append((cn, code_names.get(cn, ""), len(pairs), mm, ma, ma - mm))
    rows.sort(key=lambda r: -r[5])
    return rows


def rank_topics(project, anchor, centroids):
    """The 24 topics ranked by centroid proximity to ``anchor`` (the molecular ordering
    that replaces the hand allowlist). ``centroids`` is ``db.fetch_topic_centroids`` output:
    ``{topic_id: {'label': str, 'vec': np.ndarray}}``. Returns ``[(topic_id, label, cos)]``."""
    rows = [(tid, d["label"],
             float(project(np.asarray(d["vec"], dtype=np.float64)) @ anchor))
            for tid, d in centroids.items()]
    rows.sort(key=lambda r: -r[2])
    return rows


# --- CLI: select (Phase 1) -------------------------------------------------

def _write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    log.info("wrote %s (%d rows)", path, len(rows))


def cmd_select(args):
    import db
    conn = db.connect()
    try:
        doc_vecs, poles, codes = db.fetch_vectors(conn, run=args.run)
        centroids = db.fetch_topic_centroids(conn, run=args.run)
        control_vecs = db.fetch_control_vectors(conn, run=args.run)
    finally:
        conn.close()
    names = load_code_names(args.csv)
    anchors = anchor_pids(codes, names)
    log.info("anchor: %d genetic-code papers across %d papers / %d codes",
             len(anchors), len(doc_vecs), len({c for c in codes.values()}))
    project, anchor = molecular_anchor(doc_vecs, poles, anchors, method=args.method)

    code_rows = rank_codes(doc_vecs, codes, names, project, anchor, method=args.method)
    _write_csv(os.path.join(GOLD_DIR, "molecular_ranking.csv"),
               ["code_number", "code_name", "n_papers", "mean_mol", "max_mol"],
               [(cn, nm, n, f"{mean:.4f}", f"{mx:.4f}")
                for cn, nm, n, mean, mx in code_rows])

    if centroids:
        topic_rows = rank_topics(project, anchor, centroids)
        _write_csv(os.path.join(GOLD_DIR, "topic_ranking.csv"),
                   ["topic_id", "label", "cos_to_anchor"],
                   [(tid, lbl, f"{c:.4f}") for tid, lbl, c in topic_rows])
        log.info("topic molecular ranking (most → least molecular):")
        for tid, lbl, c in topic_rows:
            log.info("  %2d  %+.4f  %s", tid, c, lbl)
    else:
        log.warning("no topic_centroids for run=%s — skipping topic ranking", args.run)

    # Artificial-code contrast anchor (computer_code_positive), seeded from an embedded control
    # exemplar rather than corpus papers (the corpus holds no computer-code papers). Gated on a
    # one-off `embed_independent --controls-only --run %s` having embedded the seed.
    have_seed = [k for k in ARTIFICIAL_SEED_KEYS if k in control_vecs]
    if have_seed:
        art_anchor = artificial_anchor(project, control_vecs)
        contrast = rank_codes_contrast(doc_vecs, codes, names, project, anchor, art_anchor,
                                       method=args.method)
        _write_csv(os.path.join(GOLD_DIR, "artificial_contrast.csv"),
                   ["code_number", "code_name", "n_papers", "mean_mol", "mean_art", "mean_diff"],
                   [(cn, nm, n, f"{mm:.4f}", f"{ma:.4f}", f"{d:+.4f}")
                    for cn, nm, n, mm, ma, d in contrast])
        log.info("artificial anchor seeded from %s; wrote molecular↔artificial contrast", have_seed)
    else:
        log.warning("no %s control vector for run=%s — run `embed_independent --controls-only "
                    "--run %s` to embed the seed, then re-run select for the artificial contrast",
                    list(ARTIFICIAL_SEED_KEYS), args.run, args.run)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sel = sub.add_parser("select", help="Phase 1: embedding-driven molecular ranking")
    sel.add_argument("--run", default="baseline")
    sel.add_argument("--method", default="chunk")
    sel.add_argument("--csv", default=CSV_PATH)
    sel.set_defaults(func=cmd_select)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
