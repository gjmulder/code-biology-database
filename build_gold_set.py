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
import json
import logging
import os
import re
from collections import Counter

import numpy as np

import criteria_judge as cj
import embed_score as es
import pdf_text as pt
from assign_topics import paper_dominant_topic
from download_pdfs import output_path_for

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("build_gold_set")

CSV_PATH = "biological_codes.csv"
GOLD_DIR = "gold"
MOLECULAR_TOPICS_CSV = os.path.join(GOLD_DIR, "molecular_topics.csv")
GOLD_SET_PATH = "gold_set.csv"
# Phase 3 — Barbieri's own bibliographies (the authority list for the tier-1 upgrade).
CODE_BIOLOGY_PDFS = "Code_Biology_PDFs"
SEED_CSV = "code_biology_seed.csv"
# Seminal Barbieri/Major texts beyond the code-0 seed manifest whose reference lists count as
# Barbieri's own citations (filenames in CODE_BIOLOGY_PDFS; resolved best-effort, missing skipped).
SEMINAL_EXTRA = (
    "The_Organic_Codes_an_introduction_to_semantic_biol.pdf",
    "Barbieri M (2024) Codes and Evolution-The Origin of Absolute Novelties.pdf",
)
# The git-tracked gold reference set: one row per labelled (code, paper) instance, merged
# across phases. `source` namespaces each phase's rows so a re-run replaces only its own (§Phase 5).
GOLD_FIELDS = ["code_number", "pdf_path", "polarity", "tier", "source", "criterion", "evidence"]
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


# --- Phase 2: tier-2 gold positives (curated topic allowlist) --------------

def load_molecular_topics(path=MOLECULAR_TOPICS_CSV):
    """Curated allowlist → ``{topic_id: label}`` for the ``molecular == yes`` topics.

    This is the **hand-confirmed** molecular cut (Phase 1's ranking is the audit, not the
    definition — the gold positives must not be defined by the embedding molecularity score
    they help validate). Membership is the key set; the label is reused for evidence."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            if (row.get("molecular") or "").strip().lower() == "yes":
                out[int(row["topic_id"])] = (row.get("label") or "").strip()
    return out


def dominant_topics(chunk_topics):
    """``db.fetch_chunk_topics`` output → ``{pdf_path: dominant_topic_id}``.

    Reuses :func:`assign_topics.paper_dominant_topic` (the §2.1 max-pool stratifier); papers
    with no chunk assignments are dropped."""
    out = {}
    for pid, chunks in chunk_topics.items():
        dom, _aff = paper_dominant_topic(chunks)
        if dom is not None:
            out[pid] = dom
    return out


def code_dominant_topic(pids, dominant_by_pid):
    """A code's dominant topic = the **modal** per-paper dominant topic across its papers
    (ties broken to the lowest ``topic_id`` for determinism). ``None`` if none of the code's
    papers has a dominant topic."""
    votes = [dominant_by_pid[p] for p in pids if p in dominant_by_pid]
    if not votes:
        return None
    counts = Counter(votes)
    top = max(counts.values())
    return min(t for t, n in counts.items() if n == top)


def molecular_codes(codes, dominant_by_pid, allowlist):
    """Codes whose modal dominant topic ∈ ``allowlist`` → ``{code_number: (dominant_topic,
    n_papers)}``.

    ``codes`` maps ``pdf_path → code_number`` over the **embedded** corpus (``db.fetch_vectors``),
    so ``n_papers`` counts only downloaded+embedded references. **Code 0** (the foundational
    Code-Biology / Major texts) is the gold *root* — authorship, not a molecular-code reference —
    and is excluded here; it is handled by the Phase 3 ``cite`` upgrade."""
    by_code = {}
    for pid, cn in codes.items():
        by_code.setdefault(cn, []).append(pid)
    out = {}
    for cn, pids in by_code.items():
        if cn == 0:
            continue
        dom = code_dominant_topic(pids, dominant_by_pid)
        if dom in allowlist:
            out[cn] = (dom, len(pids))
    return out


def load_topic_labels(path=MOLECULAR_TOPICS_CSV):
    """All 24 topics → ``{topic_id: label}`` (both ``yes`` and ``no`` rows), for evidence
    strings on the non-molecular side (where :func:`load_molecular_topics` has no entry)."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            try:
                tid = int(row["topic_id"])
            except (KeyError, ValueError, TypeError):
                continue
            out[tid] = (row.get("label") or "").strip()
    return out


def tier2_positives(codes, mol_codes, code_names, topic_labels):
    """Gold+ / tier-2 rows: **every** embedded paper of each molecular code (the code is
    DB-endorsed molecular, so all its references are positives — §Phase 2). One ``GOLD_FIELDS``
    dict per paper; ``evidence`` carries the code name + dominant topic for hand-auditing."""
    by_code = {}
    for pid, cn in codes.items():
        by_code.setdefault(cn, []).append(pid)
    rows = []
    for cn, (dom, _n) in sorted(mol_codes.items()):
        label = topic_labels.get(dom, str(dom))
        evidence = f"{code_names.get(cn, '')} · topic {dom} {label}".strip()
        for pid in sorted(by_code[cn]):
            rows.append({"code_number": cn, "pdf_path": pid, "polarity": "pos",
                         "tier": "2", "source": "db", "criterion": "all",
                         "evidence": evidence})
    return rows


# --- gold_set.csv (git-tracked source of truth, merged across phases) ------

def read_gold_set(path=GOLD_SET_PATH):
    """Existing ``gold_set.csv`` → list of ``GOLD_FIELDS`` dicts (empty if absent)."""
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: row.get(k, "") for k in GOLD_FIELDS} for row in _csv.DictReader(f)]


def merge_gold(existing, new_rows, sources):
    """Replace every existing row whose ``source`` ∈ ``sources`` with ``new_rows``; keep the
    rest in place. Lets one phase (here ``select`` → ``{"db"}``) refresh its own rows
    idempotently without clobbering another phase's (``exclusion``/``implicit``/``barbieri-cite``)."""
    kept = [r for r in existing if r.get("source") not in sources]
    return kept + list(new_rows)


def write_gold_set(path, rows):
    _write_csv(path, GOLD_FIELDS, [[r[k] for k in GOLD_FIELDS] for r in rows])


# --- Phase 3: tier-1 upgrade (Barbieri-cited) ------------------------------
#
# A tier-2 (DB-endorsed) molecular positive upgrades to **tier-1** when Barbieri/Major *also*
# cite it in their own seminal texts — the authority signal, without the hard "which code does
# this citation support" classification (we already have the code from the DB). The match key is
# a citation **signature** ``(first-author surname, year)`` shared by the corpus citation string
# and Barbieri's reference entry. This is loose by design (surname+year), but it only ever
# *promotes within* the already-curated molecular positive set, so a stray match over-promotes a
# real molecular positive rather than admitting a non-code — an acceptable, conservative error.

# A reference entry begins at the left margin with an author block, in one of two house styles:
#   APA      "Gabius, H.-J. (2000)…"   surname, comma, initial-dot
#   Springer "Adl SM, Simpson ABG …"   surname, space, ALL-CAPS initials (no comma, no dots)
# Wrapped continuation lines ("Nature, 465, …", "Microbiol 52:399–451") match neither and so
# don't start a new entry.
_REF_ENTRY_RE = re.compile(
    r"^[A-Z][A-Za-zÀ-ÿ'’-]+"                       # surname
    r"(?:\s*,\s*[A-Z]\s*\.|\s+[A-Z]{1,4}\b)")      # ", I."  or  " SM"/" ABG"/" T"
_SURNAME_RE = re.compile(r"\s*([A-Z][A-Za-zÀ-ÿ'’.-]*)")
_YEAR_RE = re.compile(r"\b(1[89]\d\d|20\d\d)\b")


def _signature(text):
    """``(surname_lower, year)`` from a citation string — first capitalised token + first
    19xx/20xx year — or ``None`` if either is missing."""
    text = text or ""
    m = _SURNAME_RE.match(text)
    y = _YEAR_RE.search(text)
    if not m or not y:
        return None
    return (m.group(1).lower().rstrip("."), y.group(1))


def paper_signature(paper_name):
    """Citation signature of a corpus ``Paper Name`` (its first-author surname + year)."""
    return _signature(paper_name)


def parse_reference_signatures(ref_text):
    """A reference section's text → ``{(surname, year), …}`` for every entry.

    Entries are segmented on the left-margin "Surname, I." start (:data:`_REF_ENTRY_RE`);
    each entry's wrapped lines are joined before signing, so a multi-line reference yields a
    single signature and a continuation line never spawns a spurious entry."""
    entries, cur = [], []
    for raw in ref_text.splitlines():
        line = raw.strip()
        if _REF_ENTRY_RE.match(line):
            if cur:
                entries.append(" ".join(cur))
            cur = [line]
        elif cur:
            cur.append(line)
    if cur:
        entries.append(" ".join(cur))
    return {s for e in entries if (s := _signature(e))}


def seminal_pdfs(pdf_dir=CODE_BIOLOGY_PDFS, seed_csv=SEED_CSV, extra=SEMINAL_EXTRA):
    """Resolve Barbieri/Major seminal-text PDF paths = the code-0 seed manifest's ``Source File``
    column plus :data:`SEMINAL_EXTRA`, joined to ``pdf_dir``. De-duplicated, existing files only."""
    files = []
    if os.path.exists(seed_csv):
        with open(seed_csv, newline="", encoding="utf-8") as f:
            files += [(row.get("Source File") or "").strip() for row in _csv.DictReader(f)]
    files += list(extra)
    out, seen = [], set()
    for name in files:
        if not name:
            continue
        p = os.path.join(pdf_dir, name)
        if p not in seen and os.path.exists(p):
            seen.add(p)
            out.append(p)
    return out


def cited_signatures(paths):
    """Union of reference signatures across the seminal PDFs (extraction failures skipped)."""
    sigs = set()
    for p in paths:
        try:
            secs = pt.split_sections(pt.extract_text(p))
        except Exception as e:                       # noqa: BLE001 — a bad PDF must not abort the rest
            log.warning("could not read references from %s: %s", p, e)
            continue
        ref = secs.get("references") or secs.get("bibliography") or ""
        n_before = len(sigs)
        sigs |= parse_reference_signatures(ref)
        log.info("  %-50s %4d signatures (+%d new)",
                 os.path.basename(p)[:50], len(parse_reference_signatures(ref)), len(sigs) - n_before)
    return sigs


def paper_names_by_path(csv_path=CSV_PATH):
    """``{pdf_path: Paper Name}`` from ``biological_codes.csv`` (first name per path), keyed by the
    same DOI-derived path :func:`download_pdfs.output_path_for` gives the gold set's ``pdf_path``."""
    out = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            name = (row.get("Paper Name") or "").strip()
            if not name:
                continue
            pid = output_path_for({"url": row.get("URL") or ""})
            out.setdefault(pid, name)
    return out


def tier1_upgrade(rows, names_by_path, cited):
    """Promote DB-endorsed positives that Barbieri/Major also cite to tier-1.

    Returns ``(new_rows, n_upgraded)``. Only ``source == "db"``, ``polarity == "pos"`` rows whose
    paper signature ∈ ``cited`` are touched: ``tier → "1"``, ``source → "barbieri-cite"``, and the
    matched ``surname year`` appended to ``evidence``. Idempotent — an already-promoted row carries
    ``source == "barbieri-cite"`` and is skipped on a re-run."""
    out, n = [], 0
    for r in rows:
        r = dict(r)
        if r.get("source") == "db" and r.get("polarity") == "pos":
            sig = paper_signature(names_by_path.get(r["pdf_path"], ""))
            if sig and sig in cited:
                r["tier"] = "1"
                r["source"] = "barbieri-cite"
                r["evidence"] = f"{r.get('evidence', '')} | barbieri-cited: {sig[0]} {sig[1]}".strip(" |")
                n += 1
        out.append(r)
    return out, n


def code0_positives(codes):
    """Code-0 (the foundational Code Biology / Major seminal texts, §1b) embedded papers as
    **tier-1 gold+** — the gold *root*: Barbieri's/Major's own definitional texts are the
    strongest possible authority anchor, above any cited reference. ``codes`` is the embedded
    corpus map ``pdf_path → code_number`` (``db.fetch_vectors``); only code 0 qualifies. Distinct
    ``source == "code0"`` so it merges independently of the citation upgrade and survives a
    ``select`` rebuild (which owns only the molecular ``db``/``barbieri-cite`` positives)."""
    return [{"code_number": 0, "pdf_path": pid, "polarity": "pos", "tier": "1",
             "source": "code0", "criterion": "all",
             "evidence": "Code Biology foundational text (code 0)"}
            for pid, cn in sorted(codes.items()) if cn == 0]


# --- Phase 4: soft negatives (implicit) ------------------------------------
#
# Statistical-tractability negatives: an embedded paper is an **implicit gold−** when it is
# (a) not a member of any molecular code (so the DB never endorses it as molecular), and (b) its
# own dominant scientometric topic is non-molecular. Lower tier ("soft") than the authority-grounded
# hard negatives (Barbieri's explicit exclusions) — these are inferred, not stated. Code-0 papers
# (positives) and anything already labelled by an earlier phase are excluded so a soft label never
# conflicts with a positive or a hard negative.

def molecular_member_pids(codes, mol_codes):
    """Every embedded ``pdf_path`` belonging to a molecular code — the gold-positive paper set
    (``codes``: ``pdf_path → code_number``; ``mol_codes``: :func:`molecular_codes` output)."""
    return {pid for pid, cn in codes.items() if cn in mol_codes}


def implicit_negatives(codes, dominant_by_pid, allowlist, mol_codes, topic_labels,
                       exclude_pids=()):
    """Soft gold− rows: embedded papers with a **non-molecular dominant topic** that are **not**
    members of any molecular code. ``tier='soft'``, ``source='implicit'``, ``polarity='neg'``.

    Excludes code 0 (positives), molecular-code members, papers with no dominant topic (can't
    confirm non-molecular), and any ``exclude_pids`` (papers an earlier phase already labelled, so
    a soft negative never overrides a positive/hard negative). Deterministic (sorted by path)."""
    exclude = set(exclude_pids) | molecular_member_pids(codes, mol_codes)
    rows = []
    for pid, cn in sorted(codes.items()):
        if cn == 0 or pid in exclude:
            continue
        dom = dominant_by_pid.get(pid)
        if dom is None or dom in allowlist:
            continue
        label = topic_labels.get(dom, str(dom))
        rows.append({"code_number": cn, "pdf_path": pid, "polarity": "neg",
                     "tier": "soft", "source": "implicit", "criterion": "all",
                     "evidence": f"non-molecular: topic {dom} {label}".strip()})
    return rows


# --- Phase 4: hard negatives (exclude) -------------------------------------
#
# Authority-grounded negatives: an LLM pass over Barbieri's/Major's OWN seminal prose surfaces
# passages where the author argues a named candidate is **not** an organic code (mere chemistry,
# copying-not-coding, no adaptor, no arbitrariness). Each is grounded in a verbatim quote (reusing
# the §9 fuzzy grounding gate) — a hallucinated exclusion is dropped. A candidate is mapped to a
# corpus code by conservative content-token containment; matched codes' embedded papers become
# `hard` gold−. Unmapped exclusions are kept as **conceptual** negatives (audit only — no corpus
# paper to attach, so neither axis can be validated against them). Paid DeepSeek; checkpoint-first.

EXCLUDE_CHECKPOINT = "exclusions.jsonl"          # spend-safety record (never deleted, §7.6)
EXCLUSIONS_AUDIT = os.path.join(GOLD_DIR, "exclusions_audit.csv")
# Prose-window size for the extraction scan (chars, not tokens — extraction is robust to the
# window boundary and this keeps the scan tokenizer-free / offline-testable). Overlap so an
# exclusion straddling a boundary is wholly present in at least one window.
EXCLUDE_MAX_CHARS = 12000
EXCLUDE_OVERLAP = 1500

# Tokens carrying no discriminative power for candidate↔code-name matching.
_NAME_STOP = {"code", "codes", "the", "a", "an", "of", "organic", "biological"}

EXCLUDE_SYSTEM_PROMPT = (
    "You are a Code Biology analyst reading Marcello Barbieri's and Jannie Major's own writing on "
    "what is and is not an organic code. Identify every place where the author argues that a "
    "specific named phenomenon or candidate is NOT an organic code — e.g. it is mere chemistry, "
    "it is copying rather than coding, or it lacks adaptors or arbitrariness. Report ONLY explicit "
    "exclusions the author actually asserts, each grounded in a verbatim quote copied exactly from "
    "the passage. Reply with ONLY a JSON object."
)


def build_exclude_prompt(passage):
    """The extraction instruction for one prose window (verbatim-grounded exclusions as JSON)."""
    return (
        "From the passage below, extract every case where the author argues that a specific named "
        "candidate is NOT an organic code (or is copying rather than coding, or lacks the criteria "
        "of two worlds / an adaptor / arbitrariness). For each, give the candidate (the thing being "
        "excluded), a VERBATIM quote copied exactly from the passage, and a one-sentence reason. "
        "If the passage excludes nothing, return an empty list.\n\n"
        'Return exactly this JSON shape:\n'
        '{"exclusions": [{"candidate": "<short name>", "quote": "<verbatim from the passage>", '
        '"reasoning": "<1 sentence>"}]}\n\n'
        f"=== PASSAGE ===\n{passage}"
    )


def chunk_prose(text, max_chars=EXCLUDE_MAX_CHARS, overlap=EXCLUDE_OVERLAP):
    """Sliding character windows over ``text`` (step = ``max_chars - overlap``). One window if it
    fits; none for empty/whitespace text. Deterministic; the last window always reaches the end."""
    text = text or ""
    if not text.strip():
        return []
    if len(text) <= max_chars:
        return [text]
    step = max(1, max_chars - overlap)
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + max_chars])
        if i + max_chars >= len(text):
            break
        i += step
    return out


def parse_exclusions(raw):
    """A model reply → list of ``{candidate, quote, reasoning}`` dicts (tolerant).

    Drops entries with no candidate; missing quote/reasoning default to empty. Returns ``[]`` on
    no JSON, a non-list ``exclusions``, or an empty list — a window that excludes nothing is fine."""
    try:
        obj = cj._extract_json(raw)
    except cj.JudgeError:
        return []
    items = obj.get("exclusions") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return []
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        cand = str(it.get("candidate", "") or "").strip()
        if not cand:
            continue
        out.append({"candidate": cand,
                    "quote": str(it.get("quote", "") or "").strip(),
                    "reasoning": str(it.get("reasoning", "") or "").strip()})
    return out


def ground_exclusions(items, passage):
    """Keep only exclusions whose ``quote`` is fuzzily grounded in ``passage`` (reuses the §9
    :func:`criteria_judge.is_grounded` gate) — a hallucinated/paraphrased exclusion is dropped."""
    return [it for it in items if cj.is_grounded(it.get("quote", ""), passage)]


def _content_tokens(s):
    """Lowercase alphanumeric tokens of ``s`` minus :data:`_NAME_STOP` (the discriminative words)."""
    return {t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if t not in _NAME_STOP}


def match_candidate_to_code(candidate, code_names):
    """Map an excluded ``candidate`` to a corpus ``code_number`` by **conservative** content-token
    equality: the candidate's discriminative tokens must exactly equal the code name's non-empty
    discriminative tokens (lowest code_number on ties, for determinism). Extra content words in the
    candidate (e.g. ``immune system`` vs the ``Immune code``) block the match. ``None`` if nothing
    matches — kept conceptual."""
    cand = _content_tokens(candidate)
    if not cand:
        return None
    for cn in sorted(code_names):
        key = _content_tokens(code_names[cn])
        if key and key == cand:
            return cn
    return None


def exclusion_rows(grounded, code_names, codes):
    """Grounded exclusions → ``(hard_negative_rows, conceptual)``.

    A candidate mapping to a corpus code (:func:`match_candidate_to_code`) that has embedded papers
    yields one ``hard`` gold− row per such paper (deduped). Unmapped / un-embedded exclusions are
    returned as ``conceptual`` (the item dict, audit only). ``codes`` is the embedded ``pdf_path →
    code_number`` map; ``code_names`` is ``code_number → name``."""
    by_code = {}
    for pid, cn in codes.items():
        by_code.setdefault(cn, []).append(pid)
    rows, conceptual, seen = [], [], set()
    for it in grounded:
        cn = match_candidate_to_code(it["candidate"], code_names)
        if cn is None or cn not in by_code:
            conceptual.append(it)
            continue
        quote = (it.get("quote", "") or "")[:120]
        for pid in sorted(by_code[cn]):
            if (cn, pid) in seen:
                continue
            seen.add((cn, pid))
            rows.append({"code_number": cn, "pdf_path": pid, "polarity": "neg",
                         "tier": "hard", "source": "exclusion", "criterion": "all",
                         "evidence": f"Barbieri excludes '{it['candidate']}': \"{quote}\""})
    return rows, conceptual


def load_exclude_done(checkpoint_path):
    """``{(pdf_path, chunk_idx), …}`` already extracted (resumability; malformed lines skipped)."""
    done = set()
    if not os.path.exists(checkpoint_path):
        return done
    with open(checkpoint_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["pdf_path"], int(rec["chunk_idx"])))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                log.warning("skipping malformed exclusion checkpoint line")
    return done


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
        chunk_topics = db.fetch_chunk_topics(conn, run=args.run, method=args.method)
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

    # Phase 2 — tier-2 gold positives from the curated molecular allowlist (gated on it existing).
    if os.path.exists(args.molecular_topics):
        allow = load_molecular_topics(args.molecular_topics)
        dom = dominant_topics(chunk_topics)
        mol = molecular_codes(codes, dom, allow)
        rows = tier2_positives(codes, mol, names, allow)
        # Reclaim BOTH db and barbieri-cite: the Phase 3 `cite` pass relabels some db rows to
        # barbieri-cite, so a select re-run must own both provenances to rebuild the tier-2 set
        # without duplicating an already-upgraded paper (re-run `cite` afterwards to re-promote).
        merged = merge_gold(read_gold_set(args.gold_set), rows, {"db", "barbieri-cite"})
        write_gold_set(args.gold_set, merged)
        log.info("Phase 2: %d molecular codes (allowlist of %d topics) → %d tier-2 gold+ papers; "
                 "merged into %s (%d total rows)",
                 len(mol), len(allow), len(rows), args.gold_set, len(merged))
    else:
        log.warning("no molecular allowlist at %s — skipping Phase 2 tier-2 positives "
                    "(curate it from gold/topic_ranking.csv first)", args.molecular_topics)


def cmd_cite(args):
    """Phase 3: tier-1 = Barbieri's/Major's own foundational texts (code 0) + any tier-2 positive
    they also cite. Reads ``gold_set.csv`` + ``biological_codes.csv`` + the seminal-text PDFs, and
    the embedded corpus map (read-only DB) for the code-0 papers. No GPU, no spend."""
    rows = read_gold_set(args.gold_set)
    if not rows:
        log.warning("no rows in %s — run `select` (Phase 2) first (code-0 positives still added)",
                    args.gold_set)
    paths = seminal_pdfs(args.pdf_dir, args.seed_csv)
    log.info("reading Barbieri/Major citations from %d seminal PDFs", len(paths))
    cited = cited_signatures(paths)
    names = paper_names_by_path(args.csv)
    log.info("%d distinct cited (surname, year) signatures; %d corpus papers signed",
             len(cited), len(names))
    upgraded, n = tier1_upgrade(rows, names, cited)

    import db
    conn = db.connect()
    try:
        _dv, _poles, codes = db.fetch_vectors(conn, run=args.run)
    finally:
        conn.close()
    c0 = code0_positives(codes)
    final = merge_gold(upgraded, c0, {"code0"})
    write_gold_set(args.gold_set, final)
    n_t1 = sum(r["tier"] == "1" for r in final)
    log.info("Phase 3: %d code-0 foundational tier-1 positives + %d citation upgrades; "
             "%d tier-1 rows total in %s (%d rows)", len(c0), n, n_t1, args.gold_set, len(final))


def cmd_implicit(args):
    """Phase 4 (soft negatives): label every embedded non-molecular, non-molecular-code paper as
    ``implicit`` gold−. Reads persisted ``chunk_topics`` + the corpus map (read-only DB), the
    curated allowlist, and the existing ``gold_set.csv``. No GPU, no spend."""
    import db
    conn = db.connect()
    try:
        _dv, _poles, codes = db.fetch_vectors(conn, run=args.run)
        chunk_topics = db.fetch_chunk_topics(conn, run=args.run, method=args.method)
    finally:
        conn.close()
    allow = load_molecular_topics(args.molecular_topics)
    labels = load_topic_labels(args.molecular_topics)
    dom = dominant_topics(chunk_topics)
    mol = molecular_codes(codes, dom, allow)
    existing = read_gold_set(args.gold_set)
    # Exclude only non-implicit labels: a re-run rebuilds the implicit set from scratch (it owns
    # `implicit`), but must never override a positive or a hard negative for the same paper.
    keep_labelled = {r["pdf_path"] for r in existing if r.get("source") != "implicit"}
    rows = implicit_negatives(codes, dom, allow, mol, labels, exclude_pids=keep_labelled)
    merged = merge_gold(existing, rows, {"implicit"})
    write_gold_set(args.gold_set, merged)
    log.info("Phase 4: %d molecular codes; %d soft (implicit) gold− papers; "
             "merged into %s (%d total rows)", len(mol), len(rows), args.gold_set, len(merged))


def collect_grounded(checkpoint_path):
    """All grounded exclusion items recorded in the checkpoint (across every scanned chunk).

    The checkpoint is the spend-safety system-of-record (§7.6): each line is one scanned
    ``(pdf_path, chunk_idx)`` whose ``exclusions`` are already grounded (gate applied before the
    line was written). Returns the flat list of item dicts for :func:`exclusion_rows`."""
    out = []
    if not os.path.exists(checkpoint_path):
        return out
    with open(checkpoint_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                log.warning("skipping malformed exclusion checkpoint line")
                continue
            for it in rec.get("exclusions") or []:
                if isinstance(it, dict) and (it.get("candidate") or "").strip():
                    out.append({"candidate": it.get("candidate", ""),
                                "quote": it.get("quote", ""),
                                "reasoning": it.get("reasoning", ""),
                                "source_pdf": rec.get("pdf_path", "")})
    return out


def write_exclusions_audit(path, grounded, code_names, codes):
    """Audit CSV of **every** grounded exclusion (mapped or conceptual), for human review."""
    rows = []
    for it in grounded:
        cn = match_candidate_to_code(it["candidate"], code_names)
        mapped = cn if (cn is not None and cn in set(codes.values())) else ""
        rows.append([it.get("source_pdf", ""), it["candidate"], mapped,
                     (code_names.get(cn, "") if cn is not None else ""),
                     it.get("quote", ""), it.get("reasoning", "")])
    _write_csv(path, ["source_pdf", "candidate", "mapped_code", "mapped_name",
                      "quote", "reasoning"], rows)


def cmd_exclude(args):
    """Phase 4 (hard negatives): paid DeepSeek pass over the seminal-text prose surfacing every
    candidate Barbieri/Major argue is **not** an organic code. Checkpoint-first (``exclusions.jsonl``,
    never deleted, §7.6) and resumable per ``(pdf_path, chunk_idx)``. Grounded exclusions mapping to
    an embedded corpus code become ``hard`` gold− rows (over that code's papers); the rest are kept
    in ``gold/exclusions_audit.csv`` as conceptual negatives. Merges into ``gold_set.csv``."""
    from judge_pilot import load_env
    load_env()  # OPENROUTER_API_KEY for the paid DeepSeek pass; no-op without .env

    paths = seminal_pdfs(args.pdf_dir, args.seed_csv)
    log.info("scanning %d seminal PDFs for exclusions (checkpoint %s)", len(paths), args.checkpoint)
    done = load_exclude_done(args.checkpoint)
    meter = cj.UsageMeter()
    complete = cj.openrouter_graded_factory(reasoning_effort=args.reasoning, meter=meter)

    scanned = 0
    for path in paths:
        text = pt.extract_text(path)
        windows = chunk_prose(text)
        for idx, passage in enumerate(windows):
            if (path, idx) in done:
                continue
            try:
                raw = complete(EXCLUDE_SYSTEM_PROMPT, build_exclude_prompt(passage),
                               response_format={"type": "json_object"})
            except Exception as e:  # per-window isolation; checkpoint preserves progress
                log.warning("exclusion call failed for %s chunk %d: %s", path, idx, e)
                continue
            grounded = ground_exclusions(parse_exclusions(raw), passage)
            cj.append_checkpoint(args.checkpoint,
                                 {"pdf_path": path, "chunk_idx": idx, "exclusions": grounded})
            scanned += 1
            if grounded:
                log.info("%s chunk %d → %d grounded exclusion(s)", os.path.basename(path), idx,
                         len(grounded))

    log.info("scanned %d new windows; DeepSeek spend ≈ $%.4f (%d calls)",
             scanned, meter.cost(), meter.calls)

    import db
    conn = db.connect()
    try:
        _dv, _poles, codes = db.fetch_vectors(conn, run=args.run)
    finally:
        conn.close()
    code_names = load_code_names(args.csv)
    grounded = collect_grounded(args.checkpoint)
    write_exclusions_audit(args.exclusions_audit, grounded, code_names, codes)

    rows, conceptual = exclusion_rows(grounded, code_names, codes)
    existing = read_gold_set(args.gold_set)
    # A hard negative must never contradict a positive for the same paper: drop (and log) any
    # exclusion row whose pdf_path is already a gold+ instance.
    positives = {r["pdf_path"] for r in existing if r.get("polarity") == "pos"}
    kept = [r for r in rows if r["pdf_path"] not in positives]
    dropped = len(rows) - len(kept)
    if dropped:
        log.warning("dropped %d hard-negative row(s) conflicting with existing gold+ positives",
                    dropped)
    # Corpus-mapped hard negatives are advisory: Barbieri's exclusions overwhelmingly target
    # concepts with NO corpus paper, so the rare candidate↔code-name collisions are where a
    # context-misread exclusion (e.g. "prokaryotes could not evolve a splicing code" reads as
    # excluding the splicing code) silently enters the authoritative set. So the default run is
    # **audit-only** — review gold/exclusions_audit.csv and pass --merge-mapped to ratify.
    if not args.merge_mapped:
        log.info("Phase 4 (hard): %d grounded exclusions → %d candidate corpus-mapped row(s) "
                 "(%d conceptual). AUDIT-ONLY: reviewed in %s; gold_set.csv unchanged. "
                 "Re-run with --merge-mapped to ratify the mapped rows.",
                 len(grounded), len(kept), len(conceptual), args.exclusions_audit)
        for r in kept:
            log.info("  candidate hard− code %s %s | %s", r["code_number"], r["pdf_path"],
                     r["evidence"][:100])
        return
    merged = merge_gold(existing, kept, {"exclusion"})
    write_gold_set(args.gold_set, merged)
    log.info("Phase 4 (hard): %d grounded exclusions → %d corpus-mapped hard gold− rows "
             "(%d conceptual, audit-only); merged into %s (%d total rows)",
             len(grounded), len(kept), len(conceptual), args.gold_set, len(merged))


def cmd_materialise(args):
    """Phase 5: materialise the git-tracked ``gold_set.csv`` into the ``gold_labels`` DB table
    for report JOINs. Run-/judge-agnostic ground truth (no embedding run, no judge). Idempotent
    upsert — safe to re-run after any phase refreshes the CSV. **Backup the DB first** (§7.8)."""
    rows = read_gold_set(args.gold_set)
    if not rows:
        log.warning("no rows in %s — run the select/cite/implicit phases first", args.gold_set)
        return
    import db
    conn = db.connect()
    try:
        n = db.store_gold(conn, rows)
    finally:
        conn.close()
    log.info("Phase 5: materialised %d gold labels from %s into gold_labels", n, args.gold_set)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sel = sub.add_parser("select", help="Phase 1: embedding-driven molecular ranking")
    sel.add_argument("--run", default="baseline")
    sel.add_argument("--method", default="chunk")
    sel.add_argument("--csv", default=CSV_PATH)
    sel.add_argument("--molecular-topics", default=MOLECULAR_TOPICS_CSV,
                     help="curated molecular allowlist (Phase 2 tier-2 positives)")
    sel.add_argument("--gold-set", default=GOLD_SET_PATH,
                     help="git-tracked gold reference set to merge tier-2 positives into")
    sel.set_defaults(func=cmd_select)

    cite = sub.add_parser("cite", help="Phase 3: tier-1 = code-0 foundational texts + Barbieri-cited")
    cite.add_argument("--run", default="baseline", help="embedding run for the code-0 corpus map")
    cite.add_argument("--csv", default=CSV_PATH)
    cite.add_argument("--gold-set", default=GOLD_SET_PATH)
    cite.add_argument("--pdf-dir", default=CODE_BIOLOGY_PDFS,
                      help="directory of the seminal Barbieri/Major PDFs")
    cite.add_argument("--seed-csv", default=SEED_CSV,
                      help="code-0 seed manifest (its Source File column names the seminal PDFs)")
    cite.set_defaults(func=cmd_cite)

    imp = sub.add_parser("implicit", help="Phase 4: soft (implicit) gold− negatives")
    imp.add_argument("--run", default="baseline")
    imp.add_argument("--method", default="chunk")
    imp.add_argument("--molecular-topics", default=MOLECULAR_TOPICS_CSV)
    imp.add_argument("--gold-set", default=GOLD_SET_PATH)
    imp.set_defaults(func=cmd_implicit)

    exc = sub.add_parser("exclude", help="Phase 4: hard gold− = Barbieri/Major explicit exclusions "
                                         "(paid DeepSeek; checkpoint-first)")
    exc.add_argument("--run", default="baseline", help="embedding run for the corpus map")
    exc.add_argument("--csv", default=CSV_PATH)
    exc.add_argument("--gold-set", default=GOLD_SET_PATH)
    exc.add_argument("--pdf-dir", default=CODE_BIOLOGY_PDFS,
                     help="directory of the seminal Barbieri/Major PDFs")
    exc.add_argument("--seed-csv", default=SEED_CSV)
    exc.add_argument("--checkpoint", default=EXCLUDE_CHECKPOINT,
                     help="resumable per-(pdf,chunk) JSONL spend-safety record (never deleted)")
    exc.add_argument("--exclusions-audit", default=EXCLUSIONS_AUDIT,
                     help="CSV of every grounded exclusion (mapped + conceptual) for review")
    exc.add_argument("--reasoning", default="high", help="DeepSeek reasoning effort")
    exc.add_argument("--merge-mapped", action="store_true",
                     help="ratify the corpus-mapped hard negatives into gold_set.csv "
                          "(default: audit-only — review gold/exclusions_audit.csv first)")
    exc.set_defaults(func=cmd_exclude)

    mat = sub.add_parser("materialise", help="Phase 5: load gold_set.csv into the gold_labels "
                                             "DB table (run-/judge-agnostic ground truth)")
    mat.add_argument("--gold-set", default=GOLD_SET_PATH)
    mat.set_defaults(func=cmd_materialise)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
