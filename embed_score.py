"""Corpus-contrastive bipolar embedding scoring for the Code Biology criteria.

This is an **independent** axis reported alongside the LLM verdicts; it never
overrides, gates, or band-merges with them (plan decision 0). For each criterion a
paper gets

    e = cos(paper, POS_prototype) - cos(paper, NEG_prototype)

where each prototype is the mean-pooled, L2-normalised embedding of the salted,
on-axis pos/neg passages in ``prototypes.json``. Positive/negative passages are
embedded as harrier *queries* (with the per-criterion instruction); paper text is
embedded as a *document* (no instruction), per the harrier model card. Because the
shared "code biology" register sits in both poles, it cancels in the subtraction and
``e`` isolates the criterion-specific (positive-vs-its-own-opposite) axis.

The vector math here is unit-tested offline with a fake encoder; the heavy harrier
encoding runs on the GPU host via ``run_harrier_embed.py``.
"""

import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

CRITERIA = ["two_worlds", "adaptors", "arbitrariness"]


def load_prototypes(path="prototypes.json"):
    """Load the per-criterion pos/neg passage lists + instructions."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {c: data[c] for c in CRITERIA}


def _l2(v):
    """L2-normalise along the last axis (rows for a 2-D array)."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-12, None)


def format_query(task, text):
    """harrier query format: an instruction line then the passage."""
    return f"Instruct: {task}\nQuery: {text}"


def pole_vector(passages, encode):
    """Embed ``passages``, L2-normalise each, mean-pool, L2-normalise the centroid."""
    vecs = _l2(encode(passages))
    return _l2(vecs.mean(axis=0))


def build_poles(prototypes, encode_query):
    """Return ``{criterion: {'pos': vec, 'neg': vec}}``.

    Each pos/neg passage is embedded as an instructed query (the per-criterion
    ``instruct_pos`` / ``instruct_neg`` prepended via :func:`format_query`).
    """
    poles = {}
    for crit, spec in prototypes.items():
        pos_q = [format_query(spec["instruct_pos"], p) for p in spec["pos"]]
        neg_q = [format_query(spec["instruct_neg"], p) for p in spec["neg"]]
        poles[crit] = {
            "pos": pole_vector(pos_q, encode_query),
            "neg": pole_vector(neg_q, encode_query),
        }
    return poles


def contrastive_score(doc_vec, pole):
    """``e = cos(doc, pos) - cos(doc, neg)`` (inputs L2-normalised, so cos == dot)."""
    d = _l2(doc_vec)
    return float(d @ _l2(pole["pos"]) - d @ _l2(pole["neg"]))


def score_papers(doc_vecs, poles):
    """``{paper_id: {criterion: e}}`` for ``doc_vecs = {paper_id: vector}``."""
    return {pid: {c: contrastive_score(v, poles[c]) for c in poles}
            for pid, v in doc_vecs.items()}


def axis_vector(pole):
    """Unit difference axis pointing from the negative toward the positive pole.

    ``a = normalize(p̂ − n̂)`` on the L2-normalised poles. Degenerate poles (p̂ == n̂)
    collapse to a zero vector (the 1e-12 norm floor keeps it zero rather than blowing
    up)."""
    return _l2(_l2(pole["pos"]) - _l2(pole["neg"]))


def axis_score(doc_vec, pole):
    """Project the (L2-normalised) document onto the pos↔neg **difference axis**.

    ``axis_score = normalize(p̂ − n̂) · normalize(doc)`` — the cosine between the
    document and the criterion's polar axis, in ``[-1, 1]``.

    This is the Task-3 replacement for the double-cosine :func:`contrastive_score`.
    They share a direction but differ by the *pole width*: ``contrastive_score`` is
    ``doc·(p̂ − n̂)``, which scales with ``‖p̂ − n̂‖`` and so **compresses** a criterion
    whose poles overlap (narrow width). Dividing by that width (i.e. projecting onto the
    *unit* axis) puts every criterion on a common scale, so a narrow-pole criterion is
    no longer penalised relative to a wide-pole one. Degenerate poles score ``0.0``."""
    return float(_l2(doc_vec) @ axis_vector(pole))


# --- chunking-method helpers (full / abstract / 8K-overlap) ----------------
#
# The independent embedding axis is computed three ways per paper, each fed to the
# embedder as its own document, to test which granularity best tracks the verdict:
#   * full      — the whole (budget-capped) paper as one document
#   * abstract  — the abstract section only (or preamble fallback)
#   * chunk     — 8192-token windows at 50% overlap, scored per chunk then max-pooled
# These are reported side-by-side; none overrides the verdict (plan decision 0).

def token_windows(ids, size=8192, overlap=4096):
    """Split a token-id sequence into ``size``-long windows stepping by ``size-overlap``.

    The whole sequence is returned as a single window when it already fits. The walk
    stops once a window reaches the end, so no tiny trailing duplicate is produced and
    every token is covered by at least one window (no truncation loss)."""
    ids = list(ids)
    if len(ids) <= size:
        return [ids]
    stride = max(1, size - overlap)
    out = []
    i = 0
    while i < len(ids):
        out.append(ids[i:i + size])
        if i + size >= len(ids):
            break
        i += stride
    return out


def aggregate_chunks(scores):
    """Pool per-chunk contrastive scores into one ``e`` — **max** (strongest evidence
    anywhere in the paper). Empty input (no chunks) scores 0.0 (neutral)."""
    return float(max(scores)) if len(scores) else 0.0


def pole_separation(poles):
    """Diagnostic with two complementary views:

    * ``pos`` / ``neg`` — pairwise cosine *between criteria* for the pos prototypes and
      for the neg prototypes. Near-identical negatives (cosine ≳0.9) mean the poles
      muddied into a generic 'mere chemistry' axis — the acceptable exception is
      adaptors↔arbitrariness, which are theoretically coupled.
    * ``within`` — *within-criterion* cosine between that criterion's own pos and neg
      prototype. This is the **pole width**: it bounds the dynamic range of ``e``. A
      value near +1 means the two poles overlap and ``e`` is compressed (magnitudes are
      not calibrated, only ranks are trustworthy); values toward 0 or −1 mean the poles
      are well separated and magnitudes carry signal. Widen the poles (more polar,
      register-matched passages in ``prototypes.json``) until this drops."""
    crits = list(poles)

    def matrix(key):
        return {f"{a}~{b}": float(_l2(poles[a][key]) @ _l2(poles[b][key]))
                for i, a in enumerate(crits) for b in crits[i + 1:]}

    within = {c: float(_l2(poles[c]["pos"]) @ _l2(poles[c]["neg"])) for c in crits}
    return {"pos": matrix("pos"), "neg": matrix("neg"), "within": within}
