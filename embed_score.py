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

# Offline-recompute lever defaults, set from the 20-paper smoke test (see recompute):
#   * whitening off — k>=1 stripped real criterion signal on that tiny sample;
#   * partial (not full) orthogonalization — full removal nuked the genetic-code
#     paper's legitimate two_worlds along with its arbitrariness topicality halo.
# Both are exposed as flags so they can be re-tuned once the full corpus is embedded.
DEFAULT_WHITEN_K = 0
DEFAULT_SHARED_STRENGTH = 0.5


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


def corpus_mean(vectors):
    """Arithmetic mean of a set of document vectors — the centring origin μ."""
    return np.asarray(vectors, dtype=np.float64).mean(axis=0)


def center(vec, mu):
    """Subtract the corpus mean μ (broadcasts over rows for a 2-D ``vec``)."""
    return np.asarray(vec, dtype=np.float64) - np.asarray(mu, dtype=np.float64)


def whiten_basis(vecs, k):
    """Top-``k`` principal axes (right singular vectors) of the (centred) row set.

    Returned as a ``(k, dim)`` orthonormal matrix so the same basis can be applied to
    documents and poles consistently. ``k`` is capped at the available rank; ``k <= 0``
    yields an empty ``(0, dim)`` basis (the whitening identity)."""
    X = np.asarray(vecs, dtype=np.float64)
    k = min(k, *X.shape)
    if k <= 0:
        return np.zeros((0, X.shape[1]))
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:k]                       # (k, dim) orthonormal principal axes


def whiten(vecs, k):
    """Remove the top-``k`` principal components ('all-but-the-top') from a set of
    (already centred) vectors.

    ``k == 0`` is the identity. Decoder-only, last-token-pooled embeddings sit in a
    narrow cone: a few dominant shared directions carry most of the variance, so every
    in-register cosine is high and the criterion axes are compressed. Projecting out the
    top-``k`` PCs strips that common-component anisotropy and restores dynamic range.
    NB the 20-paper smoke test found ``k>=1`` *hurt* on that tiny sample (the top PC
    still carried criterion signal there), so the default in the recompute path is
    ``k=0``; raise it only once the corpus is large enough to trust the PC estimate."""
    X = np.asarray(vecs, dtype=np.float64)
    B = whiten_basis(X, k)
    if B.shape[0] == 0:
        return X
    return X - (X @ B.T) @ B            # project the top-k subspace out


def shared_direction(axes):
    """Dominant direction shared across the per-criterion difference axes.

    The three criteria's poles all sit in the same 'code biology' register, so their
    axes ``{a_c}`` share a common topicality component — the source of the run-1 halo
    where an on-topic paper (code 428) topped *every* criterion, including ones its
    verdict marked ``not_met``. This returns that common direction as the first right
    singular vector (first PC, **uncentred** — we want the shared direction, not the
    spread) of the stacked unit axes, sign-fixed to point with the bulk of the axes."""
    A = _l2(np.asarray(axes, dtype=np.float64))     # unit rows
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    s = _l2(Vt[0])
    if float((A @ s).sum()) < 0:                     # deterministic sign
        s = -s
    return s


def orthogonalize(axis, shared, strength=1.0):
    """Partial ``axis`` against the ``shared`` register direction.

    ``a⊥ = normalize(a − strength·(a·ŝ) ŝ)`` — removes the shared-topicality component so
    the projection measures only the criterion-specific (pos-vs-its-own-neg) contrast,
    not how on-topic the text is. A zero residual (axis parallel to ``shared``) stays
    zero via the norm floor.

    ``strength`` ∈ [0, 1] scales how much of the shared component is stripped. The
    20-paper smoke test showed that **full** removal (``strength=1.0``) over-corrects
    when the criteria axes are strongly co-aligned (≈0.74 there): it discards the
    *legitimate* signal of an on-topic-and-genuinely-positive paper (the genetic-code
    paper lost its true ``two_worlds`` rank along with its ``arbitrariness`` halo).
    ``strength=0`` is the identity; the recompute default is partial — see
    :data:`DEFAULT_SHARED_STRENGTH`."""
    a = np.asarray(axis, dtype=np.float64)
    s = _l2(shared)
    return _l2(a - strength * float(a @ s) * s)


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


# --- offline recompute: compose all four levers from persisted vectors ------
#
# This is the final scoring path. After the single structural GPU embed, every contrast
# number is recomputed here from the stored doc/pole vectors with **no GPU** — so the
# levers can be re-tuned and the report regenerated freely. The composition is
#   μ      = mean of the per-paper representative (full) vectors      (centring origin)
#   B      = top-k principal axes of the centred reps                 (whitening basis)
#   a_c    = normalize(p̂_c − n̂_c) on the centred poles               (difference axis)
#   ŝ      = shared register direction across {a_c}
#   a_c⊥   = normalize(a_c − strength·(a_c·ŝ) ŝ)                       (decongest halo)
#   e_c(d) = a_c⊥ · normalize( whiten(d − μ, B) )                     (axis projection)
# chunk windows are max-pooled (strongest evidence); full/abstract are a single window.

def _rep_vec(methods):
    """A paper's representative vector for the corpus geometry (μ + whitening basis):
    its ``full`` embedding, else the first vector of whatever method it has."""
    seq = methods.get("full") or next(iter(methods.values()))
    return np.asarray(seq[0], dtype=np.float64)


def build_axes(poles, mu, strength=DEFAULT_SHARED_STRENGTH):
    """Centred, shared-decongested per-criterion axes from the pole vectors.

    Returns ``(axes, shared, within)`` where ``axes[c]`` is the unit difference axis on
    the μ-centred poles, partialled against the shared register direction by ``strength``;
    ``shared`` is that direction; and ``within[c]`` is the **centred** pole width (cosine
    of the criterion's own centred pos/neg) — the dynamic-range bound, rendered in the
    report."""
    mu = np.asarray(mu, dtype=np.float64)
    cpoles = {c: {"pos": center(poles[c]["pos"], mu), "neg": center(poles[c]["neg"], mu)}
              for c in poles}
    raw = {c: axis_vector(cpoles[c]) for c in cpoles}
    shared = shared_direction(list(raw.values()))
    axes = {c: orthogonalize(raw[c], shared, strength) for c in cpoles}
    within = {c: float(_l2(cpoles[c]["pos"]) @ _l2(cpoles[c]["neg"])) for c in cpoles}
    return axes, shared, within


def recompute(doc_vecs, poles, k=DEFAULT_WHITEN_K, strength=DEFAULT_SHARED_STRENGTH):
    """Rescore every persisted paper offline with all four space-level levers.

    ``doc_vecs = {paper_id: {method: [vec, ...]}}`` (full/abstract carry one vector;
    chunk carries every window). ``poles = {criterion: {'pos': vec, 'neg': vec}}``.
    Returns ``(scores, within)`` with ``scores[paper_id][method][criterion] = e`` and the
    centred pole widths ``within[criterion]``. No GPU, no I/O — pure vector math."""
    reps = np.array([_rep_vec(doc_vecs[p]) for p in doc_vecs], dtype=np.float64)
    mu = corpus_mean(reps)
    basis = whiten_basis(center(reps, mu), k)
    axes, _, within = build_axes(poles, mu, strength)

    def project(vec):
        d = center(vec, mu)
        if basis.shape[0]:
            d = d - (d @ basis.T) @ basis
        return _l2(d)

    scores = {}
    for pid, methods in doc_vecs.items():
        scores[pid] = {
            method: {c: aggregate_chunks([float(project(v) @ axes[c]) for v in vecs])
                     for c in axes}
            for method, vecs in methods.items()
        }
    return scores, within


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
