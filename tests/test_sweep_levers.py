"""Offline tests for the lever parameter sweep (no GPU, no DB).

The sweep rescues an open TODO: the lever defaults (whiten-k, shared-strength) were
calibrated on a 20-paper smoke test and never revisited at corpus scale. ``sweep`` is a
pure function over the persisted-vector dicts so it can be exercised with a fake corpus.
"""

import numpy as np

import embed_score as es
from embed_independent import spearman
from criteria_judge import verdict_ordinal
import sweep_levers as sl


def _fake_corpus(seed=0):
    """A tiny synthetic corpus: poles, doc vectors (full+chunk), and verdicts with rank
    variation so Spearman is defined on every criterion."""
    rng = np.random.default_rng(seed)
    dim = 6
    poles = {c: {"pos": rng.standard_normal(dim), "neg": rng.standard_normal(dim)}
             for c in es.CRITERIA}
    doc_vecs = {}
    verdicts = {}
    labels = ["met", "not_met", "unclear", "not_met", "met", "not_met"]
    for i in range(6):
        pid = f"pdfs/p{i}.pdf"
        doc_vecs[pid] = {"full": [rng.standard_normal(dim)],
                         "chunk": [rng.standard_normal(dim), rng.standard_normal(dim)]}
        # rotate the label list per criterion so each has variation but not identical orders
        verdicts[pid] = {c: labels[(i + j) % len(labels)]
                         for j, c in enumerate(es.CRITERIA)}
    return doc_vecs, poles, verdicts


def test_grid_shape_and_cell_keys():
    doc_vecs, poles, verdicts = _fake_corpus()
    ks, strengths = [0, 1, 2], [0.0, 0.5, 1.0]
    results = sl.sweep(doc_vecs, poles, verdicts, ks, strengths)
    assert len(results) == len(ks) * len(strengths)
    cell = results[0]
    assert {"k", "strength", "rho", "within"} <= set(cell)
    # rho is method -> criterion -> (float | None)
    for method in ("full", "chunk"):
        assert set(cell["rho"][method]) == set(es.CRITERIA)


def test_rho_matches_embed_independent_spearman():
    """Wiring proof: a sweep cell's ρ equals an independent recompute+spearman at the
    same levers — so the sweep is on the identical scoring path as the live recompute."""
    doc_vecs, poles, verdicts = _fake_corpus()
    results = sl.sweep(doc_vecs, poles, verdicts, [0], [0.5])
    rho = results[0]["rho"]

    scores, _ = es.recompute(doc_vecs, poles, k=0, strength=0.5)
    for method in ("full", "chunk"):
        for c in es.CRITERIA:
            e_vals, o_vals = [], []
            for pid, v in verdicts.items():
                e = scores[pid][method][c]
                e_vals.append(e)
                o_vals.append(verdict_ordinal(v[c]))
            expected = spearman(e_vals, o_vals)
            assert rho[method][c] == expected


def test_within_constant_across_grid():
    """Pole width is centred-pole geometry only — independent of k and strength — so it
    must be identical in every cell (reported once)."""
    doc_vecs, poles, verdicts = _fake_corpus()
    results = sl.sweep(doc_vecs, poles, verdicts, [0, 2, 4], [0.0, 1.0])
    first = results[0]["within"]
    for cell in results[1:]:
        for c in es.CRITERIA:
            assert cell["within"][c] == first[c]


def test_rho_none_when_no_verdict_variation():
    doc_vecs, poles, _ = _fake_corpus()
    flat = {pid: {c: "not_met" for c in es.CRITERIA} for pid in doc_vecs}
    results = sl.sweep(doc_vecs, poles, flat, [0], [0.5])
    rho = results[0]["rho"]
    for method in rho:
        for c in es.CRITERIA:
            assert rho[method][c] is None


def test_sweep_needs_no_database():
    """The sweep is read-only/offline: monkeypatching db.connect to explode must not
    affect it (it never touches the database)."""
    import db
    orig = db.connect
    db.connect = lambda *a, **k: (_ for _ in ()).throw(AssertionError("sweep hit the DB"))
    try:
        doc_vecs, poles, verdicts = _fake_corpus()
        results = sl.sweep(doc_vecs, poles, verdicts, [0, 1], [0.5])
        assert len(results) == 2
    finally:
        db.connect = orig


def test_best_per_criterion_picks_argmax_cell():
    doc_vecs, poles, verdicts = _fake_corpus()
    results = sl.sweep(doc_vecs, poles, verdicts, [0, 1, 2], [0.0, 0.5, 1.0])
    best = sl.best_per_criterion(results, method="chunk")
    for c in es.CRITERIA:
        k, s, rho = best[c]["k"], best[c]["strength"], best[c]["rho"]
        # the reported best equals the true max over all cells for that method/criterion
        allvals = [cell["rho"]["chunk"][c] for cell in results
                   if cell["rho"]["chunk"][c] is not None]
        assert rho == max(allvals)
        # and it is an actual grid point
        assert any(cell["k"] == k and cell["strength"] == s for cell in results)
