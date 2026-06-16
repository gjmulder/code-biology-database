"""Offline tests for assign_topics.py (no GPU/DB; small numeric vectors).

Each paper chunk is assigned to its nearest scientometric topic centroid in the
**same centred space the §4 levers use** — reusing ``embed_score.build_scorer`` so
chunks and centroids are μ-centred / whitened / unit-normalised identically (the
exact treatment controls already get). These tests pin the pure assignment maths;
the geometry itself is covered by ``test_embed_score.py``.
"""

import numpy as np

import assign_topics as at


def test_nearest_topic_picks_highest_cosine():
    centroids = {5: np.array([1.0, 0.0, 0.0]),
                 7: np.array([0.0, 1.0, 0.0])}
    tid, sim = at.nearest_topic(np.array([0.9, 0.1, 0.0]), centroids)
    assert tid == 5
    assert sim == 0.9


def test_paper_dominant_topic_maxpools_assigned_sims():
    # per-chunk argmax votes; the paper's affinity is the per-topic max over chunks.
    chunks = [(0, 5, 0.30), (1, 7, 0.90), (2, 5, 0.40)]
    dominant, affinity = at.paper_dominant_topic(chunks)
    assert affinity == {5: 0.40, 7: 0.90}   # max-pool, not sum
    assert dominant == 7


def test_paper_dominant_topic_empty():
    assert at.paper_dominant_topic([]) == (None, {})


def _poles(dim):
    rng = np.random.default_rng(0)
    return {c: {"pos": rng.normal(size=dim), "neg": rng.normal(size=dim)}
            for c in ("two_worlds", "adaptors", "arbitrariness")}


def test_build_assignments_shape_and_uses_chunk_method():
    dim = 4
    rng = np.random.default_rng(1)
    # three papers; each carries a single `full` vec and two `chunk` windows.
    doc_vecs = {
        f"pdfs/{i}.pdf": {
            "full": [rng.normal(size=dim)],
            "chunk": [rng.normal(size=dim), rng.normal(size=dim)],
        }
        for i in range(3)
    }
    centroids = {0: rng.normal(size=dim), 1: rng.normal(size=dim),
                 2: rng.normal(size=dim)}
    out = at.build_assignments(doc_vecs, _poles(dim), centroids, method="chunk")

    assert set(out) == set(doc_vecs)
    for pid, rec in out.items():
        # one assignment per chunk window (chunk method has 2, not the 1 full vec)
        assert len(rec["chunks"]) == 2
        for idx, tid, sim in rec["chunks"]:
            assert tid in centroids
            assert isinstance(sim, float)
        assert rec["dominant"] in centroids
        assert set(rec["affinity"]).issubset(centroids)


def test_build_assignments_is_centred_not_raw_cosine():
    # A chunk equal to a centroid's RAW vector need not be assigned to it once both are
    # μ-centred: assignment must reflect the centred space, so we assert the function
    # agrees with a hand-rolled centred nearest-centroid (same project closure).
    import embed_score as es
    dim = 5
    rng = np.random.default_rng(2)
    doc_vecs = {f"pdfs/{i}.pdf": {"chunk": [rng.normal(size=dim)]} for i in range(4)}
    centroids = {10: rng.normal(size=dim), 11: rng.normal(size=dim)}
    poles = _poles(dim)

    out = at.build_assignments(doc_vecs, poles, centroids, method="chunk")

    project, _, _ = es.build_scorer(doc_vecs, poles)
    pc = {t: project(v) for t, v in centroids.items()}
    for pid, methods in doc_vecs.items():
        pv = project(methods["chunk"][0])
        want = max(pc, key=lambda t: float(pv @ pc[t]))
        assert out[pid]["chunks"][0][1] == want
