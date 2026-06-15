"""Assign each paper chunk to its nearest scientometric topic (offline, no GPU).

The 24 topic centroids (Paredes & Prinz 2025; embedded in §3's ``topic_centroids``)
are the *topicality halo* the §4 levers already strip — not a new discrimination
axis. This module assigns every persisted chunk vector to its nearest topic **in the
identical centred space the scoring uses**, by reusing :func:`embed_score.build_scorer`
(μ-centred, top-``k`` whitened, unit-normalised — exactly how controls are projected).
A paper's dominant topic is the max-pool of its chunks' assignments.

The result feeds the per-topic ρ breakdown (stratified diagnosis) and, later, gold-set
stratification — both serving the binding constraint of *label quality*, not the
embedding. No vectors are recomputed and no GPU is touched; this reads the persisted
``doc_vectors`` + ``topic_centroids`` and writes the ``chunk_topics`` table.

Run (offline):
    python3 assign_topics.py            # baseline run, chunk method
    python3 assign_topics.py --run gte-qwen2 --method full
"""

import argparse
import logging

import numpy as np

import embed_score as es

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("assign_topics")


def nearest_topic(proj_vec, proj_centroids):
    """Topic whose (projected) centroid has the highest cosine with ``proj_vec``.

    Both ``proj_vec`` and the centroids are already unit vectors in the centred space,
    so the dot product is the cosine. Returns ``(topic_id, similarity)``."""
    best_id, best_sim = None, float("-inf")
    for tid, cvec in proj_centroids.items():
        sim = float(np.asarray(proj_vec) @ np.asarray(cvec))
        if sim > best_sim:
            best_id, best_sim = tid, sim
    return best_id, best_sim


def paper_dominant_topic(chunk_assignments):
    """Max-pool a paper's per-chunk assignments into a topic affinity + dominant topic.

    ``chunk_assignments`` is ``[(chunk_idx, topic_id, sim), ...]`` (each chunk's argmax
    vote). The paper's affinity for a topic is the **max** similarity over the chunks
    that voted it (so one strongly-on-topic window dominates, matching the max-pooled
    chunk scoring in §4); the dominant topic is the argmax of that affinity. Returns
    ``(dominant_topic, affinity)``; ``(None, {})`` for a paper with no chunks."""
    affinity = {}
    for _idx, tid, sim in chunk_assignments:
        if tid not in affinity or sim > affinity[tid]:
            affinity[tid] = sim
    if not affinity:
        return None, {}
    dominant = max(affinity, key=affinity.get)
    return dominant, affinity


def build_assignments(doc_vecs, poles, centroids, method="chunk",
                      k=es.DEFAULT_WHITEN_K, strength=es.DEFAULT_SHARED_STRENGTH):
    """Assign every paper's ``method`` chunks to nearest topic in the centred space.

    ``doc_vecs = {pid: {method: [vec, ...]}}``, ``poles = {criterion: {pos, neg}}``,
    ``centroids = {topic_id: vec}``. The projection geometry (μ, whitening basis) is
    built from the **papers** via :func:`embed_score.build_scorer` and applied to both
    chunks and centroids, so assignment happens in the same μ-centred / whitened / unit
    space the criterion scores live in. Falls back to ``full`` (then any method) when a
    paper lacks the requested ``method``. Returns
    ``{pid: {'chunks': [(idx, topic_id, sim)], 'dominant': topic_id, 'affinity': {...}}}``."""
    project, _, _ = es.build_scorer(doc_vecs, poles, k, strength)
    proj_centroids = {tid: project(np.asarray(v, dtype=np.float64))
                      for tid, v in centroids.items()}
    out = {}
    for pid, methods in doc_vecs.items():
        vecs = methods.get(method) or methods.get("full") or next(iter(methods.values()))
        chunks = []
        for idx, v in enumerate(vecs):
            tid, sim = nearest_topic(project(np.asarray(v, dtype=np.float64)),
                                     proj_centroids)
            chunks.append((idx, tid, sim))
        dominant, affinity = paper_dominant_topic(chunks)
        out[pid] = {"chunks": chunks, "dominant": dominant, "affinity": affinity}
    return out


def main():
    import db
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="baseline")
    ap.add_argument("--method", default="chunk", choices=es._REP_METHODS
                    if hasattr(es, "_REP_METHODS") else ["full", "abstract", "chunk"])
    ap.add_argument("--whiten-k", type=int, default=es.DEFAULT_WHITEN_K)
    ap.add_argument("--shared-strength", type=float, default=es.DEFAULT_SHARED_STRENGTH)
    args = ap.parse_args()

    conn = db.connect()
    try:
        doc_vecs, poles, codes = db.fetch_vectors(conn, run=args.run)
        doc_vecs, dropped = es.drop_self_references(doc_vecs)
        if dropped:
            log.info("dropped %d in-corpus self-reference(s)", len(dropped))
        centroids = db.fetch_topic_centroids(conn, run=args.run)
        if not centroids:
            log.error("no topic_centroids for run=%s — embed them first", args.run)
            return
        log.info("assigning %d papers x %s chunks against %d centroids (run=%s)",
                 len(doc_vecs), args.method, len(centroids), args.run)
        cent_vecs = {tid: c["vec"] for tid, c in centroids.items()}
        assignments = build_assignments(doc_vecs, poles, cent_vecs, method=args.method,
                                        k=args.whiten_k, strength=args.shared_strength)
        n = db.store_chunk_topics(conn, assignments, codes,
                                  method=args.method, run=args.run)
        log.info("wrote %d chunk_topics rows (run=%s, method=%s)", n, args.run, args.method)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
