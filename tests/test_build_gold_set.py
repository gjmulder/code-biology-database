"""Offline tests for build_gold_set.py Phase 1 — embedding-driven molecular selection.

Gold positives are defined by molecular *code membership*; which codes/topics count as
molecular is decided **data-drivenly** by proximity to a genetic-code centroid anchor
(Barbieri's canonical molecular exemplar), not a hand allowlist. These tests pin the pure
selection maths in the same μ-centred §4 space (``embed_score.build_scorer``); no GPU/DB.
"""

import numpy as np

import build_gold_set as bgs
import embed_score as es


def _poles(dim):
    rng = np.random.default_rng(0)
    return {c: {"pos": rng.normal(size=dim), "neg": rng.normal(size=dim)}
            for c in ("two_worlds", "adaptors", "arbitrariness")}


def test_load_code_names_first_name_per_number(tmp_path):
    csv = tmp_path / "codes.csv"
    csv.write_text(
        "Code Number,Code Name,Paper Name,URL\n"
        "12,Genetic code,Paper A,http://x\n"
        "12,Genetic code,Paper B,http://y\n"
        "30,Sugar code,Paper C,http://z\n",
        encoding="utf-8")
    names = bgs.load_code_names(str(csv))
    assert names == {12: "Genetic code", 30: "Sugar code"}


def test_anchor_pids_matches_genetic_variants():
    codes = {"g1.pdf": 12, "g2.pdf": 13, "mito.pdf": 14, "sugar.pdf": 30}
    names = {12: "Genetic code", 13: "Genetic Code – C (expanded)",
             14: "Mitochondrial genetic code", 30: "Sugar code"}
    assert bgs.anchor_pids(codes, names) == {"g1.pdf", "g2.pdf", "mito.pdf"}


def test_anchor_excludes_nonmolecular():
    codes = {"lang.pdf": 200, "g.pdf": 12}
    names = {200: "Language code", 12: "Genetic code"}
    assert bgs.anchor_pids(codes, names) == {"g.pdf"}


def _toy_corpus(dim=4):
    # genetic papers point ~ +e0; non-molecular point ~ -e0; a molecular-like and a
    # non-molecular held-out paper sit on either side of the +e0 anchor direction.
    e0 = np.zeros(dim); e0[0] = 1.0
    e1 = np.zeros(dim); e1[1] = 1.0
    doc_vecs = {
        "g1.pdf": {"chunk": [e0 + 0.05 * e1, e0 - 0.05 * e1]},
        "g2.pdf": {"chunk": [e0 + 0.1 * e1]},
        "lang.pdf": {"chunk": [-e0 + 0.05 * e1]},
        "dance.pdf": {"chunk": [-e0 - 0.1 * e1]},
        "molish.pdf": {"chunk": [0.8 * e0 + 0.2 * e1]},   # leans molecular
        "neur.pdf": {"chunk": [-0.7 * e0 + 0.3 * e1]},    # leans non-molecular
    }
    codes = {"g1.pdf": 12, "g2.pdf": 12, "lang.pdf": 200,
             "dance.pdf": 201, "molish.pdf": 30, "neur.pdf": 202}
    names = {12: "Genetic code", 200: "Language code", 201: "Dance code",
             30: "Sugar code", 202: "Neural code"}
    return doc_vecs, codes, names


def test_molecular_anchor_orders_molecular_above_nonmolecular():
    doc_vecs, codes, names = _toy_corpus()
    poles = _poles(4)
    project, anchor = bgs.molecular_anchor(doc_vecs, poles, bgs.anchor_pids(codes, names))
    assert np.isclose(np.linalg.norm(anchor), 1.0)
    mol = bgs.paper_molecularness(project, anchor, doc_vecs["molish.pdf"]["chunk"])
    non = bgs.paper_molecularness(project, anchor, doc_vecs["neur.pdf"]["chunk"])
    assert mol > non


def test_paper_molecularness_is_maxpool_over_chunks():
    doc_vecs, codes, names = _toy_corpus()
    poles = _poles(4)
    project, anchor = bgs.molecular_anchor(doc_vecs, poles, bgs.anchor_pids(codes, names))
    chunks = doc_vecs["g1.pdf"]["chunk"]
    per = [float(project(np.asarray(v, float)) @ anchor) for v in chunks]
    assert bgs.paper_molecularness(project, anchor, chunks) == max(per)


def test_rank_codes_genetic_and_sugar_above_language_and_dance():
    doc_vecs, codes, names = _toy_corpus()
    poles = _poles(4)
    project, anchor = bgs.molecular_anchor(doc_vecs, poles, bgs.anchor_pids(codes, names))
    rows = bgs.rank_codes(doc_vecs, codes, names, project, anchor)
    order = [cn for cn, *_ in rows]
    assert order.index(12) < order.index(200)   # genetic above language
    assert order.index(30) < order.index(201)   # sugar above dance
    # each row: (code_number, code_name, n_papers, mean_mol, max_mol)
    gen = next(r for r in rows if r[0] == 12)
    assert gen[1] == "Genetic code" and gen[2] == 2


def test_rank_topics_orders_by_proximity_to_anchor():
    doc_vecs, codes, names = _toy_corpus()
    poles = _poles(4)
    project, anchor = bgs.molecular_anchor(doc_vecs, poles, bgs.anchor_pids(codes, names))
    e0 = np.zeros(4); e0[0] = 1.0
    centroids = {1: {"label": "Genetic", "vec": e0},
                 2: {"label": "Language", "vec": -e0}}
    rows = bgs.rank_topics(project, anchor, centroids)
    assert [tid for tid, *_ in rows] == [1, 2]
    assert rows[0][2] > rows[1][2]
