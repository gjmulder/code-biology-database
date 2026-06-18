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


def test_artificial_anchor_is_unit_and_separates_from_molecular():
    # molecular papers ~ +e0; the computer_code_positive control points ~ +e1 (a distinct
    # direction). The artificial anchor seeded from that control must rank an artificial-leaning
    # held-out paper ABOVE a genetic paper, while the molecular anchor ranks them the other way.
    dim = 4
    e0 = np.zeros(dim); e0[0] = 1.0
    e1 = np.zeros(dim); e1[1] = 1.0
    doc_vecs = {
        "g1.pdf": {"chunk": [e0 + 0.05 * e1, e0 - 0.05 * e1]},
        "g2.pdf": {"chunk": [e0]},
        "py.pdf": {"chunk": [e1 + 0.05 * e0]},   # artificial-leaning held-out paper
        "lang.pdf": {"chunk": [-e0]},
    }
    codes = {"g1.pdf": 12, "g2.pdf": 12, "py.pdf": 99, "lang.pdf": 200}
    names = {12: "Genetic code", 99: "Computer code", 200: "Language code"}
    poles = _poles(dim)
    project, mol = bgs.molecular_anchor(doc_vecs, poles, bgs.anchor_pids(codes, names))
    art = bgs.artificial_anchor(project, {"computer_code_positive": e1})
    assert np.isclose(np.linalg.norm(art), 1.0)
    # py is more artificial than the genetic paper; genetic is more molecular than py
    assert (bgs.paper_molecularness(project, art, doc_vecs["py.pdf"]["chunk"])
            > bgs.paper_molecularness(project, art, doc_vecs["g1.pdf"]["chunk"]))
    assert (bgs.paper_molecularness(project, mol, doc_vecs["g1.pdf"]["chunk"])
            > bgs.paper_molecularness(project, mol, doc_vecs["py.pdf"]["chunk"]))


def test_artificial_anchor_requires_a_present_seed():
    doc_vecs, codes, names = _toy_corpus()
    poles = _poles(4)
    project, _ = bgs.molecular_anchor(doc_vecs, poles, bgs.anchor_pids(codes, names))
    import pytest
    with pytest.raises(ValueError):
        bgs.artificial_anchor(project, {})                       # no control vectors at all
    with pytest.raises(ValueError):
        bgs.artificial_anchor(project, {"unrelated": np.ones(4)})  # seed key absent


def test_rank_codes_contrast_surfaces_artificial_lean():
    dim = 4
    e0 = np.zeros(dim); e0[0] = 1.0
    e1 = np.zeros(dim); e1[1] = 1.0
    doc_vecs = {
        "g1.pdf": {"chunk": [e0]},
        "py.pdf": {"chunk": [e1]},
        "lang.pdf": {"chunk": [-e0]},
    }
    codes = {"g1.pdf": 12, "py.pdf": 99, "lang.pdf": 200}
    names = {12: "Genetic code", 99: "Computer code", 200: "Language code"}
    poles = _poles(dim)
    project, mol = bgs.molecular_anchor(doc_vecs, poles, bgs.anchor_pids(codes, names))
    art = bgs.artificial_anchor(project, {"computer_code_positive": e1})
    rows = bgs.rank_codes_contrast(doc_vecs, codes, names, project, mol, art)
    # rows: (code_number, code_name, n_papers, mean_mol, mean_art, mean_diff); diff = art - mol,
    # sorted most-artificial-first → Computer code leads, Genetic code trails.
    assert rows[0][0] == 99 and rows[-1][0] == 12
    diffs = [r[5] for r in rows]
    assert diffs == sorted(diffs, reverse=True)


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


# --- Phase 2: tier-2 gold positives (curated topic allowlist) --------------

def test_load_molecular_topics_yes_only(tmp_path):
    csv = tmp_path / "molecular_topics.csv"
    csv.write_text(
        "topic_id,label,molecular,molecularity,basis\n"
        "3,Genetic Code,yes,0.41,anchor\n"
        "10,Synthetic Code,yes,0.38,user-confirmed-borderline\n"
        "8,Regulatory Code,no,0.08,excluded-borderline\n"
        "19,Neural Circuits,no,-0.20,non-molecular\n",
        encoding="utf-8")
    mol = bgs.load_molecular_topics(str(csv))
    assert set(mol) == {3, 10}
    assert mol[3] == "Genetic Code" and mol[10] == "Synthetic Code"


def test_dominant_topics_maxpools_per_paper():
    # p1: topic 3 wins (sim .9 > .4); p2: single chunk topic 10; p3: no chunks → dropped.
    chunk_topics = {
        "p1.pdf": [(0, 3, 0.9), (1, 8, 0.4)],
        "p2.pdf": [(0, 10, 0.7)],
        "p3.pdf": [],
    }
    dom = bgs.dominant_topics(chunk_topics)
    assert dom == {"p1.pdf": 3, "p2.pdf": 10}


def test_code_dominant_topic_is_modal_tie_to_lowest():
    dom = {"a": 3, "b": 3, "c": 10}          # code's papers: 3,3,10 → modal 3
    assert bgs.code_dominant_topic(["a", "b", "c"], dom) == 3
    dom2 = {"a": 10, "b": 3}                  # 1-1 tie → lowest topic_id
    assert bgs.code_dominant_topic(["a", "b"], dom2) == 3
    assert bgs.code_dominant_topic(["x"], {}) is None   # no dominant topic


def test_molecular_codes_filters_by_allowlist_and_excludes_code0():
    codes = {"g1.pdf": 12, "g2.pdf": 12,      # both topic 3 → molecular
             "lang.pdf": 200,                  # topic 19 → not molecular
             "seed.pdf": 0}                     # code 0 → always excluded
    dom = {"g1.pdf": 3, "g2.pdf": 3, "lang.pdf": 19, "seed.pdf": 3}
    allow = {3: "Genetic Code", 10: "Synthetic Code"}
    mol = bgs.molecular_codes(codes, dom, allow)
    assert set(mol) == {12}
    assert mol[12] == (3, 2)                   # (dominant_topic, n_embedded_papers)


def test_tier2_positives_one_row_per_embedded_paper():
    codes = {"g1.pdf": 12, "g2.pdf": 12, "lang.pdf": 200}
    mol = {12: (3, 2)}
    names = {12: "Genetic code", 200: "Language code"}
    labels = {3: "Genetic Code"}
    rows = bgs.tier2_positives(codes, mol, names, labels)
    assert [r["pdf_path"] for r in rows] == ["g1.pdf", "g2.pdf"]   # sorted, code 200 excluded
    assert all(r["polarity"] == "pos" and r["tier"] == "2" and r["source"] == "db"
               and r["criterion"] == "all" for r in rows)
    assert all(r["code_number"] == 12 for r in rows)
    assert "Genetic code" in rows[0]["evidence"] and "Genetic Code" in rows[0]["evidence"]


def test_merge_gold_replaces_only_named_sources(tmp_path):
    existing = [
        {"code_number": 12, "pdf_path": "g1.pdf", "polarity": "pos", "tier": "2",
         "source": "db", "criterion": "all", "evidence": "old"},
        {"code_number": 99, "pdf_path": "x.pdf", "polarity": "neg", "tier": "hard",
         "source": "exclusion", "criterion": "all", "evidence": "barbieri"},
    ]
    fresh = [{"code_number": 12, "pdf_path": "g2.pdf", "polarity": "pos", "tier": "2",
              "source": "db", "criterion": "all", "evidence": "new"}]
    merged = bgs.merge_gold(existing, fresh, {"db"})
    # the hard-negative (exclusion) row survives; the stale db row is replaced by the fresh one
    assert [r["pdf_path"] for r in merged] == ["x.pdf", "g2.pdf"]


def test_gold_set_csv_roundtrip(tmp_path):
    path = tmp_path / "gold_set.csv"
    rows = [{"code_number": 12, "pdf_path": "g1.pdf", "polarity": "pos", "tier": "2",
             "source": "db", "criterion": "all", "evidence": "Genetic code"}]
    bgs.write_gold_set(str(path), rows)
    back = bgs.read_gold_set(str(path))
    assert back == [{k: str(rows[0][k]) for k in bgs.GOLD_FIELDS}]
    assert bgs.read_gold_set(str(tmp_path / "absent.csv")) == []
