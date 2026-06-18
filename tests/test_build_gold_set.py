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


# --- Phase 3: tier-1 upgrade (Barbieri-cited) ------------------------------

def test_paper_signature_parses_first_author_surname_and_year():
    # APA-style: surname before the comma, year in parens after the authors
    assert bgs.paper_signature("Farina, A. (2019). Acoustic codes. Biosystems 183.") == ("farina", "2019")
    # multi-author author-date with quoted title
    assert bgs.paper_signature('Vedula, P. and A. Kashina (2018). "The actin code." JCS.') == ("vedula", "2018")
    # hyphenated initials, the surname stops at the comma
    assert bgs.paper_signature("Gabius, H.-J. (2000). The sugar code. Naturwiss. 87.") == ("gabius", "2000")
    # no parseable year → None
    assert bgs.paper_signature("Some prose with no author and no year") is None
    assert bgs.paper_signature("") is None


def test_parse_reference_signatures_splits_entries_and_ignores_continuations():
    ref = (
        "References\n"
        "Barash, Y., Calarco, J. A., et al. (2010). Deciphering the splicing code.\n"
        "Nature, 465, 53-59.\n"                       # continuation: must NOT start a new entry
        "Gabius, H.-J. (2000). The sugar code.\n"
        "Naturwissenschaften, 87, 108-121.\n"
    )
    sigs = bgs.parse_reference_signatures(ref)
    assert sigs == {("barash", "2010"), ("gabius", "2000")}


def test_parse_reference_signatures_springer_style():
    # Springer/Nature house style: "Surname IN, Surname IN … (YYYY)" — no comma after the first
    # surname, initials have no dots. The year is in parens but far into the (joined) entry.
    ref = (
        "References\n"
        "Adl SM, Simpson ABG, Farmer MA et al (2005) The new higher-level classification.\n"
        "J Eukaryot Microbiol 52:399-451\n"          # continuation: must NOT start a new entry
        "Agalioti T, Chen G, Thanos D (2002) Deciphering the histone acetylation code.\n"
    )
    assert bgs.parse_reference_signatures(ref) == {("adl", "2005"), ("agalioti", "2002")}


def test_seminal_pdfs_resolves_seed_plus_extra_existing_only(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "Intro - Barbieri (2014).pdf").write_bytes(b"%PDF")
    (pdf_dir / "Organic.pdf").write_bytes(b"%PDF")
    # the seed manifest names two files but only one exists on disk
    seed = tmp_path / "seed.csv"
    seed.write_text("Source File,Paper Name,URL\n"
                    "Intro - Barbieri (2014).pdf,x,http://x\n"
                    "Missing - Barbieri (2099).pdf,y,http://y\n", encoding="utf-8")
    got = bgs.seminal_pdfs(pdf_dir=str(pdf_dir), seed_csv=str(seed), extra=("Organic.pdf", "Organic.pdf"))
    assert got == [str(pdf_dir / "Intro - Barbieri (2014).pdf"), str(pdf_dir / "Organic.pdf")]


def test_paper_names_by_path_keyed_by_output_path(tmp_path):
    csv = tmp_path / "codes.csv"
    csv.write_text(
        "Code Number,Code Name,Paper Name,URL\n"
        "30,Sugar code,Gabius (2000). The sugar code.,https://doi.org/10.1007/sugar\n",
        encoding="utf-8")
    names = bgs.paper_names_by_path(str(csv))
    assert names == {"pdfs/10.1007_sugar.pdf": "Gabius (2000). The sugar code."}


def test_tier1_upgrade_promotes_only_cited_db_positives():
    rows = [
        {"code_number": 30, "pdf_path": "pdfs/sugar.pdf", "polarity": "pos", "tier": "2",
         "source": "db", "criterion": "all", "evidence": "Sugar code"},
        {"code_number": 12, "pdf_path": "pdfs/uncited.pdf", "polarity": "pos", "tier": "2",
         "source": "db", "criterion": "all", "evidence": "Genetic code"},
        {"code_number": 99, "pdf_path": "pdfs/neg.pdf", "polarity": "neg", "tier": "hard",
         "source": "exclusion", "criterion": "all", "evidence": "barbieri"},
    ]
    names = {"pdfs/sugar.pdf": "Gabius, H.-J. (2000). The sugar code.",
             "pdfs/uncited.pdf": "Nobody, Z. (1700). Obscure."}
    cited = {("gabius", "2000")}
    upgraded, n = bgs.tier1_upgrade(rows, names, cited)
    assert n == 1
    sug = next(r for r in upgraded if r["pdf_path"] == "pdfs/sugar.pdf")
    assert sug["tier"] == "1" and sug["source"] == "barbieri-cite"
    assert "gabius 2000" in sug["evidence"] and "Sugar code" in sug["evidence"]
    # the uncited positive and the hard negative are untouched
    assert next(r for r in upgraded if r["pdf_path"] == "pdfs/uncited.pdf")["tier"] == "2"
    assert next(r for r in upgraded if r["pdf_path"] == "pdfs/neg.pdf")["source"] == "exclusion"


def test_tier1_upgrade_is_idempotent():
    rows = [{"code_number": 30, "pdf_path": "pdfs/sugar.pdf", "polarity": "pos", "tier": "2",
             "source": "db", "criterion": "all", "evidence": "Sugar code"}]
    names = {"pdfs/sugar.pdf": "Gabius, H.-J. (2000). The sugar code."}
    cited = {("gabius", "2000")}
    once, _ = bgs.tier1_upgrade(rows, names, cited)
    twice, n2 = bgs.tier1_upgrade(once, names, cited)
    assert twice == once and n2 == 0          # already barbieri-cite, not re-upgraded


def test_code0_positives_only_code_zero_as_tier1():
    codes = {"pdfs/intro.pdf": 0, "pdfs/whatis.pdf": 0, "pdfs/sugar.pdf": 30}
    rows = bgs.code0_positives(codes)
    assert [r["pdf_path"] for r in rows] == ["pdfs/intro.pdf", "pdfs/whatis.pdf"]  # sorted, code 30 out
    assert all(r["code_number"] == 0 and r["polarity"] == "pos" and r["tier"] == "1"
               and r["source"] == "code0" and r["criterion"] == "all" for r in rows)


def test_load_topic_labels_all_rows(tmp_path):
    csv = tmp_path / "molecular_topics.csv"
    csv.write_text(
        "topic_id,label,molecular,molecularity,basis\n"
        "3,Genetic Code,yes,0.41,anchor\n"
        "19,Neural Circuits,no,-0.20,non-molecular\n"
        "bad,Garbage,no,0,junk\n",            # non-int topic_id is skipped
        encoding="utf-8")
    labels = bgs.load_topic_labels(str(csv))
    assert labels == {3: "Genetic Code", 19: "Neural Circuits"}   # both yes and no, junk dropped


def test_molecular_member_pids_union_of_molecular_codes():
    codes = {"g1.pdf": 12, "g2.pdf": 12, "sug.pdf": 30, "lang.pdf": 200}
    mol = {12: (3, 2), 30: (16, 1)}            # genetic + sugar are molecular; language is not
    assert bgs.molecular_member_pids(codes, mol) == {"g1.pdf", "g2.pdf", "sug.pdf"}


# --- Phase 4: soft negatives (implicit) ------------------------------------

def test_implicit_negatives_non_molecular_topic_and_not_molecular_member():
    codes = {"g1.pdf": 12, "lang.pdf": 200, "neur.pdf": 202}
    dom = {"g1.pdf": 3, "lang.pdf": 4, "neur.pdf": 19}   # g1 molecular topic, others not
    allow = {3: "Genetic Code", 16: "Molecular Codes"}
    mol = {12: (3, 1)}                                    # only the genetic code is molecular
    labels = {3: "Genetic Code", 4: "Language Code", 19: "Neural Circuits"}
    rows = bgs.implicit_negatives(codes, dom, allow, mol, labels)
    assert [r["pdf_path"] for r in rows] == ["lang.pdf", "neur.pdf"]   # sorted; g1 (molecular) out
    assert all(r["polarity"] == "neg" and r["tier"] == "soft" and r["source"] == "implicit"
               and r["criterion"] == "all" for r in rows)
    lang = next(r for r in rows if r["pdf_path"] == "lang.pdf")
    assert lang["code_number"] == 200 and "Language Code" in lang["evidence"]


def test_implicit_negatives_excludes_code0_members_and_labelled():
    codes = {"seed.pdf": 0, "lang.pdf": 200, "mem.pdf": 201, "done.pdf": 202}
    dom = {"seed.pdf": 4, "lang.pdf": 4, "mem.pdf": 4, "done.pdf": 4}   # all non-molecular topic 4
    allow = {3: "Genetic Code"}
    mol = {201: (3, 1)}                       # code 201 IS molecular → mem.pdf is a member (excluded)
    labels = {4: "Language Code"}
    rows = bgs.implicit_negatives(codes, dom, allow, mol, labels,
                                  exclude_pids={"done.pdf"})
    # seed.pdf (code 0), mem.pdf (molecular member), done.pdf (already labelled) all excluded
    assert [r["pdf_path"] for r in rows] == ["lang.pdf"]


def test_implicit_negatives_skips_papers_without_dominant_topic():
    codes = {"a.pdf": 200, "b.pdf": 201}
    dom = {"a.pdf": 4}                         # b.pdf has no dominant topic → skipped
    rows = bgs.implicit_negatives(codes, dom, {3: "Genetic"}, {}, {4: "Language Code"})
    assert [r["pdf_path"] for r in rows] == ["a.pdf"]


# --- Phase 4: hard negatives (exclude) -------------------------------------

def test_chunk_prose_windows_with_overlap():
    text = "abcdefghij" * 3            # 30 chars
    chunks = bgs.chunk_prose(text, max_chars=12, overlap=4)
    assert chunks[0] == text[:12]
    assert chunks[1] == text[8:20]     # advanced by step = max_chars - overlap = 8
    assert "".join(c[:8] for c in chunks[:-1]) + chunks[-1][:] is not None  # no gaps
    assert chunks[-1].endswith(text[-1])          # last window reaches the end
    # short text → single window; empty/whitespace → none
    assert bgs.chunk_prose("short", max_chars=100) == ["short"]
    assert bgs.chunk_prose("   ", max_chars=100) == []


def test_parse_exclusions_tolerant():
    raw = ('{"exclusions": ['
           '{"candidate": "metabolism", "quote": "metabolism is not a code", "reasoning": "no adaptor"},'
           '{"candidate": "", "quote": "x"},'                       # no candidate → dropped
           '{"candidate": "immune system"}]}')                     # no quote → empty quote kept
    items = bgs.parse_exclusions(raw)
    assert [it["candidate"] for it in items] == ["metabolism", "immune system"]
    assert items[0]["quote"] == "metabolism is not a code" and items[0]["reasoning"] == "no adaptor"
    assert items[1]["quote"] == ""
    # no JSON / empty list → []
    assert bgs.parse_exclusions("the model refused") == []
    assert bgs.parse_exclusions('{"exclusions": []}') == []


def test_ground_exclusions_keeps_only_verbatim_quotes():
    passage = ("In Barbieri's view, metabolism is a continuous chemical process and is "
               "therefore not an organic code at all, lacking any adaptor.")
    items = [
        {"candidate": "metabolism", "quote": "metabolism is a continuous chemical process",
         "reasoning": "no adaptor"},                                   # verbatim → kept
        {"candidate": "translation", "quote": "translation uses a wholly invented mapping",
         "reasoning": "fabricated"},                                   # not in passage → dropped
    ]
    grounded = bgs.ground_exclusions(items, passage)
    assert [it["candidate"] for it in grounded] == ["metabolism"]


def test_match_candidate_to_code_conservative_content_tokens():
    code_names = {7: "Chemical codes", 30: "Sugar code", 50: "Immune code"}
    assert bgs.match_candidate_to_code("the immune code", code_names) == 50
    assert bgs.match_candidate_to_code("immune system", code_names) is None   # 'system' != 'immune'-only key
    assert bgs.match_candidate_to_code("metabolism", code_names) is None
    assert bgs.match_candidate_to_code("", code_names) is None
    # 'code' alone is a stopword → never a match on the empty content set
    assert bgs.match_candidate_to_code("a code", code_names) is None


def test_exclusion_rows_only_mapped_db_codes_over_their_embedded_papers():
    grounded = [
        {"candidate": "the immune code", "quote": "immune is not a code", "reasoning": "r"},
        {"candidate": "metabolism", "quote": "metabolism is not a code", "reasoning": "r"},
    ]
    code_names = {50: "Immune code", 30: "Sugar code"}
    codes = {"imm1.pdf": 50, "imm2.pdf": 50, "sug.pdf": 30}    # embedded corpus map
    rows, conceptual = bgs.exclusion_rows(grounded, code_names, codes)
    # immune maps to code 50 → both its embedded papers become hard negatives; metabolism is conceptual
    assert [r["pdf_path"] for r in rows] == ["imm1.pdf", "imm2.pdf"]
    assert all(r["polarity"] == "neg" and r["tier"] == "hard" and r["source"] == "exclusion"
               and r["code_number"] == 50 and r["criterion"] == "all" for r in rows)
    assert "immune" in rows[0]["evidence"].lower()
    assert [c["candidate"] for c in conceptual] == ["metabolism"]


def test_exclude_checkpoint_resume_keys_by_pdf_and_chunk(tmp_path):
    ckpt = tmp_path / "exclusions.jsonl"
    import json
    with open(ckpt, "w", encoding="utf-8") as f:
        f.write(json.dumps({"pdf_path": "a.pdf", "chunk_idx": 0, "exclusions": []}) + "\n")
        f.write(json.dumps({"pdf_path": "a.pdf", "chunk_idx": 1, "exclusions": []}) + "\n")
        f.write("not json\n")                       # malformed line tolerated
    done = bgs.load_exclude_done(str(ckpt))
    assert done == {("a.pdf", 0), ("a.pdf", 1)}
    assert bgs.load_exclude_done(str(tmp_path / "absent.jsonl")) == set()


def test_select_merge_reclaims_barbieri_cite_rows():
    # after a cite pass some db rows became barbieri-cite; re-running select must reclaim BOTH
    # so the rebuilt tier-2 set never duplicates an upgraded paper.
    existing = [
        {"code_number": 30, "pdf_path": "pdfs/sugar.pdf", "polarity": "pos", "tier": "1",
         "source": "barbieri-cite", "criterion": "all", "evidence": "Sugar code | barbieri-cited"},
        {"code_number": 99, "pdf_path": "pdfs/neg.pdf", "polarity": "neg", "tier": "hard",
         "source": "exclusion", "criterion": "all", "evidence": "barbieri"},
    ]
    fresh = [{"code_number": 30, "pdf_path": "pdfs/sugar.pdf", "polarity": "pos", "tier": "2",
              "source": "db", "criterion": "all", "evidence": "Sugar code"}]
    merged = bgs.merge_gold(existing, fresh, {"db", "barbieri-cite"})
    assert [r["pdf_path"] for r in merged] == ["pdfs/neg.pdf", "pdfs/sugar.pdf"]
    assert sum(r["pdf_path"] == "pdfs/sugar.pdf" for r in merged) == 1   # no duplicate
