"""Offline tests for the contrastive embedding scoring math (fake encoder)."""

import numpy as np
import pytest

import embed_score as es
import pdf_text
from criteria_judge import (apply_coherence, combine_score, verdict_ordinal,
                            weighted_median)


# A fake encoder: deterministic unit-ish vectors keyed by substrings in the text,
# so we control geometry without loading a model.
def make_encoder(table, dim=8):
    def encode(texts):
        out = []
        for t in texts:
            v = np.zeros(dim)
            for key, vec in table.items():
                if key in t:
                    v = v + np.asarray(vec, dtype=float)
            out.append(v)
        return np.asarray(out, dtype=float)
    return encode


def test_pole_vector_is_unit_norm_and_mean_pooled():
    enc = make_encoder({"a": [1, 0, 0, 0, 0, 0, 0, 0],
                        "b": [0, 1, 0, 0, 0, 0, 0, 0]})
    v = es.pole_vector(["a", "b"], enc)
    assert np.isclose(np.linalg.norm(v), 1.0)
    # mean of two orthogonal unit vectors, renormalised → equal components
    assert np.isclose(v[0], v[1])


def test_contrastive_score_positive_when_doc_aligns_with_pos():
    pole = {"pos": [1, 0, 0], "neg": [0, 1, 0]}
    assert es.contrastive_score([1, 0, 0], pole) > 0.9
    assert es.contrastive_score([0, 1, 0], pole) < -0.9
    assert abs(es.contrastive_score([0, 0, 1], pole)) < 1e-9


def test_build_poles_uses_instruction_format():
    seen = {}

    def enc(texts):
        seen.setdefault("texts", []).extend(texts)
        return np.ones((len(texts), 4))

    proto = {"two_worlds": {"instruct_pos": "POS-TASK", "instruct_neg": "NEG-TASK",
                            "pos": ["p1"], "neg": ["n1"]}}
    es.build_poles(proto, enc)
    joined = "\n".join(seen["texts"])
    assert "Instruct: POS-TASK" in joined and "Query: p1" in joined
    assert "Instruct: NEG-TASK" in joined and "Query: n1" in joined


def test_pole_separation_reports_pairwise_cosines():
    poles = {
        "two_worlds": {"pos": [1, 0, 0], "neg": [1, 0, 0]},
        "adaptors": {"pos": [0, 1, 0], "neg": [1, 0, 0]},
        "arbitrariness": {"pos": [0, 0, 1], "neg": [0, 1, 0]},
    }
    sep = es.pole_separation(poles)
    assert np.isclose(sep["neg"]["two_worlds~adaptors"], 1.0)   # identical neg poles
    assert np.isclose(sep["pos"]["two_worlds~adaptors"], 0.0)   # orthogonal pos poles


def test_pole_separation_reports_within_criterion_pole_width():
    # The within-criterion pos<->neg cosine is the actual "pole width": cos near 1
    # means narrow/overlapping poles and a compressed dynamic range for e; cos near
    # -1 (or 0) means wide, well-separated poles where magnitudes are meaningful.
    poles = {
        "two_worlds": {"pos": [1, 0, 0], "neg": [1, 0, 0]},     # degenerate: zero width
        "adaptors": {"pos": [0, 1, 0], "neg": [1, 0, 0]},       # orthogonal: wide
        "arbitrariness": {"pos": [1, 0, 0], "neg": [-1, 0, 0]}, # antipodal: widest
    }
    sep = es.pole_separation(poles)
    assert np.isclose(sep["within"]["two_worlds"], 1.0)
    assert np.isclose(sep["within"]["adaptors"], 0.0)
    assert np.isclose(sep["within"]["arbitrariness"], -1.0)


def test_real_prototypes_load_with_three_criteria():
    proto = es.load_prototypes("prototypes.json")
    assert set(proto) == set(es.CRITERIA)
    for c in es.CRITERIA:
        assert proto[c]["pos"] and proto[c]["neg"]
        assert proto[c]["instruct_pos"] and proto[c]["instruct_neg"]


# --- axis-projection contrast (Task 3) ------------------------------------

def test_axis_score_antipodal_poles_match_pos_axis():
    # pos↔neg span a single axis; doc on +pos → +1, on neg → -1, orthogonal → 0
    pole = {"pos": [1, 0, 0], "neg": [-1, 0, 0]}
    assert np.isclose(es.axis_score([1, 0, 0], pole), 1.0)
    assert np.isclose(es.axis_score([-1, 0, 0], pole), -1.0)
    assert np.isclose(es.axis_score([0, 1, 0], pole), 0.0)


def test_axis_score_orthogonal_poles_project_onto_difference_axis():
    # axis = normalize(p̂ − n̂) = normalize([1,-1,0]); doc==pos projects to 1/√2
    pole = {"pos": [1, 0, 0], "neg": [0, 1, 0]}
    assert np.isclose(es.axis_score([1, 0, 0], pole), 1 / np.sqrt(2))
    assert np.isclose(es.axis_score([0, 1, 0], pole), -1 / np.sqrt(2))
    assert np.isclose(es.axis_score([1, -1, 0], pole), 1.0)  # doc == the axis itself


def test_axis_score_monotone_from_neg_to_pos():
    # rotating the doc from the neg pole toward the pos pole strictly raises the score
    pole = {"pos": [1, 0, 0], "neg": [-1, 0, 0]}
    angles = np.linspace(np.pi, 0.0, 9)   # π (==neg) → 0 (==pos)
    scores = [es.axis_score([np.cos(a), np.sin(a), 0], pole) for a in angles]
    assert all(b > a for a, b in zip(scores, scores[1:]))


def test_axis_score_divides_out_pole_width():
    # axis_score == contrastive_score / ‖p̂ − n̂‖ — narrow (overlapping) poles are
    # amplified back onto a common projection scale instead of being compressed.
    narrow = {"pos": [1, 0, 0], "neg": es._l2([1, 0.2, 0]).tolist()}
    doc = [0.3, 0.9, 0.1]
    width = float(np.linalg.norm(es._l2(narrow["pos"]) - es._l2(narrow["neg"])))
    assert np.isclose(es.axis_score(doc, narrow),
                      es.contrastive_score(doc, narrow) / width)


def test_axis_score_degenerate_poles_are_zero():
    # p̂ == n̂ → zero axis → 0.0 (no blow-up from the 1e-12 norm floor)
    pole = {"pos": [1, 0, 0], "neg": [1, 0, 0]}
    assert es.axis_score([0.5, 0.5, 0.7], pole) == 0.0


# --- centre / whiten the space (Task 4) -----------------------------------

def test_corpus_mean_is_arithmetic_mean():
    mu = es.corpus_mean([[1, 0], [3, 0], [2, 6]])
    assert np.allclose(mu, [2, 2])


def test_center_removes_common_offset_1d_and_2d():
    assert np.allclose(es.center([3, 4], [2, 3]), [1, 1])
    centred = es.center([[1, 2], [3, 4]], [2, 3])
    assert np.allclose(centred, [[-1, -1], [1, 1]])
    assert np.allclose(centred.mean(axis=0), 0)   # offset gone


def test_whiten_k0_is_identity():
    X = np.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert np.allclose(es.whiten(X, 0), X)


def test_whiten_removes_dominant_shared_direction():
    # a large shared-variance direction (axis 0) plus tiny per-sample signal (axis 1)
    rng = np.random.default_rng(0)
    n = 60
    X = np.stack([rng.normal(0, 10, n),      # dominant anisotropic direction
                  rng.normal(0, 0.1, n),     # the faint real signal
                  np.zeros(n)], axis=1)
    X = X - X.mean(axis=0)
    W = es.whiten(X, k=1)
    assert np.var(W[:, 0]) < 1e-6 * np.var(X[:, 0])   # dominant direction killed
    assert np.var(W[:, 1]) > 0.5 * np.var(X[:, 1])    # faint signal preserved


def test_whiten_k_caps_at_rank():
    # asking to remove more PCs than exist must not raise; result is well-defined
    X = np.array([[1.0, 0, 0], [0, 1, 0]])
    assert es.whiten(X, k=99).shape == X.shape


# --- orthogonalize the criteria (Task 5) ----------------------------------

def test_shared_direction_recovers_common_axis():
    # axes that all point mostly along one shared direction → first PC recovers it
    s_true = es._l2([1, 1, 0, 0])
    rng = np.random.default_rng(1)
    axes = [3 * s_true + rng.normal(0, 0.05, 4) for _ in range(5)]
    s = es.shared_direction(axes)
    assert abs(float(np.dot(s, s_true))) > 0.99
    assert np.isclose(np.linalg.norm(s), 1.0)


def test_shared_direction_sign_points_with_the_bulk():
    # deterministic sign: aligned with the mean of the axes, not its negation
    axes = [[1, 0, 0], [0.9, 0.1, 0], [0.8, 0, 0.1]]
    s = es.shared_direction(axes)
    assert float(np.mean([np.dot(es._l2(a), s) for a in axes])) > 0


def test_orthogonalize_removes_shared_component_and_unit_norms():
    shared = es._l2([1, 0, 0])
    o = es.orthogonalize([2.0, 1.0, 0.0], shared)   # x = shared part, y = unique part
    assert np.isclose(float(np.dot(o, shared)), 0.0)   # shared part partialled out
    assert np.isclose(np.linalg.norm(o), 1.0)


def test_orthogonalized_axes_are_decongested_from_shared_register():
    # three axes sharing a big common "register" direction + a small unique part;
    # after orthogonalising against the shared direction none retains it (kills the
    # cross-criterion topicality halo).
    shared = es._l2([1, 1, 1, 0, 0])
    uniques = [es._l2(v) for v in ([0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 1])]
    axes = [es._l2(2 * shared + 0.2 * u) for u in uniques]
    s = es.shared_direction(axes)
    for a in axes:
        assert abs(float(np.dot(es.orthogonalize(a, s), s))) < 1e-9


# --- chunking methods (full / abstract / 8K-overlap) ----------------------

def test_token_windows_short_sequence_is_single_window():
    assert es.token_windows(list(range(5)), size=8, overlap=4) == [list(range(5))]
    assert es.token_windows(list(range(8)), size=8, overlap=4) == [list(range(8))]


def test_token_windows_50pct_overlap_steps_by_half():
    # size 8, overlap 4 → stride 4; 20 tokens → windows at 0,4,8,12 (last reaches end)
    w = es.token_windows(list(range(20)), size=8, overlap=4)
    assert [x[0] for x in w] == [0, 4, 8, 12]
    assert w[-1][-1] == 19           # last window covers the tail
    assert all(len(x) <= 8 for x in w)


def test_token_windows_no_truncation_loss():
    # every token appears in at least one window
    w = es.token_windows(list(range(101)), size=16, overlap=8)
    covered = set().union(*[set(x) for x in w])
    assert covered == set(range(101))


def test_aggregate_chunks_is_max_strongest_evidence():
    assert es.aggregate_chunks([-0.5, 0.1, 0.8, -0.2]) == 0.8
    assert es.aggregate_chunks([0.3]) == 0.3


def test_aggregate_chunks_empty_is_zero():
    assert es.aggregate_chunks([]) == 0.0


# --- abstract extraction --------------------------------------------------

def test_extract_abstract_picks_the_abstract_section():
    text = ("Title line\nAuthors\n"
            "Abstract\nThis paper argues the genetic code is arbitrary.\n"
            "Introduction\nLong body that should not be returned here.\n")
    a = pdf_text.extract_abstract(text)
    assert "arbitrary" in a
    assert "Long body" not in a


def test_extract_abstract_falls_back_to_preamble_when_no_heading():
    text = "Some opening paragraph with the gist of the work and no headings at all."
    a = pdf_text.extract_abstract(text, max_chars=40)
    assert a.startswith("Some opening")
    assert len(a) <= 40


# --- criteria_judge scoring helpers ---------------------------------------

def test_verdict_ordinal_monotone():
    assert verdict_ordinal("not_met") < verdict_ordinal("unclear") < verdict_ordinal("met")


@pytest.mark.parametrize("e", [-1.0, -0.3, 0.0, 0.5, 1.0])
def test_combine_score_band_monotone(e):
    # met always outranks unclear always outranks not_met, regardless of e
    assert combine_score("met", -1.0) > combine_score("unclear", 1.0)
    assert combine_score("unclear", -1.0) > combine_score("not_met", 1.0)
    assert 0.0 <= combine_score("not_met", e) <= 1.0


def test_combine_score_embedding_orders_within_band():
    assert combine_score("met", 1.0) > combine_score("met", -1.0)


def test_weighted_median_equal_weights_is_plain_median():
    assert weighted_median([1, 2, 3], [1, 1, 1]) == 2.0
    assert weighted_median([1, 2, 3, 4], [1, 1, 1, 1]) == 2.5  # even → avg of middles


def test_weighted_median_single_value():
    assert weighted_median([0.7], [5.0]) == 0.7


def test_weighted_median_weight_pulls_toward_heavy_value():
    # heavy weight on 3 drags the median up off the unweighted 2
    assert weighted_median([1, 2, 3], [1, 1, 10]) == 3.0


def test_apply_coherence_marks_downstream_vacuous():
    crit = {"two_worlds": {"verdict": "not_met"},
            "adaptors": {"verdict": "not_met"},
            "arbitrariness": {"verdict": "not_met"}}
    annotated, flags = apply_coherence(crit)
    assert annotated["adaptors"]["vacuous"] and annotated["arbitrariness"]["vacuous"]
    assert flags == []


def test_apply_coherence_flags_incoherent_pattern():
    crit = {"two_worlds": {"verdict": "not_met"},
            "adaptors": {"verdict": "met"},
            "arbitrariness": {"verdict": "not_met"}}
    _, flags = apply_coherence(crit)
    assert any("adaptors=met" in f for f in flags)


def test_apply_coherence_no_gate_when_two_worlds_met():
    crit = {"two_worlds": {"verdict": "met"},
            "adaptors": {"verdict": "met"},
            "arbitrariness": {"verdict": "not_met"}}
    annotated, flags = apply_coherence(crit)
    assert "vacuous" not in annotated["adaptors"]
    assert flags == []
