"""Offline tests for compare_verdicts' pure stats (no GPU/DB/network).

Covers the parts the plan marks unit-testable: the old-vs-new categorical distribution
per criterion, the new graded value-count distribution (does gradation materialise vs the
old 0.95-1.0 clustering?), and the pooled ρ(value, e) over the pilot papers reused for both
the categorical-ordinal axis and the graded axis. The DB/printing glue in main() is
exercised manually, not here.
"""

import compare_verdicts as cv
from criteria_judge import verdict_ordinal


# --- categorical distribution ---------------------------------------------

def test_categorical_distribution_counts_per_criterion_ignoring_unknown():
    recs = [
        ("two_worlds", "met"), ("two_worlds", "not_met"), ("two_worlds", "met"),
        ("adaptors", "unclear"), ("adaptors", None),
    ]
    dist = cv.categorical_distribution(recs)
    assert dist["two_worlds"] == {"met": 2, "unclear": 0, "not_met": 1}
    assert dist["adaptors"] == {"met": 0, "unclear": 1, "not_met": 0}


# --- graded value-count distribution --------------------------------------

def test_graded_distribution_buckets_the_five_levels():
    recs = [
        ("two_worlds", 1.0), ("two_worlds", 0.5), ("two_worlds", 1.0),
        ("adaptors", -0.5),
    ]
    dist = cv.graded_distribution(recs)
    assert dist["two_worlds"] == {-1.0: 0, -0.5: 0, 0.0: 0, 0.5: 1, 1.0: 2}
    assert dist["adaptors"] == {-1.0: 0, -0.5: 1, 0.0: 0, 0.5: 0, 1.0: 0}


def test_graded_distribution_snaps_near_values_to_nearest_level():
    # graded_max is discrete, but tolerate float noise by snapping to the nearest level.
    dist = cv.graded_distribution([("c", 0.49), ("c", -0.99)])
    assert dist["c"] == {-1.0: 1, -0.5: 0, 0.0: 0, 0.5: 1, 1.0: 0}


def test_graded_spread_reports_mean_std_min_max():
    spread = cv.graded_spread([("c", 1.0), ("c", 0.0), ("c", -1.0)])
    s = spread["c"]
    assert s["n"] == 3
    assert s["mean"] == 0.0
    assert s["min"] == -1.0 and s["max"] == 1.0
    assert s["std"] > 0.0


# --- pooled spearman (one helper, two axes) -------------------------------

def _papers():
    return {
        "a": {"scores": {"chunk": {"two_worlds": 0.9}},
              "verdict": {"two_worlds": "met"}, "graded": {"two_worlds": 1.0}},
        "b": {"scores": {"chunk": {"two_worlds": 0.1}},
              "verdict": {"two_worlds": "not_met"}, "graded": {"two_worlds": -0.5}},
        "c": {"scores": {"chunk": {"two_worlds": 0.5}},
              "verdict": {"two_worlds": "unclear"}, "graded": {"two_worlds": 0.5}},
    }


def test_pooled_spearman_graded_axis_is_monotone():
    rho = cv.pooled_spearman(
        _papers(), ["a", "b", "c"], ["chunk"], ["two_worlds"],
        value_of=lambda p, c: p["graded"].get(c))
    assert rho["two_worlds"]["chunk"] == 1.0


def test_pooled_spearman_categorical_axis_via_ordinal():
    rho = cv.pooled_spearman(
        _papers(), ["a", "b", "c"], ["chunk"], ["two_worlds"],
        value_of=lambda p, c: verdict_ordinal(p["verdict"].get(c))
        if p["verdict"].get(c) in ("met", "not_met", "unclear") else None)
    # e order a>c>b, ordinal met>unclear>not_met → perfect
    assert rho["two_worlds"]["chunk"] == 1.0


def test_pooled_spearman_none_when_no_variation():
    papers = {
        "a": {"scores": {"chunk": {"x": 0.9}}, "graded": {"x": 0.0}},
        "b": {"scores": {"chunk": {"x": 0.1}}, "graded": {"x": 0.0}},
    }
    rho = cv.pooled_spearman(papers, ["a", "b"], ["chunk"], ["x"],
                             value_of=lambda p, c: p["graded"].get(c))
    assert rho["x"]["chunk"] is None


def test_pooled_spearman_skips_missing_e_or_value():
    papers = {
        "a": {"scores": {"chunk": {"x": 0.9}}, "graded": {"x": 1.0}},
        "b": {"scores": {"chunk": {}}, "graded": {"x": -1.0}},          # no e
        "c": {"scores": {"chunk": {"x": 0.5}}, "graded": {"x": None}},  # no value
        "d": {"scores": {"chunk": {"x": 0.2}}, "graded": {"x": -1.0}},
    }
    rho = cv.pooled_spearman(papers, ["a", "b", "c", "d"], ["chunk"], ["x"],
                             value_of=lambda p, c: p["graded"].get(c))
    # only a and d survive → e[0.9,0.2] vs v[1.0,-1.0] → +1.0
    assert rho["x"]["chunk"] == 1.0
