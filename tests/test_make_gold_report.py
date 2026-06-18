"""Offline tests for the Phase 6 gold-validation report (CLAUDE.md §8 / plan).

The report adjudicates the two synthetic axes against the Barbieri-anchored gold set:
the embedding axis (does gold+ outrank gold−? AUC + ρ) and the judge axis (categorical
verdict vs gold polarity — precision/recall/confusion), per criterion and split by tier.
Pure metric + join logic is tested here; the DB read is exercised by the driver.
"""

import math

import make_gold_report as mgr


# --- AUC (rank statistic: P[random gold+ outranks random gold−]) -----------

def test_auc_perfect_separation_is_one():
    assert mgr.auc([1.0, 2.0, 3.0], [-1.0, 0.0]) == 1.0


def test_auc_reversed_separation_is_zero():
    assert mgr.auc([-1.0, 0.0], [1.0, 2.0, 3.0]) == 0.0


def test_auc_ties_count_half():
    # one pos vs one neg, equal score → 0.5
    assert mgr.auc([1.0], [1.0]) == 0.5


def test_auc_empty_side_is_nan():
    assert math.isnan(mgr.auc([], [1.0]))
    assert math.isnan(mgr.auc([1.0], []))


# --- confusion matrix + precision/recall -----------------------------------

def test_confusion_counts_and_rates():
    preds = [True, True, False, False, True]
    golds = [True, False, False, True, True]
    c = mgr.confusion(preds, golds)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 1, 1)
    assert c["precision"] == 2 / 3       # 2 tp / (2 tp + 1 fp)
    assert c["recall"] == 2 / 3          # 2 tp / (2 tp + 1 fn)
    assert c["n"] == 5


def test_confusion_zero_denominator_is_nan():
    c = mgr.confusion([False, False], [False, False])  # no predicted-pos, no gold-pos
    assert math.isnan(c["precision"])
    assert math.isnan(c["recall"])


# --- end-to-end join: gold polarity × embedding e × categorical verdict ----

def _papers():
    # pid -> fetch_report-shaped record. method 'chunk' is the working granularity.
    return {
        "p_pos1.pdf": {"code": 10, "scores": {"chunk": {"two_worlds": 0.9, "adaptors": 0.8,
                       "arbitrariness": 0.5}},
                       "verdict": {"two_worlds": "met", "adaptors": "met",
                                   "arbitrariness": "not_met"}},
        "p_pos2.pdf": {"code": 11, "scores": {"chunk": {"two_worlds": 0.7, "adaptors": 0.6,
                       "arbitrariness": 0.4}},
                       "verdict": {"two_worlds": "met", "adaptors": "not_met",
                                   "arbitrariness": "not_met"}},
        "p_neg1.pdf": {"code": 12, "scores": {"chunk": {"two_worlds": 0.1, "adaptors": 0.2,
                       "arbitrariness": 0.3}},
                       "verdict": {"two_worlds": "not_met", "adaptors": "not_met",
                                   "arbitrariness": "not_met"}},
        "p_unlabelled.pdf": {"code": 99, "scores": {"chunk": {"two_worlds": 0.5,
                             "adaptors": 0.5, "arbitrariness": 0.5}},
                             "verdict": {"two_worlds": "met"}},  # not in gold → ignored
    }


def _gold():
    return {
        (10, "p_pos1.pdf", "all"): {"polarity": "pos", "tier": "2"},
        (11, "p_pos2.pdf", "all"): {"polarity": "pos", "tier": "1"},
        (12, "p_neg1.pdf", "all"): {"polarity": "neg", "tier": "soft"},
    }


def test_gold_validation_joins_only_gold_papers():
    res = mgr.gold_validation(_papers(), _gold(), method="chunk")
    # the unlabelled paper (not in gold) is excluded from every criterion
    tw = res["two_worlds"]
    assert tw["embedding"]["n_pos"] == 2 and tw["embedding"]["n_neg"] == 1


def test_gold_validation_embedding_auc_separates_pos_from_neg():
    res = mgr.gold_validation(_papers(), _gold(), method="chunk")
    # two_worlds: pos e = {0.9, 0.7}, neg e = {0.1} → perfect separation
    assert res["two_worlds"]["embedding"]["auc"] == 1.0


def test_gold_validation_judge_confusion_uses_met_as_positive():
    res = mgr.gold_validation(_papers(), _gold(), method="chunk")
    # adaptors: preds(met?) = [pos1 met=T, pos2 not_met=F, neg1 not_met=F];
    # golds(pos?) = [T, T, F] → tp=1, fp=0, fn=1, tn=1
    jc = res["adaptors"]["judge"]
    assert (jc["tp"], jc["fp"], jc["fn"], jc["tn"]) == (1, 0, 1, 1)


def test_gold_validation_reports_per_tier_counts():
    res = mgr.gold_validation(_papers(), _gold(), method="chunk")
    tiers = res["two_worlds"]["tiers"]
    # tier '1' has 1 pos, tier '2' has 1 pos, tier 'soft' has 1 neg
    assert tiers["1"]["n"] == 1 and tiers["2"]["n"] == 1 and tiers["soft"]["n"] == 1


def test_format_report_is_markdown_with_criteria_headers():
    res = mgr.gold_validation(_papers(), _gold(), method="chunk")
    md = mgr.format_report(res, judge="deepseek/deepseek-v4-pro")
    assert md.startswith("#")
    for crit in ("two_worlds", "adaptors", "arbitrariness"):
        assert crit in md
    assert "AUC" in md and "precision" in md
