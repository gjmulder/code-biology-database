"""Tests for criteria_judge — the Code Biology criteria-verification pipeline.

Fully offline: model calls are injected as plain callables, the CSV reader and
filesystem existence checks are monkeypatched. No network, no real PDFs.
"""

import json

import pytest

import criteria_judge as cj


# --- paper/PDF join --------------------------------------------------------

def test_iter_papers_joins_existing_pdfs_only(monkeypatch):
    rows = [
        {"code_number": "1", "code_name": "A", "paper_name": "P1", "url": "https://doi.org/10.1/a"},
        {"code_number": "2", "code_name": "B", "paper_name": "P2", "url": "https://doi.org/10.2/b"},
    ]
    monkeypatch.setattr(cj, "read_rows", lambda csv_path: iter(rows))
    # only the first paper's PDF exists on disk
    monkeypatch.setattr(cj.os.path, "exists", lambda p: p.endswith("10.1_a.pdf"))
    papers = list(cj.iter_papers("any.csv", "pdfs"))
    assert len(papers) == 1
    assert papers[0]["code_number"] == "1"
    assert papers[0]["pdf_path"].endswith("10.1_a.pdf")


def test_iter_papers_dedupes_shared_pdf(monkeypatch):
    rows = [
        {"code_number": "1", "code_name": "A", "paper_name": "P", "url": "https://doi.org/10.1/x"},
        {"code_number": "1", "code_name": "A", "paper_name": "P dup", "url": "https://doi.org/10.1/x"},
    ]
    monkeypatch.setattr(cj, "read_rows", lambda csv_path: iter(rows))
    monkeypatch.setattr(cj.os.path, "exists", lambda p: True)
    papers = list(cj.iter_papers("any.csv", "pdfs"))
    assert len(papers) == 1  # same DOI -> same PDF -> one paper


# --- graded per-chunk prompt (topic-grounded, control-anchored, calibrated) ---

_CONTROLS = {
    "genetic_code_positive": "codons and amino acids are two worlds bridged by tRNA, "
                             "the assignment is an arbitrary convention",
    "deterministic_chemistry_negative": "stereochemical lock-and-key, physically determined",
}


def _build(criterion):
    return cj.build_chunk_prompt(
        chunk_text="The histone marks form a combinatorial pattern read by effector proteins.",
        criterion=criterion,
        topic_label="Histonic Code",
        topic_blurb="Histone modifications and chromatin signalling.",
        controls=_CONTROLS,
    )


def test_build_chunk_prompt_injects_calibration_sections():
    p = _build("two_worlds")
    # the compressed skeptical-analyst calibration must be present
    assert cj.CALIBRATION_PREAMBLE in p
    # premise check: the research-area label is context, not evidence
    assert "context" in p.lower() and "evidence" in p.lower()
    # ground-or-abstain wording + the Low/Medium/High operational scale
    assert "Low" in p and "Medium" in p and "High" in p


def test_build_chunk_prompt_injects_topic_and_controls():
    p = _build("adaptors")
    assert "Histonic Code" in p
    assert "Histone modifications and chromatin signalling." in p
    # both control anchors appear (AGREE exemplar + DISAGREE exemplar)
    assert _CONTROLS["genetic_code_positive"] in p
    assert _CONTROLS["deterministic_chemistry_negative"] in p
    # the criterion definition is injected
    assert cj.CRITERIA_DEFS["adaptors"] in p
    # the passage under judgement is present
    assert "combinatorial pattern read by effector proteins" in p


def test_build_chunk_prompt_has_graded_json_schema():
    p = _build("two_worlds")
    for level in ("strongly_disagree", "disagree", "neutral", "agree", "strongly_agree"):
        assert level in p
    assert "agreement" in p and "confidence" in p
    assert "evidence_quote" in p and "reasoning" in p


def test_build_chunk_prompt_steelman_only_for_arbitrariness():
    assert cj.STEELMAN_ARBITRARINESS in _build("arbitrariness")
    assert cj.STEELMAN_ARBITRARINESS not in _build("two_worlds")
    assert cj.STEELMAN_ARBITRARINESS not in _build("adaptors")


# --- domain-general criteria (apply across the 24 scientometric topics) ----

def test_criteria_defs_are_domain_general_not_molecular():
    """Each definition must instantiate beyond the molecular case: name >=2 non-molecular
    domains so a neural/audio/cultural paper is judged on its own world, not rejected for
    not being molecular (the molecular-bias bug the re-pilot targets)."""
    for crit in ("two_worlds", "adaptors"):
        d = cj.CRITERIA_DEFS[crit].lower()
        non_molecular = [w for w in ("neural", "percept", "audit", "cultural", "sign")
                         if w in d]
        assert len(non_molecular) >= 2, f"{crit} names too few non-molecular domains: {d}"
        # the definition must explicitly tell the judge not to require the molecular case
        assert "not specifically" in d or "need not be molecular" in d


def test_adaptors_def_generalises_to_mediator():
    """Per Major (2025) the adaptor is the molecular instance of a domain-general mediator;
    the DB key stays 'adaptors' but its definition text must carry the generalisation."""
    assert "mediator" in cj.CRITERIA_DEFS["adaptors"].lower()


def test_chunk_prompt_anchors_labelled_illustrative():
    """The molecular control anchors must be framed as ILLUSTRATIVE of the abstract relation,
    not as the required form (else they re-impose the molecular bias)."""
    p = _build("two_worlds")
    assert "ILLUSTRATIVE" in p


# --- prompt provenance (hash persisted alongside each verdict) -------------

def test_prompt_hash_is_stable_and_per_criterion():
    """The prompt version is a deterministic 64-hex sha256 and differs per criterion (the
    molecular->domain-general rewrite changed two_worlds/adaptors but not arbitrariness, so
    per-criterion provenance is meaningful)."""
    h = cj.prompt_hash("two_worlds")
    assert len(h) == 64 and all(ch in "0123456789abcdef" for ch in h)
    assert cj.prompt_hash("two_worlds") == h  # deterministic
    assert cj.prompt_hash("two_worlds") != cj.prompt_hash("adaptors")


def test_prompt_template_is_version_bearing_but_input_free():
    """prompt_template carries the version-bearing scaffold + criterion definition (so an
    edit to either changes the hash) but excludes per-chunk inputs (passage/topic/control
    text), so the hash identifies the prompt version, not the input."""
    t = cj.prompt_template("two_worlds")
    assert cj.CRITERIA_DEFS["two_worlds"] in t
    assert cj.CALIBRATION_PREAMBLE in t
    assert cj.STEELMAN_ARBITRARINESS in cj.prompt_template("arbitrariness")
    assert cj.STEELMAN_ARBITRARINESS not in t
    assert "=== PASSAGE ===" not in t  # the chunk passage is an input, not the version


def test_prompt_hash_changes_with_criterion_definition(monkeypatch):
    """Editing a criterion's definition changes its prompt hash (provenance sensitivity)."""
    before = cj.prompt_hash("two_worlds")
    monkeypatch.setitem(cj.CRITERIA_DEFS, "two_worlds", "a wholly different definition")
    assert cj.prompt_hash("two_worlds") != before


def test_build_chunk_prompt_still_matches_template_scaffold():
    """The live prompt and the version template must share the same scaffold (anchor framing,
    schema, criterion def) so the hash can't silently drift from what is actually sent."""
    p = _build("two_worlds")
    assert cj.ANCHOR_AGREE_FRAMING in p
    assert cj.ANCHOR_DISAGREE_FRAMING in p
    assert cj.ANCHOR_AGREE_FRAMING in cj.prompt_template("two_worlds")


# --- graded parsing + grounding -------------------------------------------

def _graded(agreement, quote="", confidence="High"):
    return json.dumps({"agreement": agreement, "confidence": confidence,
                       "evidence_quote": quote, "reasoning": "r"})


def test_parse_graded_maps_agreement_to_signed_score():
    cases = {"strongly_disagree": -1.0, "disagree": -0.5, "neutral": 0.0,
             "agree": 0.5, "strongly_agree": 1.0}
    for label, score in cases.items():
        out = cj.parse_graded(_graded(label), "two_worlds")
        assert out["agreement"] == score


def test_parse_graded_maps_confidence_to_float():
    assert cj.parse_graded(_graded("agree", confidence="Low"), "adaptors")["confidence"] == 0.33
    assert cj.parse_graded(_graded("agree", confidence="Medium"), "adaptors")["confidence"] == 0.66
    assert cj.parse_graded(_graded("agree", confidence="HIGH"), "adaptors")["confidence"] == 1.0


def test_parse_graded_tolerates_fence_and_keeps_quote():
    raw = "```json\n" + _graded("strongly_agree", quote="codons map to amino acids") + "\n```"
    out = cj.parse_graded(raw, "two_worlds")
    assert out["agreement"] == 1.0
    assert out["evidence_quote"] == "codons map to amino acids"


def test_parse_graded_raises_on_invalid_agreement():
    with pytest.raises(cj.JudgeError):
        cj.parse_graded(_graded("maybe"), "two_worlds")


def test_graded_grounding_gate_keeps_grounded_positive():
    chunk = "Here codons and amino acids form two worlds."
    parsed = cj.parse_graded(_graded("agree", quote="codons and amino acids form two worlds"),
                             "two_worlds")
    gated = cj.graded_grounding_gate(parsed, chunk)
    assert gated["agreement"] == 0.5
    assert not gated.get("grounding_failed")


def test_graded_grounding_gate_neutralises_ungrounded_positive():
    chunk = "The passage is about chromatin marks."
    parsed = cj.parse_graded(_graded("strongly_agree", quote="codons and amino acids"),
                             "two_worlds")
    gated = cj.graded_grounding_gate(parsed, chunk)
    assert gated["agreement"] == 0.0          # pulled to neutral, quote not in chunk
    assert gated["grounding_failed"] is True


def test_graded_grounding_gate_ignores_non_positive():
    chunk = "anything"
    parsed = cj.parse_graded(_graded("disagree", quote="not in chunk"), "two_worlds")
    gated = cj.graded_grounding_gate(parsed, chunk)
    assert gated["agreement"] == -0.5         # negatives need no grounding
    assert not gated.get("grounding_failed")


# --- graded aggregation (per paper per criterion) -------------------------

def _cs(agreement, confidence=1.0):
    return {"agreement": agreement, "confidence": confidence}


def test_aggregate_graded_maxpools_and_takes_argmax_confidence():
    chunks = [_cs(-0.5, 0.33), _cs(0.5, 0.66), _cs(0.0, 1.0)]
    gmax, gmean, conf, cat = cj.aggregate_graded(chunks)
    assert gmax == 0.5                              # strongest evidence anywhere
    assert abs(gmean - 0.0) < 1e-9                  # (-0.5 + 0.5 + 0.0)/3
    assert conf == 0.66                             # confidence of the argmax chunk
    assert cat == "met"                             # gmax >= +0.5


def test_aggregate_graded_categorical_thresholds():
    assert cj.aggregate_graded([_cs(1.0)])[3] == "met"
    assert cj.aggregate_graded([_cs(0.5)])[3] == "met"
    assert cj.aggregate_graded([_cs(0.25)])[3] == "unclear"   # 0 < gmax < 0.5
    assert cj.aggregate_graded([_cs(0.0)])[3] == "not_met"    # gmax <= 0
    assert cj.aggregate_graded([_cs(-1.0)])[3] == "not_met"


def test_aggregate_graded_empty_is_neutral_not_met():
    gmax, gmean, conf, cat = cj.aggregate_graded([])
    assert gmax == 0.0 and gmean == 0.0 and conf == 0.0
    assert cat == "not_met"


# --- JSON parsing ----------------------------------------------------------

def test_parse_judgment_reads_plain_json():
    raw = json.dumps({"two_worlds": {"verdict": "met", "confidence": 0.9,
                                      "evidence_quote": "q", "reasoning": "r"}})
    out = cj.parse_judgment(raw, ["two_worlds"])
    assert out["two_worlds"]["verdict"] == "met"


def test_parse_judgment_tolerates_code_fence_and_prose():
    raw = ('Sure! Here is the result:\n```json\n'
           '{"adaptors": {"verdict": "not_met"}}\n```\nHope that helps.')
    out = cj.parse_judgment(raw, ["adaptors"])
    assert out["adaptors"]["verdict"] == "not_met"
    # missing fields are defaulted
    assert out["adaptors"]["evidence_quote"] == ""
    assert out["adaptors"]["confidence"] == 0.0


def test_parse_judgment_raises_on_unparseable():
    with pytest.raises(cj.JudgeError):
        cj.parse_judgment("there is no json here at all", ["two_worlds"])


def test_parse_judgment_raises_on_missing_key():
    raw = json.dumps({"two_worlds": {"verdict": "met"}})
    with pytest.raises(cj.JudgeError):
        cj.parse_judgment(raw, ["two_worlds", "adaptors"])


def test_parse_judgment_rejects_bad_verdict_value():
    raw = json.dumps({"two_worlds": {"verdict": "probably"}})
    with pytest.raises(cj.JudgeError):
        cj.parse_judgment(raw, ["two_worlds"])


# --- grounding gate --------------------------------------------------------

SOURCE = "The tRNA acts as an adaptor between codons and amino acids."


def test_grounding_gate_keeps_met_with_verbatim_quote():
    v = {"verdict": "met", "confidence": 0.8,
         "evidence_quote": "acts as an adaptor", "reasoning": "r"}
    out = cj.grounding_gate(v, SOURCE)
    assert out["verdict"] == "met"
    assert out.get("grounding_failed") is not True


def test_grounding_gate_downgrades_fabricated_quote():
    v = {"verdict": "met", "confidence": 0.8,
         "evidence_quote": "ribosomes invented language", "reasoning": "r"}
    out = cj.grounding_gate(v, SOURCE)
    assert out["verdict"] == "unclear"
    assert out["grounding_failed"] is True


def test_grounding_gate_normalises_whitespace():
    v = {"verdict": "met", "evidence_quote": "adaptor   between\ncodons"}
    out = cj.grounding_gate(v, SOURCE)
    assert out["verdict"] == "met"


def test_grounding_gate_ignores_non_met_verdicts():
    v = {"verdict": "not_met", "evidence_quote": "anything fabricated"}
    out = cj.grounding_gate(v, SOURCE)
    assert out["verdict"] == "not_met"
    assert "grounding_failed" not in out


# --- judge_criteria (model injected) ---------------------------------------

def test_judge_criteria_parses_and_grounds(monkeypatch):
    def fake_complete(system, user, response_format=None):
        return json.dumps({
            "two_worlds": {"verdict": "met", "evidence_quote": "codons and amino acids"},
            "adaptors": {"verdict": "met", "evidence_quote": "totally made up phrase"},
        })

    text = "A paper about codons and amino acids and their mapping."
    out = cj.judge_criteria(text, fake_complete, ["two_worlds", "adaptors"])
    assert out["two_worlds"]["verdict"] == "met"          # grounded -> kept
    assert out["adaptors"]["verdict"] == "unclear"        # fabricated -> downgraded
    assert out["adaptors"]["grounding_failed"] is True


# --- aggregation -----------------------------------------------------------

def _pv(code, t, a, ar):
    return {
        "code_number": code, "code_name": "X", "pdf_path": f"{code}.pdf",
        "criteria": {
            "two_worlds": {"verdict": t},
            "adaptors": {"verdict": a},
            "arbitrariness": {"verdict": ar},
        },
    }


def test_paper_qualifies_requires_all_three_met():
    assert cj.paper_qualifies(_pv("1", "met", "met", "met")["criteria"]) is True
    assert cj.paper_qualifies(_pv("1", "met", "met", "unclear")["criteria"]) is False


def test_aggregate_rolls_up_per_code_with_denominator():
    verdicts = [
        _pv("1", "met", "met", "met"),       # qualifies
        _pv("1", "met", "not_met", "met"),   # does not
        _pv("2", "met", "unclear", "met"),   # does not
    ]
    codes = cj.aggregate(verdicts)
    assert codes["1"]["supported"] == 1
    assert codes["1"]["total"] == 2
    assert codes["2"]["supported"] == 0
    assert codes["2"]["total"] == 1


# --- resumability ----------------------------------------------------------

def test_load_done_reads_pdf_paths_from_jsonl(tmp_path):
    p = tmp_path / "ckpt.jsonl"
    p.write_text(json.dumps({"pdf_path": "a.pdf"}) + "\n" +
                 json.dumps({"pdf_path": "b.pdf"}) + "\n")
    assert cj.load_done(str(p)) == {"a.pdf", "b.pdf"}


def test_load_done_missing_file_is_empty(tmp_path):
    assert cj.load_done(str(tmp_path / "nope.jsonl")) == set()


# --- criterion-3 model routing --------------------------------------------

def test_openrouter_model_is_paid_tier():
    # The batch uses the paid Nemotron (priority routing, no daily cap), so the
    # model id must NOT carry the rate-limited ":free" suffix.
    assert cj.OPENROUTER_MODEL == "nvidia/nemotron-3-ultra-550b-a55b"
    assert not cj.OPENROUTER_MODEL.endswith(":free")


# --- concurrent batch runner ----------------------------------------------

def test_run_batch_judges_all_and_checkpoints(tmp_path):
    ckpt = tmp_path / "ckpt.jsonl"
    papers = [
        {"code_number": "1", "code_name": "A", "paper_name": "P1",
         "url": "u1", "pdf_path": "a.pdf"},
        {"code_number": "2", "code_name": "B", "paper_name": "P2",
         "url": "u2", "pdf_path": "b.pdf"},
    ]

    def judge_fn(paper):
        return {"two_worlds": {"verdict": "met"},
                "adaptors": {"verdict": "met"},
                "arbitrariness": {"verdict": "met"}}

    records = cj.run_batch(papers, judge_fn, str(ckpt), max_workers=2)
    assert len(records) == 2
    # every record carries paper metadata + criteria so aggregate() works
    by_path = {r["pdf_path"]: r for r in records}
    assert by_path["a.pdf"]["code_number"] == "1"
    assert by_path["a.pdf"]["criteria"]["two_worlds"]["verdict"] == "met"
    # checkpoint persisted both papers
    assert cj.load_done(str(ckpt)) == {"a.pdf", "b.pdf"}


def test_run_batch_skips_already_done(tmp_path):
    ckpt = tmp_path / "ckpt.jsonl"
    ckpt.write_text(json.dumps({"pdf_path": "a.pdf", "code_number": "1",
                                "criteria": {}}) + "\n")
    papers = [
        {"code_number": "1", "code_name": "A", "paper_name": "P1",
         "url": "u1", "pdf_path": "a.pdf"},
        {"code_number": "2", "code_name": "B", "paper_name": "P2",
         "url": "u2", "pdf_path": "b.pdf"},
    ]
    judged = []

    def judge_fn(paper):
        judged.append(paper["pdf_path"])
        return {"arbitrariness": {"verdict": "met"}}

    records = cj.run_batch(papers, judge_fn, str(ckpt), max_workers=2)
    assert judged == ["b.pdf"]          # a.pdf skipped (already done)
    assert len(records) == 1
    assert cj.load_done(str(ckpt)) == {"a.pdf", "b.pdf"}


def test_run_batch_runs_papers_concurrently(tmp_path):
    import threading
    import time

    ckpt = tmp_path / "ckpt.jsonl"
    papers = [{"code_number": str(i), "code_name": "C", "paper_name": "P",
               "url": "u", "pdf_path": f"{i}.pdf"} for i in range(4)]

    lock = threading.Lock()
    state = {"current": 0, "peak": 0}

    def judge_fn(paper):
        with lock:
            state["current"] += 1
            state["peak"] = max(state["peak"], state["current"])
        time.sleep(0.1)
        with lock:
            state["current"] -= 1
        return {"arbitrariness": {"verdict": "met"}}

    cj.run_batch(papers, judge_fn, str(ckpt), max_workers=4)
    assert state["peak"] >= 2           # genuinely overlapped, not serialized


def test_run_batch_isolates_failing_paper(tmp_path):
    ckpt = tmp_path / "ckpt.jsonl"
    papers = [
        {"code_number": "1", "code_name": "A", "paper_name": "P1",
         "url": "u1", "pdf_path": "good.pdf"},
        {"code_number": "2", "code_name": "B", "paper_name": "P2",
         "url": "u2", "pdf_path": "bad.pdf"},
    ]

    def judge_fn(paper):
        if paper["pdf_path"] == "bad.pdf":
            raise cj.JudgeError("model gave garbage")
        return {"arbitrariness": {"verdict": "met"}}

    records = cj.run_batch(papers, judge_fn, str(ckpt), max_workers=2)
    # the good paper still succeeds and is checkpointed; the bad one is skipped
    assert cj.load_done(str(ckpt)) == {"good.pdf"}
    assert len(records) == 1
    assert records[0]["pdf_path"] == "good.pdf"
