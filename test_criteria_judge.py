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
