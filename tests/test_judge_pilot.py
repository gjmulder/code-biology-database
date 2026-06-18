"""Offline tests for judge_pilot's pure logic (no GPU/DB/network).

Covers the parts the plan marks unit-testable: augmented-topic loading, dominant-topic
selection of the top-N pilot strata, triple-keyed (pdf_path, chunk_idx, criterion)
resumability, the flat per-chunk checkpoint record, and the roll-up of those records into
db.update_verdicts-shaped per-(paper,criterion) records via criteria_judge.aggregate_graded.
The I/O driver (tokenizer, Gemma calls, MySQL) is exercised manually, not here.
"""

import json

import judge_pilot as jp


# --- augmented-topic loading ----------------------------------------------

def _write_csv(tmp_path):
    p = tmp_path / "aug.csv"
    p.write_text(
        "Topic #,Label,Abbreviation,Justification,Characteristic Terms (Fig 2),Centroid Text\n"
        "11,Cognitive Signal,COG,j,terms,brains and signalling\n"
        "13,Olfactory Code,OLF,j,terms,smell receptors map odorants\n",
        encoding="utf-8",
    )
    return str(p)


def test_load_augmented_topics_maps_id_to_label_and_blurb(tmp_path):
    topics = jp.load_augmented_topics(_write_csv(tmp_path))
    assert topics[11] == ("Cognitive Signal", "brains and signalling")
    assert topics[13] == ("Olfactory Code", "smell receptors map odorants")


# --- dominant-topic selection ---------------------------------------------

def _chunk_topics():
    # (chunk_idx, topic_id, sim); paper_dominant_topic max-pools per topic.
    return {
        "a.pdf": [(0, 11, 0.8), (1, 11, 0.9)],   # dominant 11
        "b.pdf": [(0, 11, 0.7)],                  # dominant 11
        "c.pdf": [(0, 13, 0.6), (1, 18, 0.5)],    # dominant 13
        "d.pdf": [(0, 18, 0.4)],                  # dominant 18
        "e.pdf": [(0, 19, 0.95)],                 # dominant 19
    }


def test_paper_dominant_topics():
    doms = jp.paper_dominant_topics(_chunk_topics())
    assert doms == {"a.pdf": 11, "b.pdf": 11, "c.pdf": 13, "d.pdf": 18, "e.pdf": 19}


def test_top_topic_ids_orders_by_frequency_then_id():
    # counts: 11→2, 13→1, 18→1, 19→1; ties broken by ascending topic id.
    assert jp.top_topic_ids(_chunk_topics(), n=3) == [11, 13, 18]


def test_select_pilot_papers_keeps_only_top_strata():
    selected = jp.select_pilot_papers(_chunk_topics(), n=1)  # top topic is 11 only
    assert selected == {"a.pdf": 11, "b.pdf": 11}


def test_select_rest_papers_is_complement_of_top_strata():
    # top-1 stratum is topic 11 (a, b); the "rest" is everything outside it.
    rest = jp.select_rest_papers(_chunk_topics(), n=1)
    assert rest == {"c.pdf": 13, "d.pdf": 18, "e.pdf": 19}
    # rest and pilot partition the dominant-topic papers with no overlap.
    pilot = jp.select_pilot_papers(_chunk_topics(), n=1)
    assert set(rest) | set(pilot) == {"a.pdf", "b.pdf", "c.pdf", "d.pdf", "e.pdf"}
    assert set(rest) & set(pilot) == set()


def test_select_code_papers_restricts_to_a_code_number():
    # codes maps pdf_path -> code_number; select only the papers under code "0"
    # (the foundational Code Biology texts), carrying each one's dominant topic
    # for grounding. Matches on string-equal code number (CSV/DB are stringy).
    codes = {"a.pdf": 0, "b.pdf": "12", "c.pdf": 0, "d.pdf": 5, "e.pdf": 0}
    selected = jp.select_code_papers(_chunk_topics(), codes, "0")
    assert selected == {"a.pdf": 11, "c.pdf": 13, "e.pdf": 19}


# --- triple-keyed resumability --------------------------------------------

def test_load_done_keys_on_pdf_chunk_criterion(tmp_path):
    ckpt = tmp_path / "ck.jsonl"
    ckpt.write_text(
        json.dumps({"pdf_path": "a.pdf", "chunk_idx": 0, "criterion": "two_worlds"}) + "\n"
        + json.dumps({"pdf_path": "a.pdf", "chunk_idx": 1, "criterion": "two_worlds"}) + "\n"
        + "\n"  # blank line tolerated
        + json.dumps({"pdf_path": "a.pdf", "chunk_idx": 0, "criterion": "adaptors"}) + "\n",
        encoding="utf-8",
    )
    done = jp.load_done(str(ckpt))
    assert done == {
        ("a.pdf", 0, "two_worlds"),
        ("a.pdf", 1, "two_worlds"),
        ("a.pdf", 0, "adaptors"),
    }
    # same chunk, different criterion is NOT done
    assert ("a.pdf", 1, "adaptors") not in done


def test_load_done_missing_file_is_empty(tmp_path):
    assert jp.load_done(str(tmp_path / "nope.jsonl")) == set()


# --- checkpoint record -----------------------------------------------------

def test_chunk_record_shape_matches_chunk_verdict_rows():
    meta = {"code_number": "42", "pdf_path": "a.pdf", "code_name": "x"}
    parsed = {"agreement": 0.5, "confidence": 0.66,
              "evidence_quote": "q", "reasoning": "r"}
    rec = jp.chunk_record(meta, 3, "adaptors", parsed)
    import criteria_judge as cj
    assert rec == {
        "code_number": "42", "pdf_path": "a.pdf", "chunk_idx": 3,
        "criterion": "adaptors", "agreement": 0.5, "confidence": 0.66,
        "evidence_quote": "q", "reasoning": "r",
        "prompt_hash": cj.prompt_hash("adaptors"),
    }
    # db.chunk_verdict_rows must consume it without KeyError, carrying the prompt version
    import db
    rows = db.chunk_verdict_rows([rec], run_ts="t", model="gemma-4-31b")
    assert rows == [(42, "a.pdf", "adaptors", 3, 0.5, 0.66, "q", "gemma-4-31b",
                     cj.prompt_hash("adaptors"), "t")]


# --- roll-up to verdict records -------------------------------------------

def test_aggregate_to_verdict_records_uses_graded_max_and_derived_categorical():
    chunk_records = [
        # two_worlds: max +1.0 → met
        jp.chunk_record({"code_number": "1", "pdf_path": "a.pdf"}, 0, "two_worlds",
                        {"agreement": 0.0, "confidence": 0.33, "evidence_quote": ""}),
        jp.chunk_record({"code_number": "1", "pdf_path": "a.pdf"}, 1, "two_worlds",
                        {"agreement": 1.0, "confidence": 1.0, "evidence_quote": "q"}),
        # adaptors: max 0.0 → not_met
        jp.chunk_record({"code_number": "1", "pdf_path": "a.pdf"}, 0, "adaptors",
                        {"agreement": -0.5, "confidence": 0.66, "evidence_quote": ""}),
    ]
    records = jp.aggregate_to_verdict_records(chunk_records)
    assert len(records) == 1
    rec = records[0]
    assert rec["code_number"] == 1 and rec["pdf_path"] == "a.pdf"
    tw = rec["criteria"]["two_worlds"]
    assert tw["verdict"] == "met"
    assert tw["graded"] == 1.0
    assert tw["confidence"] == 1.0          # confidence of the argmax chunk
    assert rec["criteria"]["adaptors"]["verdict"] == "not_met"
    assert rec["criteria"]["adaptors"]["graded"] == -0.5

    # feeds db.verdict_update_rows cleanly (graded + prompt version carried through)
    import db, criteria_judge as cj
    rows = db.verdict_update_rows(records, run_ts="t", model="gemma-4-31b")
    tw_row = [r for r in rows if r[2] == "two_worlds"][0]
    assert tw_row == (1, "a.pdf", "two_worlds", "met", 1.0, 1.0, "gemma-4-31b",
                      cj.prompt_hash("two_worlds"), "t")


# --- AGREE anchor ablation variants ---------------------------------------

def test_agree_anchor_variants_map_keys_and_tags():
    v = jp.AGREE_ANCHOR_VARIANTS
    # baseline is the molecular genetic 1-shot, untagged (does not shadow the existing corpus)
    assert v["genetic"] == (("genetic_code_positive",), "")
    # non-molecular 1-shot and the 2-shot both carry a distinct tag suffix so they coexist
    assert v["neural"] == (("neural_code_positive",), "@neural-1shot")
    assert v["neural-genetic"] == (
        ("neural_code_positive", "genetic_code_positive"), "@neural-genetic-2shot")


def test_judge_paper_chunks_forwards_agree_keys(tmp_path, monkeypatch):
    import threading
    monkeypatch.setattr(jp.chunk_text, "reproduce_chunks",
                        lambda text, tok: [(0, "a passage")])
    seen = {}

    def fake_build(ctext, crit, label, blurb, controls,
                   agree_keys=("genetic_code_positive",)):
        seen["agree_keys"] = agree_keys
        return "PROMPT"
    monkeypatch.setattr(jp.cj, "build_chunk_prompt", fake_build)

    def fake_complete(system, user, response_format=None):
        return json.dumps({"agreement": "neutral", "confidence": "Low",
                           "evidence_quote": "", "reasoning": "r"})

    ckpt = str(tmp_path / "c.jsonl")
    jp.judge_paper_chunks({"code_number": 1, "pdf_path": "p.pdf"}, "full text",
                          tokenizer=None, topic_label="L", topic_blurb="B", controls={},
                          complete=fake_complete, checkpoint_path=ckpt, done=set(),
                          write_lock=threading.Lock(),
                          agree_keys=("neural_code_positive",))
    assert seen["agree_keys"] == ("neural_code_positive",)


# --- judge backend selection (local Gemma vs DeepSeek V4 Pro) --------------

def test_make_judge_local_returns_gemma_tag():
    complete, model = jp.make_judge("local", host="http://x:11434")
    assert callable(complete)
    assert model == jp.cj.LOCAL_MODEL


def test_make_judge_deepseek_pins_model_and_passes_meter(monkeypatch):
    seen = {}

    def fake_factory(reasoning_effort=None, meter=None):
        seen["reasoning"] = reasoning_effort
        seen["meter"] = meter
        return lambda *a, **k: "{}"

    monkeypatch.setattr(jp.cj, "openrouter_graded_factory", fake_factory)
    meter = object()
    complete, model = jp.make_judge("deepseek", reasoning="high", meter=meter)
    assert model == jp.cj.DEEPSEEK_MODEL
    assert seen["reasoning"] == "high" and seen["meter"] is meter


def test_make_judge_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        jp.make_judge("gpt5")


def test_load_env_sets_missing_keys_without_clobbering(tmp_path, monkeypatch):
    """load_env populates a missing key from .env but never overrides an already-set env var."""
    env = tmp_path / ".env"
    env.write_text('OPENROUTER_API_KEY=sk-from-file\nexport FOO="bar"\n')
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("FOO", "already-set")
    jp.load_env(str(env))
    assert jp.os.environ["OPENROUTER_API_KEY"] == "sk-from-file"
    assert jp.os.environ["FOO"] == "already-set"  # setdefault must not clobber


def test_load_env_missing_file_is_noop(tmp_path):
    """A missing .env must not raise — local runs need no secrets file."""
    jp.load_env(str(tmp_path / "nonexistent.env"))
