"""Code Biology criteria-verification pipeline.

For each downloaded paper, decide — with grounded evidence — whether it argues
that its object satisfies the three minimum criteria for an organic code
(see CLAUDE.md):

  1. two_worlds    — two independent worlds of molecules
  2. adaptors      — a set of adaptors bridging them
  3. arbitrariness — the coding rules are conventional, not physically dictated

Model routing (per project decision):
  * criteria 1 & 2 (concrete) → local Gemma-4-31B (OpenAI-compatible server)
  * criterion 3 (subtle/contested) → nvidia/nemotron-3-ultra-550b-a55b:free via
    OpenRouter, which has a 1M-token context so it reads the *entire* paper.

Every "met" verdict must cite a verbatim quote from the source text; the
grounding gate downgrades any that don't (hallucination guard, CLAUDE.md rule 1).
The model assesses whether the *paper argues* a criterion — never whether the
criterion is objectively true.

This module's pure logic (join, JSON parsing, grounding, aggregation,
resumability) is unit-tested offline with the models injected as callables.
"""

import json
import logging
import os
import re

import pdf_text
from download_pdfs import output_path_for, read_rows

logger = logging.getLogger(__name__)

CRITERIA_12 = ["two_worlds", "adaptors"]
CRITERION_3 = ["arbitrariness"]
ALL_CRITERIA = CRITERIA_12 + CRITERION_3
VALID_VERDICTS = {"met", "not_met", "unclear"}

LOCAL_MODEL = "gemma-4-31b"
# Paid Nemotron (no ":free"): priority routing, no daily cap. The free tier judged
# correctly but at ~146 s/paper (≈19 h sequential for 471); paid + concurrency
# brings the whole criterion-3 run under an hour for ~$4. Same model/1M context.
OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
# Leave headroom under the Gemma batch context (32k) for prompt + output.
LOCAL_PAPER_TOKEN_BUDGET = 28000
# Concurrent papers in flight for the batch runner. The criterion-3 OpenRouter
# call is network/queue-bound (the bottleneck), so several overlap cleanly; the
# local Gemma is --parallel 1 and simply serialises its share at the server.
DEFAULT_WORKERS = 6


class JudgeError(RuntimeError):
    """Raised when a model response cannot be parsed into valid verdicts."""


# --- paper / PDF join ------------------------------------------------------

def iter_papers(csv_path, pdf_dir):
    """Yield ``{code_number, code_name, paper_name, url, pdf_path}`` for every
    citation whose PDF is present on disk, de-duplicated by PDF path."""
    seen = set()
    for row in read_rows(csv_path):
        path = output_path_for(row, pdf_dir)
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        yield {**row, "pdf_path": path}


# --- model-response parsing ------------------------------------------------

def _extract_json(text):
    """Pull the first JSON object out of a model response (tolerating code
    fences and surrounding prose)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            candidate = text[start:end + 1]
    if candidate is None:
        raise JudgeError("no JSON object found in model response")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise JudgeError(f"invalid JSON in model response: {exc}") from exc


def _normalise_verdict(value):
    """Coerce a per-criterion object into the canonical verdict shape."""
    if not isinstance(value, dict) or "verdict" not in value:
        raise JudgeError(f"criterion object missing 'verdict': {value!r}")
    verdict = str(value["verdict"]).lower()
    if verdict not in VALID_VERDICTS:
        raise JudgeError(f"invalid verdict {value['verdict']!r}")
    return {
        "verdict": verdict,
        "confidence": float(value.get("confidence", 0.0) or 0.0),
        "evidence_quote": str(value.get("evidence_quote", "") or ""),
        "reasoning": str(value.get("reasoning", "") or ""),
    }


def parse_judgment(raw, expected_keys):
    """Parse a model response into ``{criterion: verdict}`` for every expected
    criterion, raising :class:`JudgeError` on malformed / incomplete output."""
    obj = _extract_json(raw)
    out = {}
    for key in expected_keys:
        if key not in obj:
            raise JudgeError(f"response is missing criterion {key!r}")
        out[key] = _normalise_verdict(obj[key])
    return out


# --- grounding gate --------------------------------------------------------

def _norm_ws(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def grounding_gate(verdict, source_text):
    """Downgrade a ``met`` verdict to ``unclear`` unless its evidence quote is a
    verbatim (whitespace-normalised) substring of the source text."""
    if verdict.get("verdict") != "met":
        return verdict
    quote = _norm_ws(verdict.get("evidence_quote", ""))
    if quote and quote in _norm_ws(source_text):
        return verdict
    downgraded = dict(verdict)
    downgraded["verdict"] = "unclear"
    downgraded["grounding_failed"] = True
    logger.debug("grounding gate downgraded a 'met' verdict (quote not found)")
    return downgraded


# --- judging ---------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a Code Biology analyst. You assess whether a scientific paper "
    "ARGUES that the biological system it studies meets specific criteria for an "
    "organic code. You judge the paper's claims and evidence, NOT whether the "
    "claim is objectively true. Always ground a 'met' verdict in a verbatim quote "
    "copied exactly from the provided text. Reply with ONLY a JSON object."
)

CRITERIA_DEFS = {
    "two_worlds": "two independent worlds of molecules with no necessary "
                  "physical/chemical link between them (e.g. codons and amino acids)",
    "adaptors": "a set of adaptor molecules that physically bridge the two worlds "
                "(e.g. tRNAs)",
    "arbitrariness": "the coding rules are conventional, not dictated by physical "
                     "law — they could in principle be otherwise",
}


def build_user_prompt(text, keys):
    """Construct the per-paper instruction for the given criteria."""
    defs = "\n".join(f'- "{k}": {CRITERIA_DEFS[k]}' for k in keys)
    schema = ", ".join(
        f'"{k}": {{"verdict": "met|not_met|unclear", "confidence": 0.0-1.0, '
        f'"evidence_quote": "<verbatim>", "reasoning": "<1-2 sentences>"}}'
        for k in keys
    )
    return (
        f"Assess the paper below against these criteria:\n{defs}\n\n"
        f"Return exactly this JSON shape:\n{{{schema}}}\n\n"
        f"=== PAPER TEXT ===\n{text}"
    )


# --- graded per-chunk judging (topic-grounded, control-anchored, calibrated) ---
#
# The redesigned judge axis (CLAUDE.md §6/§8 — label quality is the binding constraint)
# scores ONE criterion against ONE 8192-token chunk at a time, on a graded agreement scale
# with a calibrated confidence, grounding any positive in a verbatim quote. The categorical
# verdict the rest of the pipeline consumes is *derived* from the aggregated graded score
# (see aggregate_graded). The prompt compresses the skeptical-analyst calibration protocol:
# premise check, ground-or-abstain, an operational Low/Medium/High scale, and (only for the
# contested arbitrariness criterion) a steelman so the model weighs the strongest counter.

GRADED_SYSTEM_PROMPT = (
    "You are a Code Biology analyst assessing whether a single passage from a paper "
    "ARGUES that the biological system it studies meets one specific criterion for an "
    "organic code. Judge the passage's claims and evidence, not whether the claim is "
    "objectively true. Reply with ONLY a JSON object."
)

CALIBRATION_PREAMBLE = (
    "Calibration protocol (follow exactly):\n"
    "1. Premise check — the research area / topic label below is CONTEXT, not EVIDENCE. "
    "A passage being about a coding-adjacent field does not by itself argue any criterion "
    "is met; judge only what THIS passage actually claims.\n"
    "2. Ground or abstain — to answer 'agree' or 'strongly_agree' you MUST copy a verbatim "
    "quote from the passage that supports it into evidence_quote. If no such quote exists, "
    "abstain to 'neutral' with an empty evidence_quote. Do not infer beyond the text.\n"
    "3. Calibrated confidence (operational): "
    "High = would act on this without further checking; "
    "Medium = directionally clear, verify before relying on it; "
    "Low = a hypothesis, not a conclusion.\n"
    "4. The two anchors below are reference poles: the AGREE anchor is a textbook passage "
    "that clearly argues the criterion; the DISAGREE anchor clearly argues against it. "
    "Calibrate your agreement level relative to these."
)

STEELMAN_ARBITRARINESS = (
    "Steelman (arbitrariness is the most contested criterion): the strongest case AGAINST "
    "arbitrariness is that the mapping is physically/chemically determined — a stereochemical "
    "or thermodynamic necessity rather than a convention. Only answer 'agree'/'strongly_agree' "
    "if the passage argues the rule is conventional and COULD be otherwise without breaking a "
    "law of chemistry, having weighed that counter-argument."
)

GRADED_AGREEMENT_LEVELS = (
    "strongly_disagree", "disagree", "neutral", "agree", "strongly_agree",
)


def build_chunk_prompt(chunk_text, criterion, topic_label, topic_blurb, controls):
    """Build the per-chunk, per-criterion graded prompt.

    Layers (plan §Design): calibration preamble → topic grounding (dominant-topic label +
    centroid blurb, as CONTEXT only) → AGREE/DISAGREE control anchors → criterion definition
    → strict graded JSON schema → the passage. The arbitrariness steelman is injected only
    for that criterion (its two controls are precisely its two poles)."""
    levels = "|".join(GRADED_AGREEMENT_LEVELS)
    schema = (
        '{"agreement": "%s", "confidence": "Low|Medium|High", '
        '"evidence_quote": "<verbatim quote from the passage, or empty>", '
        '"reasoning": "<1-2 sentences>"}' % levels
    )
    parts = [
        CALIBRATION_PREAMBLE,
        f"Research area (CONTEXT ONLY): {topic_label} — {topic_blurb}",
        "AGREE anchor (clearly argues the criterion):\n"
        f"  {controls['genetic_code_positive']}",
        "DISAGREE anchor (clearly argues against the criterion):\n"
        f"  {controls['deterministic_chemistry_negative']}",
        f'Criterion under judgement — "{criterion}": {CRITERIA_DEFS[criterion]}',
    ]
    if criterion == "arbitrariness":
        parts.append(STEELMAN_ARBITRARINESS)
    parts.append(
        "Does the passage ARGUE that this criterion is met? Answer on the graded scale.\n"
        f"Return exactly this JSON shape:\n{schema}"
    )
    parts.append(f"=== PASSAGE ===\n{chunk_text}")
    return "\n\n".join(parts)


def judge_criteria(text, complete, keys):
    """Judge ``keys`` on ``text`` using a ``complete(system, user, response_format)``
    callable that returns the model's raw text reply. Parses, then grounds each
    verdict against ``text``."""
    raw = complete(SYSTEM_PROMPT, build_user_prompt(text, keys),
                   response_format={"type": "json_object"})
    verdicts = parse_judgment(raw, keys)
    return {k: grounding_gate(v, text) for k, v in verdicts.items()}


# --- aggregation -----------------------------------------------------------

def paper_qualifies(criteria):
    """True iff all three criteria are ``met`` for a paper."""
    return all(criteria.get(k, {}).get("verdict") == "met" for k in ALL_CRITERIA)


def aggregate(paper_verdicts):
    """Roll per-paper verdicts up to per-code support counts and a dossier."""
    codes = {}
    for pv in paper_verdicts:
        code = pv["code_number"]
        entry = codes.setdefault(code, {
            "code_name": pv.get("code_name", ""), "supported": 0, "total": 0, "papers": [],
        })
        entry["total"] += 1
        qualifies = paper_qualifies(pv["criteria"])
        if qualifies:
            entry["supported"] += 1
        entry["papers"].append({
            "pdf_path": pv.get("pdf_path"),
            "qualifies": qualifies,
            "criteria": pv["criteria"],
        })
    return codes


# --- scoring (pure; embedding axis is independent of these verdicts) --------

VERDICT_ORDINAL = {"not_met": 0.0, "unclear": 0.5, "met": 1.0}
# Verdict picks the band; the embedding score positions within it, so a verdict can
# never be flipped by topical similarity (met > unclear > not_met always).
SCORE_BANDS = {"not_met": (0.0, 0.33), "unclear": (0.34, 0.66), "met": (0.67, 1.0)}


def verdict_ordinal(verdict):
    """Map a verdict string to its ordinal score (unknown → 0.5)."""
    return VERDICT_ORDINAL.get(verdict, 0.5)


def combine_score(verdict, e):
    """Combined per-criterion score in [0,1]: verdict band + embedding spread.

    ``e`` is the contrastive embedding score (≈[-1,1]); it only orders papers
    *within* the verdict's band. Deferred (plan decision 0): run 1 reports the
    embedding axis independently rather than merged.
    """
    lo, hi = SCORE_BANDS.get(verdict, SCORE_BANDS["unclear"])
    e_norm = (max(-1.0, min(1.0, float(e))) + 1.0) / 2.0
    return lo + (hi - lo) * e_norm


def weighted_median(values, weights):
    """Weighted median: the value where cumulative weight first reaches half the
    total. Reduces to the plain median for equal weights / a single value."""
    import numpy as np

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return float("nan")
    order = values.argsort(kind="mergesort")
    values, weights = values[order], weights[order]
    cum = np.cumsum(weights)
    cutoff = weights.sum() / 2.0
    idx = int(np.searchsorted(cum, cutoff, side="left"))  # first cum >= cutoff
    # boundary lands exactly atop values[idx] → average with the next (plain-median parity)
    if idx + 1 < values.size and np.isclose(cum[idx], cutoff):
        return float((values[idx] + values[idx + 1]) / 2.0)
    return float(values[min(idx, values.size - 1)])


def apply_coherence(criteria):
    """Annotate a paper's criteria dict with the two_worlds-gated coherence layer.

    Returns ``(annotated, flags)``. ``adaptors``/``arbitrariness`` are marked
    ``vacuous`` when ``two_worlds`` is firmly not_met (they presuppose two worlds).
    Logically incoherent patterns are collected in ``flags`` for the triage queue.
    """
    tw = criteria.get("two_worlds", {}).get("verdict")
    flags = []
    annotated = {k: dict(v) for k, v in criteria.items()}
    for downstream in ("adaptors", "arbitrariness"):
        v = annotated.get(downstream, {}).get("verdict")
        if tw == "not_met":
            annotated[downstream]["vacuous"] = True
            if v == "met":  # met adaptor/arbitrariness with no two worlds is incoherent
                flags.append(f"{downstream}=met but two_worlds=not_met")
    return annotated, flags


# --- resumability ----------------------------------------------------------

def load_done(checkpoint_path):
    """Return the set of pdf_paths already judged (from a JSONL checkpoint)."""
    done = set()
    if not os.path.exists(checkpoint_path):
        return done
    with open(checkpoint_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["pdf_path"])
            except (json.JSONDecodeError, KeyError):
                logger.warning("skipping malformed checkpoint line")
    return done


def append_checkpoint(checkpoint_path, record):
    """Append one judged-paper record to the JSONL checkpoint."""
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# --- concurrent batch runner -----------------------------------------------

PAPER_META_KEYS = ("code_number", "code_name", "paper_name", "url", "pdf_path")


def run_batch(papers, judge_fn, checkpoint_path, max_workers=DEFAULT_WORKERS,
              done=None):
    """Judge ``papers`` concurrently, checkpointing each as it completes.

    ``judge_fn(paper)`` returns the merged criteria dict (inject
    :func:`judge_paper` bound to its model callables in production; a fake in
    tests). Papers whose ``pdf_path`` is already in ``done`` (default: loaded
    from ``checkpoint_path``) are skipped, so the batch is resumable. The
    criterion-3 OpenRouter call dominates wall time and overlaps across workers;
    checkpoint appends are serialised under a lock. A paper that raises is logged
    and skipped (not checkpointed), so a single bad paper never aborts the run.
    Returns the list of records judged *this* invocation.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    if done is None:
        done = load_done(checkpoint_path)
    todo = [p for p in papers if p["pdf_path"] not in done]
    logger.info("batch: %d papers to judge (%d already done), %d workers",
                len(todo), len(done), max_workers)

    write_lock = threading.Lock()
    records = []

    def work(paper):
        criteria = judge_fn(paper)
        record = {k: paper.get(k) for k in PAPER_META_KEYS}
        record["criteria"] = criteria
        return record

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(work, p): p for p in todo}
        for fut in as_completed(futures):
            paper = futures[fut]
            try:
                record = fut.result()
            except Exception as exc:  # one bad paper must not kill the batch
                logger.warning("skipping %s: %s", paper["pdf_path"], exc)
                continue
            with write_lock:
                append_checkpoint(checkpoint_path, record)
                records.append(record)
            logger.info("judged %s (%d/%d)", paper["pdf_path"],
                        len(records), len(todo))
    return records


# --- model adapters (thin, not unit-tested; exercised by smoke tests) ------

def local_complete_factory(host="http://asushimu:11434", model=LOCAL_MODEL):
    """A ``complete`` callable for the local Gemma OpenAI-compatible server.

    Trims the paper to the local context budget before sending.
    """
    import requests

    def complete(system, user, response_format=None):
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.6,
            "top_p": 0.95,
        }
        if response_format is not None:
            body["response_format"] = response_format
        resp = requests.post(f"{host}/v1/chat/completions", json=body, timeout=600)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return complete


def openrouter_complete_factory(client=None, model=OPENROUTER_MODEL):
    """A ``complete`` callable for the OpenRouter criterion-3 judge (Nemotron)."""
    from openrouter_agent import OpenRouterClient

    client = client or OpenRouterClient()

    def complete(system, user, response_format=None):
        msg = client.call_model(
            model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format=response_format,
            temperature=0.4,
        )
        return msg.get("content") or ""

    return complete


def judge_paper(paper, local_complete, or_complete):
    """Judge one paper: criteria 1&2 locally on a budget-trimmed text, criterion
    3 on the full text via OpenRouter. Returns the merged criteria dict."""
    full_text = pdf_text.extract_text(paper["pdf_path"])
    local_text = pdf_text.select_for_budget(full_text, LOCAL_PAPER_TOKEN_BUDGET)
    criteria = {}
    criteria.update(judge_criteria(local_text, local_complete, CRITERIA_12))
    criteria.update(judge_criteria(full_text, or_complete, CRITERION_3))
    return criteria
