"""Code Biology criteria-verification pipeline.

For each downloaded paper, decide — with grounded evidence — whether it argues
that its object satisfies the three minimum criteria for an organic code
(see CLAUDE.md):

  1. two_worlds    — two independent worlds (sets of entities) with no necessary link
  2. adaptors      — a mediator bridging them (Barbieri's adaptor generalised, Major 2025)
  3. arbitrariness — the coding rules are conventional, not physically dictated

The criteria are domain-general: they instantiate across the 24 scientometric topics
(molecular, neural, auditory, olfactory, epigenetic, cultural, ...), with the molecular
genetic code as one exemplar rather than the requirement.

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

import hashlib
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


# --- graded parsing + grounding --------------------------------------------

AGREEMENT_SCORE = {
    "strongly_disagree": -1.0, "disagree": -0.5, "neutral": 0.0,
    "agree": 0.5, "strongly_agree": 1.0,
}
CONFIDENCE_SCORE = {"low": 0.33, "medium": 0.66, "high": 1.0}


def parse_graded(raw, criterion):
    """Parse a single-criterion graded response into a numeric record.

    Maps the ``agreement`` label to a signed score in [-1, 1] and the operational
    ``confidence`` (Low/Medium/High) to a float; raises :class:`JudgeError` on an invalid
    or missing agreement. ``criterion`` is accepted for symmetry / error context."""
    obj = _extract_json(raw)
    if not isinstance(obj, dict) or "agreement" not in obj:
        raise JudgeError(f"graded response missing 'agreement' for {criterion!r}: {obj!r}")
    label = str(obj["agreement"]).strip().lower()
    if label not in AGREEMENT_SCORE:
        raise JudgeError(f"invalid agreement {obj['agreement']!r} for {criterion!r}")
    conf = str(obj.get("confidence", "low")).strip().lower()
    return {
        "agreement": AGREEMENT_SCORE[label],
        "confidence": CONFIDENCE_SCORE.get(conf, CONFIDENCE_SCORE["low"]),
        "evidence_quote": str(obj.get("evidence_quote", "") or ""),
        "reasoning": str(obj.get("reasoning", "") or ""),
    }


def graded_grounding_gate(parsed, chunk_text):
    """Pull a *positive* graded score to neutral (0.0) unless its evidence quote is a
    verbatim (whitespace-normalised) substring of the chunk. Negatives/neutral pass through
    untouched — only an ungrounded claim of agreement is a hallucination risk."""
    if parsed.get("agreement", 0.0) <= 0.0:
        return parsed
    quote = _norm_ws(parsed.get("evidence_quote", ""))
    if quote and quote in _norm_ws(chunk_text):
        return parsed
    gated = dict(parsed)
    gated["agreement"] = 0.0
    gated["grounding_failed"] = True
    logger.debug("graded grounding gate neutralised an ungrounded positive (quote not found)")
    return gated


def aggregate_graded(chunk_scores):
    """Roll per-chunk graded records up to one per-paper-per-criterion result.

    ``chunk_scores`` is a list of parsed (and grounded) records, each with ``agreement`` and
    ``confidence``. Returns ``(graded_max, graded_mean, confidence, categorical)``:

    * ``graded_max``  — max agreement (primary; "argued anywhere", matching the embedding
      axis's max-pool, embed_score.aggregate_chunks).
    * ``graded_mean`` — mean agreement (overall stance across the paper).
    * ``confidence``  — the confidence of the argmax chunk.
    * ``categorical`` — ``met`` if graded_max ≥ +0.5, ``not_met`` if ≤ 0.0, else ``unclear``.

    Ungrounded positives are already neutralised by :func:`graded_grounding_gate`, so the
    threshold is purely on the grounded max. Empty input is neutral (0.0) → ``not_met``."""
    if not chunk_scores:
        return 0.0, 0.0, 0.0, "not_met"
    agreements = [float(c["agreement"]) for c in chunk_scores]
    graded_max = max(agreements)
    graded_mean = sum(agreements) / len(agreements)
    argmax = max(chunk_scores, key=lambda c: float(c["agreement"]))
    confidence = float(argmax.get("confidence", 0.0) or 0.0)
    if graded_max >= 0.5:
        categorical = "met"
    elif graded_max <= 0.0:
        categorical = "not_met"
    else:
        categorical = "unclear"
    return graded_max, graded_mean, confidence, categorical


# --- judging ---------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a Code Biology analyst. You assess whether a scientific paper "
    "ARGUES that the biological system it studies meets specific criteria for an "
    "organic code. You judge the paper's claims and evidence, NOT whether the "
    "claim is objectively true. Always ground a 'met' verdict in a verbatim quote "
    "copied exactly from the provided text. Reply with ONLY a JSON object."
)

CRITERIA_DEFS = {
    "two_worlds": (
        "two independent worlds: two distinct sets of entities — molecules, signals, "
        "states, or representations — with no necessary physical/chemical/causal link, "
        "so their elements could in principle be paired in more than one way. Instantiate "
        "in THIS passage's domain: codons<->amino acids (molecular); stimulus "
        "features<->neural spike patterns (neural); acoustic signal<->auditory percept "
        "(auditory); odorant chemistry<->perceptual valence (olfactory); histone "
        "marks<->gene-regulatory outcomes (epigenetic); signs<->meanings (cultural). "
        "Judge whether two such worlds are argued here — not specifically molecular worlds"
    ),
    "adaptors": (
        "a mediator: a third entity, distinct from both worlds, that physically reads and "
        "executes the correspondence, translating one world into the other (Barbieri's "
        "adaptor; Major 2025 generalises it to the domain's mediator). Need not be "
        "molecular: tRNA/ribosome (molecular); a neural circuit or the nervous system "
        "(neural, perceptual); receptor populations (sensory/auditory); imaginal function "
        "(archetypal/cultural); a computational engine (artificial). Judge whether such a "
        "mediating mechanism is argued here — not specifically an adaptor molecule"
    ),
    "arbitrariness": (
        "the coding rules are conventional: compatible with but not determined by physical "
        "law, so they could in principle be otherwise. Domain-general: a learned or cultural "
        "mapping is arbitrary; a mapping dictated by stimulus physics or stereochemistry is "
        "not"
    ),
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
    "4. The two anchors below are ILLUSTRATIVE reference poles, not the required form: the "
    "AGREE anchor clearly argues the criterion (in its own domain), the DISAGREE anchor "
    "clearly argues against it. Calibrate the abstract relation relative to these, then "
    "judge it in THIS passage's domain."
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

# Version-bearing prompt scaffold, factored out so both the live prompt
# (:func:`build_chunk_prompt`) and the provenance template (:func:`prompt_template`) share
# the exact same text — the hash then cannot silently drift from what is actually sent.
ANCHOR_AGREE_FRAMING = (
    "AGREE anchor (ILLUSTRATIVE — the genetic code is the molecular exemplar of the "
    "abstract relation; your passage's domain will differ, so match the relation, not "
    "the molecules):"
)
ANCHOR_DISAGREE_FRAMING = (
    "DISAGREE anchor (ILLUSTRATIVE — a physically-determined process, here molecular; "
    "the principle is domain-general):"
)
QUESTION_LINE = (
    "Does the passage ARGUE that this criterion is met? Answer on the graded scale."
)


def _graded_schema():
    """The strict graded JSON shape the judge must return (shared by live prompt + template)."""
    levels = "|".join(GRADED_AGREEMENT_LEVELS)
    return (
        '{"agreement": "%s", "confidence": "Low|Medium|High", '
        '"evidence_quote": "<verbatim quote from the passage, or empty>", '
        '"reasoning": "<1-2 sentences>"}' % levels
    )


def _criterion_block(criterion):
    """The version-bearing criterion section: definition + (arbitrariness only) steelman."""
    parts = [f'Criterion under judgement — "{criterion}": {CRITERIA_DEFS[criterion]}']
    if criterion == "arbitrariness":
        parts.append(STEELMAN_ARBITRARINESS)
    return parts


def build_chunk_prompt(chunk_text, criterion, topic_label, topic_blurb, controls):
    """Build the per-chunk, per-criterion graded prompt.

    Layers: calibration preamble → topic grounding (dominant-topic label + centroid blurb, as
    CONTEXT only) → AGREE/DISAGREE control anchors → **the passage** → criterion definition →
    strict graded JSON schema. The arbitrariness steelman is injected only for that criterion.

    PREFIX-CACHING ORDER: everything up to and including the passage is identical across the
    three criteria calls for a given chunk, so it forms a long shared PREFIX an implicit-caching
    provider (DeepSeek first-party) serves from cache on the 2nd/3rd call — the ~8k-token passage
    is paid full price once, cache-read (≈120× cheaper) twice. Only the small criterion-specific
    block + schema vary per call, so they are the SUFFIX (see openrouter_graded_factory). The
    prompt-version hash (prompt_template/prompt_hash) is unaffected: it excludes the passage and
    captures scaffold *content*, not this delivery ordering."""
    parts = [
        CALIBRATION_PREAMBLE,
        f"Research area (CONTEXT ONLY): {topic_label} — {topic_blurb}",
        f"{ANCHOR_AGREE_FRAMING}\n  {controls['genetic_code_positive']}",
        f"{ANCHOR_DISAGREE_FRAMING}\n  {controls['deterministic_chemistry_negative']}",
        f"=== PASSAGE ===\n{chunk_text}",
    ]
    parts += _criterion_block(criterion)
    parts.append(f"{QUESTION_LINE}\nReturn exactly this JSON shape:\n{_graded_schema()}")
    return "\n\n".join(parts)


def prompt_template(criterion):
    """Canonical, version-bearing text of the graded per-chunk prompt for ``criterion``.

    The invariant scaffold (system prompt, calibration preamble, anchor framing, graded JSON
    schema, the question) plus the criterion definition and — for arbitrariness — the steelman.
    Per-chunk inputs (the passage, topic label/blurb, control passage text) are deliberately
    excluded so the text identifies the *prompt version*, not the input. This is exactly what
    the molecular → domain-general rewrite changed, so its hash is the prompt-provenance key
    persisted next to each verdict."""
    parts = [
        GRADED_SYSTEM_PROMPT,
        CALIBRATION_PREAMBLE,
        ANCHOR_AGREE_FRAMING,
        ANCHOR_DISAGREE_FRAMING,
    ]
    parts += _criterion_block(criterion)
    parts.append(f"{QUESTION_LINE}\nReturn exactly this JSON shape:\n{_graded_schema()}")
    return "\n\n".join(parts)


def prompt_hash(criterion):
    """Stable sha256 hex digest of :func:`prompt_template` — the verdict's prompt version."""
    return hashlib.sha256(prompt_template(criterion).encode("utf-8")).hexdigest()


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


# --- DeepSeek V4 Pro graded judge (high-reasoning, implicit-caching) --------
#
# A higher-quality replacement for the local Gemma graded judge (CLAUDE.md §6/§8: label
# quality is the binding constraint). Routed via OpenRouter and PINNED to the DeepSeek
# first-party endpoint, which is both the cheapest and the only one advertising
# ``supports_implicit_caching`` — so the ~8k-token passage that build_chunk_prompt places at
# the head of the prompt is served from cache on the 2nd/3rd criterion call of each chunk at
# the cache-read rate (≈120× cheaper than fresh input). Prices are per-1M tokens, taken from
# the OpenRouter endpoints API for tag "deepseek" (2026-06-17); confirm against a small batch
# before a full corpus run (the run prints measured usage + cost via UsageMeter).

DEEPSEEK_MODEL = "deepseek/deepseek-v4-pro"
DEEPSEEK_PROVIDER = {"order": ["deepseek"], "allow_fallbacks": False}
DEEPSEEK_PRICE_IN = 0.435          # $ / 1M fresh input tokens
DEEPSEEK_PRICE_CACHE_READ = 0.003625  # $ / 1M cached (prefix-hit) input tokens
DEEPSEEK_PRICE_OUT = 0.87          # $ / 1M completion tokens (reasoning tokens included here)


class UsageMeter:
    """Thread-safe accumulator of OpenRouter token usage for real-spend reporting.

    Sums ``prompt_tokens`` / ``completion_tokens`` and, where the provider reports it, the
    cached prefix tokens (``prompt_tokens_details.cached_tokens``) and reasoning tokens. Bills
    fresh input, cache-read input and completion at separate per-1M rates so the cache discount
    is visible in :meth:`cost`."""

    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0

    def add(self, usage):
        pdet = usage.get("prompt_tokens_details") or {}
        cdet = usage.get("completion_tokens_details") or {}
        with self._lock:
            self.calls += 1
            self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            self.cached_tokens += int(pdet.get("cached_tokens", 0) or 0)
            self.reasoning_tokens += int(cdet.get("reasoning_tokens", 0) or 0)

    def cost(self, in_price=DEEPSEEK_PRICE_IN, cache_read_price=DEEPSEEK_PRICE_CACHE_READ,
             out_price=DEEPSEEK_PRICE_OUT):
        """Total $ for the accumulated usage. ``cached_tokens`` are billed at the cache-read
        rate and the remainder of ``prompt_tokens`` at the fresh-input rate."""
        fresh = max(0, self.prompt_tokens - self.cached_tokens)
        return (fresh * in_price + self.cached_tokens * cache_read_price
                + self.completion_tokens * out_price) / 1e6


def openrouter_graded_factory(client=None, model=DEEPSEEK_MODEL, reasoning_effort="high",
                              provider=None, meter=None, temperature=0.4):
    """A graded ``complete(system, user, response_format)`` callable on DeepSeek V4 Pro.

    Pins the implicit-caching DeepSeek first-party provider (``DEEPSEEK_PROVIDER``) and requests
    the given reasoning effort. If ``meter`` is a :class:`UsageMeter`, each call's token usage is
    accumulated for cost reporting."""
    from openrouter_agent import OpenRouterClient

    client = client or OpenRouterClient()
    provider = DEEPSEEK_PROVIDER if provider is None else provider
    reasoning = {"effort": reasoning_effort} if reasoning_effort else None

    def complete(system, user, response_format=None):
        msg, usage = client.call_model_usage(
            model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format=response_format, temperature=temperature,
            reasoning=reasoning, provider=provider)
        if meter is not None:
            meter.add(usage)
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
