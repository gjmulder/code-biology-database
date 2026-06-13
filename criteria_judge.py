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
OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
# Leave headroom under the Gemma batch context (32k) for prompt + output.
LOCAL_PAPER_TOKEN_BUDGET = 28000


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
