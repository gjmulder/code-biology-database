"""PDF text extraction, section splitting, and context-budget selection.

Used by the criteria-verification pipeline to turn a paper PDF into clean text
and, when a paper is too long for a model's context window, to select the
sections that carry the criteria evidence (dropping references / back-matter).

The 471-paper corpus extracts cleanly with ``pypdf`` (0 failures, 0 scanned), so
no OCR path is provided here. Token counts are *estimated* from a calibrated
chars-per-token ratio (3.58, measured against the live Gemma tokenizer) — the
authoritative tokenizer is the model server itself; this estimate only needs to
be good enough to pick a context budget.
"""

import logging
import re

from pypdf import PdfReader

logger = logging.getLogger(__name__)

# Calibrated against the live Gemma-4-31B tokenizer over a 12-paper sample.
CHARS_PER_TOKEN = 3.58

# Section headings we recognise, in the order they typically appear. Matching is
# done on a line that is *only* the heading (case-insensitive), which is how the
# extracted text renders them.
SECTION_HEADINGS = [
    "abstract",
    "introduction",
    "background",
    "methods",
    "materials and methods",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
]

# Sections that carry criteria evidence, best-first. Used to trim a paper down to
# a context budget — references and back-matter are dropped before body sections.
EVIDENCE_PRIORITY = [
    "_preamble",
    "abstract",
    "introduction",
    "background",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "methods",
    "materials and methods",
]

_HEADING_RE = re.compile(
    r"^\s*(?:\d+\.?\s*)?(" + "|".join(re.escape(h) for h in SECTION_HEADINGS) + r")\s*:?\s*$",
    re.IGNORECASE,
)


def estimate_tokens(text):
    """Estimate token count from character length (calibrated ratio)."""
    return int(len(text) / CHARS_PER_TOKEN)


def extract_text(path):
    """Extract and concatenate the text of every page in a PDF."""
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def split_sections(text):
    """Split paper text into ``{heading: body}`` by recognised heading lines.

    Text before the first recognised heading is stored under ``"_preamble"``.
    Headings are lower-cased keys; a later duplicate heading appends to the same
    key so a paper is never silently truncated at a repeated word.
    """
    sections = {"_preamble": []}
    current = "_preamble"
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            current = m.group(1).lower()
            sections.setdefault(current, [])
            sections[current].append(line)  # keep the heading line in its body
            continue
        sections[current].append(line)
    joined = {k: "\n".join(v).strip() for k, v in sections.items()}
    return {k: v for k, v in joined.items() if v or k == "_preamble"}


def select_for_budget(text, max_tokens):
    """Return text trimmed to fit ``max_tokens``.

    Strategy: if the whole paper fits, return it unchanged. Otherwise keep the
    evidence-bearing sections (preamble, abstract, intro, results, discussion,
    …) in priority order, dropping references/back-matter. If even the evidence
    sections overflow, hard-truncate to the character budget as a last resort.
    """
    if estimate_tokens(text) <= max_tokens:
        return text

    sections = split_sections(text)
    budget_chars = int(max_tokens * CHARS_PER_TOKEN)
    kept = []
    used = 0
    for name in EVIDENCE_PRIORITY:
        body = sections.get(name)
        if not body:
            continue
        block = body if used == 0 else "\n" + body
        if used + len(block) <= budget_chars:
            kept.append(body)
            used += len(block)
    out = "\n".join(kept).strip()

    if not out or estimate_tokens(out) > max_tokens:
        # Nothing fit cleanly (e.g. no headings) — hard truncate.
        out = text[:budget_chars]
        logger.debug("select_for_budget hard-truncated to %d chars", budget_chars)
    return out
