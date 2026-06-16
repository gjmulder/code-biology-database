# Criteria-Verification Pipeline — Design Plan

**Status:** Draft for review.
**Goal:** For each paper we have on disk (the citations behind the codes in
`biological_codes.csv`), decide — with grounded evidence — whether it actually
satisfies the three minimum criteria for an organic code.

> **Not to be confused with `cross-discipline-screening-design.md`.** That doc
> *discovers* criteria-meeting objects hidden in **other fields'** literature (a
> positive-unlabeled ranking problem; the criteria are never stated outright).
> This doc *verifies* the **in-field** corpus we already have, by reading each
> paper directly for criteria that the authors generally *do* claim. Shared
> evaluation philosophy; different modeling. The verified positives produced
> here become the high-quality positive set that the cross-discipline system
> trains on.

---

## 1. Problem statement & scope

The three minimum criteria (per `CLAUDE.md` / `www.codebiology.org/index.html`):

1. **Two independent worlds of molecules** — no necessary physical/chemical link.
2. **A set of adaptors** — a third molecule bridging the two worlds.
3. **Arbitrariness of the coding rules** — conventional, not dictated by law;
   the rules could in principle be otherwise.

A paper "qualifies" only if **all three** are evidenced.

**Critical scoping decision.** The model cannot establish whether a code is
*objectively real*. It can only assess whether **the paper presents a claim and
supporting evidence** for each criterion. Every output is therefore phrased as
*"the paper argues/evidences criterion X,"* never *"criterion X is true."* This
keeps us honest, makes the task a grounded reading-comprehension problem (which a
31B model can do) rather than a scientific-adjudication problem (which it
cannot), and satisfies project rule 1 (no hallucinated claims).

**Two levels of output:**
- **Per paper** — the unit the model reads. A code has 1..n papers.
- **Per code** — the aggregate. A code is "supported" if *any* of its papers
  evidences all three criteria; "partial"/"unsupported" otherwise.

## 2. Data inventory & joins

- **471 PDFs** in `pdfs/` (2.1 GB, gitignored). These are the OA-retrievable
  subset of 2240 unique references; the other ~1769 are paywalled
  (`failed_downloads.csv`) and out of scope until obtainable.
- **Join key:** reuse `download_pdfs.output_path_for(row, "pdfs")` to map each
  `biological_codes.csv` row → on-disk PDF path (DOI-slug or URL-slug filename).
  `extract_doi()` recovers the DOI for provenance. **No new join logic needed.**
- Coverage caveat to surface in every report: a verdict exists only for the ~21%
  we can read. A code whose only OA paper fails ≠ an unsupported code; it may
  have a paywalled paper that would pass. Report `(supported / total)` papers per
  code so the denominator is always visible.

## 3. Pipeline stages

```
PDF ──▶ (1) text extract ──▶ (2) segment ──▶ (3) context select
     ──▶ (4) per-criterion grounded judgment (LLM) ──▶ (5) grounding gate
     ──▶ (6) aggregate paper→code ──▶ (7) human-review queue ──▶ output
```

**(1) Text extraction.** Reuse `pdfplumber` (already a dependency). Born-digital
PDFs dominate; flag any PDF whose extracted text is < N chars as
*scanned/needs-OCR* and route to a review list rather than silently scoring empty
text. (Defer OCR — Tesseract — to a later phase; quantify how many are affected
first.)

**(2) Segmentation.** Split into labelled sections (abstract, intro,
methods, results, discussion/conclusion) via heading heuristics. The criteria
claim usually lives in **abstract + intro + discussion**; methods/results carry
the *adaptor* evidence (criterion 2).

**(3) Context selection — the binding constraint.** The local model's context is
**16k tokens** (`--ctx-size 16384`), and a full paper is 7k–20k tokens. We cannot
stuff a whole paper. **Tiered strategy:**
- *Tier A (cheap, default):* feed abstract + intro + conclusion (~3–4k tokens) in
  **one** call asking for all three criteria at once. Resolves the clear cases.
- *Tier B (escalation):* for any criterion returned `unclear`, run map-reduce
  over the remaining sections (or embedding-retrieved chunks most similar to that
  criterion's definition) and re-judge just that criterion.

This keeps the common case to ~1 LLM call/paper and spends compute only where the
paper is genuinely ambiguous.

**(4) Per-criterion grounded judgment.** One structured call returns, per
criterion, a strict JSON object:

```json
{
  "criterion": "adaptors",
  "verdict": "met | not_met | unclear",
  "confidence": 0.0,
  "evidence_quote": "verbatim span copied from the source text",
  "reasoning": "one or two sentences"
}
```

Run with **thinking mode ON** (reasoning → `reasoning_content`, kept out of the
JSON). Few-shot the prompt with two anchors: the **genetic code** (canonical
all-three-met positive) and one clear negative (e.g. a paper describing ordinary
deterministic enzyme specificity — criterion 3 fails). For the overall
"qualifies" decision, **self-consistency**: sample N=3 and majority-vote;
disagreement → `unclear` → human queue.

**(5) Grounding gate (automatable hallucination guard).** Reject any `met`
verdict whose `evidence_quote` is **not a verbatim substring** of the source text
(after whitespace normalisation). This is a cheap, deterministic check that
catches fabricated evidence with no model in the loop, and directly enforces
project rule 1. A `met` that fails the gate is downgraded to `unclear`.

**(6) Aggregation.** Paper-level: `qualifies = all three met`. Code-level: roll
up papers; record counts and the strongest supporting paper + its three quotes as
an **evidence dossier**.

**(7) Human-review queue.** Everything `unclear`, every self-consistency
disagreement, every scanned-PDF, and a random audit sample of confident verdicts.

**Output:** `criteria_verdicts.csv` (one row per paper: code, DOI, three
verdicts+confidences+quotes, overall) and `code_support.json` (per-code dossier).

## 4. Model & infrastructure

- **Model:** local Gemma-4-31B-it Q5_K_M + MTP on asushimu's 3090 Ti, served by
  llama-server, OpenAI-compatible at `http://asushimu:11434/v1`, alias
  `gemma-4-31b`. Endpoint reachable (health 200). Sampling per the validated
  thinking recipe (temp 0.6 / top_p 0.95 / top_k 20).
- **Constraints that shape the design:** 16k context (Tier-A/B split above);
  `--parallel 1` (MTP is single-sequence — **no request batching**); structured
  output via `--jinja` tool-calling or a strict-JSON response with retry-on-parse.
- **Throughput estimate (single GPU, sequential):** decode ~60–72 tok/s, prefill
  ~1000 tok/s.
  - *Tier A only:* 471 papers × 1 call × ~1.5k output tokens ≈ 0.7M decode tokens
    → **~3 hours**; prefill ~1.6M tokens → ~0.5 h. Call it a **half-day batch**.
  - *With N=3 self-consistency + Tier-B escalation on ~30% of papers:* **~1 day**.
  - Run detached overnight (`nohup`/systemd-style), checkpoint per paper so it is
    resumable (mirror `download_pdfs.py`'s skip-if-done pattern).
- **Optional speed lever:** for a pure batch run, MTP's single-sequence mode is
  the bottleneck. A throughput-mode launcher (drop `--model-draft`/MTP, raise
  `--parallel`) trades the voice-latency tuning — irrelevant for offline batch —
  for concurrency. Note as an option; don't disturb the production launcher.
- **Embeddings (Tier B only):** if we add retrieval, serve a small embedder
  (e.g. `bge-small`/`SciNCL`-class) on a spare 1080 Ti or CPU — the 3090 Ti's
  llama-server is single-model. Skippable if section-heuristic Tier B suffices.

## 5. Model quality estimate requirements

**Can a 31B dense model do this?** Yes for criteria 1 & 2; *with reservations*
for criterion 3. The task is grounded extraction, which is squarely in a SOTA
31B's range — **but the three criteria are not equally hard:**

| Criterion | Difficulty | Why |
|---|---|---|
| 1 — two worlds | Low | Concrete, near-explicit; named molecule classes. |
| 2 — adaptors | Low–Med | Concrete but sometimes implicit; needs methods/results. |
| 3 — arbitrariness | **High** | Subtle, theory-laden, *contested even among experts* (the society's own text notes "streams of objections"). The model's weak point. |

**Requirement: treat the model as a triage/assistant, not the arbiter — and
prove it clears a bar before trusting any batch output.** Concretely:

1. **Gold set.** Human-label **~50 papers** drawn from the 471 (stratified:
   clear-positive, clear-negative, near-miss), double-annotated where possible,
   with per-criterion verdict + the human's own evidence quote. Anchor it with the
   genetic code (must score all-three-met) and a known non-code.
2. **Acceptance thresholds (gate before any full run is believed):**
   - Per-criterion agreement vs human **Cohen's κ ≥ 0.6** (substantial) for
     criteria 1 & 2.
   - Criterion 3: if κ < 0.6, it ships **advisory only** — the overall
     "qualifies" flag is then *human-confirmed*, model-proposed.
   - Overall binary ("qualifies") **macro-F1 ≥ 0.80** on the gold set.
   - **Grounding validity = 100%** — every `met` quote must verbatim-match source
     (a hard correctness gate, not a soft metric; §3 stage 5 enforces it).
   - **Calibration:** confidence binned into deciles; high-confidence
     (`≥0.8`) verdicts must be *right* ≥90% of the time, or confidence is
     reported as rank-only and not used for auto-accept.
3. **Reliability probes:** self-consistency agreement rate across N=3 samples
   (low agreement on a paper = intrinsic ambiguity → human queue), and a
   sensitivity check that shuffling few-shot order doesn't flip verdicts.
4. **Failure-mode budget.** We expect the model to over-call criterion 3 (label
   ordinary specificity as "arbitrary"). The gold set must contain such hard
   negatives specifically to measure this; if precision on criterion 3 is poor,
   criterion 3 stays advisory and the whole "qualifies" decision routes to humans.

**Net:** the 31B is good enough to *do the reading and the bookkeeping at scale*
and to confidently settle criteria 1 & 2; the arbitrariness judgment must be
validated against humans and, if it doesn't clear κ ≥ 0.6, remain a
human-confirmed step. The pipeline's value is reducing 471 (eventually 2240)
papers to a ranked, evidence-quoted queue — not replacing expert judgment on the
hard criterion.

## 6. Reuse & TDD test plan

**Reuse:** `download_pdfs.output_path_for` / `extract_doi` (code↔PDF join);
`pdfplumber` + patterns from `extract_csv.py` (text extraction); the
resumable-cache pattern from `download_pdfs.py`.

**TDD (project rule 1 — failing test first), all offline with the LLM
monkeypatched:**
- PDF→text and section segmentation on a tiny fixture PDF.
- Tier-A context builder stays under a token budget.
- Strict-JSON response parser: valid, malformed (retry), missing-field.
- **Grounding gate:** a fabricated quote → `met` downgraded to `unclear`; a real
  substring → preserved.
- Self-consistency majority vote, incl. the 3-way-split → `unclear` path.
- Paper→code aggregation arithmetic and the `(supported/total)` denominator.
- A live-LLM integration test gated behind an env flag (à la `LLM_TEST=1`),
  excluded from the offline suite.

## 7. Risks

- **Coverage bias** — only 21% readable; never present code-level verdicts
  without the denominator.
- **Criterion-3 over-calling** — the central quality risk; §5 gold set + advisory
  fallback contain it.
- **Extraction noise** — multi-column/scanned PDFs; route low-yield extractions
  to review, don't score empty text.
- **Throughput** — single-sequence MTP; mitigated by Tier-A-first + overnight
  resumable batch.

## 8. Phased plan

1. **v0 — harness + gold set.** Build PDF→text→Tier-A→strict-JSON→grounding-gate
   on ~10 papers; hand-label the ~50-paper gold set; measure κ/F1/calibration.
   **Decision gate:** does the model clear §5? This determines whether criterion 3
   is automated or advisory *before* any full run.
2. **v1 — full Tier-A batch** over all 471, resumable, with the human-review
   queue and per-code dossiers. Spot-audit confident verdicts.
3. **v2 — Tier-B escalation** (map-reduce / retrieval) for the `unclear` residue;
   optional OCR for scanned PDFs; feed verified positives to the cross-discipline
   system as its positive set.

---

*Open questions for you:* (a) Who annotates the gold set — you, or model-proposes/
you-confirm? (b) Is criterion 3 acceptable as advisory-only if it can't clear
κ ≥ 0.6, or is a fully-automated verdict a hard requirement? (c) Score only the
471 now, or wait for better PDF coverage (institutional network) first?
