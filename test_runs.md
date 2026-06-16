# test_runs.md — chronological run log

Historical results from prior measurement runs, moved out of `CLAUDE.md` to keep that file a
tight statement of *current* design and state. This file is the **accurate record of what was
run, when, and what it showed**. CLAUDE.md cites the live decisions these runs produced.

**Note on retired granularities.** The embedding axis was originally embedded at three
chunking granularities — **full** (whole capped paper), **abstract** (abstract only), and
**chunk** (8192-token windows, 50% overlap, max-pooled). The project now focuses on **chunk**
for both the embedding axis and the per-chunk LLM judge (it edged out full/abstract on every
criterion — see Run 2 below). The full/abstract columns survive in the DB and in this log for
provenance, but are **no longer the working axis**.

---

## Run 0 — double-cosine baseline (under-discriminated)

The first scoring `e = cos(paper, POS) − cos(paper, NEG)` under-discriminated: a topicality
halo plus decoder-only-embedder anisotropy meant in-register text sat in a narrow cone. This
motivated the four space-level levers (CLAUDE.md §4) — the fix was space-level, not
prompt-level.

---

## Run 1 — corpus-scale lever sweep (2026-06-14, `sweep_levers.py`, free/offline, n=219)

The `(whiten-k, shared-strength)` grid `k∈{0,1,2,4,8,16} × strength∈{0,.25,.5,.75,1}` was
rescored from the persisted vectors and ρ(e, verdict) tabulated per method × criterion (no
GPU/spend/DB-write; the `(0, 0.5)` cell reproduces Run 2 exactly, proving the same path as live
`--recompute`). Findings:
1. **Whitening is a dead end at corpus scale, not just at n=20** — `k≥1` collapses the two
   concrete criteria (k=2 drops two_worlds +0.41→+0.05); only the 2-positive `arbitrariness`
   mildly prefers high k, which is noise.
2. `strength=0.5` is near-optimal for two_worlds; only `adaptors` wanted lower (s=0 → +0.343
   vs +0.310), inside the overfit margin.
3. No single cell wins all three, and **per-criterion argmax selection was rejected as
   overfitting** 217 *synthetic, unreliable* labels (esp. arbitrariness's 2 positives).

**Decision (live in CLAUDE.md §4): keep `k=0, strength=0.5`.** The binding constraint is the
model + pole separation (`within` still 0.51–0.68), not the levers. `sweep_levers.py` is kept
as the re-runnable diagnostic for a future model's vectors.

---

## Run 2 — first measurable ρ (2026-06-14)

The embedding corpus is **219 papers** (after dropping the in-corpus self-reference).
Previously only the 10-paper seed carried verdicts, so ρ was driven by one `met` paper and
`arbitrariness` had **zero positives → undefined**. Backfilling LLM verdicts for the whole
corpus (`judge_corpus.py`; **217 / 219** judged, 2 failed PDF extraction, ignored) gave real
variation in every criterion, so **ρ(e, verdict_ordinal) became measurable for the first
time** — the prior "honest gap."

Verdict distribution (217 labelled):

| criterion | met | unclear | not_met |
|---|---|---|---|
| two_worlds    | 17 | 9  | 191 |
| adaptors      | 12 | 12 | 193 |
| arbitrariness | 2  | 17 | 198 |

Spearman ρ(e, verdict_ordinal):

| criterion | full | abstract | chunk |
|---|---|---|---|
| two_worlds    | +0.397 | +0.389 | +0.409 |
| adaptors      | +0.265 | +0.293 | +0.310 |
| arbitrariness | +0.123 | +0.081 | +0.149 |

**Read:** the embedding axis tracks the verdict direction positively on all three criteria —
strongest on `two_worlds` (most concrete), weakest on `arbitrariness` (subtlest, still only 2
positives → directional, not precise). **`chunk` edges out `full`/`abstract` everywhere** —
the basis for retiring the other two granularities. Pole widths still overlap (`within`:
`arbitrariness` best-separated), so **ranks are trustworthy, absolute magnitudes less so**.

### Per-topic ρ (2026-06-15) — stratified by nearest scientometric topic
Stratifying ρ by dominant topic (CLAUDE.md §2.1; `assign_topics.py` → `report.md`) holds the
topicality halo fixed, so a positive within-topic ρ is stronger evidence than the pooled ρ
above. The largest strata are the neuro/metaphorical topics (Cognitive Signal n=39, Histonic
Code 26, Neural Circuits 21), where the concrete criteria are flat `not_met` (ρ frequently
`n/a` — no verdict variation in-stratum); only `arbitrariness` varies there (e.g. Regulatory
Code +0.52, Cognitive Signal +0.36). The molecular "met" codes sit in the **low-frequency
tail**. **Diagnostic only** — strata are small (≥10 shown) and the verdicts synthetic: read
direction, not magnitude.

---

## Run 3 — embedding-model swap shelved (2026-06-14)

The run-keyed schema (CLAUDE.md §3) was built to host a head-to-head against a second
embedding model, **gte-Qwen2-7B-instruct** (Q8_0 GGUF via a transient llama-server). The
runner (`run_gte_embed.py`, `start_llama_embed.sh`, driver `--engine llamacpp`) is **complete
and tested but the GPU pass was not run** — shelved deliberately:
- `within` (+0.68/+0.63/+0.52) indicts **pole geometry**, and gte is the *same* decoder-only
  last-token architecture as harrier — the documented source of the anisotropy/topicality halo
  — so the swap is unlikely to widen the poles. Kept as a tested artifact; the `run` column
  means a future pass is non-destructive if ever wanted.
- The remaining label-free lever is **prototype/pole quality** (sharper contrastive pos/neg
  passages, esp. arbitrariness). Iterable cheaply: edit `prototypes.json` → re-embed only the
  poles → offline recompute `within`/`e`. **Expected gain is small** — the prototypes are
  already corpus-mined and the pos/neg passages are topically collinear by construction.
- **Honest read:** the levers are exhausted, the model swap is judged unpromising, and
  prototype edits are expected to move things only marginally. The genuine constraint is now
  **label quality, not the embedding** — the verdicts are unreliable and arbitrariness has
  only 2 positives, so ρ can't even adjudicate fine gains. The next real step is a re-tuned
  judge + a gold-set validation, not more embedding-side tuning.

---

## Run 4 — judge redesign pilot, top-4 neuro topics, molecular criteria (2026-06-16)

The graded per-chunk judge (CLAUDE.md §9) was piloted on the **top-4 topics = 102 papers**
(neuro/metaphorical strata `[11,18,19,13]`); checkpoint `pilot_verdicts.jsonl`
(= `pilot_verdicts_molecular.jsonl`). 1330 chunk cells judged (2 dropped to malformed JSON,
clean per-cell isolation). Categorical (old→new): two_worlds met 0→0 / unclear 1→0 / not_met
101→102; adaptors met 1→**3** / 1→0 / 100→99; arbitrariness met 1→**2** / 5→0 / 96→100. Graded
spread: two_worlds flat 0.0 (std 0); adaptors std 0.12; arbitrariness std 0.11. Pooled
ρ(graded, e) ≈ ρ(cat, e), weak-positive where defined (two_worlds `n/a`, no variation).

**Read (honest):**
- **The two design risks did not materialise.** Halo-injection guard held — **7/7 positive
  cells are quote-grounded** (e.g. histone "reader proteins (adapters)", lncRNA "address
  code"), zero ungrounded false positives. The new judge is *more* skeptical than the old: it
  collapsed the wishy-washy `unclear` bucket while surfacing a few *more* genuinely-grounded
  `met`. Gradation moved off the dead `confidence` field onto `agreement`, as intended.
- **Gradation is inconclusive in *this* stratum by design.** The top-4 are the neuro topics
  Run 2 already flags as flat `not_met` on concrete criteria — little real positive signal to
  grade. The graded axis took distinct levels (+0.5/+1.0) exactly where warranted; the
  distribution is dominated by legitimate 0.0s.
- **This pilot exposed a prompt bug → Run 5.** `CRITERIA_DEFS` was molecular-specific, so every
  non-molecular paper was mechanically rejected (two_worlds **0** positives, 310/443 reasoning
  cells citing "not *molecular* worlds"; adaptors positives only in the histonic molecular
  topic). Only `arbitrariness` was already abstract and worked across domains — it became the
  template for the domain-general rewrite (CLAUDE.md §9.1).

---

## Run 5 — domain-general criteria re-pilot (2026-06-16, IN PROGRESS)

Re-pilot on the **same top-4 neuro topics** with the domain-general `CRITERIA_DEFS`
(CLAUDE.md §9.1); fresh checkpoint `pilot_verdicts_domaingen.jsonl`, baseline preserved in
`pilot_verdicts_molecular.jsonl`. **Pass criteria:** two_worlds gains grounded positives in
neural/audio papers (was 0); adaptors finds mediator-based positives beyond the histonic
topic; every positive still quote-grounded.

_Results pending — fill in on completion (categorical old→new per criterion, grounded-positive
counts, spot-check that each positive's quote is verbatim in its chunk)._
