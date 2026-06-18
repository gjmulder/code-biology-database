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

## Run 5 — domain-general criteria re-pilot (2026-06-16)

Re-pilot on the **same top-4 neuro topics** `[11, 18, 19, 13]` (102 papers, 1341 chunk cells)
with the domain-general `CRITERIA_DEFS` (CLAUDE.md §9.1); checkpoint
`pilot_verdicts_domaingen.jsonl`, molecular baseline preserved in `pilot_verdicts_molecular.jsonl`.
Persisted run-agnostically to `chunk_verdicts` + `verdicts` (overwriting the molecular pilot rows
for these 102 papers; the molecular baseline survives in the JSONL copy), and this run carried the
real `prompt_hash` / `prompt_registry` schema migration (CLAUDE.md §3) — `mysqldump` taken first
per §7.

**Operational note — `--predict` truncation + recovery.** The first pass left ~40 chunk cells
unparsed: the pilot server ran `--predict 2048`, and Gemma's reasoning preamble *shares* that
budget (thinking is on → `reasoning_content` decoded before the JSON), so on dense chunks the JSON
was cut off mid-object → unparseable, isolated per cell (logged + skipped, never checkpointed).
Fix: raise `start_llama_pilot.sh` to `--predict 4096` (still fits 16384/slot: ~9k chunk + ~2k
scaffold + 4096 out). A recovery re-run (resume keyed on the `(pdf_path, chunk_idx, criterion)`
triple) re-judged exactly those cells — **0 skips, 0 reconnects, VRAM steady 23.97/24.6 GB, no
OOM**. The post-pilot persist also now survives an idle-connection drop via `db.run_with_reconnect`
(the `wait_timeout` 2013 fix; this run needed 0 retries).

Categorical (molecular → domain-general); `unclear` is 0 in both throughout:

| criterion | MOL met / not_met | DOMAIN-GEN met / not_met | positive chunk cells (mol→dg) |
|---|---|---|---|
| two_worlds    | 0 / 102 | **8** / 94  | 0 → 16 |
| adaptors      | 3 / 99  | **41** / 61 | 5 → 68 |
| arbitrariness | 2 / 100 | **3** / 99  | 2 → 4  |

**Read — both pass criteria met, grounding intact.**
- **two_worlds 0 → 8 met.** The molecular-bias rejection ("not *molecular* worlds") is gone; the
  judge now recognises domain two-world mappings (stimulus↔spikes, sound↔percept) in neural/audio
  papers — exactly the stratum it mechanically zeroed in Run 4. Pass criterion met.
- **adaptors 3 → 41 met.** The mediator generalization (Major 2025; CLAUDE.md §9.1) surfaces
  nervous-system / neural-circuit / receptor mediators across the neuro topics, far beyond Run 4's
  histonic-only 3. Pass criterion met. (This is the largest swing — watch for over-broadening on
  the molecular tail / corpus-wide; the grounding gate is the guardrail.)
- **arbitrariness 2 → 3 met.** Already domain-general in Run 4, so a small change as expected — it
  was the template, not the target.
- **Grounding held perfectly.** All **88** positive chunk cells (agreement ≥ +0.5) carry a
  non-empty `evidence_quote`; **0** are flagged `grounding_failed` and **0** have an empty quote —
  every positive is verbatim-grounded by construction (the gate writes post-gate records). A
  3-cell independent re-check (re-tokenise → `chunk_text.reproduce_chunks` → substring) confirmed
  the quote is verbatim in its chunk in 3/3 (two_worlds, adaptors, arbitrariness).

**Next (backlog):** confirm gradation on the molecular "met" tail *outside* the neuro top-4 before
any corpus-wide or paid (DeepSeek V4 Pro) all-criteria run.

---

## Run 6 — molecular "met"-tail pilot, rest of corpus (2026-06-17, COMPLETE)

Acting on the Run 5 backlog: the graded domain-general judge over the **117 papers OUTSIDE the
neuro top-4** — the complement of Runs 4–5, i.e. every paper whose dominant topic is *not* in
`[11, 18, 19, 13]`. This is the molecular "met" tail where both axes should carry real signal
(Run 2's pooled ρ was driven by these codes), so it is the first chance to compare `e` against
the new judge where the verdict actually varies. **With this run the entire 219-paper corpus
now carries domain-general verdicts** (102 neuro from Run 5's `pilot_verdicts_domaingen.jsonl`
+ 117 rest from this run's `pilot_verdicts_rest.jsonl`), so the combined analysis below is the
first corpus-wide read of the new judge.

- **Driver:** `judge_pilot.py --rest --top 4 --tokenizer harrier_tokenizer --checkpoint
  pilot_verdicts_rest.jsonl --workers 3` (the `--rest` selector = the complement of
  `select_pilot_papers`; tests in `tests/test_judge_pilot.py`). Free local Gemma on
  `start_llama_pilot.sh`; prod voice agent offline for the run.
- **Clean finish.** All **117/117** papers judged (`pilot_verdicts_rest.jsonl` = 1656 chunk
  cells over 117 unique `pdf_path`); persist line landed: **1656 `chunk_verdicts` + 351
  `verdicts`** (117×3), restricted to the 117 `pids` so the 102 neuro rows are untouched. Exactly
  **1** chunk cell skipped to malformed model JSON (`10.4238_3x4qm566.pdf` chunk 0 adaptors),
  isolated per-cell and never checkpointed — the per-paper failure isolation worked as designed.

### Run-6 rest tail only (117 papers)
Categorical (`unclear` = 0 throughout, as in Runs 4–5):

| criterion | met | not_met | graded mean / std |
|---|---|---|---|
| two_worlds    | 8  | 109 | +0.030 / 0.247 |
| adaptors      | 43 | 74  | +0.282 / 0.400 |
| arbitrariness | 4  | 113 | +0.004 / 0.139 |

**Gradation materialised on the molecular tail — Run 6's gate criterion is met.** `adaptors`
spreads hard (20×`+0.5`, 23×`+1.0`, std 0.40); `two_worlds` carries real levels too
(±1.0 present, std 0.25). `arbitrariness` stays thin (4 met) — the subtlest criterion, as always.

### Combined corpus (all 219, domain-general)
Categorical:

| criterion | met | not_met | graded mean / std |
|---|---|---|---|
| two_worlds    | 16 | 203 | +0.043 / 0.232 |
| adaptors      | 84 | 135 | +0.269 / 0.389 |
| arbitrariness | 7  | 212 | +0.011 / 0.130 |

Pooled ρ(e, verdict) over all 219 — **chunk axis, categorical / graded**, vs Run 2's molecular
chunk ρ for reference:

| criterion | Run 2 chunk (molecular judge) | Run 6 ALL-219 cat / grad |
|---|---|---|
| two_worlds    | +0.409 | +0.061 / −0.015 |
| adaptors      | +0.310 | +0.078 / +0.139 |
| arbitrariness | +0.149 | +0.136 / +0.188 |

**Read — the corpus-wide ρ collapse is an axis-mismatch artefact, not signal loss.** The
domain-general judge spreads "met" roughly *evenly* across both strata (two_worlds 8 neuro / 8
molecular; adaptors 41 / 43; arbitrariness 3 / 4), but `e` is corpus-contrastive against
**molecular** prototype poles, so it is blind to the non-molecular "codes" the new judge now
recognises in the neuro strata (§9.1, Major 2025 mediator generalization). The two axes have
**diverged by design**: the judge went cross-domain, the embedding stayed molecular. The
per-stratum ρ proves it:

| criterion | molecular rest (117) cat / grad | neuro top-4 (102) cat / grad |
|---|---|---|
| two_worlds    | +0.137 / −0.004 | +0.005 / +0.009 |
| adaptors      | **+0.262 / +0.307** | **−0.140 / −0.087** |
| arbitrariness | **+0.185 / +0.252** | +0.031 / +0.033 |

- **Within the molecular tail — the only stratum where both axes measure the same thing — the
  agreement holds.** `adaptors` recovers Run 2's signal (+0.262 cat, **+0.307 graded** ≈ Run 2's
  +0.310), and `arbitrariness` *improves* on Run 2 (+0.149 → +0.185 cat / **+0.252 graded**) —
  the graded axis tracks `e` better than the categorical there, the first place gradation pays
  off. `two_worlds` is the exception (Run 2 +0.409 → +0.137 cat, graded ~0): the molecular
  two-world signal `e` keyed on is no longer what the broadened judge rewards.
- **Within the neuro stratum `e` is flat or reversed** (adaptors −0.140): the 41 neuro
  `adaptors`-met papers sit at low/average `e`, so pooling them with the molecular tail drags the
  all-219 ρ to ≈0. This is the dilution, not a regression.

**Implication.** ρ(e, domain-general verdict) is **no longer a clean corpus-wide validation
signal** — the molecular embedding poles and the cross-domain judge are measuring different
constructs (CLAUDE.md §1: the two axes are independent and reported side-by-side, neither
authoritative; this run is a concrete instance of them legitimately diverging). To compare them
apples-to-apples either (a) restrict the e-vs-judge ρ to the molecular tail (where it holds), or
(b) build domain-general prototype poles to match the broadened judge, or (c) accept the
divergence and report both axes side-by-side as designed. Embedding-side work is still judged
exhausted (§8); option (b) is the only embedding lever that would re-couple the axes, and is
untested.

**Next (backlog):** the gate is cleared (gradation confirmed on the molecular tail), but the
corpus-wide paid (DeepSeek V4 Pro) all-criteria run should be decided against the gold-set plan
(MEMORY: gold-set is the live next step), **not** against ρ(e, verdict) — which this run shows no
longer adjudicates the domain-general judge corpus-wide. `ranking_report.html` regeneration still
pending.

## Run 7 — domain-general prototype poles (2026-06-17, COMPLETE)

Executes Run 6's untested **option (b)** ("build domain-general prototype poles to match the
broadened judge") — the only remaining embedding-side lever that could re-couple the axes after
the §9.1 judge went cross-domain. `prototypes.json` rewritten (rev 2): each pole is one
domain-neutral definitional **anchor** plus five concrete instances spread across domains
(genetic, neural, semiotic/language, signalling, immune/olfactory), so the genetic code is one
instance among many and the diverse topicalities partially cancel in the mean — "neutrality by
balance, not abstraction" (a pure-abstraction rev 1 was rejected: decoder-only embedders
represent abstract relational prose weakly, which narrows the poles). Each criterion keeps its
**own** theoretical negative (two_worlds: one continuous system; adaptors: direct contact/no
mediator; arbitrariness: physically determined). Re-embedded poles + controls only
(`embed_independent.py --controls-only`, harrier on the 3090 Ti, ~1 min GPU); papers untouched,
`e` recomputed offline for all 219 from persisted vectors (k=0, strength=0.50). DB backed up
first (§7.7).

**Pole widths held — the rewrite did not degrade the poles** (the rev-1 worry didn't
materialise): `within` two_worlds **+0.636**, adaptors **+0.595**, arbitrariness **+0.591** —
all inside the prior molecular-pole 0.51–0.68 band.

**But corpus-wide ρ stayed flat — option (b) did NOT re-couple the axes.** Scored against the
Run-6 domain-general verdicts (all 219):

| criterion | chunk ρ, domain-general poles (Run 7) | chunk ρ, molecular poles (Run 6 corpus-wide) |
|---|---|---|
| two_worlds    | +0.078 | +0.061 |
| adaptors      | +0.040 | ≈flat |
| arbitrariness | +0.129 | — |

Essentially unchanged from the Run-6 collapse. Making `e` domain-general to match the
domain-general judge does **not** restore corpus-wide agreement: the synthetic verdicts spread
"met" across heterogeneous topics in a way no single contrastive axis tracks. Per-topic ρ (chunk)
still shows the axis working **within coherent strata** — `[0] Morphological Codes` two_worlds
**+0.463** / adaptors +0.244, `[2] Binding Code` adaptors **+0.342**, `[11] Cognitive Signal`
arbitrariness +0.216 — but `[19] Neural Circuits` runs negative and many strata are `n/a` (no
verdict variation), so it remains diagnostic-only.

**Conclusion.** Run 7 closes out option (b): the domain-general poles are well-formed *and*
design-consistent with the §9.1 judge, and are kept as the new `baseline`, but they confirm
rather than fix the §8 verdict — **the binding constraint is label quality, not the embedding
axis.** All three of Run 6's options are now spent on the embedding side; the real next step is
unchanged: the gold-set validation (MEMORY), not more axis tuning.

## Run 8 — AGREE-anchor ablation, paid DeepSeek (2026-06-17, COMPLETE)

Tests the §9 worry that the judge's molecular-genetic AGREE exemplar biases it toward marking
passages "met". `judge_pilot.py --agree-anchors {genetic,neural,neural-genetic}` swaps **only the
AGREE example's domain** — molecular genetic 1-shot (baseline, untagged) vs neural 1-shot
(`@neural-1shot`) vs neural+genetic 2-shot (`@neural-genetic-2shot`) — leaving the prompt
otherwise identical. Run paid on **DeepSeek V4 Pro** over the same **60-paper top-4 neuro set**
(`--limit 60`), each variant under its own judge tag + checkpoint so all three coexist with the
baseline (§3). Cost: neural-1shot $2.33, neural-genetic-2shot $2.05. 180 verdicts (60 papers × 3
criteria) per variant; comparison on the 180 shared `(code, pdf, criterion)` keys.

| criterion | mean graded (gen / n1 / ng2) | Δ mean (n1−gen / ng2−gen) | met (gen / n1 / ng2) |
|---|---|---|---|
| two_worlds    | +0.042 / +0.067 / +0.008 | +0.025 / −0.033 | 4 / 6 / 1 |
| adaptors      | +0.058 / +0.067 / +0.042 | +0.008 / −0.017 | 4 / 8 / 6 |
| arbitrariness | +0.017 / −0.017 / +0.008 | −0.033 / −0.008 | 3 / 1 / 1 |

Per-key change rate: **neural-1shot** moved graded on only 25/180 keys (13 up / 12 down — sign
balanced), 14 categorical flips (9 →met, 5 met→); **neural-genetic-2shot** moved 16/180 (6 up /
10 down), 9 flips (3 →met, 6 met→).

**Conclusion — negative result: the genetic anchor is not biasing the judge.** Swapping the AGREE
exemplar's domain moves ~9–14% of judgments with effect sizes ≤0.033 on the −1…+1 scale
(noise-level). The only directional hint is a weak **domain-matching** lift from the neural
**1-shot** anchor (net +2 met two_worlds, +4 adaptors on these neuro papers), but the **2-shot**
washes it out and trends slightly *more* skeptical — so there is no robust molecular halo. The
verdicts are robust to the exemplar's domain, which supports (does not threaten) label quality.
Variants kept non-destructively under their tags; baseline `deepseek/deepseek-v4-pro` unchanged.

## Run 9 — code-0 gold-positive calibration, paid DeepSeek (2026-06-18, COMPLETE)

A direct sanity check of the skeptical DeepSeek judge against the canonical positive: **does it
mark Barbieri's own defining texts as meeting all three criteria?** The two foundational *Code
Biology* papers (reserved **code 0**, CLAUDE.md §1b) are Barbieri's all-three-criteria exemplar,
so they are the closest thing to a gold positive available without a gold set. Topics assigned
offline (`assign_topics.py`), then judged on the same baseline genetic-anchor **DeepSeek V4 Pro**
as the 219-paper corpus via a new `judge_pilot.py --code 0` selector (`select_code_papers`, tests
in `tests/test_judge_pilot.py`) — papers under one `code_number`, ignoring the top-N strata.
7 chunks × 3 criteria; checkpoints `deepseek_verdicts.jsonl` (first run) + `deepseek_code0_regate.jsonl`
(re-gate after the ligature fix below). Cost $0.079 total ($0.061 + $0.018, ~99% prompt-cached).

Re-gated categorical / graded_max:

| paper | two_worlds | adaptors | arbitrariness | all three? |
|---|---|---|---|---|
| Introduction to Code Biology (2014) | met (+1.0) | met (+1.0) | not_met (0.0) | No |
| What Is Code Biology? (2018)        | not_met (0.0) | met (+1.0) | met (+1.0) | No |

**Read — the grounding gate, not the judge's reasoning, is the dominant factor.** DeepSeek's
*reasoning* reads both papers as exemplifying all three criteria — there is no substantive
disagreement with Barbieri. Every `not_met` is a **quoting** failure, not a judgment: the model
**reconstructs** its `evidence_quote` (stitching real, non-adjacent sentences — e.g. Barbieri's
"mapping between two independent worlds" definition spliced onto the genetic-code example that
doesn't follow it contiguously), and the verbatim grounding gate (CLAUDE.md §9) correctly refuses
to credit a spliced quote. On the 2018 two_worlds chunks the longest contiguous overlap with any
quote was 13 chars — genuine splicing, not a typographic artifact.

**Ligature-folding gate fix (label-quality, shipped this run).** Diagnosis of the *first* run's
2014→2018 cells found one cell defeated only by a `ﬁ`-ligature in the PDF source vs a plain-ASCII
quote — a real gate brittleness, not splicing. `criteria_judge._norm_ws` now NFKC-normalises
before the substring check, folding typographic ligatures (ﬁ→fi, ﬂ→fl, ﬃ→ffi, …) so a PDF
ligature can no longer false-negative an otherwise-verbatim quote (TDD, +3 offline tests, suite
287). The re-judge realised the fix on the persisted verdicts; it did **not** flip 2018 two_worlds
because that fresh stochastic sample spliced all four two_worlds quotes (the gate working as
designed, above).

**Conclusion.** The judge is *substantively* correct on the gold positive but the strict verbatim
gate withholds the categorical `met` whenever the model paraphrases/splices — sharpening the §6/§8
label-quality constraint into two faces: (a) gate brittleness to PDF artifacts (ligatures now
fixed; curly-quote folding is a similar cheap next fix), and (b) the model's tendency to
reconstruct quotes, which inflates `not_met`. A sentence-level / span-set verbatim check (accept a
quote assembled from verbatim sentences, reject true fabrication) is the substantive next gate
improvement — deferred, not chased here. No bearing on the corpus verdicts; the gold-set plan
remains the live next step (MEMORY).

## Run 10 — fuzzy grounding gate + strict-gate cost diagnostic (2026-06-18, gate shipped + free measurement)

Acting on Run 9's deferred improvement, the strict-verbatim grounding gate was replaced by a
**fuzzy** one (gold-set plan Phase 4.5 step 1, TDD). `criteria_judge.is_grounded(quote, source,
τ=0.85, L=15)` promotes the previously diagnostic-only `quote_coverage` (difflib
`SequenceMatcher` over spliced non-contiguous spans) to the gate: a quote grounds iff
**coverage ≥ 0.85 AND longest contiguous block ≥ min(15, len(quote))**. A new `_norm_fuzzy`
(superset of `_norm_ws`) folds smart quotes/dash variants and joins line-break hyphenation. This
admits whitespace/smart-quote/hyphenation/splice drift while still rejecting paraphrase and
fabrication; strict-verbatim is the `τ=1.0` special case. `grounding_gate` /
`graded_grounding_gate` rewired onto it (suite 304, fully offline).

**Free read-only diagnostic (no GPU, no spend) — how much was the strict gate costing?** Over
every stored `chunk_verdicts` cell at `agreement==0.0` with a non-empty `evidence_quote`, each
chunk was reproduced (`chunk_text.reproduce_chunks`, local harrier tokenizer) and the quote
classified: **A** passes strict (genuine 0.0), **B** fails strict but passes fuzzy (recoverable —
strict could only have zeroed it on grounding drift), **C** fails both (true paraphrase/fabrication).

| judge | zeroed-with-quote | A genuine 0.0 | **B recoverable** | C fabrication |
|---|---|---|---|---|
| DeepSeek V4 Pro | 73 | 5 | **67 (92%)** | 1 |
| Gemma-4-31b | 231 | 16 | **215 (93%)** | 0 |

**Read.** ~92–93% of strict-gate zeroings on quote-bearing cells were **formatting drift, not
hallucination** — only **1 true fabrication across 304 cells**. The criterion that lost the most
is **`adaptors`** (DeepSeek 25, Gemma 159 of the B cells), the molecular-fingerprint criterion the
molecular gold validation hinges on; B examples are unmistakably real evidence (e.g. *"genes for
tRNAs that decode both TAA and TAG were readily obtained"*, *"aminoacyl tRNA-synthetases use the
anticodon as a key identity element"*).

**Caveat (why B is an upper bound).** The pre-gate `agreement` is unrecoverable from storage — a B
cell *could* have been a genuine 0.0 that merely cited a real-but-drifted quote, in which case
fuzzy passing it through at 0.0 changes nothing. That irrecoverability is exactly why Phase 4.5
step 2 retains the raw pre-gate value (`raw_agreement`/`coverage`/`grounding_failed`) going
forward, making gate-threshold tuning offline-free (parity with the §4 embedding levers). But the
near-total dominance of B over C decisively justifies the fuzzy gate and predicts the gold
re-judge (Phase 6) will recover real positives — the `met` rate was suppressed by quote
formatting, not by the judge disagreeing with the evidence.

**Tooling (Phase 4.5 step 3, committed).** The diagnostic is now a tested, re-runnable script
`diagnose_gate_cost.py` (pure `classify_cell`/`tally` unit-tested offline; DB+tokenizer driver
read-only, no GPU/spend), replacing the ad-hoc snippet. Re-run confirms the table above exactly;
the per-criterion split of the **282 B cells** is `adaptors` 184, `two_worlds` 65,
`arbitrariness` 33 — `adaptors` (the molecular-fingerprint criterion the gold validation hinges
on) carrying the bulk, as expected.

## Run 11 — gold-set validation, paid DeepSeek (2026-06-18, COMPLETE)

The first adjudication of **both** synthetic axes against authority ground truth (the
Barbieri-anchored `gold_labels`, run-/judge-agnostic), not against each other. Gold set: **208
positives** (4 tier-1 code-0 seminal texts + 204 tier-2 DB-endorsed molecular-code papers) and
**240 soft negatives** (non-molecular by the Phase-1 topic allowlist, absent from every molecular
code's reference list — in practice **neural-heavy**, the Run 4/6 top-4 neuro strata). The whole
gold set was **re-judged fresh under the Run 10 fuzzy gate** by paid DeepSeek V4 Pro to a clean
per-chunk checkpoint (positives + negatives), then `make_gold_report.py` joined gold polarity ×
`e` × categorical verdict per criterion.

**Spend (measured).** Positives 207 papers / 868→2692 chunk_verdicts / **$7.62**; negatives 240
papers / 3479 chunk_verdicts / **$9.87** → **$17.48 total**, ~36% prefix-cached, in line with the
$0.0088/chunk smoke estimate. Both persisted under `deepseek/deepseek-v4-pro`.

**Embedding axis — does gold+ outrank gold−?** No — it **reverses**:

| criterion | AUC | ρ(e, gold) | n+ | n− |
|---|---|---|---|---|
| two_worlds | **0.189** | −0.536 | 207 | 240 |
| adaptors | 0.368 | −0.228 | 207 | 240 |
| arbitrariness | 0.352 | −0.255 | 207 | 240 |

All AUC < 0.5, all ρ negative. Spot-check confirms it is **not a sign/join bug**: the top-`e`
two_worlds papers are all **neural** soft-negatives (jneurosci/eneuro/pcbi), the bottom-`e` are
**molecular** gold positives (rna/pnas/nature-comms); mean `e` two_worlds gold+ = −0.015 vs
gold− = +0.031. **Read:** the domain-general poles (Run 7, prototypes rev 2) match the neural
"stimulus↔response / two-worlds" register **more** than dense molecular wet-lab prose, so against a
molecular-authority gold the axis is anti-correlated. This is the §5/§8 "two axes measure different
constructs" divergence **quantified against authority** — and a hard confound of the soft-negative
pool composition (neural-heavy positives-vs-molecular-positives), not a clean "not-a-code" contrast.
A topic-matched or hard-negative test is needed to separate genre/register from code-demonstration.

**Judge axis — verdict (`met`=positive) vs gold polarity** (n=447 each):

| criterion | precision | recall | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|
| two_worlds | 0.474 | 0.130 | 0.205 | 27 | 30 | 180 | 210 |
| adaptors | **0.688** | 0.213 | 0.325 | 44 | 20 | 163 | 220 |
| arbitrariness | 0.553 | 0.101 | 0.171 | 21 | 17 | 186 | 223 |

The fuzzy-gated DeepSeek judge is **precise-ish but very conservative**: `adaptors` precision 0.69
(the molecular-fingerprint criterion, strongest as in §5), but recall 0.10–0.21 everywhere — it
marks most tier-2 positives **not_met**. This is the **coarse-label** signature flagged at gold
construction: tier-2 = "paper in the supporting literature of an endorsed code," many of which
(reviews, methods, single-facet papers) genuinely don't *individually* demonstrate the criterion.
Low recall is partly the label, partly a skeptical judge; the 17–30 FP are mostly the §9.1
domain-general judge legitimately marking neural soft-negatives met (not necessarily errors).

**Takeaways.** (1) The embedding axis does **not** recover Barbieri's molecular-authority
distinction — corroborating §8 that `e` keys on topicality/register, not code-demonstration.
(2) The judge is the better-aligned axis (positive precision, esp. adaptors) but conservative
against coarse tier-2 labels. (3) The soft-negative pool's neural composition is a live confound;
hard negatives (Phase 4 exclusions) and a per-paper tier-2 audit are the next quality levers — not
more axis tuning.

**Phase 3 tier-1 upgrade — confirmed negative result (no spend, no mutation).** The plan's
authority upgrade (promote a tier-2 paper to tier-1 when Barbieri/Major *also* cite it) yields
**0 valid promotions**. `cited_signatures` extracts **427** `(surname, year)` signatures from 5
seminal texts (*The Organic Codes* 2003 is image-based/truncated — 8.7k chars, no extractable
bibliography — the one real data loss), but the intersection with the 200 tier-2 papers with
parseable signatures is **empty**, and surname-only overlap is just 10/182. Across the *entire*
embedded corpus only **9** papers match cited signatures — 4 are the code-0 texts self-citing, the
other 5 coincidental (`wang 2008`, `brandon 2009`, …), 4 of which are soft negatives. **Cause:**
Barbieri's personal bibliography (foundational/conceptual) and the ISCB `biological_codes.csv` code
list (recent niche molecular-demonstration papers, tier-2 median year 2020) are **near-disjoint
corpora**. Relaxing the match (surname-only / year ±1) would only admit the coincidental collisions
and corrupt positives, so it was rejected. **tier-1 stays = the 4 code-0 seminal texts** (the
conservative authority root); the tooling (`build_gold_set.py cite`) is correct, the data doesn't
support expansion. (Aside: code-367/splicing references `dhir 2010`/`wang 2008` landed as soft
negatives — a possible gold mis-stratification to revisit separately.)
