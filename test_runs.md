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
