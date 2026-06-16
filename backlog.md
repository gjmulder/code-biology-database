# Backlog

## Next runs (planned)
1. **Re-run the topic-subtraction A/B once Run 6 persists** (within-topic centring,
   idea #1 below). Read-only spike `/var/tmp/topic_subtract_spike.py` is written and
   tested: it builds the production geometry (μ, axes, strength=0.5, k=0), then per
   chunk subtracts that chunk's dominant-topic centroid direction (centred, unit) with
   strength λ ∈ {0, .25, .5, .75, 1.0} and Spearman-correlates `e` vs the continuous
   `verdicts.graded`. **First run (2026-06-16, 102 neuro top-4 graded papers) was
   inconclusive by design — range-restricted** (concrete criteria nearly all `not_met`
   on the neuro stratum): adaptors flat (ρ ≈ −0.06, no λ effect), two_worlds flat
   (≈ −0.03), arbitrariness the only mover (+0.140 → +0.198 monotonic to λ=1.0) but
   resting on **~2 positives among 102** → too fragile to trust (cf. [[dont-overfit-synthetic-verdicts]]).
   The clean test needs variance on two_worlds/adaptors, i.e. the 117 met-tail "rest"
   papers Run 6 is judging. **Decision rule:** if λ lifts adaptors/two_worlds there →
   implement in `embed_score.py` behind a flag (e.g. `--topic-strength`), TDD, gated on
   `--run` so baseline is untouched; if it again only nudges arbitrariness → drop it.
   Judge ρ, not `within` (the latter improves cosmetically without improving ranks).
2. **More Gemma 4 smoke runs.** A few additional smoke/pilot passes on the local
   free Gemma-4-31B judge (graded per-chunk, domain-general criteria — CLAUDE.md §9)
   to confirm gradation materialises beyond the neuro top-4 before committing paid
   spend. Free/offline GPU; checkpoint per-chunk JSONL (never deleted).
3. **Full three-criteria run on DeepSeek V4 Pro.** Once the prompts are validated,
   send **all three** criteria for **all papers** to paid **DeepSeek V4 Pro** via
   OpenRouter (CLAUDE.md §6; input $0.435/1M, output $0.87/1M — total cost TBD).
   Spend-safety: checkpoint to `sample_verdicts.jsonl` per paper (resumable APPEND)
   **before** MySQL persistence; never delete the checkpoint. Note: the code
   constant `criteria_judge.OPENROUTER_MODEL` still points at Nemotron — flip it
   (TDD: update `test_criteria_judge.py` first) before this run.


## Ideas / notes

* Chunk dynamically: Create overlapping sliding windows of 3 to 5 paragraphs (roughly 500–1000 tokens). Advance the window by only one paragraph at a time.

* An author might spend five paragraphs objectively summarizing a concept (yielding 5 positive hits) just to thoroughly debunk it in one concluding paragraph (1 negative hit). A raw sum (+4) results in a false positive.

* To resolve conflicting evidence accurately, consider these upgrades instead of a raw sum:

** Weighted Sum: Have the LLM rate the strength of each extracted claim (e.g., +3 for a core thesis statement, +1 for a passing mention, -3 for an explicit rejection). Sum the weights.

** Positional Weighting: Give override power or higher multipliers to evidence found at the end of the document (Discussion/Conclusion), as it represents the author's final stance rather than the literature review.

** The Meta-Judge (Recommended): If a paper yields conflicting chunks, concatenate only the extracted verbatim quotes. Run one final, cheap LLM pass asking: "Given this specific conflicting evidence, what is the author's final stance?"

* Three uses, ranked by payoff for this project:

  1. Within-topic centring / contrast (the real embedding lever). Instead of one global μ, score each paper against its own topic's centroid — "is this more two-worlds-y than typical
  papers of its kind?" That partials out topicality far more precisely than the single global first-PC you use now, and it directly attacks the wide within (0.51–0.68) that §8 names as
  the actual constraint. Related: topic-balanced prototype construction. §8 notes pos/neg poles are "topically collinear by construction" — that's why the poles overlap. Building matched
  pos/neg pairs that span the same topics would force the contrast axis to point at the criterion rather than shared topic vocabulary. This is the one path that could widen poles, and
  it's label-free.
  2. Stratified diagnosis. Compute ρ(e, verdict) per topic. This tells you whether arbitrariness is hopeless everywhere or just collapses outside a few topics — turning the current single
  weak ρ into an actionable map of where the axis works.
  3. Gold-set stratification (highest leverage overall). §6/§8 and your memory all conclude the binding constraint is label quality, not the embedding, and the live next step is a
  re-tuned judge + gold validation. Topic categories let you sample that gold set across topics so you don't validate only on Genetic Code papers, and ensure arbitrariness positives
  aren't all from one niche. This serves the constraint the project actually has.

