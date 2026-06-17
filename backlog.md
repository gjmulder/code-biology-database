# Backlog

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

