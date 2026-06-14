# Independent Embedding Analysis vs LLM Verdicts

Sample: 10 preselected papers. Model: `/data/vllm/harrier-oss-v1-27b` (5376-dim, 4-bit=True). Source of record: MySQL `embedding_scores` (keyed on code id).

`e = cos(paper, POS_prototype) − cos(paper, NEG_prototype)` — higher means the text reads as *arguing* the criterion. This embedding axis is **independent**: it is reported beside the LLM verdict and never overrides it (plan decision 0). Each paper is scored three ways as separate documents — **full** text, **abstract** only, and **chunk** (8192-token windows @50% overlap, max-pooled) — to test which granularity best tracks the verdict (chunk size 8192, overlap 4096).

## Per-paper verdicts (criteria_judge) + embedding columns

### Criterion: `two_worlds`

| code | paper | verdict | conf | e_full | e_abstract | e_chunk |
| --- | --- | --- | --- | --- | --- | --- |
| 21 | 10.1007_s00441-015-2202-z.pdf | not_met | 1.00 | -0.006 | +0.007 | -0.003 |
| 162 | www.oncotarget.com_article_435_text.pdf | not_met | 1.00 | +0.024 | +0.025 | +0.024 |
| 233 | www.oncotarget.com_article_5108.pdf | not_met | 1.00 | +0.000 | -0.006 | +0.000 |
| 247 | 10.1007_s11914-023-00846-y.pdf | not_met | 1.00 | -0.003 | -0.015 | -0.004 |
| 248 | 10.1038_s41586-026-10267-3.pdf | not_met | 1.00 | +0.011 | +0.009 | +0.021 |
| 259 | hdl.handle.net_1773_36496.pdf | not_met | 1.00 | -0.001 | +0.008 | +0.010 |
| 275 | www.jneurosci.org_content_33_25_10568.lo… | not_met | 1.00 | +0.020 | +0.024 | +0.022 |
| 375 | 10.3389_fpls.2021.640919_full.pdf | not_met | 1.00 | -0.003 | +0.005 | +0.008 |
| 424 | 10.1371_journal.pcbi.1002536.pdf | not_met | 1.00 | +0.015 | +0.005 | +0.016 |
| 428 | 10.1371_journal.pgen.1003076.pdf | met | 1.00 | +0.045 | +0.062 | +0.053 |

### Criterion: `adaptors`

| code | paper | verdict | conf | e_full | e_abstract | e_chunk |
| --- | --- | --- | --- | --- | --- | --- |
| 21 | 10.1007_s00441-015-2202-z.pdf | not_met | 1.00 | +0.015 | +0.021 | +0.020 |
| 162 | www.oncotarget.com_article_435_text.pdf | not_met | 1.00 | -0.007 | -0.003 | -0.007 |
| 233 | www.oncotarget.com_article_5108.pdf | not_met | 1.00 | -0.006 | -0.011 | -0.006 |
| 247 | 10.1007_s11914-023-00846-y.pdf | not_met | 1.00 | -0.005 | +0.003 | -0.001 |
| 248 | 10.1038_s41586-026-10267-3.pdf | not_met | 1.00 | +0.020 | +0.020 | +0.022 |
| 259 | hdl.handle.net_1773_36496.pdf | not_met | 1.00 | -0.002 | +0.010 | +0.010 |
| 275 | www.jneurosci.org_content_33_25_10568.lo… | not_met | 1.00 | +0.018 | +0.020 | +0.024 |
| 375 | 10.3389_fpls.2021.640919_full.pdf | not_met | 1.00 | -0.010 | -0.001 | -0.007 |
| 424 | 10.1371_journal.pcbi.1002536.pdf | not_met | 1.00 | -0.008 | -0.012 | -0.004 |
| 428 | 10.1371_journal.pgen.1003076.pdf | met | 1.00 | +0.058 | +0.066 | +0.058 |

### Criterion: `arbitrariness`

| code | paper | verdict | conf | e_full | e_abstract | e_chunk |
| --- | --- | --- | --- | --- | --- | --- |
| 21 | 10.1007_s00441-015-2202-z.pdf | not_met | 0.95 | +0.040 | +0.056 | +0.039 |
| 162 | www.oncotarget.com_article_435_text.pdf | not_met | 0.95 | +0.038 | +0.044 | +0.038 |
| 233 | www.oncotarget.com_article_5108.pdf | not_met | 0.90 | +0.003 | -0.004 | +0.003 |
| 247 | 10.1007_s11914-023-00846-y.pdf | not_met | 0.95 | -0.011 | -0.007 | -0.007 |
| 248 | 10.1038_s41586-026-10267-3.pdf | not_met | 0.90 | +0.050 | +0.052 | +0.048 |
| 259 | hdl.handle.net_1773_36496.pdf | not_met | 0.95 | -0.008 | -0.010 | -0.006 |
| 275 | www.jneurosci.org_content_33_25_10568.lo… | not_met | 0.95 | +0.030 | +0.036 | +0.036 |
| 375 | 10.3389_fpls.2021.640919_full.pdf | not_met | 0.90 | -0.001 | +0.004 | -0.002 |
| 424 | 10.1371_journal.pcbi.1002536.pdf | not_met | 0.95 | +0.049 | +0.042 | +0.051 |
| 428 | 10.1371_journal.pgen.1003076.pdf | not_met | 0.95 | +0.063 | +0.073 | +0.060 |

## Which granularity tracks the verdict? Spearman ρ(e, verdict_ordinal)

Higher ρ = the embedding ranks papers in the same order the LLM does. ρ is `n/a` when all verdicts for a criterion are identical (no rank variation).

| criterion | full | abstract | chunk |
| --- | --- | --- | --- |
| two_worlds | +0.522 | +0.522 | +0.522 |
| adaptors | +0.522 | +0.522 | +0.522 |
| arbitrariness | n/a | n/a | n/a |

## Pole separation (pairwise cosine; high neg-neg ≈ muddied poles)

- **pos**: `adaptors~arbitrariness`=+0.64, `two_worlds~adaptors`=+0.77, `two_worlds~arbitrariness`=+0.73
- **neg**: `adaptors~arbitrariness`=+0.69, `two_worlds~adaptors`=+0.72, `two_worlds~arbitrariness`=+0.71

## Control checks

genetic-code control should read high on all three; deterministic-chemistry should read low on `arbitrariness`.

| control | two_worlds | adaptors | arbitrariness |
| --- | --- | --- | --- |
| deterministic_chemistry_negative | -0.077 | -0.147 | -0.161 |
| genetic_code_positive | +0.129 | +0.150 | +0.154 |

