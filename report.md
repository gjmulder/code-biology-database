# Independent Embedding Analysis vs LLM Verdicts

Sample: 10 preselected papers. Model: `/data/vllm/harrier-oss-v1-27b` (5376-dim, 4-bit=True). Source of record: MySQL `embedding_scores` (keyed on code id).

`e = cos(paper, POS_prototype) − cos(paper, NEG_prototype)` — higher means the text reads as *arguing* the criterion. This embedding axis is **independent**: it is reported beside the LLM verdict and never overrides it (plan decision 0). Each paper is scored three ways as separate documents — **full** text, **abstract** only, and **chunk** (8192-token windows @50% overlap, max-pooled) — to test which granularity best tracks the verdict (chunk size 8192, overlap 4096).

## Per-paper verdicts (criteria_judge) + embedding columns

### Criterion: `two_worlds`

| code | paper | verdict | conf | e_full | e_abstract | e_chunk |
| --- | --- | --- | --- | --- | --- | --- |
| 21 | 10.1007_s00441-015-2202-z.pdf | not_met | 1.00 | -0.006 | +0.006 | -0.002 |
| 162 | www.oncotarget.com_article_435_text.pdf | not_met | 1.00 | +0.024 | +0.024 | +0.024 |
| 233 | www.oncotarget.com_article_5108.pdf | not_met | 1.00 | +0.003 | -0.004 | +0.003 |
| 247 | 10.1007_s11914-023-00846-y.pdf | not_met | 1.00 | -0.001 | -0.013 | -0.000 |
| 248 | 10.1038_s41586-026-10267-3.pdf | not_met | 1.00 | +0.013 | +0.012 | +0.021 |
| 259 | hdl.handle.net_1773_36496.pdf | not_met | 1.00 | +0.003 | +0.014 | +0.014 |
| 275 | www.jneurosci.org_content_33_25_10568.lo… | not_met | 1.00 | +0.021 | +0.025 | +0.023 |
| 375 | 10.3389_fpls.2021.640919_full.pdf | not_met | 1.00 | +0.002 | +0.009 | +0.013 |
| 424 | 10.1371_journal.pcbi.1002536.pdf | not_met | 1.00 | +0.016 | +0.007 | +0.018 |
| 428 | 10.1371_journal.pgen.1003076.pdf | met | 1.00 | +0.043 | +0.059 | +0.052 |

### Criterion: `adaptors`

| code | paper | verdict | conf | e_full | e_abstract | e_chunk |
| --- | --- | --- | --- | --- | --- | --- |
| 21 | 10.1007_s00441-015-2202-z.pdf | not_met | 1.00 | +0.017 | +0.022 | +0.021 |
| 162 | www.oncotarget.com_article_435_text.pdf | not_met | 1.00 | -0.009 | -0.004 | -0.009 |
| 233 | www.oncotarget.com_article_5108.pdf | not_met | 1.00 | -0.007 | -0.012 | -0.007 |
| 247 | 10.1007_s11914-023-00846-y.pdf | not_met | 1.00 | -0.003 | +0.006 | +0.000 |
| 248 | 10.1038_s41586-026-10267-3.pdf | not_met | 1.00 | +0.023 | +0.023 | +0.024 |
| 259 | hdl.handle.net_1773_36496.pdf | not_met | 1.00 | +0.000 | +0.012 | +0.012 |
| 275 | www.jneurosci.org_content_33_25_10568.lo… | not_met | 1.00 | +0.019 | +0.023 | +0.028 |
| 375 | 10.3389_fpls.2021.640919_full.pdf | not_met | 1.00 | -0.010 | -0.002 | -0.007 |
| 424 | 10.1371_journal.pcbi.1002536.pdf | not_met | 1.00 | -0.006 | -0.008 | -0.001 |
| 428 | 10.1371_journal.pgen.1003076.pdf | met | 1.00 | +0.046 | +0.053 | +0.049 |

### Criterion: `arbitrariness`

| code | paper | verdict | conf | e_full | e_abstract | e_chunk |
| --- | --- | --- | --- | --- | --- | --- |
| 21 | 10.1007_s00441-015-2202-z.pdf | not_met | 0.95 | +0.040 | +0.058 | +0.040 |
| 162 | www.oncotarget.com_article_435_text.pdf | not_met | 0.95 | +0.040 | +0.045 | +0.040 |
| 233 | www.oncotarget.com_article_5108.pdf | not_met | 0.90 | +0.004 | -0.003 | +0.004 |
| 247 | 10.1007_s11914-023-00846-y.pdf | not_met | 0.95 | -0.008 | -0.005 | -0.004 |
| 248 | 10.1038_s41586-026-10267-3.pdf | not_met | 0.90 | +0.051 | +0.052 | +0.049 |
| 259 | hdl.handle.net_1773_36496.pdf | not_met | 0.95 | -0.002 | -0.006 | -0.001 |
| 275 | www.jneurosci.org_content_33_25_10568.lo… | not_met | 0.95 | +0.031 | +0.036 | +0.035 |
| 375 | 10.3389_fpls.2021.640919_full.pdf | not_met | 0.90 | +0.002 | +0.005 | +0.003 |
| 424 | 10.1371_journal.pcbi.1002536.pdf | not_met | 0.95 | +0.051 | +0.042 | +0.052 |
| 428 | 10.1371_journal.pgen.1003076.pdf | not_met | 0.95 | +0.059 | +0.068 | +0.057 |

## Which granularity tracks the verdict? Spearman ρ(e, verdict_ordinal)

Higher ρ = the embedding ranks papers in the same order the LLM does. ρ is `n/a` when all verdicts for a criterion are identical (no rank variation).

| criterion | full | abstract | chunk |
| --- | --- | --- | --- |
| two_worlds | +0.522 | +0.522 | +0.522 |
| adaptors | +0.522 | +0.522 | +0.522 |
| arbitrariness | n/a | n/a | n/a |

## Pole separation (pairwise cosine; high neg-neg ≈ muddied poles)

- **pos**: `adaptors~arbitrariness`=+0.63, `two_worlds~adaptors`=+0.77, `two_worlds~arbitrariness`=+0.72
- **neg**: `adaptors~arbitrariness`=+0.69, `two_worlds~adaptors`=+0.72, `two_worlds~arbitrariness`=+0.69

## Control checks

genetic-code control should read high on all three; deterministic-chemistry should read low on `arbitrariness`.

| control | two_worlds | adaptors | arbitrariness |
| --- | --- | --- | --- |
| deterministic_chemistry_negative | -0.066 | -0.132 | -0.151 |
| genetic_code_positive | +0.125 | +0.137 | +0.144 |

