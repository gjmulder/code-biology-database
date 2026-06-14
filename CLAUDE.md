# CLAUDE.md

High-level design, development, testing and reporting guide for the Code Biology
project. `@environment_notes.md` holds host-specific operational detail (GPUs, MySQL
host, llama-server, model runtimes) — keep that out of this file.

## 1. What the project is

The project turns the canonical Code Biology code/citation list into an analysable
corpus and scores how strongly each paper's literature argues Barbieri's definition of
an organic code. It has two independent measurement axes over the same papers:

- **LLM verdicts** (`criteria_judge.py`) — a grounded categorical `met / not_met /
  unclear` per criterion, every `met` gated by a verbatim quote. The categorical backbone.
- **Embedding axis** (`embed_*` + `run_harrier_embed.py`) — a continuous corpus-contrastive
  score `e` per criterion, reported **side-by-side** with the verdicts. **Design Decision 0
  (load-bearing): the embedding axis is independent — it never overrides, gates, or
  band-merges the verdict.**

### The three criteria (the definition being measured)
Per Barbieri (`www.codebiology.org/index.html`), a biological code exists only if **all
three** are demonstrated (objective, experimentally falsifiable):
1. **Two independent worlds of molecules** — two distinct sets with no necessary
   physical/chemical link (e.g. codons and amino acids).
2. **A set of adaptors** — a *third* molecule that physically bridges the two worlds
   (e.g. tRNAs); the empirical "molecular fingerprint" of a code.
3. **Arbitrariness of the coding rules** — the mapping is conventional, *compatible* with
   but **not determined** by physics/chemistry. The subtlest, most contested criterion.

This separates **coding** (meaning; natural conventions) from **copying** (information;
natural selection). Not every PDF entry is a "code" in this strict sense — e.g. code 352
(SeqCode) is a nomenclature code.

### Source figures
- Source PDF `Biological_Code_List_20260531.pdf` → **435 code categories**, **~2299
  references** quoted (extraction recovers 2290). Treat the source PDF as authoritative;
  the society website's published editions (`database.pdf` 237 codes, 2022;
  `second-database.pdf` 418 codes, 2026) are upstream context, not identical to it.

### Supporting corpus (used to mine prototypes and for context)
- **`Code_Biology_PDFs/`** — the seminal texts (Barbieri's *The Organic Codes*,
  *Introduction to Code Biology*, *What Is Code Biology?*, *Codes and Evolution*, …; Major's
  *Archetypes and Code Biology*, *From Code to Archetype*).
- **`www.codebiology.org/`** — static mirror of the ISCB site (102 HTML pages, 31 PDFs).
  `index.html` (public definition), `glossary.html` (~100 terms by Barbieri, de Beule &
  Hofmeyr), `brief-history.html`, the published databases, society governance, and
  `conferences/` (every ISCB meeting). Positive/negative prototype passages are mined here.

## 2. The data & processing pipeline (reproducible, end-to-end)

Each step lists **script → input → output → tests**. Run `pytest` (162 tests, fully
offline) after any change. MySQL on asushimu is the system of record from step 5 on.

1. **Extract the code list** — `extract_csv.py`
   - in: `Biological_Code_List_20260531.pdf` · out: `biological_codes.csv`
     (`Code Number, Code Name, Paper Name, URL`).
   - Parses the PDF with `pdfplumber`; each citation is a hyperlink whose anchor text is
     the full reference, so **hyperlink runs (not text splitting) are the extraction
     anchor**. → 2290 references across all 435 codes (code 352/SeqCode has citation text
     but no hyperlink → empty URL).
   - tests: `test_extract.py` (coverage, contiguity, per-code counts, URL validity,
     cross-page continuation, cross-row hyperlink-bleed regressions).

2. **Download full-text PDFs** — `download_pdfs.py`
   - in: `biological_codes.csv` · out: `pdfs/` (gitignored), `failed_downloads.csv`,
     `crossref_cache.json` + `unpaywall_cache.json`.
   - Derives a DOI from the URL, then tries in order: Crossref `application/pdf`, every
     Unpaywall OA location, the landing page `citation_pdf_url`. **Legal-OA only — no
     Sci-Hub / paywall circumvention.** Resumable: skips PDFs on disk, reuses caches.
   - Coverage from a non-institutional home network: **471 of 2240 unique refs (~21%)**;
     1769 not retrievable (mostly hard paywall; some CDN bot-blocked OA; some PMC-only).
     Coverage would rise on an institutional network.
   - tests: `test_download_pdfs.py`.

3. **Extract text from PDFs** — `pdf_text.py` (library used by steps 4 & 6)
   - `extract_text()` (full text), `extract_abstract()` (abstract heading → preamble
     fallback), `select_for_budget()` (token-budget trim).
   - tests: `test_pdf_text.py`.

4. **Embed the papers (GPU, once per structural run)** — `run_harrier_embed.py`
   - host: asushimu 3090 Ti · in: paper texts + prototype passages · out: transient
     `embed_out.json` (transport only).
   - Emits **raw vectors only** — document vectors (three methods, below) and pooled pole
     vectors (3 criteria × pos/neg). It does **not** compute `e`. Model
     `microsoft/harrier-oss-v1-27b` (5376-dim); runtime detail in `@environment_notes.md`.
   - **Three chunking methods**, each embedded as a separate document so we can test which
     granularity tracks the verdict: **full** (whole capped paper), **abstract** (abstract
     only), **chunk** (8192-token windows, 50% overlap, scored per-window then **max-pooled**).
   - tests: `test_run_harrier_embed.py`.

5. **Persist vectors + score `e` offline (driver)** — `embed_independent.py` (+ `db.py`,
   `embed_score.py`)
   - in: `embed_out.json`, `prototypes.json` · out: rows in MySQL `codebiology`.
   - Loads vectors to MySQL via `db.store`, then computes `e` **offline** from the
     persisted vectors with the four space-level levers (§4). After one structural GPU run,
     every lever is re-tunable and the report regenerates **with no further GPU** via
     `--recompute`. `--controls-only` is a cheap GPU run that embeds only the control texts.
   - Drops any `codebiology.org` self-reference from the corpus (`drop_self_references`) so
     the in-corpus conference docs can't leak into the ranking.
   - tests: `test_embed_score.py`, `test_embed_independent.py`, `test_db.py`.

6. **Judge the papers (LLM verdicts)** — `criteria_judge.py`, driven by `run_sample.py`
   (sampling) or `judge_corpus.py` (corpus backfill)
   - in: papers on disk + `biological_codes.csv` · out: `sample_verdicts.jsonl`
     (resumable APPEND checkpoint — the file system-of-record for spend safety) →
     upserted into MySQL `embedding_scores` (verdict/confidence only; the embedding `e` is
     left untouched, via `db.update_verdicts`).
   - **Routing:** criteria 1 & 2 (concrete) → local **Gemma-4-31B** (free); criterion 3
     (*arbitrariness*, subtle) → paid **Nemotron** (`nvidia/nemotron-3-ultra-550b-a55b`,
     1M ctx, reads whole paper) via OpenRouter. `run_batch` is concurrent
     (`DEFAULT_WORKERS=6`), resumable, per-paper failure isolation.
   - **Cost:** criterion-3-only ≈ **$3 / ~220 papers**. A future higher-quality run sending
     **all three** criteria to Nemotron is ≈ **$9 / run** (do this once the prompts are
     better tuned — see §6 caveats).
   - tests: `test_criteria_judge.py`, `test_openrouter_agent.py`.

7. **Generate the report** — `embed_independent.py --report-only`
   - in: MySQL · out: `report.md` + `embedding_scores.csv` (regenerated from the DB).
   - `--report-only` reads scores; `--recompute` rescores from vectors (lever flags) then
     writes. Report sections: per-paper verdicts + embedding columns; Spearman
     ρ(e, verdict_ordinal) per method × criterion; pole separation; pole width `within`;
     control checks.

## 3. MySQL schema (`db.py`) — system of record

DB `codebiology` on asushimu (host/connection detail in `@environment_notes.md`). Vectors
are float32 LE bytes in `LONGBLOB`. Tables:
- **`doc_vectors`** (`code_number, pdf_path, method, chunk_idx, dim, vec`),
  **`pole_vectors`** (`criterion, pole, dim, vec`), **`control_vectors`** (`name, dim, vec`)
  — the **raw vectors** that make offline `--recompute` possible.
- **`embedding_scores`** — one row per `(code_number, pdf_path, method, criterion)` with
  `e`, `verdict`, `confidence`, `model`, `run_ts`. **`code_number` is the leading PK
  column.** `--recompute` upserts `e` only (preserving the verdict); `update_verdicts`
  upserts verdict/confidence only (preserving `e`).
- **`pole_separation`** (incl. centred `within` rows), **`control_scores`**, **`run_meta`**
  (lever params + scoring mode).

## 4. The four space-level levers — offline scoring (`embed_score.py`)

Run 1's double-cosine `e = cos(paper, POS) − cos(paper, NEG)` under-discriminated: a
topicality halo and the decoder-only-embedder anisotropy meant in-register text all sat in
a narrow cone. The fix is **space-level, not prompt-level**:

```
μ      = mean of all document vectors                       # centring origin
B      = whiten_basis(center(reps, μ), k)                   # top-k PCs to strip (--whiten-k)
a_c    = normalize(p̂_c − n̂_c)        on centred poles       # axis-projection contrast
ŝ      = shared_direction({a_c})     = first PC of the axes  # shared register direction
a_c⊥   = orthogonalize(a_c, ŝ, strength)                    # partial out topicality (--shared-strength)
e_c(d) = a_c⊥ · normalize( whiten(d − μ, B) )               # chunk windows max-pooled
```

`recompute(doc_vecs, poles, k, strength)` is the pure composition; `build_axes`,
`whiten_basis`, `shared_direction`, `orthogonalize`, `axis_score` are unit-tested pieces.
`within[c] = cos(centred pos, centred neg)` (pole width) is recomputed and rendered.

**Tunables (CLI flags, smoke-calibrated defaults):**
- `--shared-strength` (default `0.5`) — how hard each axis is orthogonalized against the
  shared register direction. `1.0` over-corrects; `0.5` removes the halo, keeps real signal.
- `--whiten-k` (default `0`) — number of top PCs removed. `k≥1` hurt at small sample;
  revisit at corpus scale.

## 5. First useful result (2026-06-14) — ρ is now measurable

The embedding corpus is **219 papers** (after dropping the in-corpus self-reference).
Previously only the 10-paper seed carried verdicts, so ρ was driven by one `met` paper and
`arbitrariness` had **zero positives → undefined**. Backfilling LLM verdicts for the whole
corpus (`judge_corpus.py`; **217 / 219** judged, 2 failed PDF extraction, ignored) gives
real variation in every criterion, so **ρ(e, verdict_ordinal) is measurable for the first
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

**Read:** the embedding axis tracks the verdict direction positively on all three
criteria — strongest on `two_worlds` (most concrete), weakest on `arbitrariness` (subtlest,
still only 2 positives → directional, not precise). `chunk` edges out `full`/`abstract`
everywhere. Pole widths still overlap (`within`: `arbitrariness` best-separated), so
**ranks are trustworthy, absolute magnitudes less so**.

## 6. ⚠️ Major caveats (the verdicts are not ground truth)

The ρ above measures agreement between two imperfect axes, not correctness. The verdicts
are synthetic from comparatively weak models and have **not** been validated against a gold
set:
- **Poor calibration** — confidences cluster at 0.95–1.0 with very few positives; the
  `confidence` field carries no usable gradation (which is *why* the embedding axis exists).
- **False positives/negatives** — the prompts read too literally, latching onto an isolated
  sentence rather than the sentence in the context of its paragraph. More contextual
  understanding is needed.
- **Next quality step:** re-tune the judge prompts for paragraph-level context, then a
  single higher-quality run sending **all three** criteria to Nemotron (≈ $9). A gold-set
  κ/F1 validation of the verdict backbone remains future work.
- **Backlog:** refined ranking via a cross-encoder over the top candidates.

## 7. Development, testing & reporting rules

1. **TDD** — for any new or changed functionality, write a failing test first, then the
   change. The suite is **162 tests, fully offline** (fake encoder, no GPU/DB needed).
2. **Language** — pythonic, readable; prefer numpy for data management.
3. **Logging** — pythonic `logging`, DEBUG/INFO chosen by criticality.
4. **Commit cadence** — pause and commit after each completed logical unit; small,
   self-contained commits. End commit messages with
   `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
5. **Spend safety** — paid (Nemotron) work checkpoints to `sample_verdicts.jsonl` per paper
   (resumable APPEND) **before** MySQL persistence; never delete the checkpoint.
6. **Secrets** — `.env` is gitignored and never committed; never print API keys.

### ⚠️ Deferred revert (PROJECT END only)
The production `llama-server` (Home Assistant voice agent) is **OFFLINE** — the 3090 Ti was
freed for this project's GPU/judging work. Restore **only at project end**:
`cp ~/start_llama.prod.bak ~/start_llama.sh && sudo systemctl restart llama-server`.
(Operational detail in `@environment_notes.md`.)
