# CLAUDE.md

High-level design, development, testing and reporting guide for the Code Biology
project. `@environment_notes.md` holds host-specific operational detail (GPUs, MySQL
host, llama-server, model runtimes) — keep that out of this file. `@test_runs.md` holds the
chronological **run log** (dated results, distributions, ρ tables, shelved experiments) — keep
those out of this file too; CLAUDE.md states the *current* design and the live decisions those
runs produced.

**Working axis:** the project focuses on **chunked embeddings** (8192-token windows) and a
**per-chunk LLM judge**. The earlier full-paper / abstract granularities are retired (chunk
edged them out — `@test_runs.md` Run 2); they survive in the DB/code for provenance only.

## 1. What the project is

The project turns the canonical Code Biology code/citation list into an analysable
corpus and scores how strongly each paper's literature argues Barbieri's definition of
an organic code. It has two independent measurement axes over the same papers:

- **LLM verdicts** (`criteria_judge.py`) — a grounded categorical `met / not_met /
  unclear` per criterion, every `met` gated by a verbatim quote.
- **Embedding axis** (`embed_*` + `run_harrier_embed.py`) — a continuous corpus-contrastive
  score `e` per criterion.

The two axes are **independent and reported side-by-side** — neither is authoritative.
The verdicts are *synthetic* ground truths from comparatively weak models and may suffer
the same failure modes as the embeddings (see §6), so the embedding axis is **not** subordinated
to them: it does not merely position within verdict-chosen bands. Agreement between the two
(e.g. ρ in §5) is corroboration, not validation against truth.

**Topic stratification (diagnostic, not a third axis).** The corpus is additionally mapped
onto the 24 scientometric topics of Paredes & Prinz (2025): each paper chunk is assigned to
its nearest topic centroid in the *same* centred embedding space (§2.1). This is the
topicality halo the §4 levers strip, reused as a **stratifier** — it lets us ask whether `e`
tracks the verdict *within* a topic (per-topic ρ, §5) — and does **not** measure the criteria.

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

Each step lists **script → input → output → tests**. Run `pytest` (249 tests, fully
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
   - Emits **raw vectors only** — document vectors and pooled pole vectors (3 criteria ×
     pos/neg). It does **not** compute `e`. Model `microsoft/harrier-oss-v1-27b` (5376-dim);
     runtime detail in `@environment_notes.md`.
   - **Working granularity is `chunk`** — 8192-token windows, 50% overlap, scored per-window
     then **max-pooled**. (The retired `full`/`abstract` methods are still emitted for
     provenance; `@test_runs.md` Run 2 has the comparison that retired them.)
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
     upserted into the run-agnostic MySQL `verdicts` table (judged once, shared by every
     embedding run; the embedding axis is untouched, via `db.update_verdicts`).
   - **Routing:** 
     → local **Gemma-4-31B** (free);
     → paid **DeepSeek V4 Pro** (input $0.435/1M, output $0.87/1M) via OpenRouter. `run_batch` is concurrent (`DEFAULT_WORKERS=6`), resumable, per-paper failure isolation.
   - **Cost:** TBD for DeepSeek V4 Pro
   - tests: `test_criteria_judge.py`, `test_openrouter_agent.py`.

7. **Generate the report** — `embed_independent.py --report-only`
   - in: MySQL · out: `report.md` + `embedding_scores.csv` (regenerated from the DB).
   - `--report-only` reads scores; `--recompute` rescores from vectors (lever flags) then
     writes. Report sections: per-paper verdicts + embedding columns; Spearman
     ρ(e, verdict_ordinal) per method × criterion; per-topic ρ stratified by nearest
     scientometric topic (§2.1); pole separation; pole width `within`; control checks.

### 2.1 Scientometric topic stratification (2026-06-15)

Maps the corpus onto the **24 topics** of Paredes & Prinz (2025) to stratify the ρ diagnosis
(§5). No new GPU pass beyond embedding the centroids; nothing here measures the criteria.

- **Augmented topics** — `code-categories-augmented.csv` (24 rows: label, abbreviation,
  justification, characteristic terms, **centroid text**).
- **Embed centroids (GPU, shares the harrier space)** — `run_harrier_centroids.py` → transient
  `centroids_out.json`. Each topic's centroid text is embedded as a *document* (no `Instruct:`
  query prefix), same model / 4-bit / `run` as the corpus chunks, so a chunk and a centroid are
  comparable. tests: `test_run_harrier_centroids.py`.
- **Assign chunks → topics (offline, no GPU)** — `assign_topics.py` projects each persisted
  chunk vector through the *identical* μ-centred/whitened scorer (`embed_score.build_scorer`)
  and assigns it to the nearest centroid; a paper's **dominant topic** is the max-pool of its
  chunks. Writes the `chunk_topics` table. tests: `test_assign_topics.py`.
- **Per-topic ρ** — `embed_independent.per_topic_spearman` recomputes ρ(e, verdict) within each
  topic stratum (≥10 labelled papers shown), rendered in `report.md`. tests:
  `test_embed_independent.py`.

## 3. MySQL schema (`db.py`) — system of record

DB `codebiology` on asushimu (host/connection detail in `@environment_notes.md`). Vectors
are float32 LE bytes in `LONGBLOB`.

**Run-keyed (2026-06-14):** every embedding table carries `run VARCHAR(64) NOT NULL DEFAULT
'baseline'` as its **leading PK column**, so multiple embedding models coexist
non-destructively (harrier = `baseline`; a future gte pass would be `gte-qwen2`). The
driver/`sweep_levers.py` take `--run`; `init_schema` runs an idempotent `migrate_runs` that
adds the column + rebuilds PKs on the existing DB. Tables:
- **`doc_vectors`** (`run, code_number, pdf_path, method, chunk_idx, dim, vec`),
  **`pole_vectors`** (`run, criterion, pole, dim, vec`), **`control_vectors`** (`run, name,
  dim, vec`) — the **raw vectors** that make offline `--recompute` possible.
- **`embedding_scores`** — one row per `(run, code_number, pdf_path, method, criterion)` with
  `e`, `model` (the *embedding* model), `run_ts`. **Verdict/confidence are no longer here.**
  `--recompute` upserts `e` only.
- **`verdicts`** — **run-agnostic** normalised table, one row per `(code_number, pdf_path,
  criterion)` with `verdict`, `confidence`, `graded`, `model` (the *judge* model),
  `prompt_hash`, `run_ts`. The LLM judge is independent of the embedding model, so verdicts are
  judged once and shared by every run via a JOIN on `(code_number, pdf_path, criterion)`;
  `update_verdicts` upserts here. `chunk_verdicts` (per-chunk graded diagnostics) likewise
  carries `model` + `prompt_hash`.
- **`prompt_registry`** (`prompt_hash` PK, `criterion`, `prompt_text`, `run_ts`) — the
  **prompt provenance** layer (2026-06-16): `prompt_hash = criteria_judge.prompt_hash(criterion)`
  is a sha256 over the version-bearing prompt scaffold (system prompt + calibration preamble +
  anchor framing + criterion definition + arbitrariness steelman + schema), stamped onto every
  verdict/chunk_verdict so the molecular vs domain-general criterion prompts (and any future
  re-tune) are distinguishable in the DB; `register_prompts` stores each version's full template
  text once, keyed by hash, so the DB alone recovers the exact prompt. Per-criterion, so it
  correctly shows `arbitrariness` unchanged across the molecular→domain-general rewrite.
- **`topic_centroids`** (`run, topic_id, label, dim, vec`) and **`chunk_topics`** (`run,
  pdf_path, chunk_idx, method, topic_id, sim`) — the scientometric topic-stratification layer
  (§2.1); both run-keyed.
- **`pole_separation`** (incl. centred `within` rows), **`control_scores`**, **`run_meta`**
  (lever params + scoring mode) — all run-scoped.

## 4. The four space-level levers — offline scoring (`embed_score.py`)

The original double-cosine `e = cos(paper, POS) − cos(paper, NEG)` under-discriminated: a
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
  **revisited at corpus scale (2026-06-14, `sweep_levers.py`) and the defaults stand** —
  see the sweep note below.

**Live decision: keep `k=0, strength=0.5`.** A corpus-scale `(whiten-k, shared-strength)`
sweep (`sweep_levers.py`, free/offline, n=219) found whitening a dead end (`k≥1` collapses the
concrete criteria), `strength=0.5` near-optimal, and no cell winning all three — per-criterion
argmax was rejected as overfitting synthetic labels. The binding constraint is the model + pole
separation (`within` 0.51–0.68), not the levers. Full sweep table and reasoning: `@test_runs.md`
Run 1. `sweep_levers.py` is kept as the re-runnable diagnostic for a future model's vectors.

## 5. Current result — ρ is measurable, ranks trustworthy

The embedding corpus is **219 papers** (after dropping the in-corpus self-reference); LLM
verdicts are backfilled across it (`judge_corpus.py`, 217/219 judged). On the working `chunk`
axis the embedding score tracks the verdict direction positively on all three criteria —
strongest on `two_worlds` (most concrete, ρ≈+0.41), weakest on `arbitrariness` (subtlest, only
2 positives → directional not precise, ρ≈+0.15). Pole widths overlap (`within` 0.51–0.68), so
**ranks are trustworthy, absolute magnitudes less so**. Per-topic ρ (stratified by §2.1) holds
the halo fixed but the concrete criteria are flat `not_met` in the large neuro strata, so it is
**diagnostic only**. Full distributions, ρ tables (incl. the retired full/abstract columns),
and per-topic breakdown: `@test_runs.md` Run 2.

## 6. ⚠️ Major caveats (the verdicts are not ground truth)

The ρ above measures agreement between two imperfect axes, not correctness. The verdicts
are synthetic from comparatively weak models and have **not** been validated against a gold
set:
- **Poor calibration** — confidences cluster at 0.95–1.0 with very few positives; the
  `confidence` field carries no usable gradation (which is *why* the embedding axis exists).
- **False positives/negatives** — the prompts read too literally, latching onto an isolated
  sentence rather than the sentence in the context of its paragraph. More contextual
  understanding is needed.
- **Judge redesign (graded, per-chunk, grounded) — piloted 2026-06-16, see §9.** The
  literal-latching and the dead `confidence` axis above are exactly what the redesign targets;
  the pilot confirms the grounding gate kills ungrounded positives and moves gradation onto an
  `agreement` (−1…+1) axis.

## 7. Development, testing & reporting rules

1. **TDD** — for any new or changed functionality, write a failing test first, then the
   change. The suite is **249 tests, fully offline** (fake encoder/tokenizer, no GPU/DB needed).
2. **Language** — pythonic, readable; prefer numpy for data management.
3. **Logging** — pythonic `logging`, DEBUG/INFO chosen by criticality.
4. **Commit cadence** — pause and commit after each completed logical unit; small,
   self-contained commits. 
5. **Spend safety** — paid (DeepSeek V4 Pro) work checkpoints to `sample_verdicts.jsonl` per paper
   (resumable APPEND) **before** MySQL persistence; never delete the checkpoint.
6. **Secrets** — `.env` is gitignored and never committed; never print API keys.
7. **DB backup before schema changes** — always take a compressed `mysqldump` of the
   `codebiology` DB **before** running any schema change (new table/column, `ALTER`, migration,
   or the first `init_schema` on new DDL). e.g. `mysqldump … codebiology | gzip >
   codebiology_$(date +%Y%m%d_%H%M%S).sql.gz` (connection detail in `@environment_notes.md`;
   dumps are gitignored). The migrations are idempotent and guarded, but the dump is the
   non-negotiable rollback path.

## 8. Embedding side is at its ceiling — the constraint is label quality

Embedding-side tuning is judged exhausted: the levers are set (§4), and an **embedding-model
swap to gte-Qwen2-7B-instruct was built, tested, and deliberately shelved** — gte is the *same*
decoder-only last-token architecture as harrier (the documented source of the anisotropy that
overlaps the poles), so it is unlikely to widen them; the runner is kept as a non-destructive
tested artifact (`run` column). The only remaining label-free lever is prototype/pole quality,
expected to move things marginally. **The genuine constraint is now label quality, not the
embedding** (§6) — the next real step is the re-tuned judge (§9) + a gold-set validation, not
more embedding-side work. Full shelving rationale: `@test_runs.md` Run 3.

## 9. Judge redesign — graded, per-chunk, topic-grounded, control-anchored (2026-06-16)

Acting on §6/§8 ("the constraint is label quality"), the LLM judge was rebuilt as a **graded**
(−1…+1 `agreement`), **per-chunk** (the *exact* 8192-token harrier embedding windows reproduced
via `chunk_text.reproduce_chunks` → tokenizer-aligned to `doc_vectors`), **topic-grounded**
(dominant scientometric topic injected as *context, not evidence*), **control-anchored**
(`genetic_code_positive` / `deterministic_chemistry_negative` poles), **calibrated** judge.
A grounding gate pulls any positive whose `evidence_quote` is not verbatim in the chunk back to
`0.0`; `aggregate_graded` max-pools chunks → `(graded_max, graded_mean, confidence, categorical)`
with `graded_max ≥ +0.5 → met`, `≤ 0.0 → not_met`, else `unclear`. Schema:
`verdicts.graded DOUBLE` + run-agnostic `chunk_verdicts` table (per-chunk diagnostics).
Build steps complete + tested; **distribution-comparison validation only, no gold set (locked).**

**Pipeline:** `judge_pilot.py` (driver, top-N topics, resumable per-chunk JSONL checkpoint) →
`compare_verdicts.py --snapshot old.json` *before* / `--old old.json` *after* (the new axis is
read from the never-overwritten `chunk_verdicts`). Free local Gemma-4-31B on
`start_llama_pilot.sh` (NO MTP, `--parallel 2 --ctx-size 32768`); driver runs locally with the
CPU-only harrier tokenizer (`harrier_tokenizer/`, `--tokenizer harrier_tokenizer`).

**Pilot status.** First pilot (top-4 neuro topics, 102 papers, molecular defs) validated the
design — the grounding gate held (every positive quote-grounded), the judge is *more* skeptical
than the old one, and gradation moved onto `agreement`. It also exposed the molecular-bias
prompt bug fixed in §9.1. Pilot results: `@test_runs.md` Runs 4–5. Before any corpus-wide or
paid (DeepSeek V4 Pro) all-criteria run, confirm rich gradation materialises on the molecular "met"
tail (the natural next pilot, outside the neuro top-4).

### 9.1 Domain-general criteria (2026-06-16) — molecular-bias fix
The molecular-specific `CRITERIA_DEFS` mechanically rejected every non-molecular paper
(two_worlds **0** positives, with reasoning citing "not *molecular* worlds"; adaptors positives
only in the histonic molecular topic). Only `arbitrariness` was already abstract and worked
across domains — it became the template. The three criterion definitions the judge
operationalizes are now **domain-general**: they instantiate per discipline across the 24 topics
(codons↔amino acids *or* stimulus↔spikes *or* sound↔percept …), with the molecular genetic code
as one **exemplar**, not the requirement. The "adaptor" is generalised to the domain's
**mediator** per **Major (2025), *From Code to Archetype*** (the third term that reads/executes
the mapping — tRNA/ribosome, nervous system, imaginal function, computational engine). The
molecular control anchors are relabelled **ILLUSTRATIVE**. This broadens what the *judge*
measures to match the field's own cross-domain claims; it does **not** alter Barbieri's strict
molecular definition in §1. The DB criterion **key stays `adaptors`** (schema/PK stability);
only its definition *text* changed. The §9.1 re-pilot validating it is `@test_runs.md` Run 5.
