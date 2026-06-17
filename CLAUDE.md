# CLAUDE.md

High-level design, development, testing and reporting guide for the Code Biology project —
the **current** design and the live decisions it rests on, nothing dated. Two sister files
hold what this one deliberately omits:
- `@environment_notes.md` — host-specific operational detail (GPUs, MySQL host, llama-server
  launchers, model runtimes, download recipes).
- `@test_runs.md` — the chronological **run log** (dated results, distributions, ρ tables,
  shelved experiments). CLAUDE.md cites these runs for the evidence behind each live decision.

**Working axis:** the project focuses on **chunked embeddings** (8192-token windows) and a
**per-chunk LLM judge**. The earlier full-paper / abstract granularities are retired (chunk
edged them out — `@test_runs.md` Run 2); they survive in the DB/code for provenance only.

## 1. What the project is

The project turns the canonical Code Biology code/citation list into an analysable corpus and
scores how strongly each paper's literature argues Barbieri's definition of an organic code. It
has two independent measurement axes over the same papers:

- **LLM verdicts** (`criteria_judge.py`) — a graded `agreement` (−1…+1) per criterion rolled up
  to categorical `met / not_met / unclear`, every `met` gated by a verbatim quote (§9).
- **Embedding axis** (`embed_*` + `run_harrier_embed.py`) — a continuous corpus-contrastive
  score `e` per criterion (§4).

The two axes are **independent and reported side-by-side** — neither is authoritative. The
verdicts are *synthetic* ground truths from comparatively weak models and may share the
embeddings' failure modes (§6), so the embedding axis is **not** subordinated to them. Agreement
between the two (ρ, §5) is corroboration, not validation against truth — and the two can
legitimately diverge (§5/§9: the domain-general judge vs the molecular embedding poles).

**Topic stratification (diagnostic, not a third axis).** The corpus is additionally mapped onto
the 24 scientometric topics of Paredes & Prinz (2025): each paper chunk is assigned to its
nearest topic centroid in the *same* centred embedding space (§2.1). This is the topicality halo
the §4 levers strip, reused as a **stratifier** — it lets us ask whether `e` tracks the verdict
*within* a topic (per-topic ρ, §5) — and does **not** measure the criteria.

### The three criteria (the definition being measured)
Per Barbieri (`www.codebiology.org/index.html`), a biological code exists only if **all three**
are demonstrated (objective, experimentally falsifiable):
1. **Two independent worlds** — two distinct sets with no necessary
   physical/chemical link (e.g. codons and amino acids).
2. **A set of adaptors** — a *third* molecule that physically bridges the two worlds (e.g.
   tRNAs); the empirical "molecular fingerprint" of a code.
3. **Arbitrariness of the coding rules** — the mapping is conventional, *compatible* with but
   **not determined** by physics/chemistry. The subtlest, most contested criterion.

This separates **coding** (meaning; natural conventions) from **copying** (information; natural
selection). Not every PDF entry is a "code" in this strict sense — e.g. code 352 (SeqCode) is a
nomenclature code. The §9.1 judge generalises these three across all 24 topics (the molecular
genetic code as one *exemplar*); this broadens what the *judge* measures and does **not** alter
Barbieri's strict molecular definition above.

### Source figures
- Source PDF `Biological_Code_List_20260531.pdf` → **435 code categories**, **~2299 references**
  quoted (extraction recovers 2290). Treat the source PDF as authoritative; the society
  website's published editions (`database.pdf` 237 codes, 2022; `second-database.pdf` 418 codes,
  2026) are upstream context, not identical to it.

### Supporting corpus (used to mine prototypes and for context)
- **`Code_Biology_PDFs/`** — the seminal texts (Barbieri's *The Organic Codes*, *Introduction to
  Code Biology*, *What Is Code Biology?*, *Codes and Evolution*, …; Major's *Archetypes and Code
  Biology*, *From Code to Archetype*).
- **`www.codebiology.org/`** — static mirror of the ISCB site (102 HTML pages, 31 PDFs).
  `index.html` (public definition), `glossary.html` (~100 terms), `brief-history.html`, the
  published databases, governance, and `conferences/`. Positive/negative prototype passages are
  mined here.

## 2. The data & processing pipeline (reproducible, end-to-end)

Each step lists **script → input → output → tests**. Run `pytest` (274 tests, fully offline —
fake encoder/tokenizer, no GPU/DB) after any change. MySQL on asushimu is the system of record
from step 5 on. Tests live in **`tests/`**; an empty root `conftest.py` puts the repo root on
`sys.path` so the in-root modules import from the subdir.

1. **Extract the code list** — `extract_csv.py`
   - in: `Biological_Code_List_20260531.pdf` · out: `biological_codes.csv` (`Code Number, Code
     Name, Paper Name, URL`).
   - Parses the PDF with `pdfplumber`; each citation is a hyperlink whose anchor text is the full
     reference, so **hyperlink runs (not text splitting) are the extraction anchor**. → 2290
     references across all 435 codes (code 352/SeqCode has citation text but no hyperlink → empty
     URL).
   - tests: `test_extract.py`.

2. **Download full-text PDFs** — `download_pdfs.py`
   - in: `biological_codes.csv` · out: `pdfs/` (gitignored), `failed_downloads.csv`,
     `crossref_cache.json` + `unpaywall_cache.json`.
   - Derives a DOI from the URL, then tries in order: Crossref `application/pdf`, every Unpaywall
     OA location, the landing page `citation_pdf_url`. **Legal-OA only — no Sci-Hub / paywall
     circumvention.** Resumable: skips PDFs on disk, reuses caches.
   - Coverage from a non-institutional home network: **471 of 2240 unique refs (~21%)**; the rest
     mostly hard paywall (would rise on an institutional network).
   - tests: `test_download_pdfs.py`.

3. **Extract text from PDFs** — `pdf_text.py` (library used by steps 4 & 6)
   - `extract_text()`, `extract_abstract()`, `select_for_budget()` (token-budget trim).
   - tests: `test_pdf_text.py`.

4. **Embed the papers (GPU, once per structural run)** — `run_harrier_embed.py`
   - in: paper texts + prototype passages · out: transient `embed_out.json` (transport only).
   - Emits **raw vectors only** — document vectors and pooled pole vectors (3 criteria × pos/neg).
     It does **not** compute `e`. Model `microsoft/harrier-oss-v1-27b` (5376-dim); host/runtime
     in `@environment_notes.md`.
   - **Working granularity is `chunk`** — 8192-token windows, 50% overlap, scored per-window then
     **max-pooled**. (Retired `full`/`abstract` still emitted for provenance.)
   - tests: `test_run_harrier_embed.py`.

5. **Persist vectors + score `e` offline (driver)** — `embed_independent.py` (+ `db.py`,
   `embed_score.py`)
   - in: `embed_out.json`, `prototypes.json` · out: rows in MySQL `codebiology`.
   - Loads vectors to MySQL via `db.store`, then computes `e` **offline** from the persisted
     vectors with the four space-level levers (§4). After one structural GPU run, every lever is
     re-tunable and the report regenerates **with no further GPU** via `--recompute`.
     `--controls-only` is a cheap GPU run that embeds only the control/pole texts.
   - Drops any `codebiology.org` self-reference from the corpus (`drop_self_references`) so the
     in-corpus conference docs can't leak into the ranking.
   - tests: `test_embed_score.py`, `test_embed_independent.py`, `test_db.py`.

6. **Judge the papers (LLM verdicts)** — `criteria_judge.py`, driven by `judge_pilot.py` (the
   live graded per-chunk driver, §9). (`run_sample.py` / `judge_corpus.py` drove the retired
   pre-redesign categorical judge.)
   - in: papers on disk + `biological_codes.csv` · out: a resumable per-chunk JSONL checkpoint
     (the file system-of-record for spend safety) → upserted into MySQL `chunk_verdicts` +
     `verdicts` (judge-keyed; §3).
   - **Routing:** local **Gemma-4-31B** (free) or paid **DeepSeek V4 Pro** via OpenRouter
     (concurrent, resumable, per-paper failure isolation). Model/host/pricing in
     `@environment_notes.md`.
   - tests: `test_criteria_judge.py`, `test_openrouter_agent.py`, `test_judge_pilot.py`.

7. **Generate the report** — `embed_independent.py --report-only`
   - in: MySQL · out: `report.md` + `embedding_scores.csv`. `--report-only` reads scores;
     `--recompute` rescores from vectors (lever flags) then writes. Sections: per-paper verdicts +
     embedding columns; Spearman ρ(e, verdict) per method × criterion; per-topic ρ (§2.1); pole
     separation + width `within`; control checks.

### 2.1 Scientometric topic stratification

Maps the corpus onto the **24 topics** of Paredes & Prinz (2025) to stratify the ρ diagnosis
(§5). No new GPU pass beyond embedding the centroids; nothing here measures the criteria.

- **Augmented topics** — `code-categories-augmented.csv` (24 rows incl. **centroid text**).
- **Embed centroids (GPU, shares the harrier space)** — `run_harrier_centroids.py` → transient
  `centroids_out.json`. Each centroid text is embedded as a *document* (no `Instruct:` prefix),
  same model / `run` as the corpus chunks. tests: `test_run_harrier_centroids.py`.
- **Assign chunks → topics (offline)** — `assign_topics.py` projects each persisted chunk vector
  through the *identical* μ-centred/whitened scorer and assigns the nearest centroid; a paper's
  **dominant topic** is the max-pool of its chunks. Writes `chunk_topics`. tests:
  `test_assign_topics.py`.
- **Per-topic ρ** — `embed_independent.per_topic_spearman` recomputes ρ(e, verdict) within each
  stratum (≥10 labelled papers shown). tests: `test_embed_independent.py`.

## 3. MySQL schema (`db.py`) — system of record

DB `codebiology` on asushimu (connection detail in `@environment_notes.md`). Vectors are float32
LE bytes in `LONGBLOB`.

**Two independent versioning keys** let multiple models coexist non-destructively:
- Every **embedding** table carries `run VARCHAR(64)` as its **leading PK column** (harrier =
  `baseline`; a future gte pass would be `gte-qwen2`). The driver/`sweep_levers.py` take `--run`.
- Every **verdict** table carries the judge `model` as its **trailing PK column** (the
  domain-general `gemma-4-31b` corpus + a `deepseek/deepseek-v4-pro` re-judge side by side, plus
  the §9 AGREE-anchor ablation tags), so a newer judge never overwrites an older at the same key.

`init_schema` runs an idempotent, guarded `migrate_runs` that adds these columns / rebuilds PKs
on an existing DB. Tables:
- **`doc_vectors`** (`run, code_number, pdf_path, method, chunk_idx, dim, vec`), **`pole_vectors`**
  (`run, criterion, pole, dim, vec`), **`control_vectors`** (`run, name, dim, vec`) — the **raw
  vectors** that make offline `--recompute` possible.
- **`embedding_scores`** — one row per `(run, code_number, pdf_path, method, criterion)` with `e`,
  `model` (the *embedding* model), `run_ts`. `--recompute` upserts `e` only.
- **`verdicts`** — one row per `(code_number, pdf_path, criterion, model)`; columns `verdict`,
  `confidence`, `graded`, `prompt_hash`, `run_ts`. Labels are shared across *embedding* runs
  (judged once per judge, JOINed on `(code_number, pdf_path, criterion)`). `chunk_verdicts` is
  likewise judge-keyed (PK `…, chunk_idx, model`) and carries the per-chunk diagnostics.
  `fetch_report` / `fetch_chunk_verdicts` take an optional `judge=` filter; unfiltered, the
  newest judge wins per key (`ORDER BY run_ts` last-wins).
- **`prompt_registry`** (`prompt_hash` PK, `criterion`, `prompt_text`, `run_ts`) — prompt
  provenance: `prompt_hash = criteria_judge.prompt_hash(criterion)` is a sha256 over the
  version-bearing prompt scaffold, stamped onto every verdict/chunk_verdict so prompt versions
  are distinguishable; `register_prompts` stores each version's full template text once.
- **`topic_centroids`** (`run, topic_id, label, dim, vec`), **`chunk_topics`** (`run, pdf_path,
  chunk_idx, method, topic_id, sim`) — the §2.1 layer; both run-keyed.
- **`pole_separation`** (incl. centred `within` rows), **`control_scores`**, **`run_meta`** (lever
  params + scoring mode) — all run-scoped.

## 4. The four space-level levers — offline scoring (`embed_score.py`)

The original double-cosine `e = cos(paper, POS) − cos(paper, NEG)` under-discriminated: a
topicality halo and the decoder-only-embedder anisotropy meant in-register text all sat in a
narrow cone. The fix is **space-level, not prompt-level**:

```
μ      = mean of all document vectors                       # centring origin
B      = whiten_basis(center(reps, μ), k)                   # top-k PCs to strip (--whiten-k)
a_c    = normalize(p̂_c − n̂_c)        on centred poles       # axis-projection contrast
ŝ      = shared_direction({a_c})     = first PC of the axes  # shared register direction
a_c⊥   = orthogonalize(a_c, ŝ, strength)                    # partial out topicality (--shared-strength)
e_c(d) = a_c⊥ · normalize( whiten(d − μ, B) )               # chunk windows max-pooled
```

`recompute(doc_vecs, poles, k, strength)` is the pure composition; `build_axes`, `whiten_basis`,
`shared_direction`, `orthogonalize`, `axis_score` are unit-tested pieces. `within[c] = cos(centred
pos, centred neg)` (pole width) is recomputed and rendered.

**Tunables (CLI flags):**
- `--shared-strength` (default `0.5`) — how hard each axis is orthogonalized against the shared
  register direction. `1.0` over-corrects; `0.5` removes the halo, keeps real signal.
- `--whiten-k` (default `0`) — number of top PCs removed.

**Live decision: keep `k=0, strength=0.5`.** A corpus-scale sweep (`sweep_levers.py`,
free/offline, n=219) found whitening a dead end (`k≥1` collapses the concrete criteria),
`strength=0.5` near-optimal, and no cell winning all three — per-criterion argmax was rejected as
overfitting synthetic labels. The binding constraint is the model + pole separation (`within`
~0.51–0.68), not the levers. Full sweep table: `@test_runs.md` Run 1. `sweep_levers.py` is kept
as the re-runnable diagnostic for a future model's vectors.

## 5. Current result — ρ is measurable within coherent strata, not corpus-wide

The embedding corpus is **219 papers** (after dropping the in-corpus self-reference); the whole
corpus now carries graded **domain-general** verdicts (§9). The headline:

- The embedding axis tracks the verdict direction positively, **strongest on `two_worlds`**
  (most concrete), **weakest on `arbitrariness`** (subtlest, very few positives). Pole widths
  overlap, so **ranks are trustworthy, absolute magnitudes less so**.
- **Corpus-wide ρ(e, domain-general verdict) collapsed by design** (`@test_runs.md` Run 6): the
  cross-domain judge marks non-molecular "codes" met across the neuro strata while `e` keys on
  **molecular** prototype poles — the two axes measure different constructs there. **Within the
  molecular tail, where both axes measure the same thing, the agreement holds** (`adaptors` graded
  ρ ≈ +0.31, matching the pre-redesign judge). Rebuilding the poles domain-general did *not*
  re-couple them (Run 7), confirming the §8 verdict.
- **Per-topic ρ is diagnostic only** — it holds the halo fixed and the axis works within coherent
  strata (e.g. Morphological Codes, Binding Code), but most strata are flat `not_met` / `n/a`.

Full distributions, ρ tables and per-topic breakdown: `@test_runs.md` Runs 2, 6, 7.

## 6. ⚠️ Major caveats (the verdicts are not ground truth)

ρ measures agreement between two imperfect axes, not correctness. The verdicts are synthetic and
have **not** been validated against a gold set:
- **Poor calibration** — the pre-redesign `confidence` field clustered at 0.95–1.0 with no usable
  gradation (which is *why* the embedding axis exists, and why §9 moved gradation onto
  `agreement`).
- **Literal latching** — prompts read too literally, latching onto an isolated sentence rather
  than the sentence in the context of its paragraph. The §9 grounding gate + graded axis target
  exactly this.

**The genuine binding constraint is label quality, not the embedding axis (§8).** The live next
step is a gold-set validation, not more axis tuning.

## 7. Development, testing & reporting rules

1. **TDD** — for any new or changed functionality, write a failing test first, then the change.
   The suite is **274 tests, fully offline**, under **`tests/`** (root `conftest.py` puts the
   repo root on `sys.path`).
2. **Language** — pythonic, readable; prefer numpy for data management.
3. **Run logs → `./logs/`** (gitignored). Write **all** run logs (background jobs, judge/embed
   driver logs) there; keep only the actively-running job's log elsewhere. Don't scatter logs in
   the repo root.
4. **Transient artifacts → `./json/`** (gitignored). Stale/superseded snapshots, retired
   checkpoints, and duplicate HF tokenizer copies are archived under `./json/`, not the repo
   root. MySQL is the system of record; the canonical pipeline I/O paths (`embed_out.json`,
   `prototypes.json`, the live judge checkpoint, the download caches) stay in the repo root where
   the code's defaults expect them.
5. **Commit cadence** — pause and commit after each completed logical unit; small, self-contained
   commits.
6. **Spend safety** — paid (DeepSeek V4 Pro) work checkpoints to a resumable per-chunk JSONL
   **before** MySQL persistence; **never delete the checkpoint**.
7. **Secrets** — `.env` is gitignored and never committed; never print API keys.
8. **DB backup before schema changes** — always take a compressed `mysqldump` of `codebiology`
   **before** any schema change (new table/column, `ALTER`, migration, first `init_schema` on new
   DDL): `mysqldump … codebiology | gzip > codebiology_$(date +%Y%m%d_%H%M%S).sql.gz` (connection
   detail in `@environment_notes.md`; dumps are gitignored). Migrations are idempotent and
   guarded, but the dump is the non-negotiable rollback path.

## 8. Embedding side is at its ceiling — the constraint is label quality

Embedding-side tuning is judged exhausted, and all three escape hatches are spent:
- **Levers** — set (§4); the sweep confirmed the defaults.
- **Model swap** — a gte-Qwen2-7B-instruct runner was built, tested, and **deliberately shelved**:
  gte is the *same* decoder-only last-token architecture as harrier (the documented source of the
  anisotropy that overlaps the poles), so it is unlikely to widen them. Kept as a non-destructive
  tested artifact (`run` column). `@test_runs.md` Run 3.
- **Prototype/pole quality** — the poles were rewritten **domain-general** (`prototypes.json` rev
  2, balanced multi-domain exemplars) to match the §9.1 judge and re-embedded; they came out
  well-formed (`within` 0.59–0.64) but **corpus-wide ρ stayed flat** — confirming, not fixing, the
  constraint. `@test_runs.md` Run 7.

**The genuine constraint is now label quality, not the embedding** (§6) — the next real step is
the gold-set validation, not more embedding-side work.

## 9. Judge redesign — graded, per-chunk, topic-grounded, control-anchored

Acting on §6/§8 ("the constraint is label quality"), the LLM judge was rebuilt as a **graded**
(−1…+1 `agreement`), **per-chunk** (the *exact* 8192-token harrier embedding windows reproduced
via `chunk_text.reproduce_chunks` → tokenizer-aligned to `doc_vectors`), **topic-grounded**
(dominant scientometric topic injected as *context, not evidence*), **control-anchored**
(illustrative AGREE/DISAGREE exemplars from `prototypes.json` `_controls`), **calibrated** judge.
A grounding gate pulls any positive whose `evidence_quote` is not verbatim in the chunk back to
`0.0`; `aggregate_graded` max-pools chunks → `(graded_max, graded_mean, confidence, categorical)`
with `graded_max ≥ +0.5 → met`, `≤ 0.0 → not_met`, else `unclear`. Schema: `verdicts.graded
DOUBLE` + run-agnostic `chunk_verdicts` (§3). **Validation is distribution-comparison only — no
gold set (locked).**

**Pipeline.** `judge_pilot.py` (driver, top-N topics or `--rest` complement, resumable per-chunk
JSONL checkpoint) → `compare_verdicts.py --snapshot old.json` *before* / `--old old.json` *after*
(the new axis read from the never-overwritten `chunk_verdicts`). Free local Gemma-4-31B or paid
DeepSeek V4 Pro (§6 routing). Launcher / tokenizer / runtime detail: `@environment_notes.md`.

**State.** The whole 219-paper corpus carries domain-general verdicts (102 neuro top-4 + 117
molecular tail). The grounding gate held throughout (every positive quote-grounded), the judge is
*more* skeptical than the retired one, and gradation moved off the dead `confidence` field onto
`agreement` — materialising on the molecular tail (`adaptors` graded std 0.40). Results and the
corpus-wide ρ divergence (§5): `@test_runs.md` Runs 4–7. A **corpus-wide / paid all-criteria run
is gated on the gold-set plan, not on ρ(e, verdict)** — which no longer adjudicates the
domain-general judge corpus-wide (§5).

**AGREE-anchor ablation (done — negative result).** `judge_pilot.py --agree-anchors {genetic,
neural,neural-genetic}` swaps the AGREE exemplar's *domain* (molecular genetic 1-shot baseline vs
neural 1-shot vs neural+genetic 2-shot) to test whether the molecular anchor biases the judge
toward marking molecular passages met. Variants carry distinct judge tags (`@neural-1shot`,
`@neural-genetic-2shot`) so they coexist with the baseline (`AGREE_ANCHOR_VARIANTS`, tests in
`test_judge_pilot.py`). **Result (`@test_runs.md` Run 8, paid DeepSeek, 60-paper neuro set): the
anchor domain does not bias the judge** — swapping it moves ≤14% of keys at noise-level effect
sizes (≤0.033 on −1…+1); the verdicts are robust to the exemplar's domain. Confirms label quality,
doesn't threaten it.

### 9.1 Domain-general criteria — molecular-bias fix
The molecular-specific `CRITERIA_DEFS` mechanically rejected every non-molecular paper
(two_worlds **0** positives citing "not *molecular* worlds"). The three criterion definitions are
now **domain-general**: they instantiate per discipline across the 24 topics (codons↔amino acids
*or* stimulus↔spikes *or* sound↔percept …), the molecular genetic code being one **exemplar**, not
the requirement. The "adaptor" is generalised to the domain's **mediator** per **Major (2025),
*From Code to Archetype*** (the third term that reads/executes the mapping — tRNA/ribosome, nervous
system, imaginal function, computational engine). This broadens what the *judge* measures to match
the field's cross-domain claims; it does **not** alter Barbieri's strict molecular definition (§1).
The DB criterion **key stays `adaptors`** (PK stability); only its definition *text* changed.
Validating re-pilot: `@test_runs.md` Run 5.
