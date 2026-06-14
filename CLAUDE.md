# CLAUDE.md

## Project Context
This project processes Code Biology data from `biological_codes.csv` (derived from `Biological_Code_List_20260531.pdf`).
- **Expected Code Categories:** 435
- **Expected References:** 2299

## Code Biology Definitional Database
Beyond the code/citation list, the repo holds a corpus of the foundational
literature and the canonical web presence of the field. Together with
`biological_codes.csv` these define what Code Biology *is* — the seminal texts,
the society, and its conferences.

### What qualifies as an organic code — the minimum criteria
Per Barbieri's definition (`www.codebiology.org/index.html`), a biological code
is proven to exist only if **all three** of the following are demonstrated. The
test is deliberately objective and experimentally falsifiable:
1. **Two independent worlds of molecules** — two distinct sets of objects with no
   necessary physical/chemical link between them (e.g. codons and amino acids).
2. **A set of adaptors** — a *third* type of molecule that physically bridges the
   two worlds (e.g. tRNAs). Adaptors are the "molecular fingerprints" of a code;
   their presence is the empirical signature that coding, not mere chemistry, is
   at work.
3. **Arbitrariness of the coding rules** — the mapping is conventional, not
   dictated by physical law. The rules are *compatible* with physics/chemistry
   but **not determined** by them, so they could in principle be otherwise.

This distinguishes **coding** (about *meaning*; evolution by natural conventions)
from **copying** (about *information*; evolution by natural selection) — the two
are held to be irreducible to each other. Supporting terms (*Adaptor*,
*Arbitrariness*, *Natural conventions*) are formally defined in `glossary.html`.

Note: not every entry in the source PDF is a "code" in this strict molecular
sense — e.g. code 352 (SeqCode) is a nomenclature/naming code, a looser usage.

### `Code_Biology_PDFs/` — seminal texts
The core literature defining the discipline, primarily by Marcello Barbieri
(founder of the field) and Sam Major:
- **The Organic Codes: An Introduction to Semantic Biology** (Barbieri) — the
  founding monograph. Two copies: `The_Organic_Codes_an_introduction_to_semantic_biol.pdf`
  and `The_organic_codes_An_introduction_t_z_library_sk,_1lib_sk,.pdf`.
- **Introduction to Code Biology** — Barbieri (2014).
- **What Is Code Biology?** — Barbieri (2018).
- **Codes and Evolution — The Origin of Absolute Novelties** — Barbieri (2024).
- **Life and Semiosis: The Real Nature of Information and Meaning** — Barbieri.
- **A Simple Measure for Biocomplexity** — Barbieri.
- **Codes across (life)sciences** — interdisciplinary survey (final published).
- **Biological Codes: A Field Guide for Code Hunters.**
- **Archetypes and Code Biology** — Major (2021).
- **From Code to Archetype** — Major (2025) (`.pdf` + `.txt` transcript).

### `www.codebiology.org/` — society web mirror
A static mirror of the International Society of Code Biology website (102 HTML
pages, 31 PDFs). Key content:
- **`index.html`** — the field's public definition ("more than 200 biological
  codes have been discovered"; codes as a third component of life beyond
  chemistry and information).
- **`glossary.html`** — terminology by Barbieri, de Beule & Hofmeyr (~100 terms:
  semiosis, codepoiesis, organic meaning, Umwelt, …).
- **`brief-history.html`** — Barbieri's origin story of the field (genotype/
  phenotype/ribotype trinity, 1981 onward).
- **The society's published code lists** — `database.html` indexes two editions:
  - `database.pdf` — **First Database, December 2022: 237 codes.**
  - `second-database.pdf` — **Second Database, May 2026: 418 codes.**
  These are the upstream lineage of the project's source
  `Biological_Code_List_20260531.pdf`, but **not identical to it** — that file is
  a later sibling (different md5/size) from which extraction recovers **435**
  codes, vs the 418 the website quotes for the second edition. Treat the source
  PDF as authoritative for this project; the website figures are for context.
- **Society governance** — `society.html`, `members-of-the-society.html`,
  `application.html`, the governing-board pages (2012, 2018, 2022), `pdf/constitution.pdf`.
- **`conferences/`** — every ISCB meeting, one folder per event: Jena 2015,
  Urbino 2016, Hungary 2017, Granada 2018, Friedrichsdorf 2019, Luznica 2021,
  Olomouc 2022, Guimarães 2023, La Spezia 2024, Zagreb 2025, Guimarães 2026 —
  plus calls for papers, programmes, and `conferences/pdf/` abstracts/papers.
- **`videos.html`, `photogalleries.html`, `Immagini/`** — media archive.

## Current Status
The extraction pipeline is implemented and passing its test suite.

- **`extract_csv.py`** — parses the PDF directly with `pdfplumber` and emits
  `biological_codes.csv` with columns `Code Number, Code Name, Paper Name, URL`.
  Each citation in the source is a hyperlink whose anchor text is the full
  reference, so hyperlink runs (not text splitting) are the extraction anchor.
- **`test_extract.py`** — pytest suite (18 tests) covering code coverage,
  contiguity, known per-code reference counts, URL validity, cross-page
  continuation, and cross-row hyperlink-bleed regressions. Run with `pytest`.
- **Output:** 2290 references across all **435** codes (within tolerance of the
  quoted 2299). One reference (code 352, SeqCode) has citation text but no
  hyperlink, so it carries an empty URL.

## PDF Availability
`download_pdfs.py` fetches the full-text PDF for each reference. It derives a
DOI from the URL, then tries, in order: the Crossref `application/pdf` link,
every Unpaywall open-access location, and the landing page's
`citation_pdf_url` meta tag. It is **legal-OA only** — no Sci-Hub or paywall
circumvention. See `test_download_pdfs.py` (30 tests, fully offline).

**Current state (last full run, from a non-institutional home network):**
- **471 of 2240 unique references downloaded (~21%)** → `pdfs/` (gitignored, 2.1 GB;
  regenerate with `python3 download_pdfs.py`). 49 duplicate citations share a DOI.
- **1769 not retrievable**, listed with reasons in `failed_downloads.csv`:
  - *Hard paywall, no OA copy* (the bulk) — Elsevier/ScienceDirect, Cell,
    Nature-subscription, Springer, Wiley, OUP, AAAS/Science, T&F. No legal
    source without a subscription.
  - *CDN bot-blocked but actually OA* — e.g. MDPI (~56): Cloudflare returns 403
    to scripted clients. Downloadable by hand in a browser.
  - *PMC-only* — NCBI / Europe PMC block non-interactive PDF fetches from this
    network (403 / HTML interstitial on every endpoint).

**Resumability:** re-runs skip PDFs already on disk and reuse
`crossref_cache.json` + `unpaywall_cache.json`, so effort is spent only on new
attempts. Coverage would rise substantially from an institutional network
(EZproxy/OpenAthens) or with an Unpaywall-plus-repository proxy.


## Criteria Scoring — embeddings as an independent axis
A separate analysis scores how strongly each paper's text *argues* the three
criteria (`two_worlds`, `adaptors`, `arbitrariness`), to complement the categorical
LLM verdicts from `criteria_judge.py`. The LLM `confidence` is saturated (0.9–1.0)
and carries no gradation, so a **corpus-contrastive embedding** supplies the
continuous signal.

**Decision 0 (load-bearing):** the embedding axis is **independent** — reported
side-by-side with the verdicts, never overriding, gating, or band-merging them. The
verdict stays the categorical backbone.

### Three chunking methods (each fed as separate documents)
Each paper is embedded **three ways** to test which granularity best tracks the
verdict, surfaced as three columns (`e_full` / `e_abstract` / `e_chunk`) in
`report.md` + a `PER-PAPER VERDICTS` table with the verdict and its confidence:
- **full** — the whole (char-budget-capped) paper as one document.
- **abstract** — the abstract section only.
- **chunk** — 8192-token windows at 50% overlap (stride 4096), scored per window then
  **max-pooled** (strongest evidence anywhere).

### MySQL is the system of record (not JSON)
All embedding output is stored in **MySQL on asushimu** (conda mysqld, data dir
`asushimu:/nvme/mysql/data`, DB `codebiology`), **not** JSON. `db.py` owns the schema.
The GPU host returns a transient `embed_out.json` purely as transport — the driver
loads it into MySQL and deletes it. Connection params live in gitignored `.env`
(`DB_HOST/PORT/NAME/USER/PASS`); never commit it. Tables (vectors are float32 LE bytes):
- **`doc_vectors`** (`code_number, pdf_path, method, chunk_idx, dim, vec LONGBLOB`) and
  **`pole_vectors`** (`criterion, pole, dim, vec`) — the **raw vectors** that make the
  offline `--recompute` levers possible (the Run 2 structural unlock).
- **`embedding_scores`** — one row per `(code_number, pdf_path, method, criterion)` with
  `e` / `verdict` / `confidence`; **`code_number` is the leading PK column**.
  `--recompute` upserts `e` only, preserving the verdict.
- **`pole_separation`** — pole widths incl. the centred `within` rows; **`control_scores`**;
  **`run_meta`** (lever params + scoring mode). `report.md` regenerates from the DB
  (`--report-only` reads scores; `--recompute` rescores then writes).

### Architecture (Run 2): GPU emits vectors, the driver scores offline
Run 1's double-cosine `e = cos(paper, POS) − cos(paper, NEG)` under-discriminated
(a topicality halo let code 428 top *all three* criteria; controls barely separated —
the classic decoder-only-embedder anisotropy). The fix is **space-level, not
prompt-level**, so the architecture split in two:

- **`run_harrier_embed.py` (GPU, asushimu's 3090 Ti)** emits **raw vectors only** — the
  document vectors (full / abstract / each chunk) and the pooled pole vectors (3 criteria
  × pos/neg). It no longer computes `e`. Loads `microsoft/harrier-oss-v1-27b` (Gemma3-27B
  decoder-only embedder, 5376-dim, MIT) via sentence-transformers in **4-bit**
  (bitsandbytes nf4, bf16 compute, ≈13.5 GB).
- **`embed_independent.py` (driver, this host)** persists the vectors to MySQL, then
  computes `e` **offline** with the four levers below. After **one** structural GPU
  re-run every lever is re-tunable and the report regenerates with **no further GPU** via
  `--recompute` (sibling of `--report-only`).

### Four space-level levers — the offline `--recompute` scoring (`embed_score.py`)
```
μ      = mean of all document (rep) vectors                 # centring origin
B      = whiten_basis(center(reps, μ), k)                   # top-k PCs to strip (lever: --whiten-k)
a_c    = normalize(p̂_c − n̂_c)        on centred poles       # axis-projection contrast (lever)
ŝ      = shared_direction({a_c})     = first PC of the axes  # shared register direction
a_c⊥   = orthogonalize(a_c, ŝ, strength)                    # partial out topicality (lever: --shared-strength)
e_c(d) = a_c⊥ · normalize( whiten(d − μ, B) )               # chunk windows max-pooled
```
- `recompute(doc_vecs, poles, k, strength)` is the pure composition; `build_axes`,
  `whiten_basis`, `shared_direction`, `orthogonalize(axis, shared, strength)` and
  `axis_score` are the unit-tested pieces. `within[c] = cos(centred pos, centred neg)`
  (pole width) is recomputed on the centred poles and **rendered** in `report.md`.

#### Tunables (smoke-test-calibrated defaults, both CLI flags)
- **`--shared-strength` (default `DEFAULT_SHARED_STRENGTH = 0.5`)** — how hard each
  criterion axis is orthogonalized against the shared register direction. `1.0`
  over-corrects (collapses 428's *legitimate* `two_worlds` along with its `arbitrariness`
  halo); `0.5` removes the halo while keeping real signal.
- **`--whiten-k` (default `DEFAULT_WHITEN_K = 0`)** — number of top PCs removed. `k≥1`
  hurt on the 20-paper sample (the top PC still carried signal); revisit at corpus scale
  where the PC estimate is trustworthy.
- **`--recompute`** — rescore from the persisted vectors with the above flags, no GPU;
  upserts `e` only (verdict/confidence preserved) and regenerates `report.md`.
- `pdf_text.py` — `extract_abstract()` (abstract heading, preamble fallback).

### Critical environment assumptions (hard-won)
- **GPU pinning:** the 3090 Ti is **GPU index 2** under
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`. The two GTX 1080 Tis are sm_61, **unsupported** by
  torch 2.8 — always run with `CUDA_VISIBLE_DEVICES=2`.
- **VRAM ceiling → token cap:** a **32k-token forward pass OOMs** 27B/4-bit on the
  24 GB card (a 115k-char paper at 32k tokens fails; ~23k tokens used 20.8 GB).
  The `full`/`abstract` methods are therefore capped at **`--max-seq 16384`**, and the
  run sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + `torch.cuda.empty_cache()`
  per doc. `full` thus embeds the first ~16k tokens of long papers; `chunk` gives full
  coverage. The 8192-token chunk windows are proven to fit.
- **Dependency pins on asushimu:** `peft>=0.11` (the bundled `0.4.0.dev0` lacks
  `PeftModelForFeatureExtraction` that ST 5.x imports), `numpy<2` (ABI), `pyarrow<17`.
- **Run logging:** each run logs total embeds per method up front, then per doc a
  stable `id=<pdf-stem>`, `[doc i/N]`, and a running `done/total` per method.

### Findings — Run 2 leverred (220-paper corpus, 2026-06-14)
The one structural GPU re-run landed (220 papers × 3 methods = 962 chunk windows, 5376-dim;
6 pole vectors). Scored offline (`--recompute`, `k=0`, `strength=0.5`).
- **The 428 halo collapsed (the headline win).** In Run 1 code 428 topped *all three*
  criteria including `arbitrariness` (`not_met`). At corpus scale (111 codes) it is now
  mid-pack everywhere: `two_worlds` (met) rank 32/111, `adaptors` (met) rank 12, and
  crucially `arbitrariness` (not_met) rank 22 — no longer dominating. Its strongest
  criterion is now a `met` one (adaptors). Mild over-correction on `full` (its met
  `two_worlds` at rank 32 dips below its not_met `arbitrariness` at 22), but both middling
  and the absolute `e` gap is tiny.
- **Pole widths `within` (centred cos pos↔neg, lower = better separated):** `arbitrariness`
  +0.515 (best), `adaptors` +0.629, `two_worlds` +0.680. Poles still overlap enough that
  **ranks are trustworthy, magnitudes less so**.
- **A self-corpus leak surfaced:** code 321 (`www.codebiology.org/conferences/Guimaraes…`,
  an in-corpus Code Biology *conference* document, not a primary paper) now tops most
  criteria — it reads maximally in-register because the poles are mined from that same
  corpus. Such meta-documents should be excluded/flagged.
- **Two honest gaps.** (1) Control scores in `report.md` are still the **pre-lever Run-1
  contrastive** values — control document vectors aren't persisted, so `--recompute` can't
  re-score them; refreshing them needs one re-embed that captures control vectors.
  (2) Spearman ρ is unchanged (+0.522 / +0.522 / n-a) — the labelled-verdict subset is too
  small and imbalanced (`arbitrariness` has no positives) to *measure* the improvement, so
  the levers' gain shows in the 428 ranks, not in ρ.

### Findings — Run 1 contrastive (10-paper, 2026-06-14), the motivation for Run 2
These are the **pre-lever, double-cosine** results; the three structural weaknesses here
are what the four levers above target.
- Controls behave: genetic-code reads positive on all three criteria; deterministic-
  chemistry negative on all three and **most negative on `arbitrariness`**. The lone
  `met` paper (code 428) tops every method on `two_worlds`/`adaptors` (Spearman ρ ≈ +0.52).
- **The topicality halo (lever target):** 428 *also* tops `arbitrariness`, where its
  verdict is `not_met` — `e` partly ranks "how genetic-code-flavoured" not "argues *this*
  criterion." On the 20-paper DB the criteria axes are ~0.74 co-aligned with the shared
  direction; `--shared-strength 0.5` knocks 428's `arbitrariness` off the podium (rank 5)
  without losing its legitimate `adaptors` (rank 2) / `two_worlds` (rank 6).
- **full ≈ abstract ≈ chunk** (identical ρ) — granularity is not where signal hides.
- Absolute `e` is small (±0.05) and pole-pair cosines high (poles overlap) — the
  corpus-mined poles + centring/whitening aim to widen this before magnitudes are trusted.

## Code development rules
1. **Testing:** For any new functionality or changes to existing functionality always write or expand code using TDD. Write a failing test first, then the feature or change
2. **Language** Always write in pythonic readable python and prefer numpy for data management
3. **Logging** Always use pythonic logging and choose DEBUG, INFO, levels suitably depending on criticality and informationality of the code
4. **Commit cadence** Pause and commit after each completed task (one logical unit of work). Keep commits small and self-contained; do not batch multiple tasks into one commit.

### ⚠️ Deferred revert (PROJECT END)
The production `llama-server` (Home Assistant voice agent) is **OFFLINE** — the
3090 Ti was freed for embedding. Restore **at project end only**:
`cp ~/start_llama.prod.bak ~/start_llama.sh && sudo systemctl restart llama-server`.

