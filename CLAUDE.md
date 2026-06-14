# CLAUDE.md

## Project Context
This project processes Code Biology data from `biological_codes.csv` (derived from `Biological_Code_List_20260531.pdf`).
- **Expected Code Categories:** 435
- **Expected References:** 2299

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

### Key implementation notes
- A code's citation list can spill across pages; the "current" code is carried
  over page boundaries since continuation pages lack a left-column number.
- A hyperlink rectangle sits a few points *above* its row's number digit, so row
  bands are shifted up by `ANCHOR_SLACK` to stop a row stealing the next row's
  first link.
- Data-integrity logging: a `(YYYY)` year heuristic cross-checks the text
  citation count against the URL count and logs a WARNING on mismatch.

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

## Criteria Scoring — embeddings as an independent axis
A separate analysis scores how strongly each paper's text *argues* the three
criteria (`two_worlds`, `adaptors`, `arbitrariness`), to complement the categorical
LLM verdicts from `criteria_judge.py`. The LLM `confidence` is saturated (0.9–1.0)
and carries no gradation, so a **corpus-contrastive embedding** supplies the
continuous signal: `e = cos(paper, POS_prototype) − cos(paper, NEG_prototype)`, with
register-matched negative poles so shared "code biology" vocabulary cancels.

**Decision 0 (load-bearing):** the embedding axis is **independent** — reported
side-by-side with the verdicts, never overriding, gating, or band-merging them. The
verdict stays the categorical backbone.

### Pipeline
- **`embed_score.py`** — pure scoring math (poles, contrastive score, `token_windows`,
  `aggregate_chunks`). Fully unit-tested offline.
- **`pdf_text.py`** — adds `extract_abstract()` (abstract heading, preamble fallback).
- **`run_harrier_embed.py`** — runs **on asushimu's 3090 Ti**. Loads
  `microsoft/harrier-oss-v1-27b` (Gemma3-27B decoder-only embedder, 5376-dim, MIT)
  via sentence-transformers in **4-bit** (bitsandbytes nf4, bf16 compute, ≈13.5 GB).
- **`embed_independent.py`** — driver on this host: extracts paper text → ships
  input + runner to asushimu over SSH → runs pinned to the 3090 Ti → loads results
  to MySQL → generates `report.md` from the DB.

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
`asushimu:/nvme/mysql/data`, DB `codebiology`), **not** JSON. `db.py` owns the schema;
the main table `embedding_scores` has **`code_number` as the leading primary-key
column**, one row per `(code_number, pdf_path, method, criterion)`. The GPU host
returns a transient `embed_out.json` purely as transport — the driver loads it into
MySQL and deletes it. `report.md` is regenerated from the DB (`--report-only`).
Connection params live in gitignored `.env` (`DB_HOST/PORT/NAME/USER/PASS`); never commit it.

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

### Findings (10-paper run, 2026-06-14)
- Controls behave (validity check passes): genetic-code reads positive on all three
  criteria; deterministic-chemistry reads negative on all three and **most negative on
  `arbitrariness`**. The lone `met` paper (code 428) tops every method on `two_worlds`
  and `adaptors` → embedding doesn't contradict the verdict (Spearman ρ ≈ +0.52).
- **full ≈ abstract ≈ chunk** (identical ρ, near-identical magnitudes) — abstract-only
  is as informative as full text on this sample.
- Caveat: absolute `e` is small (±0.05) and pole-pair cosines are high (poles partly
  overlap) — fine for ranking/triage, widen poles before trusting magnitudes.

### ⚠️ Deferred revert (PROJECT END)
The production `llama-server` (Home Assistant voice agent) is **OFFLINE** — the
3090 Ti was freed for embedding. Restore **at project end only**:
`cp ~/start_llama.prod.bak ~/start_llama.sh && sudo systemctl restart llama-server`.

## AI Goals & Responsibilities
- **Primary Task:** Parse the CSV to process the codes and their associated citations.
- **Specific Extraction:** For every code, parse the references to isolate the **paper name** and map it directly to its corresponding **hyperlink/URL**.
- **Output:** Generate a clean, structured format (e.g., JSON or cleaned CSV) mapping `Code -> Paper Name -> URL`.

## Rules for AI Agents
1. **Data Integrity:** Extract exactly what is in the CSV columns. Do not hallucinate references or URLs.
2. **Data Parsing:** Handle string splitting carefully, as multiple citations and URLs are bundled in single cells.
3. **Libraries:** Default to Python (`pandas`, `re`) for string manipulation and data extraction. 
4. **Error Handling:** Log any code categories where the number of parsed paper names does not match the number of parsed URLs.

## Code development rules
1. **Testing:** For any new functionality or changes to existing functionality always write or expand code using TDD. Write a failing test first, then the feature or change
2. **Language** Always write in pythonic readable python and prefer numpy for data management
3. **Logging** Always use pythonic logging and choose DEBUG, INFO, levels suitably depending on criticality and informationality of the code
4. **Commit cadence** Pause and commit after each completed task (one logical unit of work). Keep commits small and self-contained; do not batch multiple tasks into one commit.
