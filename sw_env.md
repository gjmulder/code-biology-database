# sw_env.md — Code Biology database

> Project-specific operational reference: runbooks, command lists, per-tool
> tables, the GPU-host launcher/model map, MySQL and backup mechanics. Read
> lazily — a CLAUDE.md § ending "→ sw_env.md" names the heading to read before
> that op. **New reference is appended here, never to CLAUDE.md.** Hardware /
> host facts go to the shared `hw_env.md` (a symlink to `~/Work/homelab/hw_env.md`),
> never here. The chronological run log lives in `test_runs.md`.

**Nature of this project.** There is **no service of its own and no deploy
cycle** — this is a batch research pipeline run **locally on miniconda Python**
that drives GPU inference on **asushimu** over SSH/LAN and persists to **MySQL**
(the system of record). The "services" below are the asushimu GPU processes the
pipeline *consumes*; the only files that land on a remote host are the launcher
scripts and the GPU-side embed script (see "Remote-installed file map").

## Services & running

The 3090 Ti hosts a single `llama-server` at a time; production (the marvinha
voice agent) and this project's batch profiles are **mutually exclusive** on that
card. Running any batch profile takes the voice agent **offline** until prod is
restored. GPU pinning, VRAM ceilings and card UUIDs are in `hw_env.md`.

### `llama-server` — LLM judge (three launcher profiles)

- Host **asushimu** · systemd unit `llama-server.service` · ExecStart
  `/home/mulderg/start_llama.sh` · pinned to the 3090 Ti (UUID
  `GPU-d355aaa9-…` via `CUDA_VISIBLE_DEVICES`; indices are unstable — see
  `hw_env.md`).
- Health: `http://asushimu:11434/health` · log `/var/tmp/llama-server.log`
- Controls:
  ```bash
  ssh asushimu 'sudo systemctl restart llama-server'   # apply a launcher swap
  ssh asushimu 'sudo systemctl status  llama-server'   # state + main PID
  ssh asushimu 'tail -F /var/tmp/llama-server.log'     # server log
  ```
- **Production profile** `start_llama.sh` — Gemma-4-31B-it (dense) Q5_K_M **+ MTP**
  @ `--ctx-size 16384`. This belongs to the marvinha voice agent, **not** this
  project. Backed up md5-identical to `~/start_llama.prod.bak`. Restore with:
  ```bash
  ssh asushimu 'cp ~/start_llama.prod.bak ~/start_llama.sh && sudo systemctl restart llama-server'
  ```
- **Batch profile** `start_llama_batch.sh` (repo file of the same name) — for the
  full-paper criteria-1&2 judge (`criteria_judge.py`). Gemma-4-31B Q5_K_M, alias
  `gemma-4-31b`, **MTP dropped** (`NO --model-draft`/`--spec-type`),
  **`--ctx-size 32768`**, else identical to prod (q8_0 KV, jinja, deepseek
  reasoning). Trades ~2× decode (~33 tok/s dense) for the 32k context that fits
  ~87 % of papers whole (16k fits only ~24 %). At 32k, VRAM ≈ 23.3/24.6 GB (~1.2 GB
  free). If it OOMs, drop `--ctx-size` to 24576 (still >50 % of papers whole).
- **Pilot profile** `start_llama_pilot.sh` (repo file of the same name) — for the
  graded per-chunk judge (CLAUDE.md §9). Gemma-4-31B Q5_K_M, alias `gemma-4-31b`,
  **MTP off** (MTP forces `--parallel 1`, blocking continuous batching), **`--parallel
  2 --ctx-size 32768`** (16384 tokens/slot = two 8192-token chunks; ctx-size must be
  an integer multiple of `--parallel` — `--parallel 3 --ctx-size 40960` SIGSEGV'd on
  the non-integer 13653.33/slot), **`--predict 4096`** (prod has none — bounds
  per-call output so Gemma's reasoning preamble + JSON can't run away and truncate;
  `--predict 2048` cut dense-chunk JSON mid-object, `test_runs.md` Run 5). Else
  mirrors prod. VRAM steady ~23.97/24.6 GB.
- **Swap runbook:**
  ```bash
  scp start_llama_pilot.sh asushimu:/home/mulderg/           # ship the profile
  ssh asushimu 'cp ~/start_llama_pilot.sh ~/start_llama.sh && sudo systemctl restart llama-server'
  # … run the batch/pilot job to completion + persistence …
  ssh asushimu 'cp ~/start_llama.prod.bak ~/start_llama.sh && sudo systemctl restart llama-server'  # restore prod
  ```
  **Never restore prod while a judge/embed run is still writing.** The script's
  `pkill` cleanup is a no-op under a clean systemd start; `> $LOG 2>&1` survives the
  `exec`, so the unit's Main PID is the `llama-server` binary itself.

### `criteria_judge.py` routing (local Gemma + off-box DeepSeek)

- Criteria 1 & 2 → **local Gemma-4-31B** on the batch/pilot `llama-server` (free).
- Criterion 3 (*arbitrariness*, the subtle one) → **DeepSeek V4 Pro** via
  **OpenRouter**, off-box (key in `.env`, gitignored). `criteria_judge.run_batch`
  runs `DEFAULT_WORKERS=6` papers in a `ThreadPoolExecutor`; the OpenRouter call is
  the overlapping bottleneck while local Gemma (`--parallel 1/2`) serialises its
  share. Resumable JSONL checkpoint + per-paper failure isolation. **Paid models
  return 402 without credit — confirm the OpenRouter balance before a full batch.**

### harrier embedder (embedding axis, GPU)

- Model `microsoft/harrier-oss-v1-27b` at `/data/vllm/harrier-oss-v1-27b`
  (Gemma3-27B decoder-only embedder, 5376-dim, MIT). Loaded via
  sentence-transformers in **4-bit** (bitsandbytes nf4, bf16 compute, ≈13.5 GB;
  bf16 would be ~54 GB).
- **Not a persistent server** — `embed_independent.py` runs `run_harrier_embed.py`
  **ON asushimu over SSH** (mkdir remote dir → run → fetch transient `embed_out.json`
  → `rm` remote copy). Pin `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2`,
  `--max-seq 16384` (VRAM cap → "GPU / VRAM invariants").
- **gte model-swap variant (shelved, CLAUDE.md §8)** — `run_gte_embed.py` talks to a
  **transient** gte-Qwen2-7B `llama-server` embedding server (`start_llama_embed.sh`,
  OpenAI-compatible `/v1/embeddings` on `:11600`, last-token pooling, dim 3584). This
  is **not** the systemd unit; start it only at project-end when prod is already
  offline and freed.

## Python environment

<!-- This project runs on ambient miniconda Python — NOT uv. There is no
     pyproject.toml/uv.lock/requirements.txt; deps live in the active conda env. -->

- **Interpreter:** `/home/mulderg/miniconda3/bin/python`. Run everything through it
  (`python …`, `pytest …`); there is no project-local `.venv`.
- **Dependencies are ambient in the conda env** — there is no lockfile. Core
  runtime deps: `pdfplumber`, `numpy`, `sentence-transformers`,
  `mysql-connector`/`db.py`'s driver, `openai`/OpenRouter client, `pytest` +
  `pytest-cov`. The offline test suite needs none of the GPU/DB/network stack (fake
  encoder/tokenizer; see CLAUDE.md § Testing).
- **asushimu-side pins (hard-won, GPU host only):** `peft>=0.11` (the bundled
  `0.4.0.dev0` lacks `PeftModelForFeatureExtraction` that ST 5.x imports),
  `numpy<2` (ABI), `pyarrow<17`. Any CUDA torch build must still ship Pascal
  (sm_61) kernels for the 1080 Tis — the cu121 wheels do.

## Quality gate

<!-- No ruff/mypy configured for this project — the gate is the offline suite. -->

```bash
python -m pytest                       # full offline suite — run before EVERY commit
python -m pytest --cov --cov-report=term-missing   # coverage (config in .coveragerc)
```

`.coveragerc` scopes coverage to the **live pipeline only** — retired drivers
(`judge_corpus.py`, `run_sample.py`) and the live-API smoke script
(`smoke_openrouter.py`) are omitted, and `main()`/`__main__`/live-I/O glue is
excluded from the report (exercised by the drivers, not the offline suite).

## Environment variables

<!-- Every env var the project reads. All live in the gitignored .env; the DB
     vars are loaded by run_sample.load_env(). New var ⇒ add a conftest dummy in
     the same commit. -->

| Var | Purpose | Set where |
|---|---|---|
| `OPENROUTER_API_KEY` | DeepSeek V4 Pro (criterion 3 + gold re-judge) | `.env` |
| `DB_HOST` | MySQL host (asushimu, over LAN) | `.env` |
| `DB_PORT` | MySQL port | `.env` |
| `DB_NAME` | Database (`codebiology`) | `.env` |
| `DB_USER` | MySQL user | `.env` |
| `DB_PASS` | MySQL password | `.env` |

**Never commit `.env`; never print keys.**

## Remote-installed file map

<!-- This project deploys no service, but these files are copied to asushimu and
     run there. The canonical copy is in this repo; the asushimu copy is a
     transient artefact. New GPU-host file ⇒ new row, same commit. -->

| Repo path | asushimu path | Installed by |
|---|---|---|
| `start_llama_batch.sh` | `~/start_llama_batch.sh` → `cp` over `~/start_llama.sh` | `scp` (swap runbook) |
| `start_llama_pilot.sh` | `~/start_llama_pilot.sh` → `cp` over `~/start_llama.sh` | `scp` (swap runbook) |
| `start_llama_embed.sh` | `~/start_llama_embed.sh` | `scp` (shelved gte swap) |
| `run_harrier_embed.py` | staged in a remote work dir | `embed_independent.run_remote` (SSH) |
| `run_harrier_centroids.py` | staged in a remote work dir | SSH (topic centroids) |

## Runtime state files

<!-- MySQL is the system of record; these on-disk files are checkpoints, caches,
     and transient transport. Design invariants (never-delete the paid checkpoint;
     canonical I/O paths stay in repo root) live in CLAUDE.md §7. -->

| File | Purpose | Writer | Retention |
|---|---|---|---|
| `*_deepseek_verdicts.jsonl`, `sample_verdicts.jsonl`, etc. | Resumable per-chunk judge checkpoints (spend safety — **never delete**) | `judge_pilot.py` / `criteria_judge.py` | permanent |
| `embed_out.json` | Transient GPU→driver vector transport (deleted after load) | `run_harrier_embed.py` (remote) | ephemeral |
| `crossref_cache.json`, `unpaywall_cache.json` | DOI→PDF resolution caches (resumable downloads) | `download_pdfs.py` | permanent |
| `prototypes.json` | Pos/neg pole passages + `_controls` (canonical input) | hand-curated | tracked-ish (repo root) |
| `gold_set.csv` | Git-tracked gold ground truth → `gold_labels` | `build_gold_set.py` | git-tracked |

Stale/superseded snapshots and duplicate tokenizer copies are archived under
`./json/` (gitignored), not the repo root; run logs under `./logs/` (gitignored).

## Per-tool reference

### MySQL — system of record

- Runs on **asushimu** (conda `mysqld`, data dir `asushimu:/nvme/mysql/data`), DB
  **`codebiology`**. `db.py` owns the schema and connects via `db.connect()` using
  the `.env` `DB_*` params; the driver host reaches it over the LAN.
- Vectors are stored as **float32 little-endian bytes** in `LONGBLOB` columns. The
  full schema and versioning keys (`run` PK-leading on embedding tables, judge
  `model` PK-trailing on verdict tables) are in CLAUDE.md §3.

### DB backup before schema changes (CLAUDE.md §7.8)

Take a compressed dump **before** any schema change (new table/column, `ALTER`,
migration, first `init_schema` on new DDL):

```bash
mysqldump --single-transaction --no-tablespaces … codebiology | gzip > codebiology_$(date +%Y%m%d_%H%M%S).sql.gz
```

- `--no-tablespaces` is **required** — the pipeline DB user lacks the global
  `PROCESS` privilege, so without it `mysqldump` errors probing tablespaces.
- `--single-transaction` gives a consistent snapshot without locking.
- Dumps are gitignored. Run with the `.env` connection params.

### GPU / VRAM invariants (drive the token caps)

- **GPU pinning:** the 3090 Ti is **CUDA index 2** under `CUDA_DEVICE_ORDER=PCI_BUS_ID`;
  the two 1080 Tis are sm_61, **unsupported by torch 2.8** — always run harrier with
  `CUDA_VISIBLE_DEVICES=2`. (Full card table / UUIDs → `hw_env.md`.)
- **VRAM ceiling → token cap:** a **32k-token forward pass OOMs** 27B/4-bit on the
  24 GB card (a 115k-char paper at 32k tokens fails; ~23k tokens used 20.8 GB). The
  `full`/`abstract` methods are capped at **`--max-seq 16384`**, and the run sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + `torch.cuda.empty_cache()` per
  doc. `full` thus embeds the first ~16k tokens of long papers; `chunk` (the live
  granularity) gives full coverage — the 8192-token windows are proven to fit.
- **Run logging:** each embed run logs total embeds per method up front, then per doc
  a stable `id=<pdf-stem>`, `[doc i/N]`, and a running `done/total` per method.

### Downloading a new model to `/data/vllm/`

Always detach via `nohup` and pipe to a log — downloads take 10+ minutes and a
dropped SSH session kills an attached transfer. Any env with `huggingface_hub`
works (the old `vllm` conda env does):

```bash
ssh asushimu 'nohup /home/mulderg/anaconda3/envs/vllm/bin/python -c "
import os; os.environ[\"HF_HOME\"]=\"/data/hfcache\"
from huggingface_hub import snapshot_download
snapshot_download(repo_id=\"<org>/<repo>\", local_dir=\"/data/vllm/<dir>\", max_workers=2)
" > /tmp/hf-download.log 2>&1 </dev/null & echo $!'
```

**Use `max_workers=2`, not the default 8.** asushimu's resolver saturates under 8
parallel HEAD requests to `huggingface.co`; the symptom is repeated
`'[Errno -3] Temporary failure in name resolution' … HEAD https://huggingface.co/…`
while the big shards that started first *do* finish, and the remaining shards sit in
exponential backoff indefinitely. Kill and restart with `max_workers=2`; completed
shards are skipped via `.cache/huggingface/download/*.lock` markers (no re-download).
For single GGUF files: `huggingface-cli download <repo> <file.gguf> --local-dir <dir>`.
Verify progress rather than trust the PID:

```bash
ssh asushimu 'tail -5 /tmp/hf-download.log; du -sh /data/vllm/<dir>; ls /data/vllm/<dir>/*.safetensors 2>/dev/null | wc -l'
```

`DONE <path>` on the final log line means all files landed; a missing `DONE` means
check for `.incomplete` or missing shards.

## Code conventions — parked reference

<!-- Model-known practice kept out of CLAUDE.md's per-session token budget. -->

- `pathlib` over `os.path` string surgery; stringify only at the outermost API that
  demands it.
- `subprocess` with list args and `check=True` (or a deliberate return-code check) —
  never `shell=True` around interpolated input.
- Prefer `numpy` for the vector/tabular math (already the house style in
  `embed_score.py`); no nested dict/list gymnastics for columnar work.
