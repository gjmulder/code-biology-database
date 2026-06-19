# Environment Notes

Reference for operating the infrastructure. Not needed for normal code editing.

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

### MySQL (system of record for the embedding/verdict pipeline)
- Runs on **asushimu** (conda `mysqld`, data dir `asushimu:/nvme/mysql/data`), DB
  **`codebiology`**. `db.py` owns the schema and connects via `db.connect()`.
- Connection params live in the gitignored **`.env`**
  (`DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASS`) — **never commit `.env`**. The driver
  host reaches it over the LAN; loaded by `run_sample.load_env()`.
- The GPU host returns a transient `embed_out.json` purely as transport; the driver
  loads it into MySQL and deletes it. Vectors are stored as **float32 little-endian
  bytes** in `LONGBLOB` columns.

#### DB backup before schema changes (CLAUDE.md §7.8)

Take a compressed dump **before** any schema change (new table/column, `ALTER`, migration,
first `init_schema` on new DDL):

```bash
mysqldump --single-transaction --no-tablespaces … codebiology | gzip > codebiology_$(date +%Y%m%d_%H%M%S).sql.gz
```

- `--no-tablespaces` is **required** — the pipeline DB user lacks the global `PROCESS`
  privilege, so without it `mysqldump` errors out probing tablespaces.
- `--single-transaction` gives a consistent snapshot without locking.
- Dumps are gitignored. Run with the `.env` connection params (host/port/user/pass).

### Harrier embedder runtime (asushimu 3090 Ti)
- Model `microsoft/harrier-oss-v1-27b` at `/data/vllm/harrier-oss-v1-27b` (Gemma3-27B
  decoder-only embedder, 5376-dim, MIT). Loaded via sentence-transformers in **4-bit**
  (bitsandbytes nf4, bf16 compute, ≈13.5 GB) — bf16 would be ~54 GB.
- Pin `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2`; `--max-seq 16384` (see VRAM
  cap above). `embed_independent.py` drives `run_harrier_embed.py` over SSH to this host.

Old info:

## asushimu server

- CPU: AMD Ryzen Threadripper 1950X (16c/32t)
- RAM: 128 GB
- GPUs (with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, listed in PCI bus order — current as of 2026-05-20):

  | CUDA idx | Card | PCI bus | VRAM | UUID |
  |---|---|---|---|---|
  | 0 | GTX 1080 Ti | `00000000:09:00.0` | 11 GB | `GPU-c360dd9e-4bbc-ea91-ad04-759c8d286c9f` |
  | 1 | GTX 1080 Ti | `00000000:0B:00.0` | 11 GB | `GPU-d8ac07ab-e77c-a300-1778-39871603d407` |
  | 2 | RTX 3090 Ti | `00000000:41:00.0` | 24.6 GB | `GPU-d355aaa9-0680-bcca-233f-ae7adb3acbd3` |

  Order changed when the second 1080 Ti was added — the 3090 Ti was previously at index 1 and is now at index 2. Pin GPU selection by UUID, not index, to survive future PCI reshuffles.
- The 1080 Tis are **not used for inference** — no BF16, crippled FP16. All LLM work on the 3090 Ti.
- Data dirs: `/data/vllm/` (model weights), `/data/hfcache/` (HuggingFace cache)

## llama.cpp (llama-server)

Runs as a systemd service on asushimu (`/etc/systemd/system/llama-server.service`, ExecStart `/home/mulderg/start_llama.sh`). Auto-starts at boot. Replaced vLLM in April 2026 for a ~9× speedup on the 3090 Ti (15.6 → 141 tok/s) thanks to llama.cpp's MoE-specific expert-gating scheduler. The active model is now **Gemma-4-31B-it (dense) + MTP** since 2026-06-09 (replaced Qwen3.6-27B-MTP — see "Model swap (June 2026)" below). The prior Qwen launcher is preserved as `/home/mulderg/start_llama_qwen.sh` (repo `start_llama_qwen.sh`); roll back by `cp`-ing it over `start_llama.sh` + restart and setting `LLM_MODEL=qwen3.6-27b`.

```bash
ssh asushimu 'sudo systemctl restart llama-server'   # apply flag changes
ssh asushimu 'sudo systemctl status llama-server'    # state + main PID
ssh asushimu 'tail -F /var/tmp/llama-server.log'     # server log
```

The script's `pkill` cleanup is a no-op when systemd starts it cleanly; the script's `> $LOG 2>&1` redirection survives the `exec` so the unit's `Main PID` is the llama-server binary itself.

The script runs `${LLAMA_DIR}/build/bin/llama-server` on the 3090 Ti (pinned by UUID `GPU-d355aaa9-…` via `CUDA_VISIBLE_DEVICES`; numeric indices are unstable when GPUs are added/removed). Key flags:
- `--model /data/vllm/gemma-4-31B-it-Q5_K_M/gemma-4-31B-it-Q5_K_M.gguf` — unsloth Q5_K_M (21.7 GB). Q5 not Q4 because the 4-bit quant has ~15 % function-calling format errors and tool calling is production-critical; Q6_K/Q8_0 don't fit 24.5 GB.
- `--model-draft /data/vllm/gemma-4-31B-it-Q5_K_M/MTP/gemma-4-31B-it-MTP-Q8_0.gguf` — Gemma 4 MTP uses a **SEPARATE 0.5B drafter** (`google/gemma-4-31B-it-assistant`; unsloth Q8_0, 491 MB), unlike Qwen3.6's embedded self-draft heads. A benign `[spec] failed to measure draft model memory` warning prints at startup; the real draft context loads fine.
- `--spec-type draft-mtp --spec-draft-n-max 4` — MTP speculative decode (PRs #23398 + #24282; needs llama.cpp ≥ 2026-06-07). n-max 4 per unsloth's Gemma MTP docs. Measured ~60-80 % draft acceptance under load, ~60-72 tok/s sustained decode (up from ~33 dense).
- `-ngl 99` — all layers on GPU
- `-fa on` — flash attention
- `--ctx-size 16384` — Q5 weights (21.7 GB) + drafter (491 MB) leave thin headroom (~23.2 GB used at 16k); 32k risks OOM. Agent prompts top out ~2.5k tokens.
- `--cache-type-k q8_0 --cache-type-v q8_0` — quantized KV cache, saves VRAM with no measurable quality loss
- `--jinja` — enables OpenAI-style tool calling via the Gemma 4 chat template (first-class llama.cpp gemma4 tool-call parser)
- `--reasoning-format deepseek` — separates Gemma 4's reasoning channel into `message.reasoning_content` (verified clean: no `<think>`/`<channel>` leak into visible content)
- `--alias gemma-4-31b` — must match `LLM_MODEL` in `.env`
- `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0` — **kept the Qwen3.6 thinking-mode sampler**, NOT Gemma's published 1.0/0.95/64. config.py overrides temp/top_p/max_tokens per call and the 62/63 benchmark ran on this profile, so it's the validated production sampler. Native Gemma sampler is untested here.
- `--no-mmap --parallel 1 --threads 8` — `--parallel 1` is **required** for MTP (single-sequence mode only)
- Listens on `0.0.0.0:11434`. Logs at `/var/tmp/llama-server.log`.

### Code Biology criteria-verification batch (added 2026-06-13)

For the criteria-verification pipeline (`criteria_judge.py`), criteria 1 & 2 run
on the local Gemma. Full papers are large (median ~18.5k tokens, p90 ~31k), so the
production 16k context only fits ~24% whole; **32k fits ~87%**. 32k won't fit
alongside the MTP drafter (prod sits at ~23.2/24.6 GB at 16k), so an offline
**batch launcher drops MTP** to reclaim the VRAM.

- **`~/start_llama_batch.sh`** (repo: `code-biology-database/start_llama_batch.sh`)
  — Gemma-4-31B Q5_K_M, **`--ctx-size 32768`, NO `--model-draft`/`--spec-type`**,
  else identical to prod (alias `gemma-4-31b`, q8_0 KV, jinja, deepseek reasoning).
  Trades ~2× decode (back to ~33 tok/s dense) for the larger context — fine for an
  overnight batch.
- **State as of 2026-06-13 (ACTIVE):** batch launcher is **live** — swapped in and
  `llama-server` restarted (user OK'd). Verified: `--ctx-size 32768`, **no
  `--model-draft`/`--spec-type`**, `n_ctx = 32768`, health 200, VRAM **23333/24564
  MiB** (~1.2 GB free — no OOM). End-to-end local criteria-1&2 judge on a real paper
  ran in 35.3 s (dense, no MTP). **Production voice agent is OFFLINE while this batch
  runs** — restore prod when the batch finishes.
  - Restore prod:  `cp ~/start_llama.prod.bak ~/start_llama.sh && sudo systemctl restart llama-server`
  - Re-activate batch: `cp ~/start_llama_batch.sh ~/start_llama.sh && sudo systemctl restart llama-server`
  - Production (`start_llama.sh`, Gemma+MTP@16k) backed up to **`~/start_llama.prod.bak`**
    (md5-verified identical to the pre-swap `start_llama.sh`).
  - If 32k OOMs, drop `--ctx-size` to 24576 (still >50% of papers fit whole).

### Graded per-chunk judge pilot launcher (`start_llama_pilot.sh`, added 2026-06-16)

For the graded per-chunk judge (CLAUDE.md §9), the driver fires **concurrent** per-chunk
Gemma calls, so the pilot server drops MTP (which forces `--parallel 1`) to unlock continuous
batching.

- **`~/start_llama_pilot.sh`** (repo: `code-biology-database/start_llama_pilot.sh`) —
  Gemma-4-31B Q5_K_M, alias `gemma-4-31b`, **NO `--model-draft`/`--spec-type` (MTP off)**,
  **`--parallel 2 --ctx-size 32768`** (16384 tokens/slot = two 8192-token chunks; ctx-size
  must be an integer multiple of parallel — a `--parallel 3 --ctx-size 40960` attempt SIGSEGV'd
  on the non-integer 13653.33/slot), **`--predict 4096`** (prod has none — bounds per-call
  output so Gemma's reasoning preamble + JSON can't run away and truncate; the `--predict 2048`
  first pass cut dense-chunk JSON mid-object, see `@test_runs.md` Run 5). Else mirrors prod
  (sampler, q8_0 KV, jinja, deepseek reasoning).
- **State as of 2026-06-16 (ACTIVE):** the pilot launcher is **live** on the 3090 Ti and the
  **production voice agent is OFFLINE** — an overnight `judge_pilot.py --rest` run (the
  molecular tail, `@test_runs.md` Run 6) is judging against it. **Do NOT restore prod** until
  that run finishes and persists. VRAM steady ~23.97/24.6 GB, no OOM. Restore prod the usual
  way (`cp ~/start_llama.prod.bak ~/start_llama.sh && sudo systemctl restart llama-server`).

### OpenRouter criterion-3 judge (DeepSeek V4 Pro, off-box)

Criterion 3 (*arbitrariness* — the subtle, contested criterion) is judged by
**`DeepSeek V4 Pro`** via OpenRouter (key in repo `.env`, gitignored). This runs off-box, not on
asushimu.

**Resolved 2026-06-13 (user-approved):** batch now uses the **paid**
`DeepSeek V4 Pro`  **with concurrency**. `criteria_judge.run_batch`
runs `DEFAULT_WORKERS=6` papers in a `ThreadPoolExecutor`; the OpenRouter call is the
overlapping bottleneck while the local Gemma (`--parallel 1`) serialises its share.
Resumable JSONL checkpoint + per-paper failure isolation. This turns the ~19 h
sequential run into well under an hour. **Requires a funded OpenRouter account**
(paid models 402 without credit) — confirm balance before the full 471 batch.


### Performance (3090 Ti, Gemma-4-31B Q5_K_M + MTP — current model)

From the 2026-06-09 benchmark (`tests/test_llm.py`, thinking-mixed):

| Phase | rate |
|---|---|
| Prefill | ~950-1030 tok/s (long prompts) |
| Decode  | ~60-72 tok/s sustained (~90 peak on short gens) |
| Draft acceptance | 60-80 % under load, 92 % on short gens |


### Downloading a new model to `/data/vllm/`

Always detach via `nohup` and pipe to a log — downloads can take 10+ minutes and dropped SSH sessions will kill an attached transfer. Use any Python env with `huggingface_hub` installed (the old `vllm` conda env still exists and works for this):

```bash
ssh asushimu 'nohup /home/mulderg/anaconda3/envs/vllm/bin/python -c "
import os; os.environ[\"HF_HOME\"]=\"/data/hfcache\"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=\"<org>/<repo>\",
    local_dir=\"/data/vllm/<dir>\",
    max_workers=2,
)" > /tmp/hf-download.log 2>&1 </dev/null & echo $!'
```

For GGUF files (llama.cpp), you can also use the single-file form with `huggingface-cli download <repo> <file.gguf> --local-dir <dir>`.

**Use `max_workers=2`, not the default `8`.** asushimu's resolver gets saturated when eight parallel HEAD requests to `huggingface.co` fire at once; the symptom is `'[Errno -3] Temporary failure in name resolution' thrown while requesting HEAD https://huggingface.co/…` repeatedly in the log, while the big `.safetensors` shards that started first *do* finish. The process stays alive in `Sl` state with only 3 open `.lock` file descriptors, and the remaining shards sit in exponential-retry backoff indefinitely. Kill it and restart with `max_workers=2`; completed shards are skipped via the `.cache/huggingface/download/*.lock` markers, so no re-download.

Verify progress rather than trust the PID:
```bash
ssh asushimu 'tail -5 /tmp/hf-download.log; du -sh /data/vllm/<dir>; ls /data/vllm/<dir>/*.safetensors 2>/dev/null | wc -l'
```

`DONE <path>` on the final log line means all files landed. If the process exits without `DONE`, check for `.incomplete` or missing shards in `ls /data/vllm/<dir>/`.


