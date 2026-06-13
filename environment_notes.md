# Environment Notes

Reference for operating the infrastructure. Not needed for normal code editing.

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

### OpenRouter criterion-3 judge (Nemotron, off-box)

Criterion 3 (*arbitrariness* — the subtle, contested criterion) is judged by
**`nvidia/nemotron-3-ultra-550b-a55b:free`** via OpenRouter (1M context → reads
the whole paper; key in repo `.env`, gitignored). This runs off-box, not on
asushimu. **Smoke test 2026-06-13: correct & grounded verdict, but 145.8 s for one
10k-token paper on the FREE tier** (low-priority queue + long internal reasoning).
Sequential, 471 papers ≈ 19 h, and the free tier has a daily request cap.

**Resolved 2026-06-13 (user-approved):** batch now uses the **paid**
`nvidia/nemotron-3-ultra-550b-a55b` (no `:free` — same model/1M context, priority
routing, no daily cap; whole run ≈ $4) **with concurrency**. `criteria_judge.run_batch`
runs `DEFAULT_WORKERS=6` papers in a `ThreadPoolExecutor`; the OpenRouter call is the
overlapping bottleneck while the local Gemma (`--parallel 1`) serialises its share.
Resumable JSONL checkpoint + per-paper failure isolation. This turns the ~19 h
sequential run into well under an hour. **Requires a funded OpenRouter account**
(paid models 402 without credit) — confirm balance before the full 471 batch.

### Model swap (June 2026): Qwen3.6-27B → Gemma-4-31B + MTP

Switched from Qwen3.6-27B-MTP to Gemma-4-31B-it (dense) + MTP on 2026-06-09 after a head-to-head benchmark (`project_gemma4_benchmark` memory). Gemma scored **62/63 on `tests/test_llm.py`** (vs Qwen passing the differentiator), with **faster wall-clock (7m44s vs ~18 min)**, on-par decode with MTP (~60-72 vs ~59 tok/s), ~3.7× faster prefill, dense + multimodal + 256K ctx, and clean reasoning separation. Gemma is the first non-A3B-MoE candidate tried and the first to beat Qwen — the three A3B-MoE reasoning models before it (35B-A3B, Nemotron, Nex) all lost on grounded tool calling.

**Two operational deltas vs Qwen MTP:** (1) Gemma 4 MTP is a **separate 0.5B drafter** loaded via `--model-draft` (Qwen self-drafted from embedded heads, no `--model-draft`); (2) it needs llama.cpp ≥ 2026-06-07 — the local checkout was rebuilt 382 commits forward (39cf5d6 → d6d0ce8) to get PRs #23398 + #24282. The old `build/bin/` is backed up at `~/llamacpp-bin-39cf5d6.bak` (instant revert: `cp -a ~/llamacpp-bin-39cf5d6.bak/* ~/llama.cpp/build/bin/`). Qwen3.6-27B + MTP was re-validated on the new binary (22/22 canary+voice, self-draft healthy), so the rebuilt binary runs both models and rollback is safe.

**Known open issue (fix-later):** one voice test (`query_office_temperature`) fails because Gemma over-uses the `room=` filter in `ha_find_entities`, which structurally can't surface sensor entities (the registry has no per-room sensor bucket). Real but narrow; fixable model-agnostically by adding a keyword fallback when a room-filtered search returns 0, or a per-room `sensors` bucket. Qwen passed it via a plain keyword search.

### Model swap (May 2026): 35B-A3B → 27B-dense + MTP

Switched from Qwen3.6-35B-A3B-UD-Q3_K_M (MoE) to Qwen3.6-27B-Q5_K_M-mtp (dense) for two reasons: (1) Qwen3.6-27B's coding/reasoning benchmarks beat the 35B-A3B despite fewer total params, and (2) the higher Q5_K_M quantization fits comfortably in 24 GB VRAM (Q3_K_M was forced by the 35B size). MTP keeps decode within tolerable range despite the dense architecture being weights-bound.

**Trade-off:** decode is **slower** post-swap (~59 tok/s vs ~137 tok/s previously). MoE A3B was cache-bandwidth-bound (~3 B active per token); 27B dense is weights-bound and MTP recovers ~2× of that but not all of it. End-to-end agent latency is dominated by thinking-mode reasoning anyway, so the absolute decode rate is rarely the bottleneck. Quality at Q5_K_M (vs prior Q3_K_M) should be a real step up for tool calling and reflection.

### llama.cpp build (MTP now on master)

The local checkout tracks master and currently sits at **HEAD `d6d0ce8` (2026-06-09)**, rebuilt forward from `39cf5d6` to pick up Gemma 4 MTP (PRs #23398 + #24282). Qwen MTP (PR #22673, merged 2026-05-16) and Gemma MTP both live in this build. Rebuild after pulling:

```bash
cd ~/llama.cpp
git pull
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j 8 --target llama-server llama-cli   # ~15-25 min; CUDA FA kernels dominate
```

Before any large rebuild, back up the working `build/bin/` (it holds the CUDA `.so`s, not just the 7.8 MB server binary) for instant revert: `cp -a build/bin ~/llamacpp-bin-<commit>.bak`. The `--spec-type` enum is `none,draft-simple,draft-eagle3,draft-mtp,ngram-*`; Qwen self-drafts (`draft-mtp`, embedded heads, no `--model-draft`), Gemma uses `draft-mtp` + an explicit `--model-draft` assistant GGUF.

### Performance (3090 Ti, Gemma-4-31B Q5_K_M + MTP — current model)

From the 2026-06-09 benchmark (`tests/test_llm.py`, thinking-mixed):

| Phase | rate |
|---|---|
| Prefill | ~950-1030 tok/s (long prompts) |
| Decode  | ~60-72 tok/s sustained (~90 peak on short gens) |
| Draft acceptance | 60-80 % under load, 92 % on short gens |

Full 63-test suite wall-clock: **7m44s** with MTP (14m18s dense, no drafter). Re-measure live with the `prompt eval time` / `eval time` jq one-liner below.

### Model-candidate baseline (Gemma-4-31B Q5_K_M + MTP, 2026-06-09)

**This is the reference a new model candidate is compared against.** Re-run the same commands against the candidate (after `cp`-ing its launcher over `start_llama.sh` and restarting `llama-server`) and diff the numbers. A candidate must match or beat correctness *and* the voice-latency profile to be adopted — see `feedback_model_swap_checklist`.

Captured on the 3090 Ti, llama-server warm, `LLM_HOST=http://asushimu:11434`, `LLM_MODEL=gemma-4-31b`.

| Metric | Baseline | How to reproduce |
|---|---|---|
| Correctness (full suite) | **81/81 cases pass**, 8m22s wall-clock | `LLM_TEST=1 conda run -n ha-agent python -m pytest tests/test_llm.py -q` |
| Voice latency — no tool (chitchat) | **283 ms** median (3 runs; first-run cold ~3.5 s discarded by median) | `LLM_TEST=1 ... -k latency -o log_cli=true --log-cli-level=INFO` |
| Voice latency — single tool (lamp on) | **1782 ms** median | (same `-k latency` run) |
| Voice vs thinking speedup (same prompt) | **74% faster** (voice 669 ms vs thinking 2599 ms) | (same `-k latency` run) |

The `-k latency` tests (`test_voice_latency_*`) run with HA mocked, so wall-clock is dominated by LLM prefill+decode — the clean model-comparison signal. Gates are env-overridable (`VOICE_LATENCY_BUDGET_MS`, `VOICE_TOOL_LATENCY_BUDGET_MS`, `VOICE_SPEEDUP_MIN`, `VOICE_LATENCY_RUNS`); the absolute ceilings catch a gross regression on this box, while `test_voice_latency_faster_than_thinking` is hardware-independent (a thinking-leak collapses the 74% speedup regardless of GPU speed). Suite case count is 81 (43 test functions × parametrization); the "62/63" figure in the swap note above counts a differently-sliced subset and predates the expanded voice suite — treat **81/81** as the current correctness baseline.

### Performance (3090 Ti, Qwen3.6-27B Q5_K_M-mtp — PRIOR model, historical)

Quick spot-check 2026-05-07 with a 200-word generation:

| Phase | rate |
|---|---|
| Prefill | ~290 tok/s (cold, n=17 prompt tokens) |
| Decode  | ~59 tok/s (n=264 generated tokens) |

MTP draft acceptance from `/var/tmp/llama-server.log` `statistics mtp` lines: **86 % draft accept, 70 % token accept** (running averages over the session). Re-pull live with:
```bash
ssh asushimu 'grep "statistics mtp" /var/tmp/llama-server.log | tail -1'
```

For comparison, the prior Qwen3.6-35B-A3B Q3_K_M baseline measured 2026-04-24 (n ≥ 180):

| Phase | p10 | p50 | p90 | max |
|---|---|---|---|---|
| Prompt eval (prefill) | 408 tok/s | 1460 tok/s | 2770 tok/s | 3420 tok/s |
| Generation (decode)   | 137 tok/s | 139 tok/s  | 140 tok/s  | 141 tok/s  |

A like-for-like (n ≥ 100, post-swap) measurement should be re-run after the new model has a few days of production traffic.

Quick re-measure:
```bash
ssh asushimu 'grep -E "prompt eval time|^\s*eval time" /var/tmp/llama-server.log | tail -1000' \
  | python3 -c "import sys,re; P,G=[],[]; [ (P if 'prompt eval' in l else G).append(float(m.group(2))) for l in sys.stdin for m in [re.search(r'/\s+(\d+)\s+tokens.*?([\d.]+)\s+tokens per second', l)] if m and int(m.group(1))>5 ]; import statistics as s; print('prefill', s.median(P) if P else '-'); print('decode ', s.median(G) if G else '-')"
```

**Thinking mode** (Qwen3.6): controlled per-call via `extra_body={"chat_template_kwargs": {"enable_thinking": bool}}`. With `--reasoning-format deepseek` configured, thinking tokens are stripped to `message.reasoning_content`; visible content stays in `message.content`. Per-call site (May 2026):

| Call site | Thinking | Sampling recipe |
|---|---|---|
| Telegram `agent_reply` | ON | thinking (0.6 / 0.95) |
| Voice `agent_reply` | **OFF** | non-thinking (0.7 / 0.8) |
| Reflect propose | ON | thinking (0.6 / 0.95) |
| Entity classify | ON | thinking (0.6 / 0.95) |
| `/audit` | ON | thinking (0.6 / 0.95) |
| LEARN pipeline | OFF | thinking constants (specialised path) |
| `agent.py` warm-up | OFF | n/a |

Per-call `max_tokens` values live in `config.py` (`LLM_MAX_TOKENS_VOICE`, `LLM_MAX_TOKENS_THINK`) — single source of truth. Voice runs without thinking to keep wall-clock latency low (~0.5–2 s end-to-end), bounded by `LLM_MAX_TOKENS_VOICE` (cap is output-only when thinking is off). Thinking-mode calls share a single token budget covering `<think>` plus the visible answer. Lowering thinking-mode temperature below ~0.5 is unsafe — Qwen docs warn this causes endless repetitions.

### Post-start verification

```bash
ssh asushimu 'curl -s -o /dev/null -w "%{http_code}\n" http://localhost:11434/health'            # expect 200
ssh asushimu 'curl -s http://localhost:11434/v1/models | head -c 300'                            # expect alias gemma-4-31b
ssh asushimu 'nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv'                # 3090 Ti (currently index 2; pinned by UUID) memory should be ~18–22 GB
ssh asushimu 'grep -E "failed|error|Traceback|CUDA error" /var/tmp/llama-server.log | tail -20'         # expect empty
```

If the server fails to start, read `tail -100 /var/tmp/llama-server.log`. Typical remediation:
1. Rebuild llama.cpp (`cd ~/llama.cpp && cmake --build build -j`)
2. Lower `--ctx-size` (32768 → 16384 → 8192) if KV cache allocation fails
3. Lower `-ngl` (99 → 35 → 20) to offload some layers to CPU
4. Pick a smaller GGUF quant — on-disk GGUF size must leave ~3 GB headroom under the 24.6 GB VRAM ceiling on the 3090 Ti

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

## Whisper STT (asushimu, 1080 Ti)

`whisper_server.py` runs as a systemd unit (`/etc/systemd/system/whisper-server.service`) on asushimu's first 1080 Ti (CUDA index 0 with `CUDA_DEVICE_ORDER=PCI_BUS_ID`). It exposes a FastAPI HTTP API on port 11500 (`POST /transcribe`, `GET /health`); `voice_bridge.py` on sanzaru POSTs WAV bytes here and falls back to local `base.en` on any exception. Conda env is `whisper-stt`. See [[project_stt_offload]] memory for the offload rationale.

```bash
ssh asushimu 'sudo systemctl restart whisper-server'      # apply changes
ssh asushimu 'sudo systemctl status whisper-server'       # state + main PID
ssh asushimu 'tail -F /var/tmp/whisper-server.log'        # decode lines: STT audio=… decode=…
curl -s http://asushimu:11500/health                      # {model, device, compute_type}
```

Service unit env vars (single source of truth — `whisper_server.py` defaults match):
- `WHISPER_MODEL=Systran/faster-whisper-large-v3` — full large-v3, not the distilled variant. The distilled version was used 2026-05-16 → 2026-05-28; switched out because distil's 2-decoder-layer stack mis-recognises short imperative commands (alarm/control verbs, numbers) — exactly the voice path's failure surface.
- `WHISPER_COMPUTE_TYPE=int8_float32` — **Pascal-specific.** `int8_float16` errors with "target device or backend do not support efficient int8_float16 computation" on the 1080 Ti (Pascal has crippled native FP16). Use `int8_float32`: int8 weights via DP4A matmul + FP32 activations.
- `WHISPER_DEVICE=cuda`, `WHISPER_DEVICE_INDEX=0` — 1080 Ti at PCI bus `09:00.0`; the 3090 Ti hosts llama-server, so STT on the 1080 Ti is contention-free.

**Performance** (large-v3, int8_float32, 1080 Ti):
- VRAM: ~1.7 GB (vs. ~0.95 GB for distil-large-v3); 9+ GB free
- Decode: ~1.0–1.5 s for typical 3–5 s clips (vs. ~0.6 s for distil); still under sanzaru CPU `base.en` baseline of ~2.3 s end-to-end after LAN round-trip
- Quality gate: voice_bridge's `voice_utils.should_reject_transcript` consumes the four returned fields (`text`, `language_probability`, `avg_logprob`, `no_speech_prob`) — wire format is stable across model swaps

To revert (rollback to distil): set `WHISPER_MODEL=Systran/faster-distil-whisper-large-v3` in `whisper-server.service`, `daemon-reload`, restart.



## Optional environment variables

- `LLM_MODEL` (current `.env` value: `gemma-4-31b`; config.py fallback default also `gemma-4-31b`) — must match llama-server's `--alias`
- `LLM_HOST` (default: `http://localhost:11434`)
- `VLLM_API_KEY` (default: `EMPTY`) — legacy name; llama-server ignores auth by default
- `MAX_ITERATIONS` (default: `10`)
- `LLM_TEMPERATURE` (default: `0.6`) — Telegram `agent_reply` (thinking on); Qwen3.6 thinking recipe
- `LLM_TEMPERATURE_VOICE` (default: `0.7`) — voice `agent_reply` (thinking off); Qwen3.6 non-thinking recipe
- `LLM_TEMPERATURE_REFLECT` (default: `0.6`) — entity classify + `/audit` (both run with thinking on); was `0.2`, but Qwen docs warn that near-greedy temps cause endless repetition in thinking mode
- `LLM_TEMPERATURE_REFLECT_THINK` (default: `0.6`) — reflection hypothesis (thinking on)
- `LLM_TOP_P_THINK` (default: `0.95`) — top_p for all thinking-mode calls
- `LLM_TOP_P_VOICE` (default: `0.8`) — top_p for the non-thinking voice path
- `LLM_MAX_TOKENS_VOICE` (default: `256`) — voice cap (non-thinking, output-only); ~2–3 sentences
- `LLM_MAX_TOKENS_THINK` — single budget for CoT + answer on thinking calls; see `config.py` for the current default
- `ALLOWED_CHAT_IDS` — comma-separated Telegram chat IDs
- `DEBUG` — `true`/`false`, hot-reloaded
- `VOICE_CHAT_ID` — Telegram chat ID for voice query mirrors

### Qwen3.6 sampling recipes (also used as-is for Gemma-4-31B)

> The current Gemma-4-31B production model runs under these same recipes — the 62/63 benchmark validated them, and config.py is the single source of truth for per-call temp/top_p/max_tokens. Gemma's published native sampler (1.0/0.95/top_k 64) is untested here; don't switch without re-running `tests/test_llm.py`.


| | Thinking on | Thinking off |
|---|---|---|
| temperature | 0.6 | 0.7 |
| top_p | 0.95 | 0.8 |
| top_k | 20 | 20 |
| min_p | 0 | 0 |

`top_k` and `min_p` come from `start_llama.sh` (`--top-k 20 --min-p 0.0`); the agent only overrides `temperature`, `top_p`, and `max_tokens` per call. Do not lower thinking-mode temperature below ~0.5 — Qwen's docs explicitly warn this can cause endless repetitions.
