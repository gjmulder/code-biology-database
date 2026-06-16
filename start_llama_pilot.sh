#!/usr/bin/env bash
# PILOT llama.cpp launcher — Gemma-4-31B-it (dense), NO MTP, batched for concurrency.
#
# Purpose: the graded per-chunk judge pilot (judge_pilot.py). This is a DIFFERENT
# workload from the older full-paper batch (start_llama_batch.sh):
#
#   * judge_pilot fires up to 6 CONCURRENT requests — ThreadPoolExecutor(
#     max_workers=cj.DEFAULT_WORKERS=6), one paper per worker, each issuing its
#     chunk x criterion calls sequentially.
#   * each request is SMALL, not a whole paper: one 8192-token embedding window +
#     the calibrated/topic/control scaffold in, a short JSON object out — but the
#     client sets no max_tokens and thinking is on, so Gemma decodes reasoning
#     (-> reasoning_content, discarded) before the JSON. That makes each call
#     DECODE-heavy, and dense decode without MTP is only ~33 tok/s.
#
# Why drop MTP here: MTP requires --parallel 1 (single-sequence mode), which would
# serialise all 6 client workers onto one slot. Dropping the drafter frees that
# constraint AND its VRAM, letting us run continuous batching with --parallel 3:
# the slow dense decodes batch together and overlap the next chunk's prefill. For
# this decode-heavy, concurrent workload that beats MTP's single-stream ~2x by a
# wide margin. (MTP's decode-latency win is irrelevant to an offline batch anyway.)
#
# Sizing (3090 Ti, 24.6 GB; Q5_K_M weights 21.7 GB, q8_0 KV):
#   --parallel 2, --ctx-size 32768  ->  32768/2 = 16384 tokens per slot = exactly
#   TWO 8192-token chunks, ample room for one chunk (~9k after Gemma re-tokenisation)
#   + ~2k scaffold + reasoning/JSON output. 32768 is the proven-safe KV footprint
#   (the full-paper batch ran here at 23.3/24.6 GB).
#
#   NB: --ctx-size MUST be an integer multiple of --parallel. An earlier attempt at
#   --parallel 3 --ctx-size 40960 (40960/3 = 13653.33, non-integer) SIGSEGV'd in the
#   "-fit on" device-memory fitting step. 32768/2 = 16384 divides cleanly.
#
# Deploy (temporary; restore production after the pilot):
#   scp start_llama_pilot.sh asushimu:/home/mulderg/start_llama_pilot.sh
#   ssh asushimu 'cp ~/start_llama.sh ~/start_llama.prod.bak \
#                 && cp ~/start_llama_pilot.sh ~/start_llama.sh \
#                 && sudo systemctl restart llama-server'
# Restore production (Gemma + MTP, 16k):
#   ssh asushimu 'cp ~/start_llama.prod.bak ~/start_llama.sh \
#                 && sudo systemctl restart llama-server'
#
# Drive with --workers 3 to lightly oversubscribe the 2 slots (a queued request
# starts the instant a slot frees, no network round-trip gap):
#   python3 judge_pilot.py --top 4 --workers 3
#
# If VRAM allows more concurrency, --parallel 3 --ctx-size 49152 (16384/slot, also
# clean) or --parallel 4 --ctx-size 49152 (12288/slot) — verify it loads first.
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Pin to the 3090 Ti by UUID (numeric indices shift when GPUs are added/removed).
export CUDA_VISIBLE_DEVICES=GPU-d355aaa9-0680-bcca-233f-ae7adb3acbd3
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}

LLAMA_DIR=/home/mulderg/llama.cpp
MODEL=/data/vllm/gemma-4-31B-it-Q5_K_M/gemma-4-31B-it-Q5_K_M.gguf
PORT=11434
LOG=/var/tmp/llama-server.log

# Kill any running inference server on the port (match the binary, not this shell).
pkill -f "${LLAMA_DIR}/build/bin/llama-server" 2>/dev/null || true
sleep 2

# Differences from production start_llama.sh:
#   * NO --model-draft / --spec-type / --spec-draft-n-max  (MTP disabled -> VRAM
#     reclaimed AND --parallel > 1 unlocked).
#   * --parallel 2  (was 1): continuous batching for the concurrent driver.
#   * --ctx-size 32768  (was 16384): 16384 tokens per slot (two chunks) for
#     chunk + scaffold + reasoning output. Must be a multiple of --parallel.
#   * --predict 4096  (prod has none): bound per-call output so a runaway reasoning
#     generation cannot exhaust a slot's context. Raised 2048 -> 4096 because at 2048
#     Gemma's reasoning preamble (thinking shares this budget) truncated the JSON on
#     dense chunks -> ~163 unparseable cells in the domain-general re-pilot. 4096 out
#     + ~9k chunk + ~2k scaffold ~= 15k < 16384 per-slot, still fits.
# Everything else mirrors production (alias, sampler, quantized KV, jinja, deepseek
# reasoning split) so criteria_judge.py / judge_pilot.py talk to it unchanged.
exec "${LLAMA_DIR}/build/bin/llama-server" \
    --model "${MODEL}" \
    --alias gemma-4-31b \
    --host 0.0.0.0 \
    --port "${PORT}" \
    -ngl 99 \
    -fa on \
    --ctx-size 32768 \
    --parallel 2 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --jinja \
    --reasoning-format deepseek \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.0 \
    --predict 4096 \
    --no-mmap \
    --threads 8 \
    > "${LOG}" 2>&1
