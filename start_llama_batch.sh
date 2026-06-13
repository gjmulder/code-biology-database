#!/usr/bin/env bash
# BATCH llama.cpp launcher — Gemma-4-31B-it (dense) at 32k context, NO MTP.
#
# Purpose: offline criteria-verification batch (criteria_judge.py, criteria 1&2).
# Full papers in this corpus are large (median ~18.5k tokens; p90 ~31k), and the
# production 16k context only fits ~24% of them whole. 32k fits ~87% — but the
# production launcher already sits at ~23.2/24.6 GB on the 3090 Ti at 16k (Q5_K_M
# weights 21.7 GB + the 491 MB MTP drafter + KV), so doubling the KV cache to 32k
# would OOM.
#
# Resolution: this is an OFFLINE batch, so MTP's decode-latency win is irrelevant.
# Dropping the separate 0.5B drafter (--model-draft / --spec-type) reclaims the
# drafter weights AND its draft KV, freeing the headroom 32k needs. We trade
# ~2x decode speed (back to ~33 tok/s dense) for the larger context — fine for an
# overnight batch.
#
# Deploy (temporary; restore production after the batch):
#   scp start_llama_batch.sh asushimu:/home/mulderg/start_llama_batch.sh
#   ssh asushimu 'cp ~/start_llama.sh ~/start_llama.prod.bak \
#                 && cp ~/start_llama_batch.sh ~/start_llama.sh \
#                 && sudo systemctl restart llama-server'
# Restore production (Gemma + MTP, 16k):
#   ssh asushimu 'cp ~/start_llama.prod.bak ~/start_llama.sh \
#                 && sudo systemctl restart llama-server'
#
# If 32k still OOMs, drop --ctx-size to 24576 (still >50% of papers fit whole).
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
#   * NO --model-draft / --spec-type / --spec-draft-n-max  (MTP disabled → VRAM).
#   * --ctx-size 32768  (was 16384) to fit ~87% of papers whole.
#   * --parallel 1 kept: VRAM is the binding constraint at 32k, so we do not
#     widen the batch (each extra sequence multiplies the 32k KV cache).
# Everything else mirrors production (alias, sampler, quantized KV, jinja,
# deepseek reasoning split) so criteria_judge.py talks to it unchanged.
exec "${LLAMA_DIR}/build/bin/llama-server" \
    --model "${MODEL}" \
    --alias gemma-4-31b \
    --host 0.0.0.0 \
    --port "${PORT}" \
    -ngl 99 \
    -fa on \
    --ctx-size 32768 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --jinja \
    --reasoning-format deepseek \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.0 \
    --predict 8000 \
    --no-mmap \
    --parallel 1 \
    --threads 8 \
    > "${LOG}" 2>&1
