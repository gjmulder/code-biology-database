#!/usr/bin/env bash
# EMBED llama.cpp launcher — gte-Qwen2-7B-instruct (Q8_0 GGUF) as an embedding server.
#
# Purpose: the model-swap head-to-head against harrier (run_gte_embed.py talks to this
# over the OpenAI-compatible /v1/embeddings endpoint). gte-Qwen2 uses LAST-token pooling
# and L2-normalised outputs (dim 3584); the contrast/lever code L2-normalises again, so
# server-side normalisation is idempotent.
#
# This is TRANSIENT — NOT the systemd llama-server unit. The production voice agent is
# already offline for this project (3090 Ti freed); judging (criteria 1&2 on Gemma) is
# done, so the batch server can be stopped to free VRAM before this runs:
#   ssh asushimu 'sudo systemctl stop llama-server'      # do NOT restore prod — PROJECT END only
# gte-Q8_0 (8.1 GB) + a 16k embedding context fits comfortably on the 24 GB card alone.
#
# Deploy + run (transient, foreground or nohup):
#   scp start_llama_embed.sh asushimu:/home/mulderg/start_llama_embed.sh
#   ssh asushimu 'nohup bash ~/start_llama_embed.sh > /var/tmp/llama-embed.log 2>&1 &'
# Health check:
#   curl -s localhost:11600/v1/embeddings -H 'Content-Type: application/json' \
#        -d '{"input":"codons map to amino acids"}' | head -c 200
# Stop when the embed pass is done:
#   pkill -f 'llama-server.*--embeddings'
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Pin to the 3090 Ti by UUID (numeric indices shift when GPUs are added/removed).
export CUDA_VISIBLE_DEVICES=GPU-d355aaa9-0680-bcca-233f-ae7adb3acbd3
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}

LLAMA_DIR=/home/mulderg/llama.cpp
MODEL=/data/vllm/gte-qwen2-q8/gte-qwen2-7b-instruct-q8_0.gguf
# Spare port — NOT prod 11434 (voice agent) nor 11500 (whisper).
PORT=11600
LOG=/var/tmp/llama-embed.log

# Kill any embedding server we previously started (match --embeddings, not this shell).
pkill -f "${LLAMA_DIR}/build/bin/llama-server.*--embeddings" 2>/dev/null || true
sleep 2

# Embedding-server flags:
#   --embeddings        expose /v1/embeddings (pooled vectors, no generation).
#   --pooling last      gte-Qwen2 is a decoder-only last-token embedder.
#   -c/-ub/-b 16384     one input per request (run_gte_embed posts singly), so the
#                       unbatched/physical/logical batch all equal the 16k token budget;
#                       a doc longer than this is truncated by the server (full/abstract
#                       are pre-capped upstream; chunk windows are 4k).
#   --no-webui          headless.
exec "${LLAMA_DIR}/build/bin/llama-server" \
    --model "${MODEL}" \
    --alias gte-qwen2 \
    --host 0.0.0.0 \
    --port "${PORT}" \
    -ngl 99 \
    -fa on \
    --embeddings \
    --pooling last \
    --ctx-size 16384 \
    --ubatch-size 16384 \
    --batch-size 16384 \
    --no-webui \
    --threads 8 \
    > "${LOG}" 2>&1
