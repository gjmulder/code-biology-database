"""GPU-host side of the embedding analysis using gte-Qwen2-7B-instruct (Q8_0 GGUF).

A head-to-head against run_harrier_embed.py: same corpus, poles, levers and ρ
diagnostic — **only the embedding model changes**. Instead of loading the model in
this process (as harrier does via sentence-transformers), the GGUF is served by a
transient ``llama-server`` (start_llama_embed.sh) and embedded over its OpenAI-compatible
``/v1/embeddings`` endpoint. Everything downstream is the harrier runner's, reused
unchanged:

  * ``score_paper_methods`` / ``plan_windows`` / ``embed_controls`` /
    ``build_controls_only_output`` / ``METHODS`` — they take an injected ``encode`` and
    ``tokenizer``, so swapping both is all that's needed.
  * ``embed_score`` — poles, contrast, windowing, pole-separation; gte's last-token
    pooling + L2 norm and the ``Instruct: {task}\nQuery: {text}`` query format are exactly
    what ``build_poles``/``format_query`` already emit, so the contrast/lever code is
    untouched.

gte-Qwen2-7B-instruct: dim **3584** (harrier 5376; the schema stores ``dim`` per row),
last-token pooling, L2-normalised, 32K context. The contrast math L2-normalises again
downstream, so server-side normalisation is idempotent.

  # on asushimu, after start_llama_embed.sh is up on :11600
  python3 run_gte_embed.py --endpoint http://localhost:11600 \
      --in embed_in.json --out embed_out.json [--controls-only]
"""

import argparse
import json
import logging
import os

import numpy as np
import requests

import embed_score
# Reuse the harrier helpers verbatim — they are model-agnostic (injected encode/tokenizer).
from run_harrier_embed import (METHODS, build_controls_only_output, embed_controls,
                               plan_windows, score_paper_methods)
from embed_score import token_windows  # re-exported for the offline tests' convenience

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("gte")

MODEL_NAME = "gte-qwen2-7b-instruct-q8_0"
DIM = 3584


def make_http_encoder(endpoint, timeout=600):
    """Return ``encode(texts) -> np.float64 (n, dim)`` that POSTs **one input per request**
    to the llama.cpp OpenAI-compatible ``/v1/embeddings`` endpoint.

    One-input-per-request is deliberate: under ``--pooling last`` a long ``full`` doc
    batched with others would exceed ``--ubatch-size`` and be silently truncated; a single
    input per forward keeps each within the configured batch. The server already L2-
    normalises; the contrast math normalises again (idempotent)."""
    url = endpoint.rstrip("/") + "/v1/embeddings"

    def encode(texts):
        vecs = []
        for text in texts:
            r = requests.post(url, json={"input": text}, timeout=timeout)
            r.raise_for_status()
            vecs.append(r.json()["data"][0]["embedding"])
        return np.asarray(vecs, dtype=np.float64)

    return encode


def load_tokenizer():
    """The Qwen2 tokenizer (CPU, tokenizer-only) so chunk windows align to the gte vocab
    that the served model actually sees. Downloaded once from the HF hub."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Alibaba-NLP/gte-Qwen2-7B-instruct",
                                         trust_remote_code=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:11600",
                    help="llama-server base URL serving the gte GGUF (/v1/embeddings)")
    ap.add_argument("--in", dest="inp", default="embed_in.json")
    ap.add_argument("--out", dest="out", default="embed_out.json")
    ap.add_argument("--max-seq", type=int, default=16384,
                    help="kept for parity/logging; the server's -c/-ub cap the real budget")
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--chunk-overlap", type=int, default=2048)
    ap.add_argument("--controls-only", action="store_true",
                    help="embed only the control texts (capture control_vectors) and skip "
                         "the paper corpus; storing this upserts control_vectors only")
    args = ap.parse_args()

    with open(args.inp, encoding="utf-8") as f:
        data = json.load(f)
    prototypes = data["prototypes"]
    papers = data["papers"]
    controls = data.get("controls", {})

    log.info("encoder -> %s/v1/embeddings; loading Qwen2 tokenizer (CPU)", args.endpoint)
    encode = make_http_encoder(args.endpoint)
    tokenizer = load_tokenizer()

    log.info("building %d criterion poles", len(prototypes))
    poles = embed_score.build_poles(prototypes, encode)
    dim = int(len(next(iter(poles.values()))["pos"]))
    if dim != DIM:
        log.warning("served embedding dim %d != expected %d (using actual)", dim, DIM)

    if args.controls_only:
        out = build_controls_only_output(controls, encode, poles, MODEL_NAME, dim)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        log.info("wrote %s (controls-only: %d control vectors, no papers)",
                 args.out, len(out["control_vectors"]))
        return

    windows = plan_windows(papers, tokenizer, args.chunk_size, args.chunk_overlap)
    n_docs = len(papers)
    total = {"full": n_docs, "abstract": n_docs,
             "chunk": sum(len(w) for w in windows.values())}
    progress = {"done": {m: 0 for m in METHODS}, "total": total}
    log.info("embedding %d papers — total embeds: full=%d, abstract=%d, chunk=%d "
             "(grand total %d)", n_docs, total["full"], total["abstract"],
             total["chunk"], sum(total.values()))

    scores, doc_vectors = {}, {}
    for idx, (pid, doc) in enumerate(papers.items(), 1):
        full = doc["full"] if isinstance(doc, dict) else doc
        abstract = doc.get("abstract", "") if isinstance(doc, dict) else ""
        doc_id = os.path.splitext(os.path.basename(pid))[0]
        scores[pid], doc_vectors[pid] = score_paper_methods(
            doc_id, idx, n_docs, full, abstract, windows[pid],
            encode, tokenizer, poles, progress)
    log.info("embedding complete: %d/%d full, %d/%d abstract, %d/%d chunk",
             progress["done"]["full"], total["full"],
             progress["done"]["abstract"], total["abstract"],
             progress["done"]["chunk"], total["chunk"])

    control_scores, control_vectors = embed_controls(controls, encode, poles)
    pole_vectors = {c: {"pos": poles[c]["pos"].tolist(),
                        "neg": poles[c]["neg"].tolist()} for c in poles}
    out = {
        "scores": scores,
        "doc_vectors": doc_vectors,
        "pole_vectors": pole_vectors,
        "pole_separation": embed_score.pole_separation(poles),
        "controls": control_scores,
        "control_vectors": control_vectors,
        "model": MODEL_NAME,
        "dim": dim,
        "methods": METHODS,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info("wrote %s (%d papers x %d methods, %d controls)", args.out,
             len(scores), len(METHODS), len(control_scores))


if __name__ == "__main__":
    main()
