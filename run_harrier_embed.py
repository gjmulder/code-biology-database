"""GPU-host side of the independent embedding analysis (runs ON asushimu).

Loads microsoft/harrier-oss-v1-27b (Gemma3-27B decoder-only embedder) via
sentence-transformers in 4-bit, embeds the criterion poles as instructed queries,
and scores each paper **three ways** as separate documents, to test which
granularity best tracks the LLM verdict:

  * ``full``     — the whole (budget-capped) paper as one document
  * ``abstract`` — the abstract section only (or preamble fallback)
  * ``chunk``    — 8192-token windows at 50% overlap, scored per chunk, then
                   max-pooled (strongest evidence anywhere in the paper)

It also writes the pole-separation diagnostic + control checks. Nothing here
overrides the verdict — this is an independent axis (plan decision 0).

Input JSON  : {"papers": {pid: {"full": str, "abstract": str}},
               "prototypes": {...}, "controls": {name: str}}
Output JSON : {"scores": {pid: {method: {crit: e}}}, "pole_separation": {...},
               "controls": {name: {crit: e}}, "model": str, "dim": int,
               "methods": [...], "chunk_size": int, "chunk_overlap": int}

Run pinned to the 3090 Ti (sm_86); the 1080 Tis are sm_61 and unsupported by this
torch build:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      python3 run_harrier_embed.py --model /data/vllm/harrier-oss-v1-27b \
      --in embed_in.json --out embed_out.json [--no-4bit] [--max-seq 32768]
"""

import argparse
import json
import logging
import os

import numpy as np

import embed_score

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("harrier")

METHODS = ["full", "abstract", "chunk"]


def load_model(model_path, use_4bit=True, max_seq=32768):
    import torch
    from sentence_transformers import SentenceTransformer

    model_kwargs = {"dtype": "auto"}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["dtype"] = torch.bfloat16
    log.info("loading %s (4bit=%s)", model_path, use_4bit)
    model = SentenceTransformer(model_path, model_kwargs=model_kwargs,
                                device="cuda", trust_remote_code=True)
    model.max_seq_length = max_seq
    log.info("loaded; max_seq_length=%s, dim=%s",
             model.max_seq_length, model.get_sentence_embedding_dimension())
    return model


def make_encoder(model, batch_size):
    def encode(texts):
        # sentence-transformers applies harrier's last-token pooling + L2 norm from
        # the repo config; any instruction is already baked into `texts`.
        return np.asarray(model.encode(list(texts), batch_size=batch_size,
                                       show_progress_bar=False), dtype=np.float64)
    return encode


def _crit_scores(doc_vec, poles):
    return {c: embed_score.contrastive_score(doc_vec, poles[c]) for c in poles}


def plan_windows(papers, tokenizer, chunk_size, chunk_overlap):
    """Pre-tokenize every paper's full text so the chunk-method total is known up
    front. Returns ``{pid: [window_ids, ...]}`` (one tokenisation pass, reused for
    the actual embedding so we never tokenise twice)."""
    windows = {}
    for pid, doc in papers.items():
        full = doc["full"] if isinstance(doc, dict) else doc
        ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        windows[pid] = embed_score.token_windows(ids, chunk_size, chunk_overlap)
    return windows


def score_paper_methods(doc_id, doc_idx, n_docs, full, abstract, windows,
                        encode, tokenizer, poles, progress):
    """Return ``(scores, vectors)`` for one paper across the three methods.

    ``scores`` is ``{method: {criterion: e}}``; ``vectors`` is ``{"full": vec,
    "abstract": vec, "chunk": [vec, ...]}`` (raw document embeddings as plain float
    lists) so the contrast math can be recomputed offline from the DB. Each method's
    progress is logged against the per-method embed total. ``progress`` is a mutable
    ``{"done": {method: int}, "total": {method: int}}`` carried across papers."""
    done, total = progress["done"], progress["total"]
    tag = f"[doc {doc_idx}/{n_docs} id={doc_id}]"
    out, vecs = {}, {}

    full_vec = encode([full])[0]
    vecs["full"] = full_vec.tolist()
    out["full"] = _crit_scores(full_vec, poles)
    done["full"] += 1
    log.info("%s full     embed %d/%d (%d chars)",
             tag, done["full"], total["full"], len(full))

    abs_vec = encode([abstract or ""])[0]
    vecs["abstract"] = abs_vec.tolist()
    out["abstract"] = _crit_scores(abs_vec, poles)
    done["abstract"] += 1
    log.info("%s abstract embed %d/%d (%d chars)",
             tag, done["abstract"], total["abstract"], len(abstract or ""))

    chunk_texts = [tokenizer.decode(w) for w in windows]
    per_crit = {c: [] for c in poles}
    chunk_vecs = []
    for wi, text in enumerate(chunk_texts, 1):
        vec = encode([text])[0]
        chunk_vecs.append(vec.tolist())
        for c in poles:
            per_crit[c].append(embed_score.contrastive_score(vec, poles[c]))
        done["chunk"] += 1
        log.info("%s chunk    embed %d/%d (window %d/%d of this doc)",
                 tag, done["chunk"], total["chunk"], wi, len(chunk_texts))
    vecs["chunk"] = chunk_vecs
    out["chunk"] = {c: embed_score.aggregate_chunks(per_crit[c]) for c in poles}
    log.info("%s done: full+abstract+%d chunks", tag, len(chunk_texts))
    return out, vecs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/vllm/harrier-oss-v1-27b")
    ap.add_argument("--in", dest="inp", default="embed_in.json")
    ap.add_argument("--out", dest="out", default="embed_out.json")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--max-seq", type=int, default=32768)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=8192)
    ap.add_argument("--chunk-overlap", type=int, default=4096)
    args = ap.parse_args()

    with open(args.inp, encoding="utf-8") as f:
        data = json.load(f)
    prototypes = data["prototypes"]
    papers = data["papers"]
    controls = data.get("controls", {})

    model = load_model(args.model, use_4bit=not args.no_4bit, max_seq=args.max_seq)
    encode = make_encoder(model, args.batch_size)
    tokenizer = model.tokenizer

    log.info("building %d criterion poles", len(prototypes))
    poles = embed_score.build_poles(prototypes, encode)

    # Pre-tokenise every paper so the chunk-method denominator is known up front;
    # full/abstract are exactly one embed per paper.
    windows = plan_windows(papers, tokenizer, args.chunk_size, args.chunk_overlap)
    n_docs = len(papers)
    total = {"full": n_docs, "abstract": n_docs,
             "chunk": sum(len(w) for w in windows.values())}
    progress = {"done": {m: 0 for m in METHODS}, "total": total}
    log.info("embedding %d papers — total embeds: full=%d, abstract=%d, chunk=%d "
             "(grand total %d)", n_docs, total["full"], total["abstract"],
             total["chunk"], sum(total.values()))

    import torch
    scores, doc_vectors = {}, {}
    for idx, (pid, doc) in enumerate(papers.items(), 1):
        full = doc["full"] if isinstance(doc, dict) else doc
        abstract = doc.get("abstract", "") if isinstance(doc, dict) else ""
        doc_id = os.path.splitext(os.path.basename(pid))[0]
        scores[pid], doc_vectors[pid] = score_paper_methods(
            doc_id, idx, n_docs, full, abstract, windows[pid],
            encode, tokenizer, poles, progress)
        torch.cuda.empty_cache()  # release per-doc activation memory before the next
    log.info("embedding complete: %d/%d full, %d/%d abstract, %d/%d chunk",
             progress["done"]["full"], total["full"],
             progress["done"]["abstract"], total["abstract"],
             progress["done"]["chunk"], total["chunk"])

    control_scores = {}
    if controls:
        for name, text in controls.items():
            control_scores[name] = _crit_scores(encode([text])[0], poles)

    # Raw vectors travel as plain float lists; the driver packs them to float32 BLOBs
    # in MySQL so the contrast math can be recomputed offline (no GPU re-embed).
    pole_vectors = {c: {"pos": poles[c]["pos"].tolist(),
                        "neg": poles[c]["neg"].tolist()} for c in poles}
    out = {
        "scores": scores,
        "doc_vectors": doc_vectors,
        "pole_vectors": pole_vectors,
        "pole_separation": embed_score.pole_separation(poles),
        "controls": control_scores,
        "model": args.model,
        "dim": int(model.get_sentence_embedding_dimension()),
        "use_4bit": not args.no_4bit,
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
