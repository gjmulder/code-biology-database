"""GPU-host side: embed the 24 scientometric topic centroids (runs ON asushimu).

The scientometric paper (Paredes & Prinz 2025) clusters the "code" literature into
24 topics. We embed each topic's augmented centroid text with harrier in the
**exact same space as the corpus chunk vectors** — same model, same 4-bit
precision, same ``run`` — so a paper chunk can be assigned to its nearest topic
(centred nearest-centroid, downstream in ``assign_topics.py``).

Centroids are embedded as **documents** (plain text, no ``Instruct:`` query
prefix), matching how the paper chunks were embedded. The output is a small
transport JSON the offline driver loads into the run-keyed ``topic_centroids``
table; no scoring happens here.

Input  : code-categories-augmented.csv (the "Centroid Text" column)
Output : {"centroids": [{"topic_id": int, "label": str, "vec": [float,...]}, ...],
          "model": str, "dim": int, "run": str}

Run pinned to the 3090 Ti (sm_86), 4-bit, matching the corpus run:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      python3 run_harrier_centroids.py --model /data/vllm/harrier-oss-v1-27b \
      --csv code-categories-augmented.csv --out centroids_out.json [--no-4bit]
"""

import argparse
import csv
import json
import logging

import numpy as np

import run_harrier_embed as rhe

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("centroids")


def read_centroids(csv_path):
    """Read ``code-categories-augmented.csv`` → ``[{topic_id, label, text}, ...]``.

    ``topic_id`` is the int ``Topic #``; ``text`` is the ``Centroid Text`` column
    (the string actually embedded). Rows with a blank/whitespace centroid text are
    skipped so a partially-filled CSV can't inject empty vectors."""
    out = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = (row.get("Centroid Text") or "").strip()
            if not text:
                continue
            out.append({"topic_id": int(row["Topic #"]),
                        "label": (row.get("Label") or "").strip(),
                        "text": text})
    return out


def embed_centroids(centroids, encode):
    """Embed each centroid's text once → list with a raw ``vec`` (plain float list).

    Embedded as documents (the ``text`` carries no instruction), so the vectors
    live in the same space as the corpus chunk document vectors."""
    out = []
    for c in centroids:
        vec = encode([c["text"]])[0]
        out.append({"topic_id": c["topic_id"], "label": c["label"],
                    "vec": np.asarray(vec, dtype=np.float64).tolist()})
    return out


def build_centroids_output(centroids, encode, model, dim, run="baseline"):
    """Assemble the transport payload the driver loads into ``topic_centroids``."""
    return {
        "centroids": embed_centroids(centroids, encode),
        "model": model,
        "dim": dim,
        "run": run,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/vllm/harrier-oss-v1-27b")
    ap.add_argument("--csv", default="code-categories-augmented.csv")
    ap.add_argument("--out", dest="out", default="centroids_out.json")
    ap.add_argument("--run", default="baseline")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--max-seq", type=int, default=16384)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    centroids = read_centroids(args.csv)
    log.info("read %d centroids from %s", len(centroids), args.csv)

    model = rhe.load_model(args.model, use_4bit=not args.no_4bit, max_seq=args.max_seq)
    encode = rhe.make_encoder(model, args.batch_size)
    dim = int(model.get_sentence_embedding_dimension())

    out = build_centroids_output(centroids, encode, args.model, dim, args.run)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info("wrote %s (%d centroids, dim=%d, run=%s)",
             args.out, len(out["centroids"]), dim, args.run)


if __name__ == "__main__":
    main()
