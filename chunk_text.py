"""Reproduce the *text* of the embedding axis's chunk windows, aligned by index.

The independent embedding axis (run_harrier_embed.plan_windows) splits each paper into
8192-token windows at 50% overlap and embeds each as its own document; chunk_idx is the
positional window index, persisted into doc_vectors / chunk_topics. The judge axis needs
the readable text of those same spans so it can score *per chunk* on exactly the units the
embedding scored. This module re-walks the identical path — tokenize with
add_special_tokens=False, split with embed_score.token_windows, decode each window — so
chunk_idx lines up one-for-one with the persisted vectors for the same paper.

The tokenizer is injected (the caller passes the harrier AutoTokenizer, CPU-only; tests pass
a fake), so this module pulls in no model weights and no GPU.
"""

import embed_score


def reproduce_chunks(full_text, tokenizer, size=8192, overlap=4096):
    """Return ``[(chunk_idx, text), ...]`` for ``full_text``, matching the embedding walk.

    ``tokenizer`` must expose the HF surface used by the embedder:
    ``tokenizer(text, add_special_tokens=False)["input_ids"]`` and ``tokenizer.decode(ids)``.
    Empty / whitespace-only input yields no chunks (the embedding axis never stores a vector
    for a text-less paper, so there is no index to align to)."""
    ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if not ids:
        return []
    windows = embed_score.token_windows(ids, size, overlap)
    return [(idx, tokenizer.decode(window)) for idx, window in enumerate(windows)]
