"""Offline tests for chunk_text.reproduce_chunks (fake tokenizer, no transformers/GPU).

The chunk text spans must be reproduced with the *identical* walk used by the embedding
axis (run_harrier_embed.plan_windows): tokenize with add_special_tokens=False, split with
embed_score.token_windows, decode each window. chunk_idx is the positional window index, so
it lines up one-for-one with doc_vectors / chunk_topics rows for the same paper.
"""

import chunk_text


class FakeTokenizer:
    """Whitespace tokenizer mirroring the HF call surface chunk_text relies on:
    ``tok(text, add_special_tokens=False)["input_ids"]`` and ``tok.decode(ids)``.
    Each whitespace token is one id (its index in a per-instance vocab)."""

    def __init__(self):
        self.vocab = []
        self.index = {}

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for word in text.split():
            if word not in self.index:
                self.index[word] = len(self.vocab)
                self.vocab.append(word)
            ids.append(self.index[word])
        return {"input_ids": ids}

    def decode(self, ids):
        return " ".join(self.vocab[i] for i in ids)


def test_short_text_is_single_chunk():
    tok = FakeTokenizer()
    text = "alpha beta gamma"
    out = chunk_text.reproduce_chunks(text, tok, size=8192, overlap=4096)
    assert out == [(0, "alpha beta gamma")]


def test_windows_match_token_windows_count_and_indexing():
    tok = FakeTokenizer()
    # 10 distinct tokens, size 4 / overlap 2 -> stride 2 -> windows at 0,2,4,6 (last reaches end)
    text = " ".join(f"w{i}" for i in range(10))
    out = chunk_text.reproduce_chunks(text, tok, size=4, overlap=2)
    idxs = [c for c, _ in out]
    assert idxs == [0, 1, 2, 3]                    # positional, contiguous from 0
    # first window is the first 4 tokens, decoded back verbatim
    assert out[0][1] == "w0 w1 w2 w3"
    # last window ends at the final token (no truncation loss)
    assert out[-1][1].endswith("w9")


def test_chunk_count_equals_token_windows():
    # the reproduction must agree with embed_score.token_windows exactly (the embedding walk)
    import embed_score
    tok = FakeTokenizer()
    text = " ".join(f"t{i}" for i in range(25))
    ids = tok(text, add_special_tokens=False)["input_ids"]
    expected = embed_score.token_windows(ids, size=8, overlap=4)
    out = chunk_text.reproduce_chunks(text, tok, size=8, overlap=4)
    assert len(out) == len(expected)


def test_empty_text_yields_no_chunks():
    tok = FakeTokenizer()
    assert chunk_text.reproduce_chunks("", tok, size=8, overlap=4) == []
