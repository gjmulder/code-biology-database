"""Offline tests for run_gte_embed.py (no GPU, no llama-server, fake encoder).

The gte runner reuses the harrier helpers (score_paper_methods, plan_windows,
embed_controls, build_controls_only_output, METHODS) unchanged — only the encoder
(an HTTP call to llama.cpp /v1/embeddings instead of a local sentence-transformers
model) and the tokenizer (Qwen2 vocab) differ. These tests pin the two new pieces and
prove the reused helpers still produce the 3-method shape with a gte-shaped encoder.
"""

import numpy as np

import run_gte_embed as rge


def _fake_encode(texts):
    # gte-shaped fake: length-3 vectors, distinct per text, shape (n, 3).
    return np.array([[len(t), len(t) % 7, 1.0] for t in texts], dtype=np.float64)


def _poles():
    return {
        "two_worlds": {"pos": np.array([1.0, 0.0, 0.0]), "neg": np.array([0.0, 1.0, 0.0])},
        "adaptors": {"pos": np.array([0.0, 0.0, 1.0]), "neg": np.array([1.0, 0.0, 0.0])},
    }


class _TinyTokenizer:
    """Whitespace tokenizer standing in for the Qwen2 AutoTokenizer: ids are word
    indices, decode joins them back. Enough for plan_windows/score_paper_methods."""
    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for w in text.split():
            ids.append(self.vocab.setdefault(w, len(self.vocab)))
        return {"input_ids": ids}

    def decode(self, ids):
        inv = {i: w for w, i in self.vocab.items()}
        return " ".join(inv[i] for i in ids)


def test_constants_are_gte_shaped():
    # the runner advertises the gte model + its native dim/chunking
    assert rge.MODEL_NAME == "gte-qwen2-7b-instruct-q8_0"
    assert rge.DIM == 3584
    assert rge.METHODS == ["full", "abstract", "chunk"]


def test_make_http_encoder_posts_one_per_request_and_parses(monkeypatch):
    posted = []

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    def fake_post(url, json=None, timeout=None):
        # one input per request (a string, not a list) — keeps each forward within ubatch
        posted.append((url, json["input"]))
        # OpenAI embeddings response shape; embedding length = the input length so the
        # test can tell vectors apart
        return _Resp({"data": [{"embedding": [float(len(json["input"])), 0.5, 2.0],
                                "index": 0, "object": "embedding"}]})

    monkeypatch.setattr(rge.requests, "post", fake_post)
    encode = rge.make_http_encoder("http://localhost:11600")
    out = encode(["aa", "bbbb"])

    # one POST per text, hitting the /v1/embeddings path
    assert [p[1] for p in posted] == ["aa", "bbbb"]
    assert all(p[0].endswith("/v1/embeddings") for p in posted)
    # float64 array, shape (n, dim) — distinct rows for distinct inputs
    assert isinstance(out, np.ndarray) and out.dtype == np.float64
    assert out.shape == (2, 3)
    assert out[0][0] == 2.0 and out[1][0] == 4.0


def test_make_http_encoder_strips_trailing_slash(monkeypatch):
    seen = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"embedding": [1.0]}]}

    monkeypatch.setattr(rge.requests, "post",
                        lambda url, json=None, timeout=None: seen.update(url=url) or _Resp())
    rge.make_http_encoder("http://localhost:11600/")(["x"])
    # no double slash regardless of trailing slash on the endpoint
    assert seen["url"] == "http://localhost:11600/v1/embeddings"


def test_score_paper_methods_three_method_shape_with_gte_encoder():
    # the reused harrier helper must produce the same 3-method shape under the gte encoder
    tok = _TinyTokenizer()
    full = "alpha beta gamma delta epsilon zeta eta theta"
    windows = rge.token_windows(tok(full)["input_ids"], size=3, overlap=1)
    progress = {"done": {m: 0 for m in rge.METHODS},
                "total": {m: 99 for m in rge.METHODS}}
    scores, vecs = rge.score_paper_methods(
        "p1", 1, 1, full, "alpha beta", windows, _fake_encode, tok, _poles(), progress)
    assert set(scores) == {"full", "abstract", "chunk"}
    assert set(scores["full"]) == {"two_worlds", "adaptors"}
    # raw vectors: full/abstract single, chunk a list (one per window)
    assert isinstance(vecs["full"], list) and isinstance(vecs["full"][0], float)
    assert isinstance(vecs["chunk"], list) and isinstance(vecs["chunk"][0], list)
    assert len(vecs["chunk"]) == len(windows)


def test_build_controls_only_output_uses_gte_model_and_dim():
    controls = {"genetic_code_positive": "codons map to amino acids"}
    out = rge.build_controls_only_output(controls, _fake_encode, _poles(),
                                         model=rge.MODEL_NAME, dim=rge.DIM)
    assert out["scores"] == {} and out["doc_vectors"] == {}
    assert set(out["control_vectors"]) == set(controls)
    assert out["model"] == "gte-qwen2-7b-instruct-q8_0" and out["dim"] == 3584
    assert out["methods"] == rge.METHODS
