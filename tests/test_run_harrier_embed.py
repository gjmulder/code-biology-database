"""Offline tests for run_harrier_embed.py pure helpers (no GPU, fake encoder)."""

import numpy as np

import run_harrier_embed as rhe


def _fake_encode(texts):
    # deterministic per-text vector: length-3, seeded by text length so distinct
    # texts get distinct vectors; shape (n, 3) like sentence-transformers
    return np.array([[len(t), len(t) % 7, 1.0] for t in texts], dtype=np.float64)


def _poles():
    return {
        "two_worlds": {"pos": np.array([1.0, 0.0, 0.0]), "neg": np.array([0.0, 1.0, 0.0])},
        "adaptors": {"pos": np.array([0.0, 0.0, 1.0]), "neg": np.array([1.0, 0.0, 0.0])},
    }


def test_embed_controls_returns_scores_and_raw_vectors():
    controls = {"genetic_code_positive": "codons map to amino acids",
                "deterministic_chemistry": "sterics fix the pairing"}
    scores, vectors = rhe.embed_controls(controls, _fake_encode, _poles())
    # one entry per control in both outputs
    assert set(scores) == set(controls)
    assert set(vectors) == set(controls)
    # raw vectors are plain float lists (JSON-transportable), one per control
    for name, v in vectors.items():
        assert isinstance(v, list)
        assert all(isinstance(x, float) for x in v)
    # scores carry one e per criterion
    assert set(scores["genetic_code_positive"]) == {"two_worlds", "adaptors"}


def test_embed_controls_empty_is_empty():
    scores, vectors = rhe.embed_controls({}, _fake_encode, _poles())
    assert scores == {} and vectors == {}


def test_build_controls_only_output_has_vectors_no_papers():
    controls = {"genetic_code_positive": "codons map to amino acids"}
    out = rhe.build_controls_only_output(controls, _fake_encode, _poles(),
                                         model="harrier", dim=3)
    # no paper-level payload — the 219 papers are untouched on store
    assert out["scores"] == {}
    assert out["doc_vectors"] == {}
    # control vectors present so the driver can persist + rescore them leverred
    assert set(out["control_vectors"]) == set(controls)
    assert out["model"] == "harrier" and out["dim"] == 3
    assert out["methods"] == rhe.METHODS
