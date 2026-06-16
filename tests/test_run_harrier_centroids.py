"""Offline tests for run_harrier_centroids.py (no GPU, fake encoder).

The centroids are the 24 scientometric topic categories embedded with harrier in
the **same space as the corpus chunk vectors** (documents, no instruction), so a
paper chunk can be assigned to its nearest topic. These tests pin the pure CSV
read + embed transforms; the heavy harrier encode runs on the GPU host.
"""

import csv

import numpy as np

import run_harrier_centroids as rhc


def _fake_encode(texts):
    # deterministic per-text vector seeded by text length so distinct texts get
    # distinct vectors; shape (n, 3) like sentence-transformers
    return np.array([[len(t), len(t) % 7, 1.0] for t in texts], dtype=np.float64)


def _write_csv(path, rows):
    cols = ["Topic #", "Label", "Abbreviation", "Justification",
            "Characteristic Terms (Fig 2)", "Centroid Text"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_read_centroids_reads_id_label_text(tmp_path):
    p = tmp_path / "cats.csv"
    _write_csv(p, [
        {"Topic #": "3", "Label": "Genetic Code", "Abbreviation": "GeC",
         "Justification": "j", "Characteristic Terms (Fig 2)": "t",
         "Centroid Text": "Genetic Code. Codons map to amino acids."},
        {"Topic #": "18", "Label": "Histone Code", "Abbreviation": "HiC",
         "Justification": "j", "Characteristic Terms (Fig 2)": "t",
         "Centroid Text": "Histone Code. Chromatin modifications."},
    ])
    cents = rhc.read_centroids(str(p))
    assert [c["topic_id"] for c in cents] == [3, 18]
    assert cents[0]["label"] == "Genetic Code"
    assert cents[0]["text"].startswith("Genetic Code.")


def test_read_centroids_skips_blank_text(tmp_path):
    p = tmp_path / "cats.csv"
    _write_csv(p, [
        {"Topic #": "3", "Label": "Genetic Code", "Abbreviation": "GeC",
         "Justification": "j", "Characteristic Terms (Fig 2)": "t",
         "Centroid Text": "Genetic Code. Codons."},
        {"Topic #": "9", "Label": "Code Theory", "Abbreviation": "CT",
         "Justification": "j", "Characteristic Terms (Fig 2)": "t",
         "Centroid Text": "   "},
    ])
    cents = rhc.read_centroids(str(p))
    assert [c["topic_id"] for c in cents] == [3]


def test_embed_centroids_returns_raw_vectors():
    cents = [{"topic_id": 3, "label": "Genetic Code", "text": "codons to amino acids"},
             {"topic_id": 18, "label": "Histone Code", "text": "chromatin marks"}]
    out = rhc.embed_centroids(cents, _fake_encode)
    assert [c["topic_id"] for c in out] == [3, 18]
    for c in out:
        assert isinstance(c["vec"], list)
        assert all(isinstance(x, float) for x in c["vec"])
    # distinct texts → distinct vectors
    assert out[0]["vec"] != out[1]["vec"]


def test_build_centroids_output_shape():
    cents = [{"topic_id": 3, "label": "Genetic Code", "text": "codons"}]
    out = rhc.build_centroids_output(cents, _fake_encode,
                                     model="harrier", dim=3, run="baseline")
    assert out["model"] == "harrier"
    assert out["dim"] == 3
    assert out["run"] == "baseline"
    assert len(out["centroids"]) == 1
    assert out["centroids"][0]["topic_id"] == 3
    assert out["centroids"][0]["label"] == "Genetic Code"
    assert len(out["centroids"][0]["vec"]) == 3
