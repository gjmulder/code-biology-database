"""Offline tests for embed_independent.py pure helpers (no GPU, no MySQL)."""

import embed_independent as ei


def _pool(n):
    return [{"code_number": str(i), "pdf_path": f"pdfs/p{i}.pdf"} for i in range(n)]


def test_sample_extra_papers_excludes_existing_and_is_seed_deterministic():
    pool = _pool(10)
    existing = {"pdfs/p0.pdf", "pdfs/p1.pdf"}
    a = ei.sample_extra_papers(pool, existing, n=3, seed=42)
    b = ei.sample_extra_papers(pool, existing, n=3, seed=42)
    # same seed -> identical selection (reproducible runs)
    assert [p["pdf_path"] for p in a] == [p["pdf_path"] for p in b]
    assert len(a) == 3
    # never re-picks a paper that already has a verdict
    assert all(p["pdf_path"] not in existing for p in a)
    # carries the code id so it can be keyed in doc_vectors
    assert all("code_number" in p for p in a)


def test_sample_extra_papers_caps_at_available():
    pool = [{"code_number": "1", "pdf_path": "pdfs/a.pdf"}]
    # the only candidate is excluded -> nothing left, even though n=5 requested
    assert ei.sample_extra_papers(pool, {"pdfs/a.pdf"}, n=5, seed=1) == []
    # n larger than the pool returns the whole (shuffled) pool, no error
    assert len(ei.sample_extra_papers(_pool(3), set(), n=99, seed=1)) == 3


def test_sample_extra_papers_different_seeds_differ():
    pool = _pool(20)
    a = [p["pdf_path"] for p in ei.sample_extra_papers(pool, set(), 5, seed=1)]
    b = [p["pdf_path"] for p in ei.sample_extra_papers(pool, set(), 5, seed=2)]
    assert a != b


def test_sample_extra_papers_zero_returns_empty():
    assert ei.sample_extra_papers(_pool(5), set(), n=0, seed=1) == []


def test_merge_all_papers_keeps_verdicts_and_appends_rest():
    # labelled recs carry verdict criteria; pool is the whole on-disk corpus
    recs = [{"code_number": "1", "pdf_path": "pdfs/a.pdf", "criteria": {"two_worlds": "met"}}]
    pool = [{"code_number": "1", "pdf_path": "pdfs/a.pdf"},   # dup of a labelled paper
            {"code_number": "2", "pdf_path": "pdfs/b.pdf"},
            {"code_number": "3", "pdf_path": "pdfs/c.pdf"}]
    merged = ei.merge_all_papers(recs, pool)
    # every on-disk paper is present exactly once, deduped by path
    assert [m["pdf_path"] for m in merged] == ["pdfs/a.pdf", "pdfs/b.pdf", "pdfs/c.pdf"]
    # the labelled paper keeps its criteria (drives ρ); the rest are bare
    assert merged[0].get("criteria") == {"two_worlds": "met"}
    assert "criteria" not in merged[1] and "criteria" not in merged[2]


def test_merge_all_papers_is_idempotent_on_paths():
    recs = [{"code_number": "1", "pdf_path": "pdfs/a.pdf"}]
    pool = [{"code_number": "1", "pdf_path": "pdfs/a.pdf"}]
    assert ei.merge_all_papers(recs, pool) == recs


def test_build_controls_input_has_no_papers():
    # controls-only embed: no paper text extracted, just prototypes + controls so the
    # GPU run captures the 2 control vectors without re-embedding the whole corpus
    proto = {"two_worlds": {"pos": ["a"], "neg": ["b"]}}
    controls = {"genetic_code_positive": "the genetic code maps codons to amino acids",
                "deterministic_chemistry": "stereochemistry fixes the pairing"}
    payload = ei.build_controls_input(proto, controls)
    assert payload["papers"] == {}
    assert payload["prototypes"] is proto
    assert payload["controls"] is controls


def test_run_remote_controls_only_appends_flag(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, check=True, **kw):
        calls.append(cmd)
        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(ei.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "embed_score.py").write_text("")
    (tmp_path / "run_harrier_embed.py").write_text("")
    (tmp_path / "in.json").write_text("{}")
    ei.run_remote("host", "/remote", "in.json", "out.json", "model",
                  use_4bit=True, cuda_devices="2", max_seq=16384, controls_only=True)
    remote_cmd = next(c for c in calls if c[:2] == ["ssh", "host"]
                      and "run_harrier_embed.py" in c[-1])
    assert "--controls-only" in remote_cmd[-1]


def test_run_remote_default_omits_controls_only(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ei.subprocess, "run",
                        lambda cmd, check=True, **kw: calls.append(cmd) or type("R", (), {"returncode": 0})())
    monkeypatch.chdir(tmp_path)
    for f in ("embed_score.py", "run_harrier_embed.py", "in.json"):
        (tmp_path / f).write_text("")
    ei.run_remote("host", "/remote", "in.json", "out.json", "model",
                  use_4bit=True, cuda_devices="2", max_seq=16384)
    remote_cmd = next(c for c in calls if c[:2] == ["ssh", "host"]
                      and "run_harrier_embed.py" in c[-1])
    assert "--controls-only" not in remote_cmd[-1]


def test_run_remote_gte_scps_runner_and_runs_against_endpoint(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ei.subprocess, "run",
                        lambda cmd, check=True, **kw: calls.append(cmd) or type("R", (), {"returncode": 0})())
    monkeypatch.chdir(tmp_path)
    # gte runner imports the harrier helpers, so both must be shipped alongside embed_score
    for f in ("embed_score.py", "run_harrier_embed.py", "run_gte_embed.py", "in.json"):
        (tmp_path / f).write_text("")
    ei.run_remote_gte("host", "/remote", "in.json", "out.json",
                      "http://localhost:11600", max_seq=16384,
                      chunk_size=4096, chunk_overlap=2048)
    scped = [c[2] for c in calls if c[:2] == ["scp", "-q"]]
    assert {"embed_score.py", "run_harrier_embed.py", "run_gte_embed.py", "in.json"} <= set(scped)
    remote_cmd = next(c for c in calls if c[:2] == ["ssh", "host"]
                      and "run_gte_embed.py" in c[-1])
    assert "--endpoint http://localhost:11600" in remote_cmd[-1]
    assert "--controls-only" not in remote_cmd[-1]


def test_run_remote_gte_controls_only_appends_flag(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ei.subprocess, "run",
                        lambda cmd, check=True, **kw: calls.append(cmd) or type("R", (), {"returncode": 0})())
    monkeypatch.chdir(tmp_path)
    for f in ("embed_score.py", "run_harrier_embed.py", "run_gte_embed.py", "in.json"):
        (tmp_path / f).write_text("")
    ei.run_remote_gte("host", "/remote", "in.json", "out.json",
                      "http://localhost:11600", max_seq=16384,
                      chunk_size=4096, chunk_overlap=2048, controls_only=True)
    remote_cmd = next(c for c in calls if c[:2] == ["ssh", "host"]
                      and "run_gte_embed.py" in c[-1])
    assert "--controls-only" in remote_cmd[-1]


def test_run_embed_remote_dispatches_on_engine(monkeypatch):
    # the dispatcher routes --engine st -> harrier runner, llamacpp -> gte runner
    seen = {}
    monkeypatch.setattr(ei, "run_remote",
                        lambda *a, **k: seen.update(which="st", kw=k))
    monkeypatch.setattr(ei, "run_remote_gte",
                        lambda *a, **k: seen.update(which="gte", kw=k))

    class A:
        host = "h"; remote_dir = "/r"; model = "m"; no_4bit = False
        cuda_devices = "2"; max_seq = 16384; endpoint = "http://localhost:11600"
        chunk_size = 4096; chunk_overlap = 2048

    a = A()
    a.engine = "st"
    ei.run_embed_remote(a, "in.json", "out.json")
    assert seen["which"] == "st"
    a.engine = "llamacpp"
    ei.run_embed_remote(a, "in.json", "out.json", controls_only=True)
    assert seen["which"] == "gte" and seen["kw"]["controls_only"] is True


def _topic_paper(code, e, verdict):
    return {"code": code,
            "scores": {"chunk": {"two_worlds": e}},
            "verdict": {"two_worlds": verdict},
            "confidence": {"two_worlds": 1.0}}


def test_per_topic_spearman_stratifies_and_thresholds():
    # topic 5: e rises monotonically with the verdict ordinal -> ρ = +1
    # topic 9: only one paper -> below min_n, excluded entirely
    papers = {
        "a": _topic_paper(1, 0.1, "not_met"),
        "b": _topic_paper(2, 0.2, "unclear"),
        "c": _topic_paper(3, 0.3, "met"),
        "z": _topic_paper(9, 0.9, "met"),
    }
    order = ["a", "b", "c", "z"]
    dominant = {"a": 5, "b": 5, "c": 5, "z": 9}
    out = ei.per_topic_spearman(papers, order, dominant,
                                methods=["chunk"], criteria=["two_worlds"], min_n=3)
    topics = {r["topic"]: r for r in out}
    assert set(topics) == {5}                      # topic 9 dropped (n=1 < min_n)
    assert topics[5]["n"] == 3
    assert abs(topics[5]["rho"]["two_worlds"]["chunk"] - 1.0) < 1e-9


def test_per_topic_spearman_marks_no_variation_none():
    # a stratum where every verdict is identical -> no rank variation -> ρ is None (n/a)
    papers = {
        "a": _topic_paper(1, 0.1, "not_met"),
        "b": _topic_paper(2, 0.2, "not_met"),
        "c": _topic_paper(3, 0.3, "not_met"),
    }
    order = ["a", "b", "c"]
    dominant = {"a": 5, "b": 5, "c": 5}
    out = ei.per_topic_spearman(papers, order, dominant,
                                methods=["chunk"], criteria=["two_worlds"], min_n=3)
    assert out[0]["rho"]["two_worlds"]["chunk"] is None


def test_per_topic_spearman_sorted_by_n_desc():
    papers = {}
    order = []
    dominant = {}
    for i in range(5):                              # topic 1: 5 papers
        k = f"t1_{i}"; papers[k] = _topic_paper(i, i / 10, "met" if i else "not_met")
        order.append(k); dominant[k] = 1
    for i in range(3):                              # topic 2: 3 papers
        k = f"t2_{i}"; papers[k] = _topic_paper(i, i / 10, "met" if i else "not_met")
        order.append(k); dominant[k] = 2
    out = ei.per_topic_spearman(papers, order, dominant,
                                methods=["chunk"], criteria=["two_worlds"], min_n=3)
    assert [r["topic"] for r in out] == [1, 2]     # larger stratum first


def test_report_from_db_threads_run_to_fetch_report(monkeypatch):
    # report_from_db must read the embedding columns for the requested run, so a gte pass
    # reports gte vectors (not baseline). Capture the run passed to db.fetch_report.
    seen = {}

    def fake_fetch_report(conn, run="baseline", judge=None):
        seen["run"] = run
        return {"papers": {}, "order": [], "meta": {},
                "pole_separation": {}, "controls": {}}

    monkeypatch.setattr(ei.db, "fetch_report", fake_fetch_report)
    ei.report_from_db(None, "/dev/null", "/dev/null", run="gte-qwen2")
    assert seen["run"] == "gte-qwen2"


def test_report_from_db_threads_judge_to_fetch_report(monkeypatch):
    # report_from_db must pass --judge through to db.fetch_report so a report can be
    # scoped to one judge's verdicts (gemma vs deepseek) instead of newest-wins.
    seen = {}

    def fake_fetch_report(conn, run="baseline", judge=None):
        seen["judge"] = judge
        return {"papers": {}, "order": [], "meta": {},
                "pole_separation": {}, "controls": {}}

    monkeypatch.setattr(ei.db, "fetch_report", fake_fetch_report)
    ei.report_from_db(None, "/dev/null", "/dev/null", run="baseline",
                      judge="deepseek/deepseek-v4-pro")
    assert seen["judge"] == "deepseek/deepseek-v4-pro"


def test_recompute_from_db_threads_run_through(monkeypatch):
    # recompute_from_db must scope every vector read/write to the requested run so the
    # offline rescore targets one model's vectors. Capture the run on each db call.
    seen = {}
    monkeypatch.setattr(ei.db, "init_schema", lambda conn: None)

    def fake_fetch_vectors(conn, run="baseline"):
        seen["fetch_vectors"] = run
        return {}, {}, {}      # empty -> recompute returns early after the read

    monkeypatch.setattr(ei.db, "fetch_vectors", fake_fetch_vectors)
    # empty vectors -> recompute_from_db returns early after fetch_vectors; that's enough
    # to prove the run is threaded to the read path.
    ei.recompute_from_db(None, "/dev/null", "/dev/null", k=0, strength=0.5, run="gte-qwen2")
    assert seen["fetch_vectors"] == "gte-qwen2"


def test_build_pool_keeps_only_on_disk_and_dedupes(tmp_path, monkeypatch):
    # two rows share a DOI (same pdf path); a third has no file on disk
    rows = [
        {"code_number": "10", "url": "https://doi.org/10.1038/x"},
        {"code_number": "11", "url": "https://doi.org/10.1038/x"},   # dup path
        {"code_number": "12", "url": "https://doi.org/10.1038/y"},   # missing file
    ]
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    monkeypatch.setattr(ei, "read_rows", lambda csv_path: iter(rows))
    (pdf_dir / "10.1038_x.pdf").write_bytes(b"%PDF-1.4")
    pool = ei.build_pool("ignored.csv", str(pdf_dir))
    # one entry: deduped by path, missing-file row dropped, first code wins
    assert pool == [{"code_number": "10", "pdf_path": str(pdf_dir / "10.1038_x.pdf")}]


def test_load_verdicts_reads_jsonl_skipping_blank_lines(tmp_path):
    path = tmp_path / "v.jsonl"
    path.write_text('{"pdf_path": "a.pdf", "criteria": {}}\n'
                    '\n'   # blank line ignored
                    '{"pdf_path": "b.pdf", "criteria": {}}\n')
    recs = ei.load_verdicts(str(path))
    assert [r["pdf_path"] for r in recs] == ["a.pdf", "b.pdf"]


def test_fmt_renders_none_as_blank_and_signs_floats():
    assert ei._fmt(None) == ""
    assert ei._fmt(0.5) == "+0.500"
    assert ei._fmt(-0.25) == "-0.250"


# --- report_from_db: the full markdown + CSV assembly (faked DB) ------------
#
# The existing report_from_db tests pass an empty payload (they only check run/judge are
# threaded). This one feeds a populated payload + topic strata through faked db reads so the
# per-paper tables, pooled ρ, per-topic ρ, pole separation/width and control sections all
# render — exercising the report assembler offline, no MySQL.

CRITS = ["two_worlds", "adaptors", "arbitrariness"]


def _populated_payload(n=12):
    # n papers, all dominant topic 3, e ascending, verdicts cycling so ρ has rank variation.
    cycle = ["met", "unclear", "not_met"]
    papers, order = {}, []
    for i in range(n):
        pid = f"pdfs/p{i}.pdf"
        order.append(pid)
        e = i / n
        v = cycle[i % 3]
        papers[pid] = {
            "code": i,
            "scores": {"chunk": {c: e for c in CRITS}},
            "verdict": {c: v for c in CRITS},
            "confidence": {c: 0.8 for c in CRITS},
        }
    return {
        "papers": papers, "order": order,
        "meta": {"model": "harrier", "dim": 5376, "use_4bit": "False",
                 "chunk_size": 8192, "chunk_overlap": 4096,
                 "scoring": "leverred-axis", "whiten_k": 0, "shared_strength": 0.5,
                 "controls_scoring": "leverred-axis"},
        "pole_separation": {"pos": {"tw|ad": 0.61}, "neg": {"tw|ad": 0.52},
                            "within": {c: 0.55 for c in CRITS}},
        "controls": {"genetic-code": {c: 0.4 for c in CRITS},
                     "deterministic-chemistry": {c: -0.1 for c in CRITS}},
    }


def test_report_from_db_writes_full_markdown_and_csv(monkeypatch, tmp_path):
    payload = _populated_payload(12)
    pids = payload["order"]
    monkeypatch.setattr(ei.db, "fetch_report",
                        lambda conn, run="baseline", judge=None: payload)
    # every paper sits in topic 3 → one stratum of 12 (≥ MIN_TOPIC_N) renders the per-topic table
    monkeypatch.setattr(ei.db, "fetch_chunk_topics",
                        lambda conn, run="baseline", method="chunk":
                        {pid: [(0, 3, 0.6)] for pid in pids})
    monkeypatch.setattr(ei.db, "fetch_topic_centroids",
                        lambda conn, run="baseline":
                        {3: {"label": "Genetic Code", "vec": [0.0]}})

    md_path = tmp_path / "report.md"
    csv_path = tmp_path / "scores.csv"
    ei.report_from_db(None, str(md_path), str(csv_path), run="baseline")

    md = md_path.read_text()
    assert "# Independent Embedding Analysis vs LLM Verdicts" in md
    assert "## Per-paper verdicts" in md
    assert "### Criterion: `two_worlds`" in md
    assert "Spearman ρ(e, verdict_ordinal)" in md
    assert "Per-topic ρ" in md and "Genetic Code" in md       # the stratum rendered
    assert "Pole separation" in md and "Pole width `within`" in md
    assert "Control checks" in md and "genetic-code" in md
    assert "**Scoring:** offline recompute" in md             # leverred-axis branch

    import csv as _csv
    with open(csv_path, encoding="utf-8") as fh:
        csv_rows = list(_csv.DictReader(fh))
    assert len(csv_rows) == 12
    assert {"two_worlds_verdict", "two_worlds_e_chunk"} <= set(csv_rows[0])
