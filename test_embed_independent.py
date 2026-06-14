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


def test_report_from_db_threads_run_to_fetch_report(monkeypatch):
    # report_from_db must read the embedding columns for the requested run, so a gte pass
    # reports gte vectors (not baseline). Capture the run passed to db.fetch_report.
    seen = {}

    def fake_fetch_report(conn, run="baseline"):
        seen["run"] = run
        return {"papers": {}, "order": [], "meta": {},
                "pole_separation": {}, "controls": {}}

    monkeypatch.setattr(ei.db, "fetch_report", fake_fetch_report)
    ei.report_from_db(None, "/dev/null", "/dev/null", run="gte-qwen2")
    assert seen["run"] == "gte-qwen2"


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
