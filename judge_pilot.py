"""Pilot driver — graded, per-chunk, topic-grounded, control-anchored judging.

The binding constraint on the project is *label quality*, not the embedding (CLAUDE.md §6,
§8). This driver pilots the redesigned judge axis on the top-N scientometric topics
(free local Gemma 4): it selects papers whose **dominant** topic is in the most frequent
strata, reproduces the *identical* 8192-token embedding windows (so ``chunk_idx`` lines up
one-for-one with ``doc_vectors`` / ``chunk_topics``), and runs **one Gemma call per
criterion per chunk** with a calibrated, topic-grounded, control-anchored prompt.

Every per-chunk judgement is checkpointed to a resumable JSONL keyed on the
``(pdf_path, chunk_idx, criterion)`` triple — APPEND, **never deleted** — *before* MySQL
persistence (the spend-safety discipline of CLAUDE.md §7.5; free here, same pattern). The
per-chunk graded records are then aggregated per ``(paper, criterion)`` via
:func:`criteria_judge.aggregate_graded` into ``chunk_verdicts`` + ``verdicts(graded`` plus a
derived categorical ``verdict``), so the existing ρ / report pipeline keeps working.

Reuses: :mod:`criteria_judge` (prompt / parse / gate / aggregate / checkpoint helpers),
:mod:`assign_topics` (dominant topic), :mod:`chunk_text` (window reproduction), :mod:`db`
(persistence). Pure selection / resumability / roll-up logic is unit-tested offline; the
tokenizer + Gemma + MySQL I/O is exercised manually (end-to-end on asushimu).

Run (asushimu, after freeing the prod GPU and starting Gemma 4):
    python3 judge_pilot.py --top 4
"""

import argparse
import csv
import json
import logging
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import assign_topics
import chunk_text
import criteria_judge as cj

logger = logging.getLogger(__name__)

DEFAULT_TOP = 4
DEFAULT_CHECKPOINT = "pilot_verdicts.jsonl"
DEEPSEEK_CHECKPOINT = "deepseek_verdicts.jsonl"

# AGREE-anchor ablation (CLAUDE.md §6 follow-up): each variant selects which `_controls`
# entries fill the AGREE anchor + a model-tag/checkpoint suffix so its verdicts coexist with
# the molecular baseline instead of overwriting it. `genetic` is the existing baseline (no
# suffix); `neural` swaps in the non-molecular 1-shot exemplar; `neural-genetic` shows both
# (2-shot). Maps name -> (agree_keys, tag_suffix).
AGREE_ANCHOR_VARIANTS = {
    "genetic": (("genetic_code_positive",), ""),
    "neural": (("neural_code_positive",), "@neural-1shot"),
    "neural-genetic": (("neural_code_positive", "genetic_code_positive"),
                       "@neural-genetic-2shot"),
}
DEFAULT_TOKENIZER = "/data/vllm/harrier-oss-v1-27b"
AUGMENTED_CSV = "code-categories-augmented.csv"
PROTOTYPES_JSON = "prototypes.json"


def load_env(path=".env"):
    """Populate missing env vars from a dotenv file (e.g. OPENROUTER_API_KEY for the paid judge).

    Uses ``setdefault`` so a value already in the environment is never clobbered, and is a
    no-op if the file is absent — a free local run needs no secrets file.
    """
    import re
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            m = re.match(r'\s*(?:export\s+)?([A-Z_]+)\s*=\s*["\']?([^"\'\n]+)', line)
            if m:
                os.environ.setdefault(m.group(1), m.group(2).strip())


# --- judge backend selection ----------------------------------------------

def make_judge(judge, host="http://asushimu:11434", reasoning="high", meter=None):
    """Return ``(complete_callable, model_tag)`` for the selected judge backend.

    * ``"local"``    — free local Gemma (:func:`criteria_judge.local_complete_factory`),
      tagged :data:`criteria_judge.LOCAL_MODEL`.
    * ``"deepseek"`` — OpenRouter DeepSeek V4 Pro at the given reasoning effort, provider-pinned
      to the implicit-caching first-party endpoint, usage fed to ``meter``; tagged
      :data:`criteria_judge.DEEPSEEK_MODEL`. The ``model_tag`` is persisted on every
      chunk_verdict / verdict so the two judges' labels are distinguishable in the DB."""
    if judge == "local":
        return cj.local_complete_factory(host=host), cj.LOCAL_MODEL
    if judge == "deepseek":
        return cj.openrouter_graded_factory(reasoning_effort=reasoning, meter=meter), cj.DEEPSEEK_MODEL
    raise ValueError(f"unknown judge {judge!r} (expected 'local' or 'deepseek')")


def report_usage(meter, papers, corpus=219):
    """Log measured DeepSeek token usage + real cost, and a linear corpus extrapolation.

    The extrapolation is per-paper (``cost / papers * corpus``) — a confirmation aid, not a
    contract: reasoning-token output varies by paper, so treat the full-corpus figure as a
    planning estimate to be re-checked once the corpus mix is known."""
    cost = meter.cost()
    cached_frac = (meter.cached_tokens / meter.prompt_tokens) if meter.prompt_tokens else 0.0
    logger.info("DeepSeek usage: %d calls over %d papers", meter.calls, papers)
    logger.info("  input tokens   : %d (%d cached = %.1f%% served at cache-read rate)",
                meter.prompt_tokens, meter.cached_tokens, 100 * cached_frac)
    logger.info("  output tokens  : %d (of which %d reasoning)",
                meter.completion_tokens, meter.reasoning_tokens)
    logger.info("  measured cost  : $%.4f  ($%.5f / paper)",
                cost, cost / papers if papers else 0.0)
    if papers:
        logger.info("  -> extrapolated full corpus (%d papers): ~$%.2f", corpus,
                    cost / papers * corpus)
    return cost


# --- augmented-topic loading (label + centroid blurb for grounding) --------

def load_augmented_topics(csv_path=AUGMENTED_CSV):
    """Load ``code-categories-augmented.csv`` → ``{topic_id: (label, centroid_blurb)}``.

    The ``Label`` and ``Centroid Text`` columns are the dominant-topic grounding fed into
    :func:`criteria_judge.build_chunk_prompt` (as CONTEXT only, never evidence)."""
    out = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = int(row["Topic #"])
            out[tid] = (row["Label"].strip(), row["Centroid Text"].strip())
    return out


# --- dominant-topic selection (top-N pilot strata) -------------------------

def paper_dominant_topics(chunk_topics):
    """``{pdf_path: [(chunk_idx, topic_id, sim), ...]}`` (db.fetch_chunk_topics) →
    ``{pdf_path: dominant_topic_id}`` via :func:`assign_topics.paper_dominant_topic`."""
    return {pid: assign_topics.paper_dominant_topic(chunks)[0]
            for pid, chunks in chunk_topics.items()}


def top_topic_ids(chunk_topics, n=DEFAULT_TOP):
    """The ``n`` most frequent dominant topics, ties broken by ascending topic id
    (deterministic)."""
    doms = paper_dominant_topics(chunk_topics)
    counts = Counter(d for d in doms.values() if d is not None)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [tid for tid, _ in ordered[:n]]


def select_pilot_papers(chunk_topics, n=DEFAULT_TOP):
    """``{pdf_path: dominant_topic_id}`` restricted to the top-``n`` strata."""
    doms = paper_dominant_topics(chunk_topics)
    top = set(top_topic_ids(chunk_topics, n))
    return {pid: d for pid, d in doms.items() if d is not None and d in top}


def select_code_papers(chunk_topics, codes, code_number):
    """Papers under a single ``code_number``, each carrying its dominant topic.

    Selects independently of the top-N strata: used to judge the foundational **code 0**
    Code Biology texts (CLAUDE.md §1b) as a gold-positive calibration set — does the
    skeptical judge mark Barbieri's own defining papers met? — and any future code-keyed
    subset. ``codes`` is the ``db.fetch_vectors`` pdf_path→code_number map; the comparison
    is string-equal because code numbers are stringy across the CSV / DB."""
    doms = paper_dominant_topics(chunk_topics)
    return {pid: d for pid, d in doms.items()
            if d is not None and str(codes.get(pid)) == str(code_number)}


def select_rest_papers(chunk_topics, n=DEFAULT_TOP):
    """The complement of :func:`select_pilot_papers`: every paper with a dominant topic
    **outside** the top-``n`` strata. This is the molecular "met" tail (CLAUDE.md §9 /
    Run 5 backlog) — the strata where both axes should carry real signal, judged after the
    neuro top-``n`` pilot so the two axes can be compared corpus-wide."""
    doms = paper_dominant_topics(chunk_topics)
    top = set(top_topic_ids(chunk_topics, n))
    return {pid: d for pid, d in doms.items() if d is not None and d not in top}


# --- triple-keyed resumability ---------------------------------------------

def _chunk_key(record):
    return (record["pdf_path"], int(record["chunk_idx"]), record["criterion"])


def load_done(checkpoint_path):
    """Return the set of ``(pdf_path, chunk_idx, criterion)`` triples already judged.

    Per-chunk-per-criterion grain (unlike the per-paper :func:`criteria_judge.load_done`), so
    a resumed run re-judges only the exact chunk/criterion cells still missing."""
    done = set()
    if not os.path.exists(checkpoint_path):
        return done
    with open(checkpoint_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(_chunk_key(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                logger.warning("skipping malformed checkpoint line")
    return done


# --- records ---------------------------------------------------------------

def chunk_record(paper_meta, chunk_idx, criterion, parsed):
    """Flat per-chunk graded record (checkpoint line + ``db.chunk_verdict_rows`` input).

    ``parsed`` is a :func:`criteria_judge.parse_graded` (and grounded) record."""
    return {
        "code_number": paper_meta["code_number"],
        "pdf_path": paper_meta["pdf_path"],
        "chunk_idx": int(chunk_idx),
        "criterion": criterion,
        "agreement": parsed["agreement"],
        "confidence": parsed["confidence"],
        "evidence_quote": parsed.get("evidence_quote", ""),
        "reasoning": parsed.get("reasoning", ""),
        "prompt_hash": cj.prompt_hash(criterion),
    }


def aggregate_to_verdict_records(chunk_records):
    """Roll flat per-chunk records up to per-``(paper, criterion)`` ``db.update_verdicts``
    records via :func:`criteria_judge.aggregate_graded`.

    For each paper the criteria dict carries the derived categorical ``verdict``, the
    aggregated ``confidence`` (argmax chunk), ``graded`` (graded_max — the persisted axis)
    and ``graded_mean`` (overall stance, diagnostic)."""
    by_paper = {}
    for r in chunk_records:
        key = (int(r["code_number"]), r["pdf_path"])
        by_paper.setdefault(key, {}).setdefault(r["criterion"], []).append(r)
    out = []
    for (code, pid), crits in by_paper.items():
        criteria = {}
        for crit, recs in crits.items():
            gmax, gmean, conf, categorical = cj.aggregate_graded(recs)
            criteria[crit] = {"verdict": categorical, "confidence": conf,
                              "graded": gmax, "graded_mean": gmean,
                              "prompt_hash": cj.prompt_hash(crit)}
        out.append({"code_number": code, "pdf_path": pid, "criteria": criteria})
    return out


# --- per-paper judging (I/O; exercised manually) ---------------------------

def judge_paper_chunks(paper_meta, full_text, tokenizer, topic_label, topic_blurb,
                       controls, complete, checkpoint_path, done, write_lock,
                       agree_keys=("genetic_code_positive",)):
    """Judge every (chunk × criterion) cell of one paper not already in ``done``.

    Reproduces the embedding windows, builds the calibrated per-chunk prompt, calls Gemma,
    parses + grounds, and checkpoints each cell (under ``write_lock``) before returning. The
    grounding gate neutralises ungrounded positives; a per-cell model/parse failure is logged
    and skipped so one bad chunk never aborts the paper. ``agree_keys`` selects the AGREE-anchor
    exemplar(s) (anchor ablation). Returns the records judged here."""
    chunks = chunk_text.reproduce_chunks(full_text, tokenizer)
    records = []
    for idx, ctext in chunks:
        for crit in cj.ALL_CRITERIA:
            if (paper_meta["pdf_path"], idx, crit) in done:
                continue
            try:
                raw = complete(cj.GRADED_SYSTEM_PROMPT,
                               cj.build_chunk_prompt(ctext, crit, topic_label,
                                                     topic_blurb, controls,
                                                     agree_keys=agree_keys),
                               response_format={"type": "json_object"})
                parsed = cj.graded_grounding_gate(cj.parse_graded(raw, crit), ctext)
            except Exception as exc:  # one bad cell must not kill the paper
                logger.warning("skipping %s chunk %d %s: %s",
                               paper_meta["pdf_path"], idx, crit, exc)
                continue
            rec = chunk_record(paper_meta, idx, crit, parsed)
            with write_lock:
                cj.append_checkpoint(checkpoint_path, rec)
            records.append(rec)
    return records


def _paper_meta_from_codes(pid, codes):
    """Minimal paper meta for a selected pdf_path (code_number from the vectors join)."""
    return {"code_number": codes.get(pid), "pdf_path": pid}


def main():
    import pdf_text
    import db

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=DEFAULT_TOP,
                    help="number of most-frequent dominant topics to pilot")
    ap.add_argument("--rest", action="store_true",
                    help="judge the COMPLEMENT of the top-N strata (the molecular tail "
                         "outside the neuro top-N) instead of the top-N themselves")
    ap.add_argument("--code", default=None,
                    help="judge only papers under this code_number (e.g. 0 = the foundational "
                         "Code Biology gold-positive set), ignoring the top-N strata selection")
    ap.add_argument("--run", default="baseline", help="embedding run whose chunk_topics to use")
    ap.add_argument("--method", default="chunk")
    ap.add_argument("--judge", choices=("local", "deepseek"), default="local",
                    help="judge backend: free local Gemma, or OpenRouter DeepSeek V4 Pro")
    ap.add_argument("--reasoning", default="high",
                    help="DeepSeek reasoning effort (high|medium|low); ignored for local")
    ap.add_argument("--agree-anchors", choices=tuple(AGREE_ANCHOR_VARIANTS), default="genetic",
                    help="AGREE-anchor ablation: 'genetic' (molecular baseline), 'neural' "
                         "(non-molecular 1-shot), or 'neural-genetic' (2-shot). Non-baseline "
                         "variants are tagged + checkpointed separately so they coexist with "
                         "the baseline corpus")
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint JSONL (default: per-judge — pilot_verdicts / deepseek_verdicts)")
    ap.add_argument("--no-persist", action="store_true",
                    help="skip the MySQL write (checkpoint only) — for pricing / smoke runs")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                    help="harrier tokenizer path (CPU only — no model weights / GPU)")
    ap.add_argument("--host", default="http://asushimu:11434",
                    help="Gemma OpenAI-compatible endpoint")
    ap.add_argument("--workers", type=int, default=cj.DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=0, help="cap papers (0 = all selected)")
    args = ap.parse_args()

    load_env()  # OPENROUTER_API_KEY for the paid DeepSeek judge; no-op without .env

    agree_keys, tag_suffix = AGREE_ANCHOR_VARIANTS[args.agree_anchors]

    base_checkpoint = DEEPSEEK_CHECKPOINT if args.judge == "deepseek" else DEFAULT_CHECKPOINT
    if args.checkpoint:
        checkpoint = args.checkpoint
    elif tag_suffix:
        # a non-baseline variant must NOT share the baseline checkpoint or it would resume
        # off baseline cells; give it its own file (slug from the tag suffix).
        root, ext = os.path.splitext(base_checkpoint)
        checkpoint = f"{root}_{tag_suffix.lstrip('@')}{ext}"
    else:
        checkpoint = base_checkpoint

    topics = load_augmented_topics()
    controls = json.load(open(PROTOTYPES_JSON, encoding="utf-8"))["_controls"]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    meter = cj.UsageMeter() if args.judge == "deepseek" else None
    complete, model_tag = make_judge(args.judge, host=args.host,
                                     reasoning=args.reasoning, meter=meter)
    model_tag += tag_suffix  # anchor-ablation variants coexist under a distinct judge tag
    logger.info("judge=%s model=%s agree_anchors=%s checkpoint=%s%s",
                args.judge, model_tag, args.agree_anchors, checkpoint,
                " [no-persist]" if args.no_persist else "")

    # Initial reads (topic assignments + codes) on a short-lived connection. We do NOT hold
    # this open across the multi-hour judging loop: an idle conn past MySQL ``wait_timeout``
    # (8h) is silently closed server-side and the next query dies with "lost connection" —
    # which is exactly what aborted the post-pilot persist. Reads + persist each get their
    # own fresh connection via db.run_with_reconnect (reconnect + retry on a transient drop).
    def _read(conn):
        return (db.fetch_chunk_topics(conn, run=args.run, method=args.method),
                db.fetch_vectors(conn, run=args.run)[2])
    chunk_topics, codes = db.run_with_reconnect(_read)

    if args.code is not None:
        selected = select_code_papers(chunk_topics, codes, args.code)
    elif args.rest:
        selected = select_rest_papers(chunk_topics, n=args.top)
    else:
        selected = select_pilot_papers(chunk_topics, n=args.top)
    pids = sorted(selected)
    if args.limit:
        pids = pids[:args.limit]
    if args.code is not None:
        logger.info("pilot: %d papers under code %s", len(pids), args.code)
    else:
        logger.info("pilot: %d papers %s top-%d topics %s",
                    len(pids), "OUTSIDE" if args.rest else "across", args.top,
                    top_topic_ids(chunk_topics, n=args.top))

    done = load_done(checkpoint)
    write_lock = threading.Lock()

    def work(pid):
        tid = selected[pid]
        label, blurb = topics.get(tid, (str(tid), ""))
        full_text = pdf_text.extract_text(pid)
        meta = _paper_meta_from_codes(pid, codes)
        return judge_paper_chunks(meta, full_text, tokenizer, label, blurb,
                                  controls, complete, checkpoint, done, write_lock,
                                  agree_keys=agree_keys)

    # No DB connection is held during judging — all output goes to the JSONL checkpoint
    # (the system of record), so a drop here costs nothing and persistence reads it back.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, pid): pid for pid in pids}
        for n, fut in enumerate(as_completed(futures), 1):
            pid = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                logger.warning("skipping paper %s: %s", pid, exc)
            logger.info("paper %d/%d done (%s)", n, len(pids), pid)

    if meter is not None:
        report_usage(meter, papers=len(pids), corpus=219)

    if args.no_persist:
        logger.info("--no-persist: skipping MySQL write (checkpoint %s is the record)", checkpoint)
        return

    # Persist from the *full* checkpoint (resilient to resumed runs), not just this
    # invocation's records, so aggregation always sees every judged chunk.
    ckpt_records = _read_checkpoint(checkpoint, set(pids))
    # Stamp prompt provenance: records from an older checkpoint predate the prompt_hash
    # field, so back-fill it from the live prompt version (correct because persistence
    # runs under the same code that produced these verdicts), and register the templates.
    for r in ckpt_records:
        r.setdefault("prompt_hash", cj.prompt_hash(r["criterion"]))
    crits = sorted({r["criterion"] for r in ckpt_records})
    prompt_entries = [
        {"prompt_hash": cj.prompt_hash(c), "criterion": c,
         "prompt_text": cj.prompt_template(c)} for c in crits]
    verdict_records = [
        {**r, "criteria": {k: {**v} for k, v in r["criteria"].items()}}
        for r in aggregate_to_verdict_records(ckpt_records)]

    # The persist sequence (register_prompts -> store_chunk_verdicts -> update_verdicts) is
    # idempotent (guarded init_schema + upserts), so run_with_reconnect can safely re-run it
    # from the top on a fresh connection if the link drops mid-write — no data is lost.
    def _persist(conn):
        db.register_prompts(conn, prompt_entries)
        n_chunks = db.store_chunk_verdicts(conn, ckpt_records, model=model_tag)
        n_verdicts = db.update_verdicts(conn, verdict_records, model=model_tag)
        return n_chunks, n_verdicts
    n_chunks, n_verdicts = db.run_with_reconnect(_persist)
    logger.info("persisted %d chunk_verdicts, %d verdicts (graded + categorical)",
                n_chunks, n_verdicts)


def _read_checkpoint(checkpoint_path, pids=None):
    """Read all checkpoint records (optionally restricted to ``pids``) for persistence."""
    out = []
    if not os.path.exists(checkpoint_path):
        return out
    with open(checkpoint_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping malformed checkpoint line")
                continue
            if pids is None or rec.get("pdf_path") in pids:
                out.append(rec)
    return out


if __name__ == "__main__":
    main()
