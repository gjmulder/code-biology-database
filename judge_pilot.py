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
DEFAULT_TOKENIZER = "/data/vllm/harrier-oss-v1-27b"
AUGMENTED_CSV = "code-categories-augmented.csv"
PROTOTYPES_JSON = "prototypes.json"


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
                       controls, complete, checkpoint_path, done, write_lock):
    """Judge every (chunk × criterion) cell of one paper not already in ``done``.

    Reproduces the embedding windows, builds the calibrated per-chunk prompt, calls Gemma,
    parses + grounds, and checkpoints each cell (under ``write_lock``) before returning. The
    grounding gate neutralises ungrounded positives; a per-cell model/parse failure is logged
    and skipped so one bad chunk never aborts the paper. Returns the records judged here."""
    chunks = chunk_text.reproduce_chunks(full_text, tokenizer)
    records = []
    for idx, ctext in chunks:
        for crit in cj.ALL_CRITERIA:
            if (paper_meta["pdf_path"], idx, crit) in done:
                continue
            try:
                raw = complete(cj.GRADED_SYSTEM_PROMPT,
                               cj.build_chunk_prompt(ctext, crit, topic_label,
                                                     topic_blurb, controls),
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
    ap.add_argument("--run", default="baseline", help="embedding run whose chunk_topics to use")
    ap.add_argument("--method", default="chunk")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                    help="harrier tokenizer path (CPU only — no model weights / GPU)")
    ap.add_argument("--host", default="http://asushimu:11434",
                    help="Gemma OpenAI-compatible endpoint")
    ap.add_argument("--workers", type=int, default=cj.DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=0, help="cap papers (0 = all selected)")
    args = ap.parse_args()

    topics = load_augmented_topics()
    controls = json.load(open(PROTOTYPES_JSON, encoding="utf-8"))["_controls"]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    complete = cj.local_complete_factory(host=args.host)

    # Initial reads (topic assignments + codes) on a short-lived connection. We do NOT hold
    # this open across the multi-hour judging loop: an idle conn past MySQL ``wait_timeout``
    # (8h) is silently closed server-side and the next query dies with "lost connection" —
    # which is exactly what aborted the post-pilot persist. Reads + persist each get their
    # own fresh connection via db.run_with_reconnect (reconnect + retry on a transient drop).
    def _read(conn):
        return (db.fetch_chunk_topics(conn, run=args.run, method=args.method),
                db.fetch_vectors(conn, run=args.run)[2])
    chunk_topics, codes = db.run_with_reconnect(_read)

    selected = select_pilot_papers(chunk_topics, n=args.top)
    pids = sorted(selected)
    if args.limit:
        pids = pids[:args.limit]
    logger.info("pilot: %d papers across top-%d topics %s",
                len(pids), args.top, top_topic_ids(chunk_topics, n=args.top))

    done = load_done(args.checkpoint)
    write_lock = threading.Lock()

    def work(pid):
        tid = selected[pid]
        label, blurb = topics.get(tid, (str(tid), ""))
        full_text = pdf_text.extract_text(pid)
        meta = _paper_meta_from_codes(pid, codes)
        return judge_paper_chunks(meta, full_text, tokenizer, label, blurb,
                                  controls, complete, args.checkpoint, done, write_lock)

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

    # Persist from the *full* checkpoint (resilient to resumed runs), not just this
    # invocation's records, so aggregation always sees every judged chunk.
    ckpt_records = _read_checkpoint(args.checkpoint, set(pids))
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
        n_chunks = db.store_chunk_verdicts(conn, ckpt_records, model=cj.LOCAL_MODEL)
        n_verdicts = db.update_verdicts(conn, verdict_records)
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
