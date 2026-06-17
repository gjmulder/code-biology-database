"""Generate a single self-contained, sortable HTML ranking report.

Reads the **DB** (system of record) for the two axes and joins the canonical
citation list (``biological_codes.csv``), then writes one static HTML page
(``ranking_report.html``) with all data inlined as JSON:

- **pages** — the per-chunk ("page") embedding score ``e`` per criterion (the
  ``baseline`` run, ``chunk`` method in ``embedding_scores``; the retired
  full/abstract granularities are no longer shown).
- **verdicts** — the domain-general graded LLM judge (CLAUDE.md §9/§9.1): the rows
  in ``verdicts`` carrying a ``graded`` value (the ~100-paper re-pilot). Papers not
  yet re-judged show a blank verdict.
- **topics** — the scientometric topics each paper *covers* (§2.1), from ``chunk_topics``
  + ``topic_centroids``: the distinct nearest-centroid topics across a paper's chunks,
  dominant (max-pool affinity) first, then by chunk count. Shown in a sortable/filterable
  Topics column; diagnostic context, not a measurement axis.

The page lets the user rank the papers by either axis, collapsing the three criteria
to one number via a selectable metric (mean / median / min). All sorting, filtering
and ranking is client-side vanilla JS; the file opens straight from ``file://``.

Scores and verdicts come from the DB; only the citation metadata (code/paper name,
URL) is read from ``biological_codes.csv`` because no DB table holds it.
"""

import argparse
import csv
import datetime
import json
import logging
import os

import assign_topics
import db
from download_pdfs import output_path_for, read_rows
from embed_score import CRITERIA
from criteria_judge import VERDICT_ORDINAL

log = logging.getLogger(__name__)

DEFAULT_CODES = "biological_codes.csv"
DEFAULT_OUT = "ranking_report.html"
DEFAULT_RUN = "baseline"
EMBED_METHOD = "chunk"  # the working granularity; surfaced in the UI as "pages"


# --- aggregation -----------------------------------------------------------

def verdict_ordinal(verdict):
    """Ordinal for a verdict string, or ``None`` if absent/unknown.

    Unlike ``criteria_judge.verdict_ordinal`` (which floors unknowns to 0.5 for
    Spearman), an absent verdict here stays ``None`` so it is excluded from the
    paper's aggregate rather than silently counted as half-met.
    """
    return VERDICT_ORDINAL.get(verdict)


def aggregate(vals, metric):
    """Collapse per-criterion values to one number, ignoring ``None``.

    ``mean`` / ``median`` / ``min`` over the present values; ``None`` if none.
    ``min`` is the weakest-link reading (Barbieri: a code needs all three criteria).
    """
    present = [v for v in vals if v is not None]
    if not present:
        return None
    if metric == "mean":
        return sum(present) / len(present)
    if metric == "min":
        return min(present)
    if metric == "median":
        s = sorted(present)
        n = len(s)
        mid = n // 2
        return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2
    raise ValueError(f"unknown metric: {metric}")


# --- DB inputs -------------------------------------------------------------

def _blank_rec(code, pdf_path):
    return {
        "code": code,
        "pdf_path": pdf_path,
        "e": {c: None for c in CRITERIA},
        "verdict": {c: None for c in CRITERIA},
        "graded": {c: None for c in CRITERIA},
        "conf": {c: None for c in CRITERIA},
        "topics": [],
        "dominant_topic": "",
    }


def _assemble(score_rows, verdict_rows):
    """Pivot the two DB result sets into one record per ``(code, pdf_path)``.

    ``score_rows``: ``(code_number, pdf_path, criterion, e)`` (chunk method).
    ``verdict_rows``: ``(pdf_path, criterion, verdict, graded, confidence)`` — only
    the re-judged papers; papers absent here keep ``None`` verdict fields.
    """
    papers = {}
    for code, pdf, crit, e in score_rows:
        if crit not in CRITERIA:
            continue
        papers.setdefault((code, pdf), _blank_rec(code, pdf))["e"][crit] = e

    v_by_pdf = {}
    for pdf, crit, verdict, graded, conf in verdict_rows:
        v_by_pdf.setdefault(pdf, {})[crit] = (verdict, graded, conf)
    for rec in papers.values():
        for crit, (verdict, graded, conf) in v_by_pdf.get(rec["pdf_path"], {}).items():
            if crit in CRITERIA:
                rec["verdict"][crit] = verdict
                rec["graded"][crit] = graded
                rec["conf"][crit] = conf

    return sorted(papers.values(), key=lambda r: (r["code"], r["pdf_path"]))


def paper_topics(chunk_assignments, labels):
    """Order the scientometric topics a paper *covers* into labels for the report.

    ``chunk_assignments``: the paper's ``[(chunk_idx, topic_id, sim), ...]`` (each chunk's
    nearest-centroid vote, §2.1); ``labels``: ``{topic_id: label}``. Returns the distinct
    topic labels the paper's chunks touch, **dominant first** (the max-pool affinity topic,
    consistent with every other dominant-topic use in the project), then the remaining
    topics by descending chunk count (ties: descending affinity, then topic id). An
    unlabelled topic id degrades to ``"topic N"``. Empty list for a paper with no chunks."""
    if not chunk_assignments:
        return []
    dominant, affinity = assign_topics.paper_dominant_topic(chunk_assignments)
    counts = {}
    for _idx, tid, _sim in chunk_assignments:
        counts[tid] = counts.get(tid, 0) + 1
    rest = sorted((t for t in affinity if t != dominant),
                  key=lambda t: (-counts[t], -affinity[t], t))
    return [labels.get(t) or f"topic {t}" for t in [dominant] + rest]


def attach_topics(records, chunk_topics, centroids):
    """Attach ``topics`` (ordered labels) + ``dominant_topic`` to each record.

    ``chunk_topics``: ``{pdf_path: [(idx, topic_id, sim), ...]}`` (db.fetch_chunk_topics
    shape); ``centroids``: ``{topic_id: label}`` or ``{topic_id: {'label': ...}}`` (so the
    db.fetch_topic_centroids shape works too). Papers with no chunk topics degrade to an
    empty list / blank dominant."""
    labels = {tid: (c.get("label") if isinstance(c, dict) else c)
              for tid, c in centroids.items()}
    for rec in records:
        topics = paper_topics(chunk_topics.get(rec["pdf_path"], []), labels)
        rec["topics"] = topics
        rec["dominant_topic"] = topics[0] if topics else ""
    return records


def load_papers_from_db(run=DEFAULT_RUN):
    """Read chunk embedding ``e`` + the domain-general graded verdicts + the topic
    coverage from the DB."""
    def work(conn):
        cur = conn.cursor()
        cur.execute(
            "SELECT code_number, pdf_path, criterion, e FROM embedding_scores "
            "WHERE run=%s AND method=%s",
            (run, EMBED_METHOD),
        )
        scores = cur.fetchall()
        cur.execute(
            "SELECT pdf_path, criterion, verdict, graded, confidence FROM verdicts "
            "WHERE graded IS NOT NULL"
        )
        verdicts = cur.fetchall()
        cur.execute(
            "SELECT pdf_path, chunk_idx, topic_id, sim FROM chunk_topics "
            "WHERE run=%s AND method=%s ORDER BY pdf_path, chunk_idx",
            (run, EMBED_METHOD),
        )
        chunk_topic_rows = cur.fetchall()
        cur.execute("SELECT topic_id, label FROM topic_centroids WHERE run=%s", (run,))
        centroid_rows = cur.fetchall()
        return scores, verdicts, chunk_topic_rows, centroid_rows

    scores, verdicts, chunk_topic_rows, centroid_rows = db.run_with_reconnect(work)
    recs = _assemble(scores, verdicts)

    chunk_topics = {}
    for pid, idx, tid, sim in chunk_topic_rows:
        chunk_topics.setdefault(pid, []).append((idx, tid, sim))
    centroids = {tid: label for tid, label in centroid_rows}
    attach_topics(recs, chunk_topics, centroids)

    judged = sum(any(v is not None for v in r["verdict"].values()) for r in recs)
    topiced = sum(1 for r in recs if r["topics"])
    log.info("loaded %d papers from DB (%d carry domain-general verdicts, %d topic-assigned)",
             len(recs), judged, topiced)
    return recs


# --- citation join (the one piece not in the DB) ---------------------------

def load_citations(path):
    """Map ``basename(pdf_path) -> {code_name, paper_name, url}`` from the code list."""
    cmap = {}
    for row in read_rows(path):
        base = os.path.basename(output_path_for(row, ""))
        cmap[base] = {
            "code_name": row["code_name"],
            "paper_name": row["paper_name"],
            "url": row["url"],
        }
    return cmap


def attach_citations(records, codes_path):
    """Attach citation metadata to each record; unmatched papers degrade gracefully."""
    cmap = load_citations(codes_path)
    matched = 0
    for rec in records:
        cite = cmap.get(os.path.basename(rec["pdf_path"]))
        if cite:
            matched += 1
        rec["code_name"] = cite["code_name"] if cite else ""
        rec["paper_name"] = cite["paper_name"] if cite else ""
        rec["url"] = cite["url"] if cite else ""
    log.info("joined %d/%d papers to a citation", matched, len(records))
    return records


# --- HTML ------------------------------------------------------------------

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Code Biology — paper ranking</title>
<style>
  :root {{ --bg:#fff; --fg:#1a1a1a; --line:#ddd; --muted:#666; }}
  body {{ font:14px/1.45 system-ui,sans-serif; margin:0; color:var(--fg); background:var(--bg); }}
  header {{ padding:14px 20px; border-bottom:1px solid var(--line); }}
  h1 {{ font-size:18px; margin:0 0 4px; }}
  .sub {{ color:var(--muted); font-size:12px; }}
  .controls {{ display:flex; flex-wrap:wrap; gap:18px; align-items:center;
               padding:12px 20px; border-bottom:1px solid var(--line); position:sticky;
               top:0; background:var(--bg); z-index:2; }}
  .controls label {{ font-weight:600; margin-right:6px; }}
  .seg button {{ border:1px solid var(--line); background:#f6f6f6; padding:5px 10px;
                 cursor:pointer; font:inherit; }}
  .seg button.on {{ background:#1a1a1a; color:#fff; }}
  .seg button:first-child {{ border-radius:6px 0 0 6px; }}
  .seg button:last-child {{ border-radius:0 6px 6px 0; }}
  input[type=search] {{ padding:6px 9px; border:1px solid var(--line); border-radius:6px;
                        font:inherit; min-width:220px; }}
  .count {{ color:var(--muted); font-size:12px; margin-left:auto; }}
  table {{ border-collapse:collapse; width:100%; }}
  th,td {{ padding:6px 9px; border-bottom:1px solid var(--line); text-align:left;
           vertical-align:top; }}
  th {{ position:sticky; top:53px; background:#fafafa; cursor:pointer;
        user-select:none; white-space:nowrap; }}
  th.sorted::after {{ content:" \\25BC"; font-size:10px; }}
  th.sorted.asc::after {{ content:" \\25B2"; }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td.rank {{ font-weight:700; }}
  .paper {{ max-width:520px; }}
  .paper a {{ color:#1a4f8b; text-decoration:none; }}
  .paper a:hover {{ text-decoration:underline; }}
  .topics {{ max-width:280px; color:#555; font-size:12px; }}
  .topics b {{ color:#1a1a1a; font-weight:600; }}
  .topics .more {{ color:#999; }}
  .v-met {{ color:#0a7d24; font-weight:600; }}
  .v-unclear {{ color:#b07000; }}
  .v-not_met {{ color:#999; }}
  details {{ margin:12px 20px; color:var(--muted); font-size:12px; }}
  summary {{ cursor:pointer; font-weight:600; }}
  .idx {{ color:var(--muted); }}
</style>
</head>
<body>
<header>
  <h1>Code Biology — paper ranking</h1>
  <div class="sub">{n} papers · baseline run · generated {date}. Rank by
    <b>similarity to genetic code</b> (the per-chunk embedding score <code>e</code>,
    max-pooled) or the domain-general <b>LLM verdicts</b> (graded judge, {judged} papers
    judged); the three criteria collapse to one score via the chosen metric. Click any
    column to sort.</div>
</header>

<div class="controls">
  <span><label>Rank by</label>
    <span class="seg" id="source">
      <button data-v="pages">similarity to genetic code</button><button data-v="verdicts" class="on">LLM verdicts</button>
    </span>
  </span>
  <span><label>Metric</label>
    <span class="seg" id="metric">
      <button data-v="mean">mean</button><button data-v="median">median</button><button data-v="min" class="on">min (weakest)</button>
    </span>
  </span>
  <input type="search" id="filter" placeholder="filter code / citation…">
  <span class="count" id="count"></span>
</div>

<table>
  <thead><tr id="head"></tr></thead>
  <tbody id="body"></tbody>
</table>

<details>
  <summary>About this report &amp; caveats</summary>
  <p>Two independent axes over the same papers, reported side by side &mdash; neither is
  ground truth. <b>Similarity to genetic code</b> = the per-chunk (8192-token "page")
  corpus-contrastive embedding score <code>e</code> per criterion, max-pooled over a
  paper's chunks (positive = reads, in the embedding space, like the genetic-code
  archetype for that criterion). <b>LLM verdicts</b> are the domain-general graded
  judge's <code>met / unclear / not_met</code> per criterion (CLAUDE.md §9/§9.1), each
  <code>met</code> gated by a verbatim quote; papers not yet judged show &ndash;. Hover a
  verdict cell for its graded value and confidence. The verdicts are still
  <i>synthetic</i> labels from a comparatively weak judge, so <b>ranks are more
  trustworthy than absolute magnitudes</b>. The two axes can legitimately diverge: the
  embedding keys on the molecular archetype while the judge scores codes across domains
  (test_runs.md Run 6). The <i>min</i> metric is the weakest-link reading: per Barbieri a
  biological code requires <i>all three</i> criteria.</p>
</details>

<script>
const PAPERS = {data};
const CRITERIA = {criteria};
const ORD = {{not_met:0.0, unclear:0.5, met:1.0}};

let source = "verdicts", metric = "min";
let sortKey = "rank", sortAsc = false;

function critVals(p) {{
  // per-criterion numeric values for the active source
  if (source === "verdicts")
    return CRITERIA.map(c => {{ const v = p.verdict[c]; return v == null ? null : ORD[v]; }});
  return CRITERIA.map(c => p.e[c]);
}}
function agg(vals) {{
  const present = vals.filter(v => v != null);
  if (!present.length) return null;
  if (metric === "mean") return present.reduce((a,b)=>a+b,0)/present.length;
  if (metric === "min") return Math.min(...present);
  const s = present.slice().sort((a,b)=>a-b), n = s.length, m = n>>1;
  return n % 2 ? s[m] : (s[m-1]+s[m])/2;
}}
function fmtNum(x) {{ return x == null ? "–" : (x>=0?"+":"") + x.toFixed(3); }}
function esc(s) {{ return (s||"").replace(/[&<>"]/g, c => (
  {{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c])); }}

function cellHTML(p, c) {{
  if (source === "verdicts") {{
    const v = p.verdict[c];
    if (v == null) return '<td class="num">–</td>';
    const bits = [];
    if (p.graded[c] != null) bits.push("graded " + p.graded[c]);
    if (p.conf[c] != null) bits.push("confidence " + p.conf[c]);
    const t = bits.length ? ' title="'+bits.join(", ")+'"' : "";
    return '<td class="num v-'+v+'"'+t+'>'+v+'</td>';
  }}
  const x = p.e[c];
  const bg = x == null ? "" : (x>=0
    ? 'background:rgba(10,125,36,'+Math.min(Math.abs(x)*4,0.5).toFixed(3)+')'
    : 'background:rgba(190,30,30,'+Math.min(Math.abs(x)*4,0.5).toFixed(3)+')');
  return '<td class="num" style="'+bg+'">'+fmtNum(x)+'</td>';
}}

function topicsHTML(p) {{
  // dominant topic in bold, the rest (other topics the paper's chunks touch) muted
  const t = p.topics || [];
  if (!t.length) return '<td class="topics">–</td>';
  const parts = t.map((x,i) => i === 0
    ? '<b>'+esc(x)+'</b>'
    : '<span class="more">'+esc(x)+'</span>');
  return '<td class="topics">'+parts.join(", ")+'</td>';
}}

function rowsView() {{
  const q = document.getElementById("filter").value.toLowerCase();
  let rows = PAPERS.map(p => ({{p, rank: agg(critVals(p)), cv: critVals(p)}}));
  if (q) rows = rows.filter(r =>
    (r.p.code_name+" "+r.p.paper_name+" "+r.p.code+" "+(r.p.topics||[]).join(" "))
      .toLowerCase().includes(q));
  rows.sort((a,b) => {{
    let av, bv;
    if (sortKey === "rank") {{ av=a.rank; bv=b.rank; }}
    else if (sortKey === "code") {{ av=a.p.code; bv=b.p.code; }}
    else if (sortKey === "code_name") {{ av=a.p.code_name; bv=b.p.code_name; }}
    else if (sortKey === "paper") {{ av=a.p.paper_name; bv=b.p.paper_name; }}
    else if (sortKey === "topics") {{ av=a.p.dominant_topic; bv=b.p.dominant_topic; }}
    else if (sortKey.startsWith("c:")) {{ const i=+sortKey.slice(2); av=a.cv[i]; bv=b.cv[i]; }}
    else {{ av=0; bv=0; }}
    av = av==null?-Infinity:av; bv = bv==null?-Infinity:bv;
    const cmp = (typeof av==="string") ? av.localeCompare(bv) : (av-bv);
    return sortAsc ? cmp : -cmp;
  }});
  return rows;
}}

function render() {{
  const head = document.getElementById("head");
  const th = (key,label,num) =>
    '<th class="'+(num?"num ":"")+(sortKey===key?("sorted "+(sortAsc?"asc":"")):"")+
    '" data-key="'+key+'">'+label+'</th>';
  let h = th("idx","#",true)+th("code","Code",true)+th("code_name","Code name",false)+
          th("paper","Paper",false)+th("topics","Topics",false);
  CRITERIA.forEach((c,i)=> h += th("c:"+i, c, true));
  h += th("rank", "RANK ("+metric+")", true);
  head.innerHTML = h;

  const rows = rowsView();
  const body = document.getElementById("body");
  body.innerHTML = rows.map((r,i) => {{
    const p = r.p;
    const cite = p.url
      ? '<a href="'+esc(p.url)+'" target="_blank" rel="noopener" title="'+esc(p.paper_name)+'">'+esc(p.paper_name||p.url)+'</a>'
      : esc(p.paper_name);
    return '<tr><td class="num idx">'+(i+1)+'</td><td class="num">'+p.code+'</td>'+
      '<td>'+esc(p.code_name)+'</td>'+
      '<td class="paper">'+cite+'</td>'+
      topicsHTML(p)+
      CRITERIA.map(c => cellHTML(p,c)).join("")+
      '<td class="num rank">'+fmtNum(r.rank)+'</td></tr>';
  }}).join("");
  document.getElementById("count").textContent = rows.length+" / "+PAPERS.length+" papers";
}}

document.getElementById("source").addEventListener("click", e => {{
  if (e.target.dataset.v) {{ source = e.target.dataset.v;
    [...e.currentTarget.children].forEach(b=>b.classList.toggle("on", b===e.target));
    sortKey = "rank"; sortAsc = false;  // re-rank by the new axis
    render(); }}
}});
document.getElementById("metric").addEventListener("click", e => {{
  if (e.target.dataset.v) {{ metric = e.target.dataset.v;
    [...e.currentTarget.children].forEach(b=>b.classList.toggle("on", b===e.target));
    sortKey = "rank"; sortAsc = false;  // re-rank by the new metric
    render(); }}
}});
document.getElementById("filter").addEventListener("input", render);
document.getElementById("head").addEventListener("click", e => {{
  const k = e.target.dataset.key; if (!k) return;
  if (sortKey === k) sortAsc = !sortAsc; else {{ sortKey = k; sortAsc = false; }}
  render();
}});
render();
</script>
</body>
</html>
"""


def build_html(papers):
    """Render the self-contained HTML page string with ``papers`` inlined as JSON."""
    judged = sum(any(v is not None for v in p["verdict"].values()) for p in papers)
    return _TEMPLATE.format(
        data=json.dumps(papers, separators=(",", ":")),
        criteria=json.dumps(CRITERIA),
        n=len(papers),
        judged=judged,
        date=datetime.date.today().isoformat(),
    )


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codes", default=DEFAULT_CODES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--run", default=DEFAULT_RUN, help="embedding run key (default: baseline)")
    args = ap.parse_args(argv)

    papers = attach_citations(load_papers_from_db(args.run), args.codes)
    html = build_html(papers)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("wrote %s (%d papers)", args.out, len(papers))


if __name__ == "__main__":
    main()
