"""Extract Code -> Paper Name -> URL mappings from the Biological Code List PDF.

The PDF is a four-column table (number | code name | description | citations).
Every citation in the right-hand column is a hyperlink whose anchor text is the
full reference, so the hyperlinks are the most reliable signal for isolating
individual papers and mapping each to its URL.

Two structural properties of the source drive the parsing logic:

  * A code's citation list can spill over several pages. Continuation pages have
    no number in the left column, so the "current" code is carried across page
    boundaries (see ``_page_segments``).
  * A hyperlink rectangle sits a few points *above* the number digit on the same
    visual row, so row bands are shifted up by ``ANCHOR_SLACK`` at both ends to
    stop a row from stealing the next row's first link.
"""

import csv
import logging
import re
from collections import OrderedDict, defaultdict

import pdfplumber

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# x-coordinate column boundaries (points).
COL_BOUNDS = {
    "number": (35, 92),
    "code_name": (92, 202),
    "descriptive": (202, 344),
    "citations": (344, 850),
}
FOOTER_TOP = 540  # exclude footer text below this y-coordinate
ANCHOR_SLACK = 5  # points a hyperlink may sit above its row's number digit

# A reference is counted in the citation text by its "(YYYY)" publication year.
YEAR_RE = re.compile(r"\(\d{4}[a-z]?\)")


def get_col_chars(chars, col):
    """Characters that fall inside a named column and above the footer."""
    x0, x1 = COL_BOUNDS[col]
    return [c for c in chars if x0 <= c["x0"] < x1 and c["top"] < FOOTER_TOP]


def chars_to_text(chars):
    """Join characters into reading-order text, one space between visual lines."""
    if not chars:
        return ""
    lines = defaultdict(list)
    for c in chars:
        lines[round(c["top"])].append(c)
    out = []
    for top in sorted(lines):
        line = "".join(c["text"] for c in sorted(lines[top], key=lambda c: c["x0"])).strip()
        if line:
            out.append(line)
    return " ".join(out)


def count_text_citations(text):
    """Estimate the number of citations in a block of citation text.

    Heuristic: each reference carries one parenthesised publication year. Used
    only to cross-check the number of extracted URLs for data-integrity logging.
    """
    return len(YEAR_RE.findall(text))


def find_anchors(chars):
    """Return ``[(top, code_number), ...]`` for numbered rows on a page."""
    num_by_top = defaultdict(list)
    for c in get_col_chars(chars, "number"):
        num_by_top[round(c["top"])].append((c["x0"], c["text"]))
    anchors = []
    for top, items in sorted(num_by_top.items()):
        text = "".join(t for _, t in sorted(items)).strip()
        if re.fullmatch(r"\d+", text):
            anchors.append((top, int(text)))
    return anchors


def _is_citation_link(h):
    return (
        h.get("uri", "").startswith("http")
        and h["x0"] >= COL_BOUNDS["citations"][0]
    )


def _page_segments(anchors, carried_code):
    """Vertical bands ``(top0, top1, code_number)`` covering a page.

    ``carried_code`` is the code continued from the previous page (or ``None``);
    content above the first anchor belongs to it.
    """
    segments = []
    if anchors:
        first_top = anchors[0][0]
        if carried_code is not None and first_top - ANCHOR_SLACK > 0:
            segments.append((0, first_top - ANCHOR_SLACK, carried_code))
        for i, (top, num) in enumerate(anchors):
            top1 = anchors[i + 1][0] - ANCHOR_SLACK if i + 1 < len(anchors) else FOOTER_TOP
            segments.append((top - ANCHOR_SLACK, top1, num))
    elif carried_code is not None:
        segments.append((0, FOOTER_TOP, carried_code))
    return segments


def _link_runs(links):
    """Group hyperlink fragments into references.

    ``links`` are tagged with a page index and sorted into reading order;
    consecutive fragments sharing a URI (one link split across lines) collapse
    into a single reference.
    """
    links = sorted(links, key=lambda h: (h["_page"], round(h["top"]), h["x0"]))
    runs = []
    for h in links:
        if runs and runs[-1]["uri"] == h["uri"] and runs[-1]["rects"][-1]["_page"] == h["_page"]:
            runs[-1]["rects"].append(h)
        else:
            runs.append({"uri": h["uri"], "rects": [h]})
    return runs


def _rects_text(rects, cite_chars):
    """Text of the characters covered by a hyperlink's rectangles."""
    page = rects[0]["_page"]
    picked = []
    for c in cite_chars:
        if c["_page"] != page:
            continue
        cx, cy = (c["x0"] + c["x1"]) / 2, (c["top"] + c["bottom"]) / 2
        for r in rects:
            if r["x0"] - 2 <= cx <= r["x1"] + 2 and r["top"] - 2 <= cy <= r["bottom"] + 2:
                picked.append(c)
                break
    return chars_to_text(picked)


def extract_references(pdf_path):
    """Parse the PDF into ``{code_number, code_name, paper_name, url}`` rows."""
    # Per code (kept in first-seen order): name, citation chars, hyperlinks.
    codes = OrderedDict()

    def bucket(num):
        return codes.setdefault(num, {"name": "", "cite_chars": [], "links": []})

    with pdfplumber.open(pdf_path) as pdf:
        carried_code = None
        for page_index, page in enumerate(pdf.pages):
            chars = page.chars
            for c in chars:
                c["_page"] = page_index
            links = [h for h in page.hyperlinks if _is_citation_link(h)]
            for h in links:
                h["_page"] = page_index

            anchors = find_anchors(chars)
            for top0, top1, num in _page_segments(anchors, carried_code):
                band_chars = [c for c in chars if top0 <= round(c["top"]) < top1]
                entry = bucket(num)
                if not entry["name"]:
                    entry["name"] = chars_to_text(get_col_chars(band_chars, "code_name"))
                entry["cite_chars"].extend(get_col_chars(band_chars, "citations"))
                entry["links"].extend(
                    h for h in links if top0 <= round(h["top"]) < top1
                )
            if anchors:
                carried_code = anchors[-1][1]

    references = []
    for num, entry in codes.items():
        runs = _link_runs(entry["links"])
        citation_text = chars_to_text(entry["cite_chars"])
        expected = count_text_citations(citation_text)
        if expected != len(runs):
            logger.warning(
                "Code %d (%r): %d citations parsed from text but %d URLs found.",
                num, entry["name"], expected, len(runs),
            )

        # Codes whose citations carry no hyperlink still belong in the output;
        # record the citation text with an empty URL so no code is dropped.
        if not runs and citation_text:
            references.append({
                "code_number": num,
                "code_name": entry["name"],
                "paper_name": citation_text,
                "url": "",
            })
            continue

        for run in runs:
            paper_name = _rects_text(run["rects"], entry["cite_chars"])
            if not paper_name:
                logger.warning("Code %d (%r): hyperlink %s has no anchor text.",
                               num, entry["name"], run["uri"])
                continue
            references.append({
                "code_number": num,
                "code_name": entry["name"],
                "paper_name": paper_name,
                "url": run["uri"],
            })

    logger.info("Extracted %d references across %d codes.", len(references), len(codes))
    return references


def write_csv(references, out_path="biological_codes.csv"):
    """Write the Code -> Paper Name -> URL mapping as a clean CSV."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Code Number", "Code Name", "Paper Name", "URL"])
        for ref in references:
            writer.writerow([
                ref["code_number"],
                ref["code_name"],
                ref["paper_name"],
                ref["url"],
            ])


def run():
    pdf_path = "Biological_Code_List_20260531.pdf"
    logger.info("Extracting data from %s ...", pdf_path)
    references = extract_references(pdf_path)
    out_csv = "biological_codes.csv"
    write_csv(references, out_csv)
    logger.info("Data successfully written to %s", out_csv)


if __name__ == "__main__":
    run()
