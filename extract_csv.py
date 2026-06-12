import pdfplumber
import csv
import json
import re
from collections import defaultdict

# x-coordinate column boundaries (points) - using the same template boundaries
COL_BOUNDS = {
    "number":      (35, 92),
    "code_name":   (92, 202),
    "descriptive": (202, 344),
    "citations":   (344, 850),
}
FOOTER_TOP = 540  # exclude footer text below this y-coordinate


def get_col_chars(chars, col):
    x0, x1 = COL_BOUNDS[col]
    return [c for c in chars if x0 <= c["x0"] < x1 and c["top"] < FOOTER_TOP]


def chars_to_text(chars):
    if not chars:
        return ""
    lines = defaultdict(list)
    for c in chars:
        lines[round(c["top"])].append(c)
    result = []
    for top in sorted(lines):
        line_text = "".join(
            c["text"] for c in sorted(lines[top], key=lambda c: c["x0"])
        ).strip()
        if line_text:
            result.append(line_text)
    return " ".join(result)


def extract_rows(pdf):
    all_rows = []
    # We will iterate over all pages. If there are no numbers found (e.g. on a title page), it will just safely skip.
    for page in pdf.pages:
        chars = page.chars
        hyperlinks = page.hyperlinks

        # Build a map of citation URIs by y-range
        uri_by_top = defaultdict(set)
        for h in hyperlinks:
            uri = h.get("uri") or ""
            if uri.startswith("http") and h["x0"] >= COL_BOUNDS["citations"][0]:
                uri_by_top[(round(h["top"]), round(h["bottom"]))].add(h["uri"])

        # Find row anchors: entries in the number column that are pure digits
        num_chars = get_col_chars(chars, "number")
        num_by_top = defaultdict(list)
        for c in num_chars:
            num_by_top[round(c["top"])].append(c["text"])

        row_tops = []
        for top, text_list in sorted(num_by_top.items()):
            text = "".join(text_list).strip()
            if re.match(r"^\d+$", text):
                row_tops.append((top, int(text)))

        page_bottom = FOOTER_TOP
        for i, (top, num) in enumerate(row_tops):
            next_top = row_tops[i + 1][0] if i + 1 < len(row_tops) else page_bottom
            row_chars = [
                c for c in chars
                if top - 5 <= round(c["top"]) < next_top and c["top"] < FOOTER_TOP
            ]

            code = chars_to_text(get_col_chars(row_chars, "code_name"))
            desc = chars_to_text(get_col_chars(row_chars, "descriptive"))
            cite = chars_to_text(get_col_chars(row_chars, "citations"))

            # Collect all URIs whose y-range falls within this row
            urls = set()
            for (h_top, h_bottom), uris in uri_by_top.items():
                if top - 5 <= h_top < next_top:
                    urls.update(uris)

            all_rows.append({
                "number": num,
                "code_name": code,
                "descriptive": desc,
                "citations": cite,
                "urls": sorted(urls),
            })

    return all_rows


def write_csv(rows, out_path="biological_codes.csv"):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Number", "Code Name", "Description", "Citations", "URLs"])
        for row in rows:
            writer.writerow([
                row["number"],
                row["code_name"],
                row["descriptive"],
                row["citations"],
                json.dumps(row["urls"])
            ])


def run():
    pdf_path = "Biological_Code_List_20260531.pdf"
    print(f"Extracting data from {pdf_path} ...")
    with pdfplumber.open(pdf_path) as pdf:
        rows = extract_rows(pdf)

    print(f"Extracted {len(rows)} codes. Writing to CSV ...")
    out_csv = "biological_codes.csv"
    write_csv(rows, out_csv)
    print(f"Data successfully written to {out_csv}")


if __name__ == "__main__":
    run()
