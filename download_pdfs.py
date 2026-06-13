"""Download the PDF for every reference in ``biological_codes.csv``.

The CSV maps every code's paper to a URL. Some URLs are ``doi.org`` links;
others are publisher landing pages that may still embed a DOI. The download
strategy mirrors the ``fix_references_v2.py`` template -- Crossref is the source
of truth for the machine-readable PDF location:

  1. **DOIs first.** For any URL from which a DOI can be derived, ask Crossref
     for its registered links and download the one advertised as
     ``application/pdf``.
  2. **Everything else.** For URLs with no derivable DOI, try the URL directly;
     it is kept only if the server actually returns a PDF.

Politeness and robustness:
  * A per-domain rate limiter spaces out repeated requests to the same host, so
     downloading many papers from one publisher does not hammer it.
  * Responses are verified to be real PDFs (magic bytes / content-type) so HTML
     paywall pages are never saved with a ``.pdf`` name.
  * Crossref lookups and downloaded files are cached, so re-runs are cheap and
     resumable.

Anything that cannot be downloaded is collected and written to
``failed_downloads.csv`` with a reason, so the gaps are explicit.
"""

import csv
import json
import logging
import os
import re
import time
from urllib.parse import urljoin, urlparse

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CSV_PATH = "biological_codes.csv"
OUTPUT_DIR = "pdfs"
CACHE_FILE = "crossref_cache.json"
UNPAYWALL_CACHE = "unpaywall_cache.json"
FAILURES_CSV = "failed_downloads.csv"
MAILTO = "gjmulder@perficientur.co.uk"  # Crossref / Unpaywall contact email
CROSSREF_PAUSE = 0.2  # seconds between Crossref API calls
UNPAYWALL_PAUSE = 0.1  # seconds between Unpaywall API calls
DOMAIN_INTERVAL = 2.0  # min seconds between requests to the same download host
TIMEOUT = 30
# Some OA hosts reject non-browser user agents, so present as a real browser.
BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0"
MAX_HTML_BYTES = 2_000_000  # cap when slurping a landing page to find its PDF link

# A DOI is "10." followed by a registrant code and a slash-delimited suffix.
# The suffix runs until a character that cannot appear in a bare DOI in a URL.
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s?#\"'<>]+")
# Characters often glued onto the end of a DOI in running text / URLs.
_DOI_TRAILING = ").,;'\""


# --- DOI / path helpers ----------------------------------------------------

def extract_doi(url):
    """Return the DOI embedded in ``url``, or ``None`` if there isn't one."""
    if not url:
        return None
    match = DOI_RE.search(url)
    if not match:
        return None
    return match.group(0).rstrip(_DOI_TRAILING)


def pdf_path_for(doi, output_dir=OUTPUT_DIR):
    """Filesystem-safe path for a DOI's PDF (slashes become underscores)."""
    return os.path.join(output_dir, f"{doi.replace('/', '_')}.pdf")


def url_slug(url):
    """A short filesystem-safe slug for a URL with no derivable DOI."""
    slug = re.sub(r"^https?://", "", url)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", slug).strip("_")
    return slug[:150]


def output_path_for(row, output_dir=OUTPUT_DIR):
    """Destination PDF path for a reference row (DOI-named when possible)."""
    doi = extract_doi(row["url"])
    if doi:
        return pdf_path_for(doi, output_dir)
    return os.path.join(output_dir, f"{url_slug(row['url'])}.pdf")


# --- per-domain rate limiting ----------------------------------------------

class DomainRateLimiter:
    """Enforce a minimum interval between requests to the *same* host.

    Requests to different hosts never wait on each other; only a repeat hit on a
    host seen less than ``min_interval`` seconds ago is delayed.
    """

    def __init__(self, min_interval=DOMAIN_INTERVAL):
        self.min_interval = min_interval
        self._last = {}

    def wait(self, url):
        domain = urlparse(url).netloc
        last = self._last.get(domain)
        now = time.monotonic()
        if last is not None:
            elapsed = now - last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
        self._last[domain] = time.monotonic()


# --- Crossref --------------------------------------------------------------

def load_cache(cache_file=CACHE_FILE):
    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache, cache_file=CACHE_FILE):
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def get_crossref_item(doi, cache):
    """Fetch Crossref metadata for ``doi`` (cached). ``None`` if not found."""
    if doi in cache:
        return cache[doi]

    url = f"https://api.crossref.org/works/{doi}"
    item = None
    try:
        resp = requests.get(url, params={"mailto": MAILTO}, timeout=TIMEOUT)
        if resp.status_code == 200:
            item = resp.json().get("message")
        elif resp.status_code == 404:
            logger.debug("Crossref has no record for %s", doi)
        else:
            logger.warning("Crossref returned %d for %s", resp.status_code, doi)
    except (requests.RequestException, ValueError) as e:
        logger.warning("Crossref lookup failed for %s: %s", doi, e)

    cache[doi] = item
    save_cache(cache)
    return item


def pick_pdf_url(item):
    """Return the ``application/pdf`` link URL from a Crossref item, or ``None``."""
    if not item:
        return None
    for link in item.get("link", []):
        if link.get("content-type") == "application/pdf":
            return link.get("URL")
    return None


# --- Unpaywall -------------------------------------------------------------

def get_unpaywall_data(doi, cache):
    """Fetch Unpaywall OA metadata for ``doi`` (cached). ``None`` if not found.

    Unpaywall indexes legally-hosted open-access copies (publisher OA, PubMed
    Central, institutional repositories, preprint servers) that Crossref's link
    list often omits.
    """
    if doi in cache:
        return cache[doi]

    url = f"https://api.unpaywall.org/v2/{doi}"
    data = None
    try:
        resp = requests.get(url, params={"email": MAILTO}, timeout=TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
        elif resp.status_code == 404:
            logger.debug("Unpaywall has no record for %s", doi)
        else:
            logger.warning("Unpaywall returned %d for %s", resp.status_code, doi)
    except (requests.RequestException, ValueError) as e:
        logger.warning("Unpaywall lookup failed for %s: %s", doi, e)

    cache[doi] = data
    save_cache(cache, UNPAYWALL_CACHE)
    return data


def unpaywall_pdf_urls(data):
    """Ordered, de-duplicated OA download URLs from an Unpaywall response.

    Direct PDF links (``url_for_pdf``) come first, best location leading, then
    landing-page URLs as a fallback. Empty when there is no OA copy.
    """
    if not data:
        return []
    locations = []
    best = data.get("best_oa_location")
    if best:
        locations.append(best)
    locations.extend(loc for loc in (data.get("oa_locations") or []) if loc)

    urls = []
    for key in ("url_for_pdf", "url"):
        for loc in locations:
            candidate = loc.get(key)
            if candidate and candidate not in urls:
                urls.append(candidate)
    return urls


# --- download --------------------------------------------------------------

def _looks_like_pdf(resp, first_chunk):
    """True if the response body is actually a PDF (magic bytes or MIME type)."""
    if first_chunk[:5].startswith(b"%PDF"):
        return True
    ctype = getattr(resp, "headers", {}).get("content-type", "").lower()
    return "application/pdf" in ctype


def _looks_like_html(resp, first_chunk):
    ctype = getattr(resp, "headers", {}).get("content-type", "").lower()
    if "html" in ctype:
        return True
    head = first_chunk[:256].lstrip().lower()
    return head.startswith((b"<!doctype", b"<html")) or b"<head" in head


# Publishers embed <meta name="citation_pdf_url" content="..."> so reference
# managers (Zotero, Mendeley, Google Scholar) can locate the full-text PDF.
_META_TAG_RE = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
_CITATION_NAME_RE = re.compile(r"""name\s*=\s*['"]?citation_pdf_url['"]?""", re.IGNORECASE)
_CONTENT_RE = re.compile(r"""content\s*=\s*['"]([^'"]+)['"]""", re.IGNORECASE)


def extract_citation_pdf_url(html, base_url=None):
    """Return the ``citation_pdf_url`` advertised in an HTML page, or ``None``.

    Relative URLs are resolved against ``base_url`` when given.
    """
    for tag in _META_TAG_RE.findall(html):
        if _CITATION_NAME_RE.search(tag):
            content = _CONTENT_RE.search(tag)
            if content:
                url = content.group(1)
                return urljoin(base_url, url) if base_url else url
    return None


def download_pdf(pdf_url, filepath, limiter=None, _depth=0):
    """Stream ``pdf_url`` to ``filepath`` if it is a real PDF.

    Skips when the file already exists. If the URL serves an HTML landing page
    that advertises a ``citation_pdf_url`` meta tag, that link is followed once.
    Returns the path on success (or cache hit), ``None`` otherwise. A ``limiter``
    (``DomainRateLimiter``) spaces out repeat requests to the same host.
    """
    if os.path.exists(filepath):
        logger.debug("Already have %s", filepath)
        return filepath

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    if limiter is not None:
        limiter.wait(pdf_url)

    headers = {"User-Agent": BROWSER_UA, "Accept": "application/pdf,*/*"}
    try:
        resp = requests.get(
            pdf_url, stream=True, headers=headers, timeout=TIMEOUT, allow_redirects=True
        )
        if resp.status_code != 200:
            logger.info("Download returned %d for %s", resp.status_code, pdf_url)
            return None

        chunks = resp.iter_content(8192)
        first = next((c for c in chunks if c), b"")

        if _looks_like_pdf(resp, first):
            with open(filepath, "wb") as f:
                f.write(first)
                for chunk in chunks:
                    if chunk:
                        f.write(chunk)
            return filepath

        # Not a PDF. If it's a landing page, follow its citation_pdf_url once.
        if _depth == 0 and _looks_like_html(resp, first):
            body = bytearray(first)
            for chunk in chunks:
                body.extend(chunk)
                if len(body) >= MAX_HTML_BYTES:
                    break
            target = extract_citation_pdf_url(
                body.decode("utf-8", "ignore"), getattr(resp, "url", pdf_url)
            )
            if target and target != pdf_url:
                logger.debug("Following citation_pdf_url -> %s", target)
                return download_pdf(target, filepath, limiter, _depth=1)

        ctype = getattr(resp, "headers", {}).get("content-type", "?")
        logger.info("Not a PDF (%s): %s", ctype, pdf_url)
        return None
    except requests.RequestException as e:
        logger.warning("Download failed for %s: %s", pdf_url, e)
        if os.path.exists(filepath):
            os.remove(filepath)  # don't leave a truncated file behind
        return None


# --- CSV in / out ----------------------------------------------------------

def read_rows(csv_path=CSV_PATH):
    """Yield reference rows ``{code_number, code_name, paper_name, url}``."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = (row.get("URL") or "").strip()
            if url:
                yield {
                    "code_number": row["Code Number"],
                    "code_name": row["Code Name"],
                    "paper_name": row["Paper Name"],
                    "url": url,
                }


def write_failures(failures, path=FAILURES_CSV):
    """Write the list of references that could not be downloaded."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Code Number", "Code Name", "Paper Name", "URL", "Reason"])
        for r in failures:
            writer.writerow([
                r["code_number"], r["code_name"], r["paper_name"], r["url"], r["reason"],
            ])


# --- orchestration ---------------------------------------------------------

def _candidate_pdf_urls(row, cr_cache, uw_cache):
    """All URLs worth trying for a reference, best first.

    For DOIs: the Crossref ``application/pdf`` link, then every Unpaywall OA
    location. For URL-only rows: the URL itself (it might already be a PDF).
    """
    doi = extract_doi(row["url"])
    if not doi:
        return [row["url"]], doi

    candidates = []
    crossref_pdf = pick_pdf_url(get_crossref_item(doi, cr_cache))
    time.sleep(CROSSREF_PAUSE)
    if crossref_pdf:
        candidates.append(crossref_pdf)

    for url in unpaywall_pdf_urls(get_unpaywall_data(doi, uw_cache)):
        if url not in candidates:
            candidates.append(url)
    time.sleep(UNPAYWALL_PAUSE)

    # Last resort: the original landing page itself. Many OA publishers expose
    # a citation_pdf_url meta tag there even when Crossref/Unpaywall list no PDF.
    if row["url"] not in candidates:
        candidates.append(row["url"])
    return candidates, doi


def _attempt(row, cr_cache, uw_cache, limiter, output_dir):
    """Try to download one reference. Returns ``(path_or_None, reason_or_None)``."""
    filepath = output_path_for(row, output_dir)
    if os.path.exists(filepath):
        return filepath, None

    candidates, doi = _candidate_pdf_urls(row, cr_cache, uw_cache)
    if not candidates:
        return None, "no OA copy found (Crossref/Unpaywall)"

    for url in candidates:
        if download_pdf(url, filepath, limiter):
            return filepath, None
    if doi:
        return None, "no working OA PDF (paywalled or links unreachable)"
    return None, "PDF not retrievable (paywall, non-PDF response, or HTTP error)"


def run(csv_path=CSV_PATH, output_dir=OUTPUT_DIR, min_interval=DOMAIN_INTERVAL):
    """Download every reference's PDF; DOIs first, then the rest.

    Returns a counts dict and writes ``failed_downloads.csv``.
    """
    cr_cache = load_cache(CACHE_FILE)
    uw_cache = load_cache(UNPAYWALL_CACHE)
    limiter = DomainRateLimiter(min_interval)
    rows = list(read_rows(csv_path))

    # DOIs first (the reliable batch), then URL-only landing pages.
    rows.sort(key=lambda r: extract_doi(r["url"]) is None)

    seen = set()
    failures = []
    counts = {"unique": 0, "downloaded": 0, "failed": 0, "duplicates": 0}

    for i, row in enumerate(rows, 1):
        key = extract_doi(row["url"]) or row["url"]
        if key in seen:
            counts["duplicates"] += 1
            continue
        seen.add(key)
        counts["unique"] += 1

        path, reason = _attempt(row, cr_cache, uw_cache, limiter, output_dir)
        if path:
            counts["downloaded"] += 1
        else:
            counts["failed"] += 1
            failures.append({**row, "reason": reason})

        if i % 50 == 0:
            logger.info(
                "Progress %d/%d: %d downloaded, %d failed.",
                i, len(rows), counts["downloaded"], counts["failed"],
            )

    write_failures(failures)
    logger.info(
        "Done. %d unique references (%d duplicate citations skipped): "
        "%d downloaded, %d failed. Failures listed in %s.",
        counts["unique"], counts["duplicates"], counts["downloaded"],
        counts["failed"], FAILURES_CSV,
    )
    return counts


if __name__ == "__main__":
    run()
