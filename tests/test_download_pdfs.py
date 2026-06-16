"""Tests for the DOI -> PDF downloader.

Network access is never exercised here: Crossref lookups and HTTP downloads
are monkeypatched so the suite is deterministic and offline. The pure helpers
(DOI extraction, path naming, PDF-link selection) are tested directly.
"""

import csv
import os

import pytest

import download_pdfs as dl


# --- extract_doi -----------------------------------------------------------

def test_extract_doi_from_doi_org_url():
    assert (
        dl.extract_doi("https://doi.org/10.1038/sj.emboj.7601954")
        == "10.1038/sj.emboj.7601954"
    )


def test_extract_doi_from_url_with_embedded_doi():
    assert (
        dl.extract_doi("https://link.springer.com/article/10.1007/s00018-008-8027-0")
        == "10.1007/s00018-008-8027-0"
    )


def test_extract_doi_strips_trailing_punctuation_and_query():
    assert (
        dl.extract_doi("https://doi.org/10.1016/j.biosystems.2009.01.001?foo=bar")
        == "10.1016/j.biosystems.2009.01.001"
    )


@pytest.mark.parametrize("url", ["", "https://www.nature.com/articles/nature12373", "not a url"])
def test_extract_doi_returns_none_when_absent(url):
    assert dl.extract_doi(url) is None


# --- pdf_path_for ----------------------------------------------------------

def test_pdf_path_is_filesystem_safe():
    assert dl.pdf_path_for("10.1038/sj.emboj.7601954", "pdfs") == os.path.join(
        "pdfs", "10.1038_sj.emboj.7601954.pdf"
    )


# --- pick_pdf_url ----------------------------------------------------------

def test_pick_pdf_url_prefers_application_pdf():
    item = {
        "link": [
            {"content-type": "text/html", "URL": "html"},
            {"content-type": "application/pdf", "URL": "pdf"},
        ]
    }
    assert dl.pick_pdf_url(item) == "pdf"


@pytest.mark.parametrize("item", [{}, {"link": [{"content-type": "text/html", "URL": "h"}]}])
def test_pick_pdf_url_none_when_no_pdf(item):
    assert dl.pick_pdf_url(item) is None


# --- download_pdf ----------------------------------------------------------

def test_download_pdf_skips_when_file_already_exists(tmp_path, monkeypatch):
    target = tmp_path / "10.1_x.pdf"
    target.write_bytes(b"%PDF-1.4 already here")

    def boom(*a, **k):  # network must not be touched
        raise AssertionError("requests.get should not be called for a cached PDF")

    monkeypatch.setattr(dl.requests, "get", boom)
    path = dl.download_pdf("http://example/x.pdf", str(target))
    assert path == str(target)


def test_download_pdf_writes_streamed_content(tmp_path, monkeypatch):
    target = tmp_path / "out.pdf"

    class FakeResp:
        status_code = 200

        def iter_content(self, n):
            yield b"%PDF-1.4 "
            yield b"body"

    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: FakeResp())
    path = dl.download_pdf("http://example/x.pdf", str(target))
    assert path == str(target)
    assert target.read_bytes() == b"%PDF-1.4 body"


def test_download_pdf_returns_none_on_http_error(tmp_path, monkeypatch):
    target = tmp_path / "out.pdf"

    class FakeResp:
        status_code = 403

        def iter_content(self, n):
            return iter(())

    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: FakeResp())
    assert dl.download_pdf("http://example/x.pdf", str(target)) is None
    assert not target.exists()


def test_download_pdf_rejects_non_pdf_content(tmp_path, monkeypatch):
    """A 200 that returns HTML (a paywall/landing page) must not be saved."""
    target = tmp_path / "out.pdf"

    class FakeResp:
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8"}

        def iter_content(self, n):
            yield b"<!DOCTYPE html><html>paywall</html>"

    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: FakeResp())
    assert dl.download_pdf("http://example/landing", str(target)) is None
    assert not target.exists()


# --- citation_pdf_url meta tag ---------------------------------------------

def test_extract_citation_pdf_url_from_meta():
    html = '<head><meta name="citation_pdf_url" content="https://x.org/y.pdf"></head>'
    assert dl.extract_citation_pdf_url(html) == "https://x.org/y.pdf"


def test_extract_citation_pdf_url_handles_swapped_attr_order():
    html = "<meta content='https://x.org/z.pdf' name='citation_pdf_url' />"
    assert dl.extract_citation_pdf_url(html) == "https://x.org/z.pdf"


def test_extract_citation_pdf_url_resolves_relative_against_base():
    html = '<meta name="citation_pdf_url" content="/articles/a.pdf">'
    assert (
        dl.extract_citation_pdf_url(html, "https://pub.org/x/y")
        == "https://pub.org/articles/a.pdf"
    )


def test_extract_citation_pdf_url_none_when_absent():
    assert dl.extract_citation_pdf_url("<html><body>no meta here</body></html>") is None


def test_download_pdf_follows_citation_pdf_url_when_landing_is_html(tmp_path, monkeypatch):
    """An HTML landing page advertising a citation_pdf_url is followed once."""
    target = tmp_path / "out.pdf"
    landing = "https://pub.org/article/1"
    pdf = "https://pub.org/article/1.pdf"

    class HtmlResp:
        status_code = 200
        headers = {"content-type": "text/html"}
        url = landing

        def iter_content(self, n):
            yield (
                b'<html><head><meta name="citation_pdf_url" '
                b'content="https://pub.org/article/1.pdf"></head></html>'
            )

    class PdfResp:
        status_code = 200
        headers = {"content-type": "application/pdf"}
        url = pdf

        def iter_content(self, n):
            yield b"%PDF-1.5 real"

    def fake_get(url, **kwargs):
        return PdfResp() if url == pdf else HtmlResp()

    monkeypatch.setattr(dl.requests, "get", fake_get)
    assert dl.download_pdf(landing, str(target)) == str(target)
    assert target.read_bytes().startswith(b"%PDF")


# --- url_slug --------------------------------------------------------------

def test_url_slug_is_filesystem_safe():
    slug = dl.url_slug("https://www.nature.com/articles/nature12373")
    assert slug == "www.nature.com_articles_nature12373"
    assert "/" not in slug


def test_output_path_prefers_doi_then_falls_back_to_url(tmp_path):
    doi_row = {"url": "https://doi.org/10.1038/x", "code_number": "1"}
    url_row = {"url": "https://www.nature.com/articles/nature12373", "code_number": "2"}
    assert dl.output_path_for(doi_row, "pdfs") == os.path.join("pdfs", "10.1038_x.pdf")
    assert dl.output_path_for(url_row, "pdfs").startswith(os.path.join("pdfs", "www.nature.com"))


# --- DomainRateLimiter -----------------------------------------------------

def test_rate_limiter_delays_only_repeat_domains(monkeypatch):
    clock = [0.0]
    slept = []
    monkeypatch.setattr(dl.time, "monotonic", lambda: clock[0])

    def fake_sleep(seconds):
        slept.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(dl.time, "sleep", fake_sleep)

    rl = dl.DomainRateLimiter(min_interval=2.0)
    rl.wait("https://a.com/one")   # first hit on a.com -> no wait
    rl.wait("https://b.com/two")   # different domain -> no wait
    rl.wait("https://a.com/three")  # a.com again, immediately -> wait 2s

    assert len(slept) == 1
    assert slept[0] == pytest.approx(2.0)


# --- unpaywall_pdf_urls ----------------------------------------------------

def test_unpaywall_pdf_urls_lists_pdf_links_best_first():
    data = {
        "best_oa_location": {"url_for_pdf": "best.pdf", "url": "best"},
        "oa_locations": [
            {"url_for_pdf": "best.pdf", "url": "best"},
            {"url_for_pdf": "repo.pdf", "url": "repo"},
        ],
    }
    urls = dl.unpaywall_pdf_urls(data)
    assert urls[0] == "best.pdf"
    assert "repo.pdf" in urls
    assert urls.count("best.pdf") == 1  # de-duplicated


def test_unpaywall_pdf_urls_falls_back_to_landing_url():
    data = {"best_oa_location": {"url": "landing"}, "oa_locations": [{"url": "landing"}]}
    assert dl.unpaywall_pdf_urls(data) == ["landing"]


@pytest.mark.parametrize("data", [None, {}, {"best_oa_location": None, "oa_locations": []}])
def test_unpaywall_pdf_urls_empty_when_no_oa(data):
    assert dl.unpaywall_pdf_urls(data) == []


def test_get_unpaywall_data_is_cached(monkeypatch):
    calls = []

    class FakeResp:
        status_code = 200

        def json(self):
            return {"doi": "10.1/x", "is_oa": True}

    def fake_get(url, **kwargs):
        calls.append(url)
        return FakeResp()

    monkeypatch.setattr(dl.requests, "get", fake_get)
    monkeypatch.setattr(dl, "save_cache", lambda *a, **k: None)  # no disk writes
    cache = {}
    a = dl.get_unpaywall_data("10.1/x", cache)
    b = dl.get_unpaywall_data("10.1/x", cache)
    assert a == b == {"doi": "10.1/x", "is_oa": True}
    assert len(calls) == 1  # second call served from cache


def test_candidates_fall_back_to_landing_url_for_doi_rows(monkeypatch):
    """Even when Crossref and Unpaywall list no PDF, the original landing URL
    must remain a candidate so its citation_pdf_url meta tag can be followed."""
    monkeypatch.setattr(dl, "get_crossref_item", lambda doi, cache: None)
    monkeypatch.setattr(dl, "get_unpaywall_data", lambda doi, cache: None)
    monkeypatch.setattr(dl.time, "sleep", lambda *a, **k: None)
    row = {"url": "https://www.frontiersin.org/articles/10.3389/x/full"}
    candidates, doi = dl._candidate_pdf_urls(row, {}, {})
    assert doi == "10.3389/x/full"
    assert candidates == ["https://www.frontiersin.org/articles/10.3389/x/full"]


# --- write_failures --------------------------------------------------------

def test_write_failures_writes_csv_with_reason(tmp_path):
    out = tmp_path / "failed.csv"
    dl.write_failures(
        [
            {
                "code_number": "3",
                "code_name": "Actin code",
                "paper_name": "Some paper",
                "url": "https://example/x",
                "reason": "paywall",
            }
        ],
        str(out),
    )
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert rows[0]["Reason"] == "paywall"
    assert rows[0]["Code Number"] == "3"
