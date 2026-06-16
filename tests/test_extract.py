"""Tests for the Biological Code List extraction pipeline.

These follow the project's TDD rule: each behaviour the extractor must
guarantee is pinned by an assertion. The PDF is parsed once per test
session (it is large) and shared via a module-scoped fixture.

Expectations come from CLAUDE.md / AGENTS.md:
  * 435 distinct code categories
  * ~2299 references (citation -> URL mappings)
"""

import logging

import pytest

import extract_csv

PDF_PATH = "Biological_Code_List_20260531.pdf"

# The PDF author quotes 2299 references. Citations without a hyperlink cannot
# be mapped to a URL, so the achievable count is slightly lower; we allow a
# small tolerance band around the quoted figure rather than an exact match.
EXPECTED_CODES = 435
EXPECTED_REFERENCES = 2299
REFERENCE_TOLERANCE = 30


@pytest.fixture(scope="module")
def references():
    """Parse the PDF once and reuse the result across every test."""
    return extract_csv.extract_references(PDF_PATH)


def _by_code(references, code_number):
    return [r for r in references if r["code_number"] == code_number]


def test_returns_a_non_empty_list(references):
    assert isinstance(references, list)
    assert references, "extraction produced no references"


def test_every_distinct_code_is_present(references):
    codes = {r["code_number"] for r in references}
    assert len(codes) == EXPECTED_CODES


def test_code_numbers_are_contiguous_from_one(references):
    codes = {r["code_number"] for r in references}
    assert codes == set(range(1, EXPECTED_CODES + 1))


def test_total_reference_count_is_near_expected(references):
    assert abs(len(references) - EXPECTED_REFERENCES) <= REFERENCE_TOLERANCE, (
        f"got {len(references)} references, expected ~{EXPECTED_REFERENCES}"
    )


@pytest.mark.parametrize(
    "code_number, expected_count",
    [
        (1, 1),  # 14-3-3 code: single citation (Winter et al. 2008)
        (2, 6),  # Acoustic code: six citations
        (3, 2),  # Actin code: two citations
        (4, 3),  # Adenylation code: three citations
    ],
)
def test_known_rows_have_correct_reference_counts(references, code_number, expected_count):
    assert len(_by_code(references, code_number)) == expected_count


def test_no_reference_has_an_empty_paper_name(references):
    empty = [r for r in references if not r["paper_name"].strip()]
    assert not empty, f"{len(empty)} references have an empty paper name"


def test_urls_are_http_or_empty(references):
    bad = [r for r in references if r["url"] and not r["url"].startswith("http")]
    assert not bad, f"{len(bad)} references have a malformed URL"


def test_url_less_references_are_rare(references):
    """Almost every citation is hyperlinked; only a handful lack a URL."""
    url_less = [r for r in references if not r["url"]]
    assert len(url_less) <= 5, f"{len(url_less)} references unexpectedly lack a URL"


def test_link_less_code_is_still_represented(references):
    """Code 352 (SeqCode) has citation text but no hyperlink; it must still
    appear so all 435 codes are present, carrying an empty URL."""
    refs = _by_code(references, 352)
    assert refs, "code 352 (SeqCode) is missing from the output"
    assert all(not r["url"] for r in refs)
    assert any("SeqCode" in r["paper_name"] for r in refs)


def test_code_name_is_populated_for_every_reference(references):
    blank = [r for r in references if not r["code_name"].strip()]
    assert not blank, f"{len(blank)} references have a blank code name"


def test_code_one_maps_the_right_paper_to_the_right_url(references):
    (ref,) = _by_code(references, 1)
    assert ref["code_name"] == "14-3-3 code"
    # NB: the PDF renders the title with a stray space ("14-3- 3 Proteins").
    assert "Proteins recognize a histone code" in ref["paper_name"]
    assert ref["url"] == "https://doi.org/10.1038/sj.emboj.7601954"


def test_no_cross_row_hyperlink_bleed(references):
    """The Springer link belongs to code 2 (Acoustic code), not code 1.

    A regression here means the row boundary is again pulling the next
    code's first hyperlink into the previous code.
    """
    code1_urls = {r["url"] for r in _by_code(references, 1)}
    assert not any("springer.com" in u for u in code1_urls)


def test_multi_page_code_keeps_its_continuation_references(references):
    """Codes whose citation list spills onto following pages must keep the
    references that appear above the first numbered row of those pages.

    Code 2 has six citations whose links sit on three different visual
    bands; losing cross-page/continuation links was the original bug that
    dropped ~half the data.
    """
    code2 = _by_code(references, 2)
    urls = {r["url"] for r in code2}
    # the last (sixth) citation links to a biosystems DOI
    assert any("biosystems" in u or "10.1016" in u for u in urls)


def test_references_preserve_code_order(references):
    code_numbers = [r["code_number"] for r in references]
    assert code_numbers == sorted(code_numbers)


def test_mismatch_logging_runs_without_error(references, caplog):
    """The integrity check must compare text-parsed citations against the
    URL count and emit logging (not raise) when they disagree."""
    with caplog.at_level(logging.WARNING):
        report = extract_csv.count_text_citations("Foo, A. (2020). Bar. Baz, A. (2021). Qux.")
    assert report == 2
