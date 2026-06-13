"""Tests for pdf_text — PDF text extraction, section splitting, budget selection.

No real PDFs and no tokenizer network calls: the pypdf reader is monkeypatched
and a fixed chars-per-token ratio keeps token estimates deterministic.
"""

import pdf_text


# --- estimate_tokens -------------------------------------------------------

def test_estimate_tokens_uses_chars_per_token_ratio():
    # 358 chars / 3.58 == 100 tokens
    assert pdf_text.estimate_tokens("x" * 358) == 100


def test_estimate_tokens_empty_is_zero():
    assert pdf_text.estimate_tokens("") == 0


# --- extract_text ----------------------------------------------------------

def test_extract_text_joins_page_text(monkeypatch):
    class FakePage:
        def __init__(self, t):
            self._t = t
        def extract_text(self):
            return self._t

    class FakeReader:
        def __init__(self, path):
            self.pages = [FakePage("page one"), FakePage("page two")]

    monkeypatch.setattr(pdf_text, "PdfReader", FakeReader)
    assert pdf_text.extract_text("whatever.pdf") == "page one\npage two"


def test_extract_text_tolerates_none_pages(monkeypatch):
    class FakePage:
        def extract_text(self):
            return None

    class FakeReader:
        def __init__(self, path):
            self.pages = [FakePage()]

    monkeypatch.setattr(pdf_text, "PdfReader", FakeReader)
    assert pdf_text.extract_text("x.pdf") == ""


# --- split_sections --------------------------------------------------------

SAMPLE = (
    "A Title Line\n"
    "Abstract\n"
    "We show a code exists.\n"
    "Introduction\n"
    "Background here.\n"
    "Methods\n"
    "We did experiments.\n"
    "Results\n"
    "Adaptors were found.\n"
    "Discussion\n"
    "It is arbitrary.\n"
    "References\n"
    "[1] Someone et al.\n"
)


def test_split_sections_finds_known_headings():
    secs = pdf_text.split_sections(SAMPLE)
    assert "abstract" in secs
    assert "We show a code exists." in secs["abstract"]
    assert "introduction" in secs
    assert "references" in secs


def test_split_sections_preamble_captured():
    secs = pdf_text.split_sections(SAMPLE)
    # text before the first recognised heading lives under "_preamble"
    assert "A Title Line" in secs["_preamble"]


def test_split_sections_no_headings_returns_preamble_only():
    secs = pdf_text.split_sections("just a blob of text with no headings")
    assert set(secs) == {"_preamble"}
    assert "blob of text" in secs["_preamble"]


# --- select_for_budget -----------------------------------------------------

def test_select_for_budget_returns_full_text_when_it_fits():
    text = "short text"
    out = pdf_text.select_for_budget(text, max_tokens=1000)
    assert out == text


def test_select_for_budget_drops_references_first_when_over_budget():
    # Build a doc whose body fits the budget but whose references push it over.
    body = "Abstract\n" + ("word " * 400) + "\nReferences\n" + ("ref " * 4000)
    budget = pdf_text.estimate_tokens("Abstract\n" + ("word " * 400)) + 50
    out = pdf_text.select_for_budget(body, max_tokens=budget)
    assert "Abstract" in out
    assert "ref ref" not in out  # references dropped
    assert pdf_text.estimate_tokens(out) <= budget


def test_select_for_budget_hard_truncates_when_still_over():
    text = "word " * 10000  # no headings, far over budget
    out = pdf_text.select_for_budget(text, max_tokens=100)
    assert pdf_text.estimate_tokens(out) <= 100
