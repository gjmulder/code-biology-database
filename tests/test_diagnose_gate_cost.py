"""Offline tests for the strict-gate-cost diagnostic (Phase 4.5 step 3).

The diagnostic re-grounds already-stored ``agreement==0.0`` cells that carry a quote and
counts those the *strict* gate would have rejected but the *fuzzy* gate admits — an upper
bound on the positives the strict gate cost. Only the pure classification logic is tested;
the DB/tokenizer driver is exercised manually (no offline fixtures for those)."""

import diagnose_gate_cost as dg


def test_exact_substring_passes_both_so_not_newly_recovered():
    chunk = "The tRNA adaptor physically bridges codons and amino acids."
    quote = "tRNA adaptor physically bridges codons and amino acids"
    c = dg.classify_cell(quote, chunk)
    assert c["strict"] and c["fuzzy"]
    assert not c["newly_passes"]


def test_smart_quote_and_hyphenation_newly_passes():
    # PDF-extracted source with a line-break hyphenation; quote rendered clean by the model.
    chunk = "The transla-\ntion apparatus uses “adaptor” tRNAs to read the genetic code."
    quote = 'translation apparatus uses "adaptor" tRNAs to read the genetic code'
    c = dg.classify_cell(quote, chunk)
    assert not c["strict"], "smart-quote/hyphenation defeats the strict substring test"
    assert c["fuzzy"], "fuzzy normalisation should ground it"
    assert c["newly_passes"]


def test_paraphrase_fails_both():
    chunk = "The tRNA adaptor physically bridges codons and amino acids."
    quote = "transfer RNA molecules connect the nucleotide triplets to their residues"
    c = dg.classify_cell(quote, chunk)
    assert not c["strict"] and not c["fuzzy"]
    assert not c["newly_passes"]


def test_empty_quote_is_inert():
    c = dg.classify_cell("", "any chunk text here")
    assert not c["strict"] and not c["fuzzy"] and not c["newly_passes"]


def test_tally_counts_newly_passing_cells():
    rows = [
        # (newly passing — fuzzy only)
        dict(quote='uses "adaptor" tRNAs to read the genetic code here today',
             chunk='the source uses “adaptor” tRNAs to read the genetic code here today'),
        # (passes both — not newly recovered)
        dict(quote="codons and amino acids",
             chunk="two worlds of codons and amino acids are linked"),
        # (paraphrase — neither)
        dict(quote="some entirely invented sentence not present at all",
             chunk="unrelated source content about morphology"),
    ]
    tally = dg.tally(
        (r["quote"], r["chunk"]) for r in rows
    )
    assert tally["candidates"] == 3
    assert tally["newly_passes"] == 1
    assert tally["strict_grounded"] == 1
