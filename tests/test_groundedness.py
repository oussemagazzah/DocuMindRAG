"""Tests for groundedness scoring module."""

from app.retrieval.groundedness import (
    _check_citation_precision,
    _check_hedging,
    compute_groundedness,
)
from app.retrieval.search import SearchResult


def _make_result(page: int = 1, text: str = "test text about revenue") -> SearchResult:
    return SearchResult(
        chunk_id=f"chunk-{page}",
        doc_id="doc1",
        page_number=page,
        chunk_index=0,
        text=text,
        char_start=0,
        char_end=100,
        score=0.8,
        source="reranked",
    )


def test_groundedness_empty_results():
    result = compute_groundedness("answer", [])
    assert result.score == 0.0
    assert result.verdict == "ungrounded"


def test_groundedness_valid_citations():
    results = [_make_result(page=1), _make_result(page=2)]
    answer = "The revenue was $100M [p.1] and growth was 20% [p.2]"
    result = compute_groundedness(answer, results)
    assert result.citation_precision == 1.0


def test_groundedness_invalid_citations():
    results = [_make_result(page=1)]
    answer = "The revenue was $100M [p.99]"
    result = compute_groundedness(answer, results)
    assert result.citation_precision == 0.0


def test_groundedness_mixed_citations():
    results = [_make_result(page=1), _make_result(page=2)]
    answer = "Revenue [p.1] and growth [p.99]"
    result = compute_groundedness(answer, results)
    assert 0.0 < result.citation_precision < 1.0


def test_groundedness_context_coverage():
    results = [_make_result(page=1, text="revenue growth profit margin")]
    answer = "The revenue and profit are strong"
    result = compute_groundedness(answer, results)
    assert result.context_coverage > 0.0


def test_groundedness_hedging():
    results = [_make_result(page=1)]
    answer = "I don't have enough information in the provided documents"
    result = compute_groundedness(answer, results)
    assert result.has_hedging is True


def test_groundedness_no_hedging():
    results = [_make_result(page=1)]
    answer = "The revenue is $100M"
    result = compute_groundedness(answer, results)
    assert result.has_hedging is False


def test_groundedness_verdict_grounded():
    results = [_make_result(page=1, text="revenue growth")]
    answer = "Revenue [p.1] grew 20%"
    result = compute_groundedness(answer, results)
    assert result.verdict in ("grounded", "partially_grounded")


def test_groundedness_score_range():
    results = [_make_result(page=1)]
    answer = "Test [p.1]"
    result = compute_groundedness(answer, results)
    assert 0.0 <= result.score <= 1.0


def test_citation_precision_no_citations():
    results = [_make_result(page=1)]
    answer = "The answer is here"
    prec = _check_citation_precision(answer, results)
    assert prec == 0.5


def test_hedging_phrases():
    assert _check_hedging("I don't have enough information", []) is True
    assert _check_hedging("cannot determine from the text", []) is True
    assert _check_hedging("The answer is clear", []) is False


def test_extract_cited_pages_handles_multi_page_citations():
    from app.retrieval.groundedness import extract_cited_pages

    answer = "Growth is 2.9% [p.6]. Tariffs rose [p.6, p.12] and [p.3; 5]. Also [P. 7]."
    assert extract_cited_pages(answer) == [6, 6, 12, 3, 5, 7]
    assert extract_cited_pages("no citations [1] here") == []


def test_multi_page_citation_counts_for_precision():
    answer = "Revenue grew [p.1, p.2]."
    results = [_make_result(1), _make_result(2)]
    assert _check_citation_precision(answer, results) == 1.0
