"""Tests for answer generation and citation extraction."""

from unittest.mock import patch

import pytest

from app.retrieval.answer import SYSTEM_PROMPT, _format_context, extract_citations
from app.retrieval.search import SearchResult


def _make_result(page: int = 1, doc_id: str = "doc1", text: str = "test text") -> SearchResult:
    return SearchResult(
        chunk_id=f"chunk-{page}",
        doc_id=doc_id,
        page_number=page,
        chunk_index=0,
        text=text,
        char_start=0,
        char_end=100,
        score=0.8,
        source="reranked",
    )


def test_extract_citations_deduplicates_by_page():
    results = [
        _make_result(page=1, doc_id="d1", text="text1"),
        _make_result(page=1, doc_id="d1", text="text2"),
        _make_result(page=2, doc_id="d1", text="text3"),
    ]
    citations = extract_citations("answer", results)
    assert len(citations) == 2


def test_extract_citations_preserves_page_number():
    results = [_make_result(page=5)]
    citations = extract_citations("answer", results)
    assert citations[0]["page_number"] == 5


def test_extract_citations_truncates_long_text():
    long_text = "x" * 1000
    results = [_make_result(page=1, text=long_text)]
    citations = extract_citations("answer", results)
    assert len(citations[0]["chunk_text"]) <= 500


def test_format_context():
    results = [_make_result(page=1, text="Hello world")]
    ctx = _format_context(results)
    assert "Hello world" in ctx
    assert "[p.1]" in ctx


def test_format_context_multiple():
    results = [_make_result(page=1, text="First"), _make_result(page=2, text="Second")]
    ctx = _format_context(results)
    assert "[p.1]" in ctx
    assert "[p.2]" in ctx


def test_system_prompt_mentions_citations():
    assert "cite" in SYSTEM_PROMPT.lower()
    assert "[p.X]" in SYSTEM_PROMPT


def test_system_prompt_mentions_hedging():
    assert "don't have enough information" in SYSTEM_PROMPT


def test_extract_citations_empty_results():
    citations = extract_citations("answer", [])
    assert citations == []


def test_extract_citations_score_rounding():
    results = [_make_result(page=1, text="test")]
    results[0].score = 0.123456
    citations = extract_citations("answer", results)
    assert citations[0]["retrieval_score"] == 0.1235


def test_extract_citations_different_documents():
    results = [
        _make_result(page=1, doc_id="d1", text="t1"),
        _make_result(page=1, doc_id="d2", text="t2"),
    ]
    citations = extract_citations("answer", results)
    assert len(citations) == 2


@patch("app.retrieval.answer._generate_with_gemini", side_effect=Exception("API error"))
def test_generate_answer_falls_back_to_ollama(mock_gemini):
    from unittest.mock import patch as p
    with p("app.retrieval.answer._generate_with_ollama", return_value="Ollama answer"):
        from app.retrieval.answer import generate_answer
        answer, model = generate_answer("test", [_make_result()])
        assert answer == "Ollama answer"


@patch("app.retrieval.answer._generate_with_ollama", side_effect=Exception("Ollama down"))
@patch("app.retrieval.answer._generate_with_gemini", side_effect=Exception("Gemini error"))
def test_generate_answer_raises_llm_error_when_all_fail(mock_gem, mock_oll):
    from app.errors import LLMError
    from app.retrieval.answer import generate_answer
    with pytest.raises(LLMError):
        generate_answer("test", [_make_result()])
