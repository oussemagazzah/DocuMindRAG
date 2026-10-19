"""Tests for semantic chunker module."""

from app.ingestion.pdf_parser import PageContent
from app.ingestion.semantic_chunker import (
    _estimate_tokens,
    _split_into_sentences,
    chunk_pages_semantic,
)


def test_split_into_sentences():
    text = "First sentence. Second sentence. Third sentence."
    sentences = _split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "First sentence."


def test_split_into_sentences_single():
    text = "Just one sentence"
    sentences = _split_into_sentences(text)
    assert len(sentences) == 1


def test_estimate_tokens():
    assert _estimate_tokens("abcd") == 1
    assert _estimate_tokens("abcdefgh") == 2


def test_chunk_pages_semantic_basic():
    pages = [PageContent(page_number=1, text="A" * 2000, char_count=2000)]
    chunks = chunk_pages_semantic(pages)
    assert len(chunks) > 0
    assert all(c.page_number == 1 for c in chunks)


def test_chunk_pages_semantic_empty():
    pages = [PageContent(page_number=1, text="", char_count=0)]
    chunks = chunk_pages_semantic(pages)
    assert len(chunks) == 0


def test_chunk_pages_semantic_multiple_pages():
    pages = [
        PageContent(page_number=1, text="The first page has enough content to create chunks " * 20, char_count=1000),
        PageContent(page_number=2, text="The second page has enough content to create chunks " * 20, char_count=1000),
    ]
    chunks = chunk_pages_semantic(pages)
    assert len(chunks) > 0
    page_nums = {c.page_number for c in chunks}
    assert len(page_nums) >= 1


def test_chunk_pages_semantic_preserves_sentences():
    text = "The quick brown fox jumps over the lazy dog. " * 50
    pages = [PageContent(page_number=1, text=text, char_count=len(text))]
    chunks = chunk_pages_semantic(pages)
    for c in chunks:
        assert c.text.endswith(".") or c.text.endswith(" ") or len(c.text) < 100


def test_chunk_pages_semantic_sequential_indices():
    pages = [PageContent(page_number=1, text="A" * 3000, char_count=3000)]
    chunks = chunk_pages_semantic(pages)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
