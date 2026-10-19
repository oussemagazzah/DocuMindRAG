"""Tests for chunker module."""

from app.ingestion.chunker import Chunk, _estimate_tokens, chunk_pages
from app.ingestion.pdf_parser import PageContent


def test_estimate_tokens():
    assert _estimate_tokens("abcd") == 1
    assert _estimate_tokens("abcdefgh") == 2
    assert _estimate_tokens("") == 0


def test_chunk_pages_basic():
    pages = [
        PageContent(page_number=1, text="A" * 2000, char_count=2000),
    ]
    chunks = chunk_pages(pages)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.page_number == 1 for c in chunks)


def test_chunk_pages_empty_text():
    pages = [
        PageContent(page_number=1, text="", char_count=0),
    ]
    chunks = chunk_pages(pages)
    assert len(chunks) == 0


def test_chunk_pages_multiple_pages():
    pages = [
        PageContent(page_number=1, text="A" * 1000, char_count=1000),
        PageContent(page_number=2, text="B" * 1000, char_count=1000),
    ]
    chunks = chunk_pages(pages)
    assert len(chunks) > 0
    page_numbers = {c.page_number for c in chunks}
    assert 1 in page_numbers


def test_chunk_pages_sequential_indices():
    pages = [
        PageContent(page_number=1, text="A" * 3000, char_count=3000),
    ]
    chunks = chunk_pages(pages)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i


def test_chunk_pages_preserves_text():
    original_text = "The quick brown fox jumps over the lazy dog. " * 50
    pages = [PageContent(page_number=1, text=original_text, char_count=len(original_text))]
    chunks = chunk_pages(pages)
    combined = " ".join(c.text for c in chunks)
    for word in original_text.split()[:10]:
        assert word in combined


def test_chunk_pages_no_negative_offsets():
    pages = [PageContent(page_number=1, text="A" * 2000, char_count=2000)]
    chunks = chunk_pages(pages)
    for c in chunks:
        assert c.char_start >= 0
        assert c.char_end > c.char_start


def test_chunk_pages_small_text_single_chunk():
    pages = [PageContent(page_number=1, text="Short text", char_count=10)]
    chunks = chunk_pages(pages)
    assert len(chunks) == 1
    assert chunks[0].text == "Short text"


def test_merged_short_pages_keep_their_own_page_number():
    # Page 1 is short, so it is buffered together with page 2; chunks made
    # mostly of page 2's text must be labelled page 2, not page 1.
    pages = [
        PageContent(page_number=1, text="A" * 300, char_count=300),
        PageContent(page_number=2, text="B" * 6000, char_count=6000),
    ]
    chunks = chunk_pages(pages)
    assert {c.page_number for c in chunks} == {2}
    for c in chunks:
        assert c.text.count("B") >= c.text.count("A")


def test_every_page_with_text_is_labelled():
    pages = [PageContent(page_number=i, text=chr(64 + i) * 1500, char_count=1500) for i in range(1, 7)]
    chunks = chunk_pages(pages)
    assert {c.page_number for c in chunks} == set(range(1, 7))
