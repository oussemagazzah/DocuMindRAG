"""Tests for PDF parser module."""

import pytest
from app.ingestion.pdf_parser import extract_pages, get_pdf_metadata, PageContent


def _make_simple_pdf() -> bytes:
    """Create a minimal valid PDF in memory."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello World - Test Document")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Page 2 content here")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_extract_pages_returns_list():
    pdf_bytes = _make_simple_pdf()
    pages = extract_pages(pdf_bytes)
    assert isinstance(pages, list)
    assert len(pages) == 2


def test_extract_pages_page_numbers():
    pdf_bytes = _make_simple_pdf()
    pages = extract_pages(pdf_bytes)
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2


def test_extract_pages_has_text():
    pdf_bytes = _make_simple_pdf()
    pages = extract_pages(pdf_bytes)
    assert "Hello World" in pages[0].text


def test_extract_pages_char_count():
    pdf_bytes = _make_simple_pdf()
    pages = extract_pages(pdf_bytes)
    for p in pages:
        assert p.char_count == len(p.text)


def test_get_pdf_metadata():
    pdf_bytes = _make_simple_pdf()
    meta = get_pdf_metadata(pdf_bytes)
    assert meta["page_count"] == 2
    assert isinstance(meta["title"], str)


def test_extract_pages_page_content_dataclass():
    pdf_bytes = _make_simple_pdf()
    pages = extract_pages(pdf_bytes)
    for p in pages:
        assert isinstance(p, PageContent)


def test_extract_pages_empty_pdf():
    import fitz

    doc = fitz.open()
    doc.new_page()
    page = doc[0]
    pdf_bytes = doc.tobytes()
    doc.close()
    pages = extract_pages(pdf_bytes)
    assert len(pages) >= 1
    assert all(p.text.strip() == "" for p in pages)
