"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from app.models import (
    QueryRequest,
    QueryResponse,
    UploadResponse,
    DocumentInfo,
    DeleteResponse,
    HealthResponse,
    Citation,
    DocumentStatus,
)


def test_query_request_valid():
    r = QueryRequest(query="What is revenue?")
    assert r.query == "What is revenue?"
    assert r.top_k == 5
    assert r.doc_ids == []


def test_query_request_custom_top_k():
    r = QueryRequest(query="test", top_k=10)
    assert r.top_k == 10


def test_query_request_rejects_empty_query():
    with pytest.raises(ValidationError):
        QueryRequest(query="")


def test_query_request_rejects_long_query():
    with pytest.raises(ValidationError):
        QueryRequest(query="x" * 2001)


def test_query_request_top_k_bounds():
    with pytest.raises(ValidationError):
        QueryRequest(query="test", top_k=0)
    with pytest.raises(ValidationError):
        QueryRequest(query="test", top_k=21)


def test_upload_response():
    r = UploadResponse(
        doc_id="abc",
        status=DocumentStatus.READY,
        name="test.pdf",
        page_count=5,
        chunk_count=10,
    )
    assert r.status == DocumentStatus.READY
    assert r.page_count == 5


def test_document_info():
    d = DocumentInfo(
        id="x",
        name="doc.pdf",
        page_count=1,
        chunk_count=2,
        status=DocumentStatus.READY,
        created_at="2025-01-01T00:00:00",
    )
    assert d.error_message is None


def test_document_info_with_error():
    d = DocumentInfo(
        id="x",
        name="doc.pdf",
        page_count=0,
        chunk_count=0,
        status=DocumentStatus.ERROR,
        created_at="2025-01-01T00:00:00",
        error_message="Parse failed",
    )
    assert d.error_message == "Parse failed"


def test_delete_response():
    r = DeleteResponse(doc_id="abc")
    assert r.status == "deleted"
    assert r.doc_id == "abc"


def test_health_response():
    h = HealthResponse()
    assert h.status == "ok"
    assert h.qdrant == "connected"


def test_citation_model():
    c = Citation(
        chunk_id="c1",
        doc_id="d1",
        doc_name="test.pdf",
        page_number=1,
        chunk_text="hello",
        char_start=0,
        char_end=5,
        retrieval_score=0.9,
    )
    assert c.retrieval_score == 0.9


def test_document_status_enum():
    assert DocumentStatus.PROCESSING == "processing"
    assert DocumentStatus.READY == "ready"
    assert DocumentStatus.ERROR == "error"
