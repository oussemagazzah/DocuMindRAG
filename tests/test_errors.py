"""Tests for error handling module."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.errors import (
    DocuMindError,
    DocumentNotFoundError,
    IngestionError,
    QueryError,
    LLMError,
    VectorStoreError,
    ErrorResponse,
)


def test_documind_error_base():
    e = DocuMindError("test error", status_code=400, detail="details")
    assert e.message == "test error"
    assert e.status_code == 400
    assert e.detail == "details"


def test_document_not_found_error():
    e = DocumentNotFoundError("abc-123")
    assert e.status_code == 404
    assert "abc-123" in e.detail


def test_ingestion_error():
    e = IngestionError("bad pdf")
    assert e.status_code == 422
    assert "bad pdf" in e.detail


def test_query_error():
    e = QueryError("search failed")
    assert e.status_code == 500


def test_llm_error():
    e = LLMError()
    assert e.status_code == 503
    assert "Gemini" in e.detail or "Ollama" in e.detail


def test_vector_store_error():
    e = VectorStoreError("connection refused")
    assert e.status_code == 503


def test_error_response_model():
    r = ErrorResponse(
        error="test",
        detail="detail",
        status_code=500,
        request_id="abc",
    )
    assert r.request_id == "abc"


def test_error_handler_returns_json():
    from fastapi.responses import JSONResponse

    test_app = FastAPI()

    async def error_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message, "detail": exc.detail},
        )

    test_app.add_exception_handler(DocuMindError, error_handler)

    @test_app.get("/test")
    async def trigger_error():
        raise IngestionError("test error")

    client = TestClient(test_app, raise_server_exceptions=False)
    resp = client.get("/test")
    assert resp.status_code == 422
    data = resp.json()
    assert "error" in data
    assert "detail" in data
