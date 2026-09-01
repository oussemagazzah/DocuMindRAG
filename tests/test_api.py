"""Integration tests for API endpoints."""

import io
import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.db import init_db


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    asyncio.get_event_loop().run_until_complete(init_db())


client = TestClient(app, raise_server_exceptions=False)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "qdrant" in data
    assert "version" in data


def test_list_documents_empty():
    resp = client.get("/documents")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_upload_rejects_non_pdf():
    file_content = b"not a pdf"
    resp = client.post(
        "/upload",
        files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
    )
    assert resp.status_code == 422


def test_upload_rejects_empty_file():
    resp = client.post(
        "/upload",
        files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
    )
    assert resp.status_code == 422


def test_upload_rejects_invalid_pdf_header():
    resp = client.post(
        "/upload",
        files={"file": ("fake.pdf", io.BytesIO(b"NOTPDF"), "application/pdf")},
    )
    assert resp.status_code == 422


def test_upload_rejects_oversized_file():
    # Test with content that exceeds limit (51MB)
    # Using a smaller mock to avoid slow memory allocation
    from unittest.mock import patch
    with patch("app.main.settings") as mock_settings:
        mock_settings.max_upload_size_mb = 0  # 0MB limit = reject everything
        mock_settings.gemini_api_key = ""
        resp = client.post(
            "/upload",
            files={"file": ("big.pdf", io.BytesIO(b"%PDF-content"), "application/pdf")},
        )
    assert resp.status_code == 422


def test_query_requires_query_field():
    resp = client.post("/query", json={})
    assert resp.status_code == 422


def test_query_rejects_empty_query():
    resp = client.post("/query", json={"query": ""})
    assert resp.status_code == 422


def test_query_rejects_long_query():
    resp = client.post("/query", json={"query": "x" * 2001})
    assert resp.status_code == 422


def test_delete_nonexistent_document():
    resp = client.delete("/documents/nonexistent-id")
    assert resp.status_code == 404


def test_frontend_served_at_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "DocuMindRAG" in resp.text or "html" in resp.headers.get("content-type", "")


def test_docs_endpoint():
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_upload_rejects_huge_filename():
    long_name = "a" * 500 + ".pdf"
    resp = client.post(
        "/upload",
        files={"file": (long_name, io.BytesIO(b"%PDF-"), "application/pdf")},
    )
    assert resp.status_code == 422


def test_health_response_structure():
    resp = client.get("/health")
    data = resp.json()
    assert isinstance(data["status"], str)
    assert isinstance(data["qdrant"], str)
    assert isinstance(data["version"], str)
