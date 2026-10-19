"""Demo mode: daily LLM budget, retrieval-only fallback, read-only endpoints, preloading."""

import asyncio
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.errors import LLMError
from app.services import document_service
from app.services.llm_budget import DailyBudget


def test_budget_unlimited_when_zero():
    b = DailyBudget(0)
    assert all(b.try_consume() for _ in range(100))
    assert b.remaining() is None


def test_budget_caps_and_resets_next_day():
    b = DailyBudget(2)
    assert b.try_consume() and b.try_consume()
    assert not b.try_consume()
    assert b.remaining() == 0
    with patch.object(DailyBudget, "_today", return_value="2999-01-01"):
        assert b.remaining() == 2
        assert b.try_consume()


def test_fallback_when_budget_spent():
    with patch.object(document_service, "llm_budget", DailyBudget(1)) as budget:
        budget.try_consume()
        text, model, generated = asyncio.run(document_service._generate_or_fallback("q", []))
    assert not generated
    assert text == document_service.RETRIEVAL_ONLY_NOTICE
    assert "limit" in model


def test_fallback_on_llm_error_only_in_demo_mode():
    with patch.object(document_service, "generate_answer", side_effect=LLMError()), \
         patch.object(document_service, "llm_budget", DailyBudget(0)):
        with patch.object(document_service.settings, "demo_mode", True):
            _, model, generated = asyncio.run(document_service._generate_or_fallback("q", []))
            assert not generated and "unavailable" in model
        with patch.object(document_service.settings, "demo_mode", False), pytest.raises(LLMError):
            asyncio.run(document_service._generate_or_fallback("q", []))


@pytest.mark.parametrize("method,path,kwargs", [
    ("post", "/upload", {"files": {"file": ("a.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")}}),
    ("delete", "/documents/abc", {}),
    ("post", "/eval/run", {"json": {"doc_id": "abc"}}),
])
def test_write_endpoints_forbidden_in_demo(method, path, kwargs):
    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)
    with patch("app.main.settings.demo_mode", True):
        resp = getattr(client, method)(path, **kwargs)
    assert resp.status_code == 403
    assert "demo" in resp.json()["detail"].lower()


def test_preload_skips_indexed_and_reingests_when_vectors_missing(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF-a")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-b")
    client = MagicMock()

    async def run(have_vectors: bool, docs: list[dict], docs_after: list[dict]):
        client.count.return_value.count = 10 if have_vectors else 0
        with patch.object(document_service, "get_qdrant_client", return_value=client), \
             patch.object(document_service, "ensure_collection"), \
             patch.object(document_service, "list_documents", AsyncMock(side_effect=[docs, docs_after])), \
             patch.object(document_service, "db_delete_document", AsyncMock()) as delete, \
             patch.object(document_service, "ingest_pdf_async", AsyncMock()) as ingest:
            loaded = await document_service.preload_documents(str(tmp_path))
        return loaded, delete, ingest

    docs = [{"id": "1", "name": "a.pdf", "status": "ready"}]
    loaded, delete, ingest = asyncio.run(run(True, docs, docs))
    assert loaded == 1 and delete.await_count == 0
    assert ingest.await_args.args[1] == "b.pdf"

    # Vectors gone (in-memory Qdrant restarted) but DB row kept: stale row is dropped.
    loaded, delete, ingest = asyncio.run(run(False, docs, []))
    assert delete.await_count == 1
    assert loaded == 2


def test_query_documents_returns_passages_when_llm_unavailable():
    from app.models import QueryRequest
    from app.retrieval.search import SearchResult

    hit = SearchResult(chunk_id="c1", doc_id="d1", page_number=6, chunk_index=0, text="Growth 2.9% in 2026",
                       char_start=0, char_end=10, score=0.9, source="reranked")
    with patch.object(document_service, "_run_retrieval", AsyncMock(return_value=[hit])), \
         patch.object(document_service, "list_documents", AsyncMock(return_value=[{"id": "d1", "name": "oecd.pdf"}])), \
         patch.object(document_service, "generate_answer", side_effect=LLMError()), \
         patch.object(document_service, "llm_budget", DailyBudget(0)), \
         patch.object(document_service.settings, "demo_mode", True):
        resp = asyncio.run(document_service.query_documents(QueryRequest(query="growth?")))
    assert resp.answer == document_service.RETRIEVAL_ONLY_NOTICE
    assert resp.groundedness.verdict == "not_applicable"
    assert resp.citations[0].doc_name == "oecd.pdf"
    assert resp.citations[0].page_number == 6
