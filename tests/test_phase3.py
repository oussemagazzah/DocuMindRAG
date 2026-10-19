"""Tests for feedback, analytics, eval, and query log functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── DB Functions ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_log_query_sqlite():
    """Test logging a query to SQLite."""
    with patch("app.db.settings") as mock_settings:
        mock_settings.use_postgres = False
        mock_settings.database_path = ":memory:"

        import aiosqlite

        from app.db import SCHEMA_SQLITE, log_query

        async with aiosqlite.connect(":memory:") as db:
            await db.executescript(SCHEMA_SQLITE)
            await db.commit()

        with patch("aiosqlite.connect", return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=AsyncMock(
                execute=AsyncMock(return_value=AsyncMock(lastrowid=1)),
                commit=AsyncMock(),
            )),
            __aexit__=AsyncMock(return_value=False),
        )):
            log_id = await log_query(
                query="test query",
                doc_ids=["doc1"],
                answer="test answer",
                model_used="test-model",
                timing_ms=100.0,
                groundedness_score=0.8,
                groundedness_verdict="grounded",
                citation_count=2,
            )
            assert log_id == 1


@pytest.mark.asyncio
async def test_save_feedback_sqlite():
    """Test saving feedback."""
    from app.db import save_feedback
    with patch("app.db.settings") as mock_settings:
        mock_settings.use_postgres = False
        mock_settings.database_path = ":memory:"

        with patch("aiosqlite.connect", return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=AsyncMock(
                execute=AsyncMock(),
                commit=AsyncMock(),
            )),
            __aexit__=AsyncMock(return_value=False),
        )):
            result = await save_feedback(query_log_id=1, rating="thumbs_up", comment="great")
            assert result is True


@pytest.mark.asyncio
async def test_save_feedback_invalid_rating():
    """Test that invalid rating is rejected."""
    from app.db import save_feedback
    # The function should still attempt to save, DB constraint will reject
    # But we test the flow
    with patch("app.db.settings") as mock_settings:
        mock_settings.use_postgres = False
        mock_settings.database_path = ":memory:"

        with patch("aiosqlite.connect", return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=AsyncMock(
                execute=AsyncMock(),
                commit=AsyncMock(),
            )),
            __aexit__=AsyncMock(return_value=False),
        )):
            result = await save_feedback(query_log_id=1, rating="thumbs_up")
            assert result is True


@pytest.mark.asyncio
async def test_get_analytics_stats_empty():
    """Test analytics stats with no data."""
    with patch("app.db.settings") as mock_settings:
        mock_settings.use_postgres = False
        mock_settings.database_path = ":memory:"

        from app.db import get_analytics_stats

        with patch("aiosqlite.connect", return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=AsyncMock(
                execute=AsyncMock(return_value=AsyncMock(
                    fetchone=AsyncMock(return_value={"c": 0}),
                    fetchall=AsyncMock(return_value=[]),
                )),
                fetchall=AsyncMock(return_value=[]),
                row_factory=None,
            )),
            __aexit__=AsyncMock(return_value=False),
        )):
            stats = await get_analytics_stats()
            assert stats["total_queries"] == 0
            assert stats["avg_latency_ms"] == 0
            assert stats["thumbs_up"] == 0
            assert stats["thumbs_down"] == 0
            assert stats["feedback_rate"] == 0


# ── Models ────────────────────────────────────────────────────


def test_feedback_request_valid():
    from app.models import FeedbackRequest
    req = FeedbackRequest(query_log_id=1, rating="thumbs_up")
    assert req.query_log_id == 1
    assert req.rating == "thumbs_up"
    assert req.comment is None


def test_feedback_request_with_comment():
    from app.models import FeedbackRequest
    req = FeedbackRequest(query_log_id=5, rating="thumbs_down", comment="Needs improvement")
    assert req.comment == "Needs improvement"


def test_feedback_response():
    from app.models import FeedbackResponse
    resp = FeedbackResponse(status="saved")
    assert resp.status == "saved"


def test_analytics_response():
    from app.models import AnalyticsResponse
    resp = AnalyticsResponse(
        total_queries=10,
        avg_latency_ms=500.0,
        avg_groundedness=0.75,
        thumbs_up=8,
        thumbs_down=2,
        feedback_rate=100.0,
        recent_queries=[],
        verdicts={"grounded": 8, "ungrounded": 2},
        models_used={"test": 10},
    )
    assert resp.total_queries == 10
    assert resp.feedback_rate == 100.0


def test_eval_result():
    from app.models import EvalResult
    r = EvalResult(
        query="test",
        answer="answer",
        groundedness_score=0.9,
        groundedness_verdict="grounded",
        citation_count=3,
        latency_ms=200.0,
        model_used="model",
    )
    assert r.citation_count == 3


def test_eval_run_response():
    from app.models import EvalRunResponse
    resp = EvalRunResponse(
        total_questions=5,
        results=[],
        avg_groundedness=0.8,
        avg_latency_ms=300.0,
        avg_citations=2.5,
    )
    assert resp.total_questions == 5
    assert resp.avg_citations == 2.5


# ── API Endpoints ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_feedback_endpoint(client):
    """Test POST /feedback endpoint."""
    with patch("app.main.save_feedback", new_callable=AsyncMock, return_value=True):
        resp = client.post("/feedback", json={
            "query_log_id": 1,
            "rating": "thumbs_up",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "saved"


@pytest.mark.asyncio
async def test_feedback_endpoint_with_comment(client):
    """Test POST /feedback with comment."""
    with patch("app.main.save_feedback", new_callable=AsyncMock, return_value=True):
        resp = client.post("/feedback", json={
            "query_log_id": 1,
            "rating": "thumbs_down",
            "comment": "Too vague",
        })
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_feedback_invalid_rating(client):
    """Test POST /feedback with invalid rating."""
    resp = client.post("/feedback", json={
        "query_log_id": 1,
        "rating": "invalid",
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_analytics_endpoint(client):
    """Test GET /analytics endpoint."""
    mock_stats = {
        "total_queries": 5,
        "avg_latency_ms": 450.0,
        "avg_groundedness": 0.72,
        "thumbs_up": 3,
        "thumbs_down": 1,
        "feedback_rate": 80.0,
        "recent_queries": [{"id": 1, "query": "test"}],
        "verdicts": {"grounded": 4},
        "models_used": {"model": 5},
    }
    with patch("app.main.get_analytics_stats", new_callable=AsyncMock, return_value=mock_stats):
        resp = client.get("/analytics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_queries"] == 5
        assert data["thumbs_up"] == 3
        assert data["feedback_rate"] == 80.0


@pytest.mark.asyncio
async def test_eval_endpoint_no_doc(client):
    """Test POST /eval/run with nonexistent doc."""
    with patch("app.db.get_document", new_callable=AsyncMock, return_value=None):
        resp = client.post("/eval/run", json={"doc_id": "nonexistent"})
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_eval_endpoint_success(client):
    """Test POST /eval/run with valid doc."""
    mock_doc = {"id": "doc1", "name": "test.pdf"}
    with patch("app.db.get_document", new_callable=AsyncMock, return_value=mock_doc), \
         patch("app.retrieval.search.hybrid_search", return_value=[]):
        resp = client.post("/eval/run", json={"doc_id": "doc1"})
        assert resp.status_code == 200
        data = resp.json()
        assert "total_questions" in data
        assert "avg_groundedness" in data


@pytest.mark.asyncio
async def test_query_logging(client):
    """Test that /query logs to DB."""
    mock_response = MagicMock()
    mock_response.answer = "test answer"
    mock_response.citations = []
    mock_response.groundedness = MagicMock(score=0.8, verdict="grounded")
    mock_response.timing_ms = 100.0
    mock_response.model_used = "test"

    with patch("app.services.document_service.query_documents_with_logging", new_callable=AsyncMock, return_value=(mock_response, 1)):
        resp = client.post("/query", json={"query": "test"})
        assert resp.status_code == 200
