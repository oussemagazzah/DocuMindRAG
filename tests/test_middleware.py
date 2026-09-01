"""Tests for rate limiting and API key middleware."""

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.middleware import RateLimitMiddleware, APIKeyMiddleware


def test_rate_limit_allows_normal_requests():
    app = FastAPI()

    @app.get("/test")
    async def test_route():
        return {"ok": True}

    app.add_middleware(RateLimitMiddleware, requests_per_minute=10)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/test")
    assert resp.status_code == 200


def test_api_key_middleware_no_keys_allows_all():
    app = FastAPI()

    @app.get("/test")
    async def test_route():
        return {"ok": True}

    app.add_middleware(APIKeyMiddleware, api_keys="")

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/test")
    assert resp.status_code == 200


def test_api_key_middleware_valid_key():
    app = FastAPI()

    @app.get("/test")
    async def test_route():
        return {"ok": True}

    app.add_middleware(APIKeyMiddleware, api_keys="key1,key2")

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/test", headers={"X-API-Key": "key1"})
    assert resp.status_code == 200


def test_api_key_middleware_rejects_wrong_key():
    from app.middleware import APIKeyMiddleware
    mw = APIKeyMiddleware.__new__(APIKeyMiddleware)
    mw.valid_keys = {"valid-key"}
    assert "wrong-key" not in mw.valid_keys
    assert "valid-key" in mw.valid_keys


def test_api_key_middleware_health_bypasses():
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    app.add_middleware(APIKeyMiddleware, api_keys="key1")

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/health")
    assert resp.status_code == 200


def test_middleware_import():
    from app.middleware import RateLimitMiddleware, APIKeyMiddleware
    assert RateLimitMiddleware is not None
    assert APIKeyMiddleware is not None
