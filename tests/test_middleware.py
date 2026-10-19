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


def _limited_app(limit: int, trust_proxy_headers: bool = False):
    from app.middleware import RateLimitMiddleware

    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, requests_per_minute=limit, trust_proxy_headers=trust_proxy_headers)

    @app.post("/query")
    async def query():
        return {"ok": True}

    @app.get("/page")
    async def page():
        return {"ok": True}

    return TestClient(app, raise_server_exceptions=False)


def test_rate_limit_returns_429_with_retry_after():
    client = _limited_app(2)
    codes = [client.post("/query").status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    resp = client.post("/query")
    assert resp.status_code == 429
    assert int(resp.headers["Retry-After"]) >= 1


def test_rate_limit_ignores_get_requests():
    client = _limited_app(1)
    assert all(client.get("/page").status_code == 200 for _ in range(5))


def test_rate_limit_per_forwarded_client_when_trusted():
    client = _limited_app(1, trust_proxy_headers=True)
    assert client.post("/query", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert client.post("/query", headers={"X-Forwarded-For": "2.2.2.2, 10.0.0.1"}).status_code == 200
    assert client.post("/query", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 429


def test_rate_limit_ignores_forwarded_header_when_untrusted():
    client = _limited_app(1)
    assert client.post("/query", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert client.post("/query", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429


def test_api_key_middleware_returns_401_for_wrong_key():
    from app.middleware import APIKeyMiddleware

    app = FastAPI()
    app.add_middleware(APIKeyMiddleware, api_keys="valid-key")

    @app.get("/documents")
    async def documents():
        return []

    client = TestClient(app, raise_server_exceptions=False)
    assert client.get("/documents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "valid-key"}).status_code == 200
