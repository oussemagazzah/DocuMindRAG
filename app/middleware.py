"""Rate limiting and API key authentication middleware."""

import time
from collections import defaultdict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings
from app.logging_config import get_logger

settings = get_settings()
logger = get_logger("middleware")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory sliding-window rate limiter for state-changing API calls.

    Only POST/PUT/PATCH/DELETE requests count (queries, uploads, feedback); page
    loads and static files do not. Behind a reverse proxy (e.g. Hugging Face
    Spaces) every request comes from the proxy's address, so the client is taken
    from X-Forwarded-For, but only when trust_proxy_headers is enabled, because
    the header is client-controlled otherwise.
    """

    LIMITED_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

    def __init__(self, app, requests_per_minute: int = 30, trust_proxy_headers: bool = False):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.trust_proxy_headers = trust_proxy_headers
        self.requests: dict[str, list[float]] = defaultdict(list)
        self._last_sweep = 0.0

    def _client_key(self, request: Request) -> str:
        if self.trust_proxy_headers:
            forwarded = request.headers.get("x-forwarded-for", "")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _sweep(self, window_start: float, now: float):
        # Drop clients with no recent requests so memory does not grow forever.
        if now - self._last_sweep < 60:
            return
        self._last_sweep = now
        for key in [k for k, ts in self.requests.items() if not ts or ts[-1] <= window_start]:
            del self.requests[key]

    async def dispatch(self, request: Request, call_next):
        if request.method not in self.LIMITED_METHODS:
            return await call_next(request)

        client = self._client_key(request)
        now = time.time()
        window_start = now - 60
        self._sweep(window_start, now)

        recent = [t for t in self.requests[client] if t > window_start]
        if len(recent) >= self.requests_per_minute:
            retry_after = max(1, int(recent[0] - window_start) + 1)
            logger.warning(f"Rate limit exceeded for {client}")
            # Return the response directly: exceptions raised inside BaseHTTPMiddleware
            # bypass FastAPI's handlers and reach the client as a 500.
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please wait before making another request."},
                headers={"Retry-After": str(retry_after)},
            )

        recent.append(now)
        self.requests[client] = recent
        return await call_next(request)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """API key authentication middleware.

    Checks for API key in X-API-Key header.
    If no API keys are configured, all requests are allowed.
    """

    def __init__(self, app, api_keys: str = ""):
        super().__init__(app)
        self.valid_keys = {
            k.strip() for k in api_keys.split(",") if k.strip()
        }

    async def dispatch(self, request: Request, call_next):
        if not self.valid_keys:
            return await call_next(request)

        if request.url.path in ("/health", "/docs", "/openapi.json", "/"):
            return await call_next(request)

        api_key = request.headers.get(settings.api_key_header, "")
        if api_key not in self.valid_keys:
            logger.warning(f"Invalid API key from {request.client.host if request.client else 'unknown'}")
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})

        return await call_next(request)
