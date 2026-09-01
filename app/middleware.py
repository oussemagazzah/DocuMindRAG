"""Rate limiting and API key authentication middleware."""

import time
from collections import defaultdict

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings
from app.logging_config import get_logger

settings = get_settings()
logger = get_logger("middleware")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter using sliding window."""

    def __init__(self, app, requests_per_minute: int = 30):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - 60

        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if t > window_start
        ]

        if len(self.requests[client_ip]) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please wait before making another request.",
            )

        self.requests[client_ip].append(now)
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
            logger.warning(f"Invalid API key from {request.client.host}")
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key",
            )

        return await call_next(request)
