"""Structured error handling for DocuMindRAG."""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.logging_config import get_logger

logger = get_logger("errors")


class ErrorResponse(BaseModel):
    """Standardized error response model."""
    error: str
    detail: str
    status_code: int
    request_id: str = ""


class DocuMindError(Exception):
    """Base exception for DocuMind application errors."""

    def __init__(self, message: str, status_code: int = 500, detail: str = ""):
        self.message = message
        self.status_code = status_code
        self.detail = detail or message
        super().__init__(self.message)


class DocumentNotFoundError(DocuMindError):
    """Raised when a document is not found."""

    def __init__(self, doc_id: str):
        super().__init__(
            message=f"Document not found: {doc_id}",
            status_code=404,
            detail=f"No document exists with ID '{doc_id}'",
        )


class IngestionError(DocuMindError):
    """Raised when PDF ingestion fails."""

    def __init__(self, detail: str):
        super().__init__(
            message="Ingestion failed",
            status_code=422,
            detail=detail,
        )


class QueryError(DocuMindError):
    """Raised when a query fails."""

    def __init__(self, detail: str):
        super().__init__(
            message="Query failed",
            status_code=500,
            detail=detail,
        )


class LLMError(DocuMindError):
    """Raised when all LLM providers fail."""

    def __init__(self):
        super().__init__(
            message="LLM unavailable",
            status_code=503,
            detail="No LLM provider is available. Please configure a valid Gemini API key or ensure Ollama is running.",
        )


class VectorStoreError(DocuMindError):
    """Raised when vector store operations fail."""

    def __init__(self, detail: str):
        super().__init__(
            message="Vector store error",
            status_code=503,
            detail=f"Vector database error: {detail}",
        )


async def documind_error_handler(request: Request, exc: DocuMindError) -> JSONResponse:
    """Global exception handler for DocuMindError subclasses."""
    from app.logging_config import REQUEST_ID

    request_id = REQUEST_ID.get()
    logger.error(
        f"{exc.message} | status={exc.status_code} | request_id={request_id}",
        extra={"extra_data": {"detail": exc.detail, "path": str(request.url)}},
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.message,
            detail=exc.detail,
            status_code=exc.status_code,
            request_id=request_id,
        ).model_dump(),
    )


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler that prevents internal details from leaking."""
    from app.logging_config import REQUEST_ID

    request_id = REQUEST_ID.get()
    logger.error(
        f"Unhandled exception: {type(exc).__name__}: {exc}",
        exc_info=True,
        extra={"extra_data": {"path": str(request.url)}},
    )
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred. Please try again later.",
            status_code=500,
            request_id=request_id,
        ).model_dump(),
    )
