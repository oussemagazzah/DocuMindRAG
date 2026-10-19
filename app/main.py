"""FastAPI application for DocuMindRAG."""

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.db import get_analytics_stats, init_db, save_feedback
from app.errors import (
    DocuMindError,
    IngestionError,
    documind_error_handler,
    generic_error_handler,
)
from app.logging_config import REQUEST_ID, get_logger, setup_logging
from app.middleware import APIKeyMiddleware, RateLimitMiddleware
from app.models import (
    AnalyticsResponse,
    DeleteResponse,
    DocumentInfo,
    EvalResult,
    EvalRunRequest,
    EvalRunResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from app.services.document_service import (
    delete_doc,
    ingest_pdf_async,
    list_all_documents,
)

settings = get_settings()
logger = get_logger("main")

STATIC_DIR = Path(__file__).parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(level=settings.log_level, json_output=settings.app_env == "production")
    logger.info("Starting DocuMindRAG", extra={"extra_data": {"env": settings.app_env}})
    await init_db()
    if settings.preload_dir:
        from app.services.document_service import preload_documents

        await preload_documents(settings.preload_dir)
    yield
    logger.info("Shutting down DocuMindRAG")


app = FastAPI(
    title="DocuMindRAG",
    description="Context-aware document intelligence platform with verifiable citations",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=settings.rate_limit_per_minute,
    trust_proxy_headers=settings.trust_proxy_headers,
)
app.add_middleware(APIKeyMiddleware, api_keys=settings.api_keys)

app.add_exception_handler(DocuMindError, documind_error_handler)
app.add_exception_handler(Exception, generic_error_handler)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:12]
    REQUEST_ID.set(request_id)

    start = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start) * 1000

    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} ({elapsed_ms:.0f}ms)",
        extra={
            "extra_data": {
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round(elapsed_ms, 2),
                "request_id": request_id,
            }
        },
    )
    return response


@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    from app.ingestion.embedder import get_qdrant_client
    from app.services.document_service import llm_budget

    qdrant_status = "disconnected"
    try:
        get_qdrant_client().get_collections()
        qdrant_status = "connected"
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {type(e).__name__}")

    status = "ok" if qdrant_status == "connected" else "degraded"
    return HealthResponse(
        status=status,
        qdrant=qdrant_status,
        version=app.version,
        demo_mode=settings.demo_mode,
        llm_calls_left_today=llm_budget.remaining(),
    )


def _reject_in_demo(action: str):
    if settings.demo_mode:
        raise DocuMindError(
            message="Disabled in demo",
            status_code=403,
            detail=f"{action} is disabled in the public demo. Run the app locally to use it.",
        )


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    _reject_in_demo("Uploading documents")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise IngestionError("Only PDF files are supported")

    content = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise IngestionError(
            f"File too large: {len(content) / (1024 * 1024):.1f}MB exceeds {settings.max_upload_size_mb}MB limit"
        )

    if len(content) == 0:
        raise IngestionError("File is empty")

    if content[:5] != b"%PDF-":
        raise IngestionError("Invalid PDF: file does not start with %PDF header")

    logger.info(
        f"Upload received: {file.filename} ({len(content) / 1024:.0f}KB)",
        extra={"extra_data": {"filename": file.filename, "size_bytes": len(content)}},
    )

    return await ingest_pdf_async(content, file.filename)


@app.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    return await list_all_documents()


@app.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str):
    _reject_in_demo("Deleting documents")
    return await delete_doc(doc_id)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    from app.services.document_service import query_documents_with_logging
    response, log_id = await query_documents_with_logging(request)
    return response


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Stream answer tokens as Server-Sent Events."""
    from app.retrieval.streaming import stream_answer
    from app.services.document_service import _run_retrieval

    reranked = await _run_retrieval(request)

    async def event_generator():
        async for chunk in stream_answer(request.query, reranked):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    saved = await save_feedback(request.query_log_id, request.rating, request.comment)
    if not saved:
        from app.errors import IngestionError
        raise IngestionError("Failed to save feedback")
    return FeedbackResponse(status="saved")


@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    stats = await get_analytics_stats()
    return AnalyticsResponse(**stats)


@app.post("/eval/run", response_model=EvalRunResponse)
async def run_evaluation(request: EvalRunRequest):
    _reject_in_demo("Running the evaluation")
    import asyncio

    from app.db import get_document
    from app.errors import DocumentNotFoundError
    from app.retrieval.answer import extract_citations, generate_answer
    from app.retrieval.groundedness import compute_groundedness
    from app.retrieval.reranker import rerank
    from app.retrieval.search import hybrid_search

    doc = await get_document(request.doc_id)
    if not doc:
        raise DocumentNotFoundError(request.doc_id)

    eval_queries = [
        "What is the main topic of this document?",
        "What are the key findings or conclusions?",
        "What data or statistics are presented?",
        "What recommendations are made?",
        "What methodology was used?",
    ]

    results: list[EvalResult] = []
    for q in eval_queries:
        try:
            search_results = await asyncio.to_thread(hybrid_search, q, [request.doc_id])
            if not search_results:
                continue
            reranked = await asyncio.to_thread(rerank, q, search_results, 5)
            answer_text, model_used = await asyncio.to_thread(generate_answer, q, reranked)
            citations = extract_citations(answer_text, reranked)
            groundedness = await asyncio.to_thread(compute_groundedness, answer_text, reranked)
            results.append(EvalResult(
                query=q,
                answer=answer_text,
                groundedness_score=groundedness.score,
                groundedness_verdict=groundedness.verdict,
                citation_count=len(citations),
                latency_ms=0,
                model_used=model_used,
            ))
        except Exception as e:
            logger.warning(f"Eval query failed: {q}: {e}")

    avg_g = sum(r.groundedness_score for r in results) / max(len(results), 1)
    avg_l = sum(r.latency_ms for r in results) / max(len(results), 1)
    avg_c = sum(r.citation_count for r in results) / max(len(results), 1)

    return EvalRunResponse(
        total_questions=len(results),
        results=results,
        avg_groundedness=round(avg_g, 3),
        avg_latency_ms=round(avg_l, 2),
        avg_citations=round(avg_c, 1),
    )
