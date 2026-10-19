"""Service layer orchestrating ingestion and retrieval pipelines."""

import asyncio
import time
from pathlib import Path

from app.config import get_settings
from app.db import (
    create_document,
    create_document_error,
    create_chunks,
    list_documents,
    get_document,
    delete_document as db_delete_document,
    log_query,
)
from app.errors import (
    DocumentNotFoundError,
    IngestionError,
    LLMError,
    QueryError,
    VectorStoreError,
)
from app.ingestion.pdf_parser import extract_pages
from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import (
    embed_chunks,
    upsert_chunks,
    delete_document_vectors,
    get_qdrant_client,
    ensure_collection,
)
from app.logging_config import get_logger
from app.models import (
    UploadResponse,
    DocumentInfo,
    DeleteResponse,
    QueryRequest,
    QueryResponse,
    Citation,
    GroundednessScore,
    DocumentStatus,
)
from app.retrieval.search import hybrid_search
from app.retrieval.reranker import rerank
from app.retrieval.answer import generate_answer, extract_citations
from app.retrieval.groundedness import compute_groundedness
from app.services.llm_budget import DailyBudget

settings = get_settings()
logger = get_logger("service")
llm_budget = DailyBudget(settings.llm_daily_limit)

RETRIEVAL_ONLY_NOTICE = (
    "No answer was generated: the demo's daily LLM quota is used up or the LLM is unavailable. "
    "The passages below are the top results of hybrid search + cross-encoder reranking for your question."
)


def _sync_ingest(file_bytes: bytes, filename: str) -> dict:
    """Synchronous ingestion pipeline — runs in thread executor."""
    pages = extract_pages(file_bytes)
    if not pages:
        raise IngestionError("PDF contains no extractable text")

    chunks = chunk_pages(pages)
    if not chunks:
        raise IngestionError("Could not generate any chunks from PDF")

    return {"pages": pages, "chunks": chunks}


async def ingest_pdf_async(file_bytes: bytes, filename: str) -> UploadResponse:
    """Async wrapper around the ingestion pipeline."""
    logger.info(f"Starting ingestion: {filename}")

    try:
        result = await asyncio.to_thread(_sync_ingest, file_bytes, filename)
    except IngestionError:
        raise
    except Exception as e:
        logger.error(f"PDF parsing failed: {e}")
        raise IngestionError(f"Failed to parse PDF: {type(e).__name__}")

    pages = result["pages"]
    chunks = result["chunks"]

    doc_id = await create_document(filename, len(pages), len(chunks))

    chunk_dicts = [
        {
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
            "text": c.text,
            "char_start": c.char_start,
            "char_end": c.char_end,
        }
        for c in chunks
    ]
    await create_chunks(doc_id, chunk_dicts)

    try:
        await asyncio.to_thread(_sync_embed_and_upsert, doc_id, chunks)
    except Exception as e:
        logger.error(f"Embedding/upsert failed for {doc_id}: {e}")
        await create_document_error(filename, str(e))
        raise VectorStoreError(str(e))

    logger.info(f"Ingestion complete: {doc_id} ({len(pages)} pages, {len(chunks)} chunks)")
    return UploadResponse(
        doc_id=doc_id,
        status=DocumentStatus.READY,
        name=filename,
        page_count=len(pages),
        chunk_count=len(chunks),
    )


def _sync_embed_and_upsert(doc_id: str, chunks):
    client = get_qdrant_client()
    ensure_collection(client)
    embedded = embed_chunks(chunks)
    upsert_chunks(doc_id, embedded)


async def list_all_documents() -> list[DocumentInfo]:
    docs = await list_documents()
    return [
        DocumentInfo(
            id=d["id"],
            name=d["name"],
            page_count=d["page_count"],
            chunk_count=d["chunk_count"],
            status=d["status"],
            created_at=d["created_at"],
            error_message=d.get("error_message"),
        )
        for d in docs
    ]


async def delete_doc(doc_id: str) -> DeleteResponse:
    doc = await get_document(doc_id)
    if not doc:
        raise DocumentNotFoundError(doc_id)

    try:
        await asyncio.to_thread(delete_document_vectors, doc_id)
    except Exception as e:
        logger.warning(f"Failed to delete vectors for {doc_id}: {e}")

    await db_delete_document(doc_id)
    return DeleteResponse(doc_id=doc_id)


async def _run_retrieval(request: QueryRequest) -> list:
    """Run search + rerank and return reranked results (for streaming)."""
    doc_ids = request.doc_ids if request.doc_ids else None

    try:
        results = await asyncio.to_thread(hybrid_search, request.query, doc_ids)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise QueryError(f"Search failed: {type(e).__name__}")

    if not results:
        raise QueryError("No matching content found in the documents")

    try:
        reranked = await asyncio.to_thread(rerank, request.query, results, request.top_k)
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        reranked = results[: request.top_k]

    return reranked


async def _generate_or_fallback(query: str, reranked: list) -> tuple[str, str, bool]:
    """Generate an answer, or fall back to retrieval-only when the daily budget is spent
    (or, in demo mode, when no LLM is reachable). Returns (text, model_used, generated)."""
    if not llm_budget.try_consume():
        logger.warning("Daily LLM limit reached; returning retrieval-only response")
        return RETRIEVAL_ONLY_NOTICE, "none (daily LLM limit reached)", False
    try:
        answer_text, model_used = await asyncio.to_thread(generate_answer, query, reranked)
        return answer_text, model_used, True
    except LLMError:
        if not settings.demo_mode:
            raise
        logger.error("LLM unavailable in demo mode; returning retrieval-only response")
        return RETRIEVAL_ONLY_NOTICE, "none (LLM unavailable)", False


async def preload_documents(directory: str) -> int:
    """Ingest every PDF in `directory` that is not already indexed. Returns how many were ingested.

    With in-memory Qdrant the vectors vanish on restart while a persisted SQLite file
    may not, so documents recorded in the DB but missing from Qdrant are re-ingested.
    """
    pdfs = sorted(Path(directory).glob("*.pdf"))
    client = get_qdrant_client()
    ensure_collection(client)
    have_vectors = client.count(settings.collection_name).count > 0

    loaded = 0
    for doc in await list_documents():
        if doc["name"] in {p.name for p in pdfs} and not have_vectors:
            await db_delete_document(doc["id"])
    existing = {d["name"] for d in await list_documents() if d["status"] == "ready"}
    for pdf in pdfs:
        if pdf.name in existing:
            continue
        await ingest_pdf_async(pdf.read_bytes(), pdf.name)
        loaded += 1
    logger.info(f"Preloaded {loaded} document(s) from {directory}")
    return loaded


async def query_documents(request: QueryRequest) -> QueryResponse:
    start = time.time()

    reranked = await _run_retrieval(request)
    answer_text, model_used, generated = await _generate_or_fallback(request.query, reranked)

    citations = extract_citations(answer_text, reranked)
    names = {d["id"]: d["name"] for d in await list_documents()}
    for c in citations:
        c["doc_name"] = names.get(c["doc_id"], "")

    if generated:
        groundedness = await asyncio.to_thread(compute_groundedness, answer_text, reranked)
        groundedness_model = GroundednessScore(
            score=groundedness.score,
            citation_precision=groundedness.citation_precision,
            context_coverage=groundedness.context_coverage,
            has_hedging=groundedness.has_hedging,
            verdict=groundedness.verdict,
        )
    else:
        groundedness_model = GroundednessScore(
            score=0.0, citation_precision=0.0, context_coverage=0.0, has_hedging=False, verdict="not_applicable"
        )

    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        f"Query complete: {elapsed_ms:.0f}ms, model={model_used}, citations={len(citations)}, "
        f"groundedness={groundedness.verdict} ({groundedness.score:.2f})"
    )

    return QueryResponse(
        answer=answer_text,
        citations=[Citation(**c) for c in citations],
        groundedness=groundedness_model,
        timing_ms=round(elapsed_ms, 2),
        model_used=model_used,
    )


async def query_documents_with_logging(request: QueryRequest) -> tuple[QueryResponse, int]:
    """Query and log to DB, returning the response and log ID."""
    response = await query_documents(request)
    log_id = await log_query(
        query=request.query,
        doc_ids=request.doc_ids if request.doc_ids else None,
        answer=response.answer,
        model_used=response.model_used,
        timing_ms=response.timing_ms,
        groundedness_score=response.groundedness.score,
        groundedness_verdict=response.groundedness.verdict,
        citation_count=len(response.citations),
    )
    return response, log_id
