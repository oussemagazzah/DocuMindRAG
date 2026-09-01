"""Service layer orchestrating ingestion and retrieval pipelines."""

import asyncio
import time

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

logger = get_logger("service")


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


async def query_documents(request: QueryRequest) -> QueryResponse:
    start = time.time()

    reranked = await _run_retrieval(request)

    try:
        answer_text, model_used = await asyncio.to_thread(generate_answer, request.query, reranked)
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        raise

    citations = extract_citations(answer_text, reranked)

    groundedness = await asyncio.to_thread(compute_groundedness, answer_text, reranked)
    groundedness_model = GroundednessScore(
        score=groundedness.score,
        citation_precision=groundedness.citation_precision,
        context_coverage=groundedness.context_coverage,
        has_hedging=groundedness.has_hedging,
        verdict=groundedness.verdict,
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
