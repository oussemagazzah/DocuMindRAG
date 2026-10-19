"""Hybrid search combining dense and sparse retrieval with RRF fusion."""

from dataclasses import dataclass
from qdrant_client.models import Filter, FieldCondition, MatchAny

from app.config import get_settings
from app.ingestion.embedder import get_embedding_model, get_qdrant_client, build_sparse_query_vector
from app.logging_config import get_logger

settings = get_settings()
logger = get_logger("search")


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    page_number: int
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    score: float
    source: str


def _build_qdrant_filter(doc_ids: list[str] | None) -> Filter | None:
    if not doc_ids:
        return None
    # A chunk belongs to exactly one document, so match ANY of the requested ids.
    return Filter(must=[FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))])


def _hit_to_result(hit, source: str) -> SearchResult:
    return SearchResult(
        chunk_id=str(hit.id),
        doc_id=hit.payload["doc_id"],
        page_number=hit.payload["page_number"],
        chunk_index=hit.payload["chunk_index"],
        text=hit.payload["text"],
        char_start=hit.payload["char_start"],
        char_end=hit.payload["char_end"],
        score=hit.score,
        source=source,
    )


def dense_search(query: str, top_k: int, doc_ids: list[str] | None = None) -> list[SearchResult]:
    model = get_embedding_model()
    query_embedding = model.encode([query])[0].tolist()

    client = get_qdrant_client()
    query_filter = _build_qdrant_filter(doc_ids)

    results = client.query_points(
        collection_name=settings.collection_name,
        query=query_embedding,
        using="dense",
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    logger.debug(f"Dense search: {len(results.points)} results")
    return [_hit_to_result(r, "dense") for r in results.points]


def sparse_search(query: str, top_k: int, doc_ids: list[str] | None = None) -> list[SearchResult]:
    sparse_vector = build_sparse_query_vector(query)

    client = get_qdrant_client()
    query_filter = _build_qdrant_filter(doc_ids)

    results = client.query_points(
        collection_name=settings.collection_name,
        query=sparse_vector,
        using="sparse",
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    logger.debug(f"Sparse search: {len(results.points)} results")
    return [_hit_to_result(r, "sparse") for r in results.points]


def rrf_fusion(dense_results: list[SearchResult], sparse_results: list[SearchResult], k: int = 60) -> list[SearchResult]:
    chunk_scores: dict[str, float] = {}
    chunk_map: dict[str, SearchResult] = {}

    for rank, result in enumerate(dense_results):
        chunk_scores[result.chunk_id] = chunk_scores.get(result.chunk_id, 0) + 1.0 / (k + rank + 1)
        chunk_map[result.chunk_id] = result

    for rank, result in enumerate(sparse_results):
        chunk_scores[result.chunk_id] = chunk_scores.get(result.chunk_id, 0) + 1.0 / (k + rank + 1)
        if result.chunk_id not in chunk_map:
            chunk_map[result.chunk_id] = result

    sorted_ids = sorted(chunk_scores.keys(), key=lambda cid: chunk_scores[cid], reverse=True)

    return [
        SearchResult(
            chunk_id=cid,
            doc_id=chunk_map[cid].doc_id,
            page_number=chunk_map[cid].page_number,
            chunk_index=chunk_map[cid].chunk_index,
            text=chunk_map[cid].text,
            char_start=chunk_map[cid].char_start,
            char_end=chunk_map[cid].char_end,
            score=chunk_scores[cid],
            source="rrf",
        )
        for cid in sorted_ids
    ]


def hybrid_search(query: str, doc_ids: list[str] | None = None) -> list[SearchResult]:
    logger.info(f"Hybrid search: '{query[:80]}...' (doc_ids={doc_ids})")
    dense_results = dense_search(query, settings.dense_top_k, doc_ids)
    sparse_results = sparse_search(query, settings.sparse_top_k, doc_ids)
    fused = rrf_fusion(dense_results, sparse_results, settings.rrf_k)
    logger.info(f"Hybrid search complete: {len(dense_results)} dense + {len(sparse_results)} sparse = {len(fused)} fused")
    return fused[: settings.dense_top_k]
