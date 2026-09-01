"""Cross-encoder reranking for search results."""

from sentence_transformers import CrossEncoder

from app.config import get_settings
from app.logging_config import get_logger
from app.retrieval.search import SearchResult

settings = get_settings()
logger = get_logger("reranker")

_reranker = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker model: {settings.reranker_model}")
        _reranker = CrossEncoder(settings.reranker_model)
        logger.info("Reranker model loaded")
    return _reranker


def rerank(query: str, results: list[SearchResult], top_k: int | None = None) -> list[SearchResult]:
    if not results:
        return []

    k = top_k or settings.rerank_top_k
    model = _get_reranker()

    pairs = [(query, r.text) for r in results]
    logger.info(f"Reranking {len(pairs)} results")
    scores = model.predict(pairs)

    for result, score in zip(results, scores):
        result.score = float(score)
        result.source = "reranked"

    reranked = sorted(results, key=lambda r: r.score, reverse=True)
    logger.info(f"Reranked top result score: {reranked[0].score:.4f}" if reranked else "No results")
    return reranked[:k]
