"""Answer groundedness and confidence scoring."""

import re
from dataclasses import dataclass

from app.logging_config import get_logger
from app.retrieval.search import SearchResult

logger = get_logger("groundedness")


@dataclass
class GroundednessResult:
    score: float  # 0.0 to 1.0
    citation_precision: float  # % of cited pages that are valid
    context_coverage: float  # % of context chunks referenced
    has_hedging: bool  # whether answer hedges appropriately
    verdict: str  # "grounded", "partially_grounded", "ungrounded"


def compute_groundedness(
    answer: str,
    context_results: list[SearchResult],
) -> GroundednessResult:
    """Compute how well the answer is grounded in the provided context.

    Uses heuristic scoring (no LLM call needed):
    1. Citation precision: are cited page numbers valid?
    2. Context coverage: does the answer reference the provided chunks?
    3. Hedging detection: does the answer appropriately hedge when uncertain?
    """
    if not context_results:
        return GroundednessResult(
            score=0.0,
            citation_precision=0.0,
            context_coverage=0.0,
            has_hedging=False,
            verdict="ungrounded",
        )

    citation_precision = _check_citation_precision(answer, context_results)
    context_coverage = _check_context_coverage(answer, context_results)
    has_hedging = _check_hedging(answer, context_results)

    score = (citation_precision * 0.5) + (context_coverage * 0.3) + (0.2 if has_hedging else 0.0)

    if score >= 0.7:
        verdict = "grounded"
    elif score >= 0.4:
        verdict = "partially_grounded"
    else:
        verdict = "ungrounded"

    logger.info(
        f"Groundedness: score={score:.2f}, citation_prec={citation_precision:.2f}, "
        f"coverage={context_coverage:.2f}, hedging={has_hedging}, verdict={verdict}"
    )

    return GroundednessResult(
        score=round(score, 3),
        citation_precision=round(citation_precision, 3),
        context_coverage=round(context_coverage, 3),
        has_hedging=has_hedging,
        verdict=verdict,
    )


def _check_citation_precision(answer: str, results: list[SearchResult]) -> float:
    """Check what fraction of cited page numbers are valid."""
    cited_pages = set(re.findall(r"\[p\.(\d+)\]", answer))
    if not cited_pages:
        return 0.5

    valid_pages = {str(r.page_number) for r in results}
    valid_citations = cited_pages & valid_pages

    return len(valid_citations) / len(cited_pages) if cited_pages else 0.0


def _check_context_coverage(answer: str, results: list[SearchResult]) -> float:
    """Check what fraction of context chunks are referenced in the answer."""
    answer_lower = answer.lower()
    referenced = 0

    for r in results:
        key_terms = _extract_key_terms(r.text)
        if any(term in answer_lower for term in key_terms):
            referenced += 1

    return referenced / len(results) if results else 0.0


def _check_hedging(answer: str, results: list[SearchResult]) -> bool:
    """Check if the answer appropriately hedges when context is insufficient."""
    hedging_phrases = [
        "i don't have enough information",
        "not enough information",
        "cannot determine",
        "not mentioned",
        "no information",
        "not found in",
        "not available",
        "unclear",
        "insufficient information",
    ]
    return any(phrase in answer.lower() for phrase in hedging_phrases)


def _extract_key_terms(text: str) -> list[str]:
    """Extract significant terms from text for coverage checking."""
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    stopwords = {
        "the", "this", "that", "with", "from", "have", "been", "were", "they",
        "their", "there", "which", "about", "would", "could", "should", "will",
        "into", "also", "than", "when", "what", "some", "more", "other", "only",
        "such", "most", "very", "each", "made", "over", "many", "after", "well",
        "part", "any", "may", "including", "based", "using", "these", "those",
    }
    significant = [w for w in words if w not in stopwords]
    return significant[:10]
