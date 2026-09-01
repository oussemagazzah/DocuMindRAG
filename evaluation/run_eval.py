"""RAG evaluation pipeline for DocuMindRAG.

Runs benchmark questions and computes:
- Citation precision: % of cited pages that are valid
- Answer relevance: keyword overlap with expected answer
- Groundedness: from the groundedness scoring module
- Latency: end-to-end query time
"""

import json
import time
import asyncio
from pathlib import Path
from dataclasses import dataclass, asdict

from app.config import get_settings
from app.logging_config import get_logger, setup_logging

settings = get_settings()
logger = get_logger("evaluation")


@dataclass
class EvalResult:
    question_id: str
    category: str
    question: str
    answer: str
    expected_contains: list[str]
    cited_pages: list[int]
    expected_pages: list[int]
    citation_precision: float
    answer_relevance: float
    groundedness_score: float
    groundedness_verdict: str
    latency_ms: float
    model_used: str
    passed: bool


def _compute_citation_precision(cited_pages: list[int], expected_pages: list[int]) -> float:
    if not expected_pages:
        return 1.0 if not cited_pages else 0.0
    if not cited_pages:
        return 0.0
    valid = sum(1 for p in cited_pages if p in expected_pages)
    return valid / len(cited_pages)


def _compute_answer_relevance(answer: str, expected_terms: list[str]) -> float:
    if not expected_terms:
        return 0.5
    answer_lower = answer.lower()
    matches = sum(1 for term in expected_terms if term.lower() in answer_lower)
    return matches / len(expected_terms)


def _extract_cited_pages(answer: str) -> list[int]:
    import re
    return [int(m) for m in re.findall(r"\[p\.(\d+)\]", answer)]


async def run_evaluation(
    questions_file: str = "evaluation/benchmark.json",
    max_questions: int | None = None,
) -> list[EvalResult]:
    """Run the evaluation benchmark."""
    from app.services.document_service import query_documents
    from app.models import QueryRequest

    with open(questions_file) as f:
        benchmark = json.load(f)

    questions = benchmark["questions"]
    if max_questions:
        questions = questions[:max_questions]

    results = []
    total = len(questions)

    for i, q in enumerate(questions):
        logger.info(f"[{i+1}/{total}] Evaluating: {q['id']} - {q['question'][:60]}...")

        start = time.time()
        try:
            response = await query_documents(QueryRequest(query=q["question"], top_k=5))
            elapsed_ms = (time.time() - start) * 1000

            cited_pages = _extract_cited_pages(response.answer)
            citation_prec = _compute_citation_precision(cited_pages, q.get("expected_pages", []))
            relevance = _compute_answer_relevance(response.answer, q.get("expected_answer_contains", []))

            passed = (
                citation_prec >= 0.5
                and relevance >= 0.3
                and response.groundedness.score >= 0.3
            )

            result = EvalResult(
                question_id=q["id"],
                category=q["category"],
                question=q["question"],
                answer=response.answer,
                expected_contains=q.get("expected_answer_contains", []),
                cited_pages=cited_pages,
                expected_pages=q.get("expected_pages", []),
                citation_precision=round(citation_prec, 3),
                answer_relevance=round(relevance, 3),
                groundedness_score=response.groundedness.score,
                groundedness_verdict=response.groundedness.verdict,
                latency_ms=round(elapsed_ms, 2),
                model_used=response.model_used,
                passed=passed,
            )
        except Exception as e:
            logger.error(f"Question {q['id']} failed: {e}")
            result = EvalResult(
                question_id=q["id"],
                category=q["category"],
                question=q["question"],
                answer=f"ERROR: {e}",
                expected_contains=q.get("expected_answer_contains", []),
                cited_pages=[],
                expected_pages=q.get("expected_pages", []),
                citation_precision=0.0,
                answer_relevance=0.0,
                groundedness_score=0.0,
                groundedness_verdict="error",
                latency_ms=0.0,
                model_used="none",
                passed=False,
            )

        results.append(result)
        logger.info(
            f"  -> passed={result.passed}, citation={result.citation_precision:.2f}, "
            f"relevance={result.answer_relevance:.2f}, groundedness={result.groundedness_score:.2f}"
        )

    return results


def generate_report(results: list[EvalResult]) -> dict:
    """Generate evaluation report with aggregate metrics."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)

    avg_citation = sum(r.citation_precision for r in results) / total if total else 0
    avg_relevance = sum(r.answer_relevance for r in results) / total if total else 0
    avg_groundedness = sum(r.groundedness_score for r in results) / total if total else 0
    avg_latency = sum(r.latency_ms for r in results) / total if total else 0

    by_category = {}
    for r in results:
        cat = r.category
        if cat not in by_category:
            by_category[cat] = {"total": 0, "passed": 0, "avg_latency": 0, "count": 0}
        by_category[cat]["total"] += 1
        if r.passed:
            by_category[cat]["passed"] += 1
        by_category[cat]["avg_latency"] += r.latency_ms
        by_category[cat]["count"] += 1

    for cat in by_category:
        c = by_category[cat]
        c["pass_rate"] = round(c["passed"] / c["total"], 3) if c["total"] else 0
        c["avg_latency"] = round(c["avg_latency"] / c["count"], 2) if c["count"] else 0
        del c["count"]

    report = {
        "summary": {
            "total_questions": total,
            "passed": passed,
            "pass_rate": round(passed / total, 3) if total else 0,
            "avg_citation_precision": round(avg_citation, 3),
            "avg_answer_relevance": round(avg_relevance, 3),
            "avg_groundedness": round(avg_groundedness, 3),
            "avg_latency_ms": round(avg_latency, 2),
        },
        "by_category": by_category,
        "results": [asdict(r) for r in results],
    }

    return report


async def main():
    setup_logging(level="INFO")
    logger.info("Starting evaluation run")

    results = await run_evaluation()
    report = generate_report(results)

    output_path = Path("evaluation/report.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation complete. Report saved to {output_path}")
    logger.info(f"Summary: {json.dumps(report['summary'], indent=2)}")

    return report


if __name__ == "__main__":
    asyncio.run(main())
