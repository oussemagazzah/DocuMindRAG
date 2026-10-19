"""Verify evaluation/benchmark.json against the corpus PDFs.

Checks that:
- metadata.total_questions and category counts match the questions;
- every expected fact (at least one variant) appears on one of its gold pages;
- every gold page exists in its document;
- unanswerable questions have no gold pages and their absent_terms appear
  nowhere in the corpus.

Usage: python -m evaluation.check_benchmark
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

from app.ingestion.pdf_parser import extract_pages
from evaluation.facts import fact_in, variant_in

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = ROOT / "evaluation" / "benchmark.json"
CATEGORIES = {"fact_retrieval", "multi_hop", "comparison", "unanswerable"}


def load_benchmark(path: Path = BENCHMARK) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache
def load_pages(pdf_path: str) -> dict[int, str]:
    return {p.page_number: p.text for p in extract_pages((ROOT / pdf_path).read_bytes())}


def check(benchmark: dict) -> list[str]:
    errors: list[str] = []
    meta = benchmark["metadata"]
    questions = benchmark["questions"]

    if meta["total_questions"] != len(questions):
        errors.append(f"metadata.total_questions={meta['total_questions']} but {len(questions)} questions exist")
    for cat, info in benchmark["categories"].items():
        actual = sum(1 for q in questions if q["category"] == cat)
        if info["count"] != actual:
            errors.append(f"categories.{cat}.count={info['count']} but {actual} questions exist")

    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        errors.append("duplicate question ids")

    docs = {doc_id: load_pages(path) for doc_id, path in meta["documents"].items()}
    corpus_text = " ".join(t for pages in docs.values() for t in pages.values())

    for q in questions:
        qid = q["id"]
        if q["category"] not in CATEGORIES:
            errors.append(f"{qid}: unknown category {q['category']}")
        if q["doc_id"] not in docs:
            errors.append(f"{qid}: unknown doc_id {q['doc_id']}")
            continue
        pages = docs[q["doc_id"]]
        group_pages = sorted({p for g in q["evidence_groups"] for p in g})
        if group_pages != sorted(q["expected_pages"]):
            errors.append(f"{qid}: expected_pages {q['expected_pages']} != union of evidence_groups {group_pages}")

        if q["category"] == "unanswerable":
            if q["expected_pages"] or q["expected_answer_contains"]:
                errors.append(f"{qid}: unanswerable question has gold pages or facts")
            if not q.get("absent_terms"):
                errors.append(f"{qid}: unanswerable question has no absent_terms")
            for term in q.get("absent_terms", []):
                if variant_in(term, corpus_text):
                    errors.append(f"{qid}: absent term {term!r} appears in the corpus")
            continue

        missing_pages = [p for p in q["expected_pages"] if p not in pages]
        if missing_pages:
            errors.append(f"{qid}: pages {missing_pages} not in {q['doc_id']}")
            continue
        gold_text = " ".join(pages[p] for p in q["expected_pages"])
        for fact in q["expected_answer_contains"]:
            if not fact_in(fact, gold_text):
                errors.append(f"{qid}: fact {fact!r} not found on pages {q['expected_pages']}")
        for group in q["evidence_groups"]:
            group_text = " ".join(pages[p] for p in group)
            if not any(fact_in(f, group_text) for f in q["expected_answer_contains"]):
                errors.append(f"{qid}: evidence group {group} contains none of the expected facts")

    return errors


def main() -> int:
    errors = check(load_benchmark())
    for e in errors:
        print("ERROR", e)
    if not errors:
        b = load_benchmark()
        print(f"OK: {b['metadata']['total_questions']} questions verified against the corpus")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
