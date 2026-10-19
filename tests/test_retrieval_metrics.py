"""Unit tests for page-level retrieval metrics."""

import math

import pytest

from app.retrieval.search import SearchResult
from evaluation.retrieval_metrics import (
    mrr_at_k,
    ndcg_at_k,
    question_metrics,
    ranked_pages,
    recall_at_k,
)


def _chunk(doc: str, page: int) -> SearchResult:
    return SearchResult(chunk_id=f"{doc}{page}", doc_id=doc, page_number=page, chunk_index=0,
                        text="", char_start=0, char_end=0, score=0.0, source="t")


def test_ranked_pages_dedupes_keeping_first_rank():
    chunks = [_chunk("a", 3), _chunk("a", 3), _chunk("b", 3), _chunk("a", 1)]
    assert ranked_pages(chunks) == [("a", 3), ("b", 3), ("a", 1)]


def test_page_from_other_document_is_not_relevant():
    groups = [{("a", 3)}]
    assert recall_at_k([("b", 3)], groups, 5) == 0.0


def test_recall_counts_groups_not_pages():
    groups = [{("a", 1), ("a", 2)}, {("a", 7)}]
    # Two restatement pages satisfy one group only.
    assert recall_at_k([("a", 1), ("a", 2)], groups, 5) == 0.5
    assert recall_at_k([("a", 2), ("x", 1), ("a", 7)], groups, 3) == 1.0
    assert recall_at_k([("a", 2), ("x", 1), ("a", 7)], groups, 1) == 0.5


def test_mrr_uses_first_relevant_rank():
    groups = [{("a", 5)}]
    assert mrr_at_k([("a", 1), ("a", 5)], groups) == 0.5
    assert mrr_at_k([("a", 1)], groups) == 0.0


def test_ndcg_perfect_and_single_group():
    groups = [{("a", 1)}, {("a", 2)}]
    assert ndcg_at_k([("a", 1), ("a", 2)], groups) == pytest.approx(1.0)
    # One group at rank 2 out of two groups
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k([("x", 9), ("a", 2)], groups) == pytest.approx(expected)


def test_ndcg_does_not_double_count_restatements():
    groups = [{("a", 1), ("a", 2)}]
    assert ndcg_at_k([("a", 1), ("a", 2)], groups) == pytest.approx(1.0)


def test_question_metrics_shape():
    q = {"doc_id": "a", "evidence_groups": [[4]]}
    m = question_metrics([_chunk("a", 9), _chunk("a", 4)], q)
    assert m["recall@1"] == 0.0 and m["recall@3"] == 1.0
    assert m["mrr@10"] == 0.5
    assert m["top_pages"] == ["a:p9", "a:p4"]


def test_bootstrap_ci_is_seeded_and_brackets_mean():
    from evaluation.retrieval_metrics import bootstrap_ci

    values = [0.0, 1.0] * 20
    lo, hi = bootstrap_ci(values)
    assert lo <= 0.5 <= hi
    assert bootstrap_ci(values) == (lo, hi)
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)
