"""Tests for end-to-end answer metrics, judge parsing and manual-label parsing (no LLM calls)."""

import pytest

from evaluation.answer_metrics import aggregate, citation_precision, heuristic_vs_judge, is_refusal, spearman
from evaluation.judge import parse_judgement
from evaluation.manual_agreement import agreement, parse_labels


@pytest.mark.parametrize("answer", [
    "I don't have enough information in the provided documents to answer this question.",
    "I don’t have enough information in the provided documents to answer this question.",
    "The documents do not contain information about Nigeria.",
    "The report does not provide a projection for 2027.",
    "A 2027 figure is not mentioned in the context.",
    "This cannot be determined from the provided context.",
])
def test_refusals_detected(answer):
    assert is_refusal(answer)


@pytest.mark.parametrize("answer", [
    "Global GDP growth is projected to be 2.9% in 2026 [p.6].",
    "BEIR uses k=0.9 and b=0.4 with Anserini [p.5].",
])
def test_normal_answers_are_not_refusals(answer):
    assert not is_refusal(answer)


def test_citation_precision():
    assert citation_precision([6, 6, 12], [6, 12]) == 1.0
    assert citation_precision([6, 3], [6]) == 0.5
    assert citation_precision([], [6]) is None


def test_aggregate_overall_correct_and_refusals():
    rows = [
        {"category": "fact_retrieval", "correct": True, "fact_recall": 1.0, "refused": False,
         "cited_pages": [1], "citation_precision": 1.0, "groundedness": 0.8, "latency_ms": 100},
        {"category": "fact_retrieval", "correct": False, "fact_recall": 0.0, "refused": True,
         "cited_pages": [], "citation_precision": None, "groundedness": 0.4, "latency_ms": 300},
        {"category": "unanswerable", "correct": True, "refused": True, "groundedness": 0.5, "latency_ms": 200},
        {"category": "unanswerable", "error": "gave up"},
    ]
    a = aggregate(rows)
    assert a["answer_correctness"] == 0.5
    assert a["false_refusal_rate"] == 0.5
    assert a["citation_rate"] == 0.5
    assert a["citation_precision"] == 1.0
    assert a["refusal_rate"] == 0.5  # an errored question counts as not refused
    assert a["overall_correct"] == 0.5
    assert a["errors"] == 1
    assert a["latency_ms"]["median"] == 200


def test_spearman_perfect_and_inverse():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0


def test_heuristic_vs_judge_confusion():
    rows = [
        {"groundedness": 0.9, "judge_faithfulness": 1.0},
        {"groundedness": 0.9, "judge_faithfulness": 0.5},
        {"groundedness": 0.3, "judge_faithfulness": 1.0},
        {"groundedness": 0.3, "judge_faithfulness": None},
    ]
    hj = heuristic_vs_judge(rows)
    assert hj["n"] == 3
    assert (hj["both_pass"], hj["heuristic_pass_judge_fail"], hj["heuristic_fail_judge_pass"]) == (1, 1, 1)


def test_parse_judgement():
    j = parse_judgement('{"is_refusal": false, "claims": [{"claim": "a", "supported": true},'
                        ' {"claim": "b", "supported": false}]}')
    assert j["judge_faithfulness"] == 0.5
    assert j["judge_unsupported"] == ["b"]
    assert parse_judgement('{"is_refusal": true, "claims": []}')["judge_faithfulness"] is None


def test_parse_manual_labels_and_agreement():
    text = ("## fr-001 (fact_retrieval)\n\nHuman faithfulness: 1\n\nHuman correct: y\n\n---\n"
            "## un-002 (unanswerable)\n\nHuman faithfulness: -\n\nHuman correct: n\n\n---\n"
            "## mh-003 (multi_hop)\n\nHuman faithfulness: \n\nHuman correct: \n")
    labels = parse_labels(text)
    assert labels["fr-001"] == {"faithfulness": 1.0, "correct": True}
    assert labels["un-002"] == {"faithfulness": None, "correct": False}
    assert labels["mh-003"] == {"faithfulness": None, "correct": None}
    rows = {"fr-001": {"groundedness": 0.8, "judge_faithfulness": 1.0, "correct": True},
            "un-002": {"correct": True}, "mh-003": {}}
    stats = agreement(labels, rows)
    assert stats["heuristic_agrees_with_human"] == 1.0
    assert stats["auto_correctness_agrees_with_human"] == 0.5
