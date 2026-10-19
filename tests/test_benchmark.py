"""Benchmark integrity: counts, gold facts on gold pages, unanswerable terms absent."""

from evaluation.check_benchmark import check, load_benchmark
from evaluation.facts import fact_in, fact_recall, normalize


def test_total_questions_matches_metadata():
    b = load_benchmark()
    assert b["metadata"]["total_questions"] == len(b["questions"])


def test_category_split_is_20_10_10_10():
    b = load_benchmark()
    counts = {c: sum(1 for q in b["questions"] if q["category"] == c) for c in b["categories"]}
    assert counts == {"fact_retrieval": 20, "multi_hop": 10, "comparison": 10, "unanswerable": 10}


def test_benchmark_verified_against_corpus():
    assert check(load_benchmark()) == []


def test_check_reports_count_mismatch():
    b = load_benchmark()
    b["metadata"]["total_questions"] += 1
    assert any("total_questions" in e for e in check(b))


def test_check_reports_fact_not_on_gold_page():
    b = load_benchmark()
    b["questions"][0]["expected_answer_contains"] = [["a fact that is nowhere"]]
    assert any("not found on pages" in e for e in check(b))


def test_normalize_dashes_and_whitespace():
    assert normalize("Touché–2020\n  x") == "touché-2020 x"


def test_fact_matching_respects_word_boundaries():
    assert fact_in(["18"], "BEIR has 18 datasets")
    assert not fact_in(["18"], "published in 2018")
    assert fact_in(["19.5%", "19.5 percent"], "about 19.5 percent")


def test_fact_recall():
    assert fact_recall([["a"], ["b"]], "a only") == 0.5
    assert fact_recall([], "anything") == 1.0
