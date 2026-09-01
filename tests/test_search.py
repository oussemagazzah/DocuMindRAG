"""Tests for search module — RRF fusion and data structures."""

from app.retrieval.search import SearchResult, rrf_fusion


def _make_result(chunk_id: str, score: float = 0.5, source: str = "dense") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        doc_id="doc1",
        page_number=1,
        chunk_index=0,
        text=f"Text for {chunk_id}",
        char_start=0,
        char_end=100,
        score=score,
        source=source,
    )


def test_rrf_fusion_empty():
    result = rrf_fusion([], [], k=60)
    assert result == []


def test_rrf_fusion_dense_only():
    dense = [_make_result("a", 0.9), _make_result("b", 0.8)]
    result = rrf_fusion(dense, [], k=60)
    assert len(result) == 2
    assert result[0].chunk_id == "a"
    assert result[0].source == "rrf"


def test_rrf_fusion_sparse_only():
    sparse = [_make_result("a", 0.7, "sparse"), _make_result("c", 0.6, "sparse")]
    result = rrf_fusion([], sparse, k=60)
    assert len(result) == 2
    assert result[0].chunk_id == "a"


def test_rrf_fusion_merges_results():
    dense = [_make_result("a", 0.9), _make_result("b", 0.8)]
    sparse = [_make_result("a", 0.7, "sparse"), _make_result("c", 0.6, "sparse")]
    result = rrf_fusion(dense, sparse, k=60)
    ids = [r.chunk_id for r in result]
    assert "a" in ids
    assert "b" in ids
    assert "c" in ids
    assert len(result) == 3


def test_rrf_fusion_deduplicates():
    dense = [_make_result("a", 0.9)]
    sparse = [_make_result("a", 0.7, "sparse")]
    result = rrf_fusion(dense, sparse, k=60)
    assert len(result) == 1
    assert result[0].score > 0.03


def test_rrf_fusion_ranking_order():
    dense = [_make_result("a", 0.9), _make_result("b", 0.8), _make_result("c", 0.7)]
    result = rrf_fusion(dense, [], k=60)
    assert result[0].chunk_id == "a"
    assert result[1].chunk_id == "b"
    assert result[2].chunk_id == "c"


def test_rrf_fusion_different_k():
    dense = [_make_result("a", 0.9)]
    r1 = rrf_fusion(dense, [], k=1)
    r2 = rrf_fusion(dense, [], k=60)
    assert r1[0].score > r2[0].score


def test_search_result_dataclass():
    r = _make_result("x", 0.5)
    assert r.chunk_id == "x"
    assert r.score == 0.5
    assert r.source == "dense"
    assert r.page_number == 1
