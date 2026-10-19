"""Tests for BM25 sparse vectors and Qdrant filtering (in-memory Qdrant, no server)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from app.config import get_settings
from app.ingestion import embedder
from app.ingestion.embedder import build_sparse_query_vector, build_sparse_vector, ensure_collection
from app.retrieval import search

settings = get_settings()


def _weights(vec) -> dict[int, float]:
    return dict(zip(vec.indices, vec.values, strict=True))


def test_sparse_indices_stable_across_processes():
    # Python's hash() is salted per process; the sparse index must not be.
    code = "from app.ingestion.embedder import build_sparse_vector as b; print(sorted(b('revenue grew 12%').indices))"
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).parent.parent,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("1", "2")
    }
    assert len(outputs) == 1


def test_query_and_document_share_indices():
    doc = _weights(build_sparse_vector("Revenue grew in 2025."))
    query = _weights(build_sparse_query_vector("revenue 2025"))
    assert set(query) <= set(doc)


def test_query_vector_weights_are_one_and_deduplicated():
    vec = build_sparse_query_vector("revenue revenue growth")
    assert len(vec.indices) == 2
    assert all(v == 1.0 for v in vec.values)


def test_bm25_term_frequency_saturates():
    one = max(build_sparse_vector("revenue").values)
    many = max(build_sparse_vector(" ".join(["revenue"] * 50)).values)
    assert many > one
    assert many < settings.bm25_k1 + 1


def test_bm25_length_normalisation():
    short = _weights(build_sparse_vector("revenue grew"))
    long = _weights(build_sparse_vector("revenue " + " ".join(f"word{i}" for i in range(600))))
    idx = build_sparse_query_vector("revenue").indices[0]
    assert long[idx] < short[idx]


def test_tokenizer_keeps_decimals():
    assert embedder.tokenize("Revenue: $4.2 billion.") == ["revenue", "4.2", "billion"]


def test_empty_text_gives_empty_vector():
    assert build_sparse_vector("").indices == []


@pytest.fixture
def memory_client(monkeypatch):
    client = QdrantClient(location=":memory:")
    ensure_collection(client)
    texts = [
        ("doc_a", "the company reported revenue growth"),
        ("doc_b", "the company reported a carbon emissions target"),
        ("doc_c", "the company reported headcount"),
    ]
    client.upsert(
        collection_name=settings.collection_name,
        points=[
            PointStruct(
                id=i,
                vector={"dense": [0.0] * (settings.embedding_dimension - 1) + [1.0], "sparse": build_sparse_vector(t)},
                payload={"doc_id": d, "page_number": 1, "chunk_index": 0, "text": t, "char_start": 0, "char_end": len(t)},
            )
            for i, (d, t) in enumerate(texts)
        ],
    )
    monkeypatch.setattr(search, "get_qdrant_client", lambda: client)
    return client


def test_sparse_search_ranks_rare_term_first(memory_client):
    results = search.sparse_search("company emissions", top_k=3)
    assert results[0].doc_id == "doc_b"


def test_filter_with_multiple_doc_ids_matches_any(memory_client):
    results = search.sparse_search("company reported", top_k=10, doc_ids=["doc_a", "doc_b"])
    assert {r.doc_id for r in results} == {"doc_a", "doc_b"}


def test_filter_with_single_doc_id(memory_client):
    results = search.sparse_search("company reported", top_k=10, doc_ids=["doc_c"])
    assert [r.doc_id for r in results] == ["doc_c"]
