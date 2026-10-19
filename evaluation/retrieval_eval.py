"""Retrieval ablation: dense vs BM25 vs hybrid (RRF) vs hybrid + cross-encoder.

All four configurations search the same in-memory Qdrant index (same PDFs,
same chunking, same embeddings) over the whole corpus, with no document filter.
Each returns a ranked list of up to 20 chunks; the reranker reorders the 20
hybrid candidates (the production pipeline keeps the top 5 of exactly this list).
"""

import time
from statistics import median

import numpy as np

from app.config import get_settings
from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import embed_chunks, get_embedding_model, upsert_chunks
from app.ingestion.pdf_parser import extract_pages
from app.retrieval.reranker import _get_reranker, rerank
from app.retrieval.search import dense_search, hybrid_search, sparse_search
from evaluation.check_benchmark import ROOT
from evaluation.retrieval_metrics import aggregate, question_metrics

settings = get_settings()

CONFIGS = {
    "dense": "Dense only (all-MiniLM-L6-v2)",
    "bm25": "Sparse only (BM25)",
    "hybrid": "Hybrid (dense + BM25, RRF k=60)",
    "hybrid_rerank": "Hybrid + cross-encoder reranker",
}


def retrieve(config: str, query: str):
    if config == "dense":
        return dense_search(query, settings.dense_top_k)
    if config == "bm25":
        return sparse_search(query, settings.sparse_top_k)
    if config == "hybrid":
        return hybrid_search(query)
    if config == "hybrid_rerank":
        candidates = hybrid_search(query)
        return rerank(query, candidates, top_k=len(candidates))
    raise ValueError(config)


def build_index(benchmark: dict) -> dict:
    """Parse, chunk, embed and upsert every corpus PDF under its benchmark doc_id."""
    stats = {}
    for doc_id, path in benchmark["metadata"]["documents"].items():
        pages = extract_pages((ROOT / path).read_bytes())
        chunks = chunk_pages(pages)
        upsert_chunks(doc_id, embed_chunks(chunks))
        stats[doc_id] = {"pages": len(pages), "chunks": len(chunks)}
    return stats


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q))


def run_ablation(benchmark: dict, repeats: int = 3) -> dict:
    questions = benchmark["questions"]
    answerable = [q for q in questions if q["category"] != "unanswerable"]

    # Load both models and run every config once before timing anything.
    get_embedding_model()
    _get_reranker()
    for config in CONFIGS:
        for q in questions[:3]:
            retrieve(config, q["question"])

    results = {}
    for config, label in CONFIGS.items():
        per_question = []
        for q in answerable:
            m = question_metrics(retrieve(config, q["question"]), q)
            per_question.append({"id": q["id"], "category": q["category"], **m})

        latencies = []
        for _ in range(repeats):
            for q in questions:
                start = time.perf_counter()
                retrieve(config, q["question"])
                latencies.append((time.perf_counter() - start) * 1000)

        by_category = {
            cat: aggregate([r for r in per_question if r["category"] == cat])
            for cat in ("fact_retrieval", "multi_hop", "comparison")
        }
        results[config] = {
            "label": label,
            "overall": aggregate(per_question, with_ci=True),
            "by_category": by_category,
            "latency_ms": {
                "median": round(median(latencies), 1),
                "p95": round(_percentile(latencies, 95), 1),
                "n": len(latencies),
            },
            "per_question": per_question,
        }
    return results


def ablation_markdown(results: dict, env: dict) -> str:
    lines = [
        "# Retrieval ablation",
        "",
        f"Benchmark: {env['answerable_questions']} answerable questions (unanswerable ones have no gold pages) over "
        f"{env['corpus_pages']} pages / {env['corpus_chunks']} chunks from 3 PDFs. Page-level metrics, whole-corpus search, "
        "20 candidates per configuration.",
        "",
        "| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@10 | nDCG@5 | Median latency | p95 latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results.values():
        o, lat = r["overall"], r["latency_ms"]
        lines.append(
            f"| {r['label']} | {o['recall@1']:.3f} | {o['recall@3']:.3f} | {o['recall@5']:.3f} | "
            f"{o['mrr@10']:.3f} | {o['ndcg@5']:.3f} | {lat['median']:.0f} ms | {lat['p95']:.0f} ms |"
        )
    lines += [
        "",
        "95% bootstrap intervals (2,000 resamples of the questions, seed 0). With n=40, one question moves Recall by 0.025, "
        "so configurations whose intervals overlap are not reliably different on this benchmark.",
        "",
        "| Configuration | Recall@5 95% CI | MRR@10 95% CI |",
        "|---|---|---|",
    ]
    for r in results.values():
        ci = r["overall"]["ci95"]
        lines.append(f"| {r['label']} | {ci['recall@5'][0]:.3f} - {ci['recall@5'][1]:.3f} | {ci['mrr@10'][0]:.3f} - {ci['mrr@10'][1]:.3f} |")
    lines += ["", "## Recall@5 by category", "", "| Configuration | fact_retrieval | multi_hop | comparison |", "|---|---|---|---|"]
    for r in results.values():
        c = r["by_category"]
        lines.append(
            f"| {r['label']} | {c['fact_retrieval']['recall@5']:.3f} (n={c['fact_retrieval']['n']}) | "
            f"{c['multi_hop']['recall@5']:.3f} (n={c['multi_hop']['n']}) | {c['comparison']['recall@5']:.3f} (n={c['comparison']['n']}) |"
        )
    lines += [
        "",
        "## Setup",
        "",
        f"- Embedding model: `{env['embedding_model']}` (revision `{env['embedding_revision']}`)",
        f"- Reranker: `{env['reranker_model']}` (revision `{env['reranker_revision']}`)",
        f"- Chunking: {env['chunk_size_tokens']} tokens (~4 chars/token), {env['chunk_overlap_tokens']} overlap; "
        f"BM25 k1={env['bm25_k1']}, b={env['bm25_b']}; RRF k={env['rrf_k']}",
        f"- Vector store: Qdrant local in-memory mode (qdrant-client {env['qdrant_client']}), exact search",
        f"- Latency: wall time per query inside the retrieval call, after warm-up; "
        f"{results[next(iter(results))]['latency_ms']['n']} samples per configuration "
        f"({env['latency_repeats']} passes over all {env['total_questions']} questions)",
        f"- Hardware: {env['cpu']}, {env['cpu_count']} logical cores, {env['ram_gb']} GB RAM, device `{env['device']}`, "
        f"torch {env['torch']} ({env['torch_threads']} threads), Python {env['python']}, {env['os']}",
        f"- Generated {env['generated_at']} by `python -m evaluation.run_eval --retrieval` at commit `{env['git_commit']}`",
        "",
    ]
    return "\n".join(lines)
