# Retrieval ablation

Benchmark: 40 answerable questions (unanswerable ones have no gold pages) over 56 pages / 146 chunks from 3 PDFs. Page-level metrics, whole-corpus search, 20 candidates per configuration.

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@10 | nDCG@5 | Median latency | p95 latency |
|---|---|---|---|---|---|---|---|
| Dense only (all-MiniLM-L6-v2) | 0.537 | 0.725 | 0.863 | 0.721 | 0.735 | 16 ms | 29 ms |
| Sparse only (BM25) | 0.688 | 0.925 | 0.963 | 0.880 | 0.885 | 6 ms | 9 ms |
| Hybrid (dense + BM25, RRF k=60) | 0.650 | 0.812 | 0.912 | 0.838 | 0.827 | 22 ms | 27 ms |
| Hybrid + cross-encoder reranker | 0.650 | 0.912 | 0.963 | 0.860 | 0.863 | 1251 ms | 1681 ms |

95% bootstrap intervals (2,000 resamples of the questions, seed 0). With n=40, one question moves Recall by 0.025, so configurations whose intervals overlap are not reliably different on this benchmark.

| Configuration | Recall@5 95% CI | MRR@10 95% CI |
|---|---|---|
| Dense only (all-MiniLM-L6-v2) | 0.750 - 0.950 | 0.615 - 0.829 |
| Sparse only (BM25) | 0.900 - 1.000 | 0.803 - 0.950 |
| Hybrid (dense + BM25, RRF k=60) | 0.825 - 0.975 | 0.751 - 0.925 |
| Hybrid + cross-encoder reranker | 0.912 - 1.000 | 0.779 - 0.925 |

## Recall@5 by category

| Configuration | fact_retrieval | multi_hop | comparison |
|---|---|---|---|
| Dense only (all-MiniLM-L6-v2) | 0.800 (n=20) | 0.850 (n=10) | 1.000 (n=10) |
| Sparse only (BM25) | 1.000 (n=20) | 0.950 (n=10) | 0.900 (n=10) |
| Hybrid (dense + BM25, RRF k=60) | 0.900 (n=20) | 0.850 (n=10) | 1.000 (n=10) |
| Hybrid + cross-encoder reranker | 1.000 (n=20) | 0.850 (n=10) | 1.000 (n=10) |

## Setup

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`)
- Chunking: 512 tokens (~4 chars/token), 128 overlap; BM25 k1=1.2, b=0.75; RRF k=60
- Vector store: Qdrant local in-memory mode (qdrant-client 1.12.1), exact search
- Latency: wall time per query inside the retrieval call, after warm-up; 150 samples per configuration (3 passes over all 50 questions)
- Hardware: 11th Gen Intel(R) Core(TM) i5-11400F @ 2.60GHz, 12 logical cores, 15.8 GB RAM, device `cpu`, torch 2.13.0+cpu (6 threads), Python 3.13.1, Windows 11
- Generated 2026-10-19 16:59 UTC by `python -m evaluation.run_eval --retrieval` at commit `fb92bbd`
