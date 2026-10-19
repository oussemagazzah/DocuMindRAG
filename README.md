# DocuMindRAG

> Question answering over PDFs with page-level citations: hybrid dense + BM25 retrieval, cross-encoder reranking and Gemini generation, measured on a checked 50-question benchmark.

[![CI](https://github.com/oussemagazzah/DocuMindRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/oussemagazzah/DocuMindRAG/actions/workflows/ci.yml)

**Live demo:** TBD (Hugging Face Space prepared in [`deploy/huggingface/`](deploy/huggingface/), not deployed yet)

**Built with:** Python 3.13 · FastAPI · Qdrant · sentence-transformers · PyMuPDF · Gemini · Docker · Pytest

## Headline results

Measured on 50 questions over 3 public PDFs (56 pages). All numbers come from [`evaluation/results/results.json`](evaluation/results/results.json).

- **Retrieval (40 answerable questions, page level):** hybrid search + cross-encoder reranking reaches **Recall@5 96.25%**, against **86.25%** for dense-only search.
  Plain **BM25 reaches the same Recall@5 (96.25%)** with the best Recall@1 (68.75%) at 5 ms per query, while the reranker costs ~1.7 s on CPU. [Why that happens](#what-the-ablation-shows) matters more than the headline.
- **End-to-end answer correctness:** TBD. The run is limited by the Gemini free tier (20 requests/day) and will be filled in from `results.json`.
- **Unanswerable questions correctly refused:** TBD.

## Retrieval ablation

Same index, chunking and 20 candidates for every configuration, searching the whole corpus with no document filter. Values are rounded to 3 decimals; exact values are in `results.json`.
Full report: [`evaluation/results/ablation.md`](evaluation/results/ablation.md).

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@10 | nDCG@5 | Median latency | p95 latency |
|---|---|---|---|---|---|---|---|
| Dense only (all-MiniLM-L6-v2) | 0.537 | 0.725 | 0.863 | 0.721 | 0.735 | 18 ms | 22 ms |
| Sparse only (BM25) | 0.688 | 0.925 | 0.963 | 0.880 | 0.885 | 5 ms | 7 ms |
| Hybrid (dense + BM25, RRF k=60) | 0.650 | 0.812 | 0.912 | 0.838 | 0.827 | 26 ms | 32 ms |
| Hybrid + cross-encoder reranker | 0.650 | 0.912 | 0.963 | 0.860 | 0.863 | 1689 ms | 1862 ms |

Recall@5 by question type:

| Configuration | fact_retrieval | multi_hop | comparison |
|---|---|---|---|
| Dense only (all-MiniLM-L6-v2) | 0.800 (n=20) | 0.850 (n=10) | 1.000 (n=10) |
| Sparse only (BM25) | 1.000 (n=20) | 0.950 (n=10) | 0.900 (n=10) |
| Hybrid (dense + BM25, RRF k=60) | 0.900 (n=20) | 0.850 (n=10) | 1.000 (n=10) |
| Hybrid + cross-encoder reranker | 1.000 (n=20) | 0.850 (n=10) | 1.000 (n=10) |

**Metrics.** Retrieved chunks are collapsed into a ranked list of unique (document, page) pairs. Each question lists *evidence groups*: a fact restated on two pages forms one group with two pages, and a two-hop question has two groups.
- **Recall@k:** fraction of groups with a page in the top k.
- **MRR@10:** reciprocal rank of the first relevant page.
- **nDCG@5:** binary gain, counting each group once.

Definitions and tests: [`evaluation/retrieval_metrics.py`](evaluation/retrieval_metrics.py).

**Latency** is wall time inside the retrieval call after warm-up: 150 samples per configuration on an Intel i5-11400F (6 cores), CPU only, Qdrant in in-memory mode. It is noisy on a desktop: the reranker's median was 1,251 ms in an earlier run of the same code and 1,689 ms in this one.

### What the ablation shows

- **The reranker fixes what fusion breaks.** Equal-weight RRF lets the weaker dense list pull good BM25 hits down: Recall@3 drops from 0.925 (BM25) to 0.812 (hybrid). Reranking the 20 hybrid candidates brings it back to 0.912 and Recall@5 to 0.963.
  It also rescues table-only questions that dense search misses. For example, Türkiye's 2025 inflation (only in an OECD table) moves from MRR 0.17 (hybrid) to 1.0.
- **BM25 alone is as good as the full pipeline here, and dense-only is clearly worse at the top rank** (Recall@1 0.537 vs 0.688). The 95% bootstrap intervals for BM25 and hybrid + reranker overlap: Recall@5 0.900–1.000 vs 0.912–1.000. With 40 questions, one question moves Recall by 0.025, so this benchmark cannot separate them.
- **Why BM25 is so strong on this benchmark:**
  - Questions were written while reading the PDFs, so they reuse the documents' exact terms ("Hole@10", "Anserini", "TREC-COVID"). That is the same lexical bias BEIR (one of the corpus documents) describes in its Section 6.
  - Many answers are numbers inside tables.
  - The corpus is tiny (146 chunks), which makes IDF very discriminative.

  A benchmark of paraphrased questions would likely favour dense retrieval more. I did not tune RRF weights or the candidate depth on these 40 questions, because that would make the numbers meaningless.
- **The reranker does not improve multi-hop recall overall** (0.850 before and after), but the total hides real changes.
  It recovers the missing hop on two questions (mh-001, mh-009) and loses a hop on two others (mh-004, mh-010).
  It scores each chunk against the whole question, so a chunk that answers only one hop can drop below the cut-off.

## End-to-end answer quality

Pipeline: hybrid search → cross-encoder rerank → top 5 chunks → `gemini-3.5-flash` at temperature 0 (max 1,024 output tokens), with the same prompt as the app.
The second-opinion judge is `gemini-3.1-flash-lite`. Configuration is pinned in [`evaluation/run_eval.py`](evaluation/run_eval.py).

| Category | n | Answer correctness | Refusal rate | Citation precision | Heuristic groundedness | Judge faithfulness |
|---|---|---|---|---|---|---|
| Overall | 50 | TBD | TBD | TBD | TBD | TBD |
| fact_retrieval | 20 | TBD | n/a | TBD | TBD | TBD |
| multi_hop | 10 | TBD | n/a | TBD | TBD | TBD |
| comparison | 10 | TBD | n/a | TBD | TBD | TBD |
| unanswerable | 10 | n/a | TBD | n/a | TBD | TBD |

The Gemini free tier allows 20 requests per model per day, so the 50-question run is spread over several days. Answers are cached in `.eval_cache/`, and no partial results are written.

- **Answer correctness:** the answer contains *every* key fact listed for the question (any accepted spelling, word-boundary match). It is string matching, so a correct but differently phrased answer counts as wrong.
- **Refusal rate:** share of unanswerable questions where the answer says the documents do not contain the answer.
- **Citation precision:** share of cited `[p.X]` pages that are gold pages.

### What "groundedness" means here

The app shows a **heuristic** groundedness score ([`app/retrieval/groundedness.py`](app/retrieval/groundedness.py)). No LLM is involved:

`0.5 × citation validity + 0.3 × context coverage + 0.2 × hedging`

- **Citation validity:** share of cited pages that were in the retrieved context (0.5 when nothing is cited).
- **Context coverage:** share of retrieved chunks whose first content words appear in the answer.
- **Hedging:** the answer contains an "insufficient information" phrase.

It measures **citation hygiene and word overlap, not whether the claims are true**. An answer that cites a retrieved page but misstates its number can still score high.

As a second opinion, an **LLM judge** splits each answer into claims and checks each against the retrieved context, Ragas-style (supported / total claims). The judge is a different model from the generator, but still a Gemini model, so it is not independent. How well the two agree (Spearman correlation and a 2×2 table), and a hand check of 20 stratified answers ([`evaluation/results/manual_review.md`](evaluation/results/manual_review.md), scored by `python -m evaluation.manual_agreement`): **TBD**.

## Benchmark

| Document | Pages | Licence |
|---|---|---|
| BEIR (Thakur et al., NeurIPS 2021), arXiv:2104.08663v4 | 24 | CC BY-SA 4.0 |
| Ragas (Es et al., 2023), arXiv:2309.15217v2 | 8 | CC BY 4.0 |
| OECD Economic Outlook, Interim Report, September 2025 | 24 | CC BY 4.0 |

[`evaluation/benchmark.json`](evaluation/benchmark.json) holds 50 questions:
- 20 fact retrieval
- 10 multi-hop (evidence on two different pages)
- 10 comparison
- 10 unanswerable but plausible (a 2027 projection when the report stops at 2026, a model BEIR does not evaluate, a metric Ragas does not report)

Each question has a `doc_id`, gold pages (1-based PDF page index), evidence groups, key facts with accepted spellings, and a difficulty. Questions answerable only from a table are marked `hard`.

**How the questions were checked:** [`python -m evaluation.check_benchmark`](evaluation/check_benchmark.py), also run in CI, verifies that:
- every key fact appears on one of its gold pages in the text extracted by the app's own parser;
- every unanswerable question's key terms appear **nowhere** in the corpus;
- `metadata.total_questions` matches the real count.

The checker caught one leaky unanswerable question on its first run ("sports" appears in a BEIR table), and it was replaced.
Sources, hashes, licences and method: [`evaluation/sources.md`](evaluation/sources.md).

### Reproduce

```bash
python -m evaluation.check_benchmark          # verify the benchmark against the PDFs
python -m evaluation.run_eval --retrieval     # ablation (no LLM, deterministic quality metrics, ~3 min on CPU)
python -m evaluation.run_eval --e2e --judge   # end-to-end + LLM judge (needs GEMINI_API_KEY)
python -m evaluation.run_eval --all           # both
```

Results are written to `evaluation/results/`, together with model revisions, settings, hardware and the git commit that produced them. The evaluation always uses its own in-memory Qdrant collection and pins the LLM configuration regardless of `.env`.

## Architecture

```
Browser (static/)
   │
FastAPI (app/main.py)
   ├── Rate limiting (POST/DELETE per client, 429 + Retry-After) and optional API keys
   ├── Upload: PyMuPDF text (ligatures expanded, hyphenation joined)
   │           → page-aware chunks (512 tokens, 128 overlap, labelled with their majority page)
   │           → all-MiniLM-L6-v2 dense vectors + BM25 sparse vectors → Qdrant
   ├── Query:  dense top 20 + BM25 top 20 → Reciprocal Rank Fusion (k=60)
   │           → cross-encoder ms-marco-MiniLM-L-6-v2 → top 5 chunks
   │           → Gemini (Ollama fallback) with [p.X] citations
   │           → heuristic groundedness score
   └── SQLite / PostgreSQL: documents, chunks, query logs, feedback, analytics
```

- **BM25 in Qdrant.** Each chunk stores the BM25 term-frequency part, saturated and length-normalised (k1=1.2, b=0.75). Qdrant applies IDF at query time (`Modifier.IDF`). Term ids come from `crc32`, so they are stable across processes; an earlier version used Python's salted `hash()` and silently broke sparse search after every restart.
- **Qdrant** runs as a server (Docker Compose) or in-process (`QDRANT_LOCATION=:memory:` or a folder), which is used by the evaluation, the tests and the demo.
- **Retrieval-only fallback.** In demo mode, or when the daily LLM budget is spent, `/query` returns the reranked passages with a notice instead of failing.

## Quick start

Prerequisites: Python 3.11+, and Docker (for the Qdrant server) or `QDRANT_LOCATION=:memory:`.

```bash
git clone https://github.com/oussemagazzah/DocuMindRAG.git
cd DocuMindRAG
python -m venv venv
venv\Scripts\activate              # Windows; on macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env             # then set GEMINI_API_KEY
docker-compose up -d qdrant        # or set QDRANT_LOCATION=:memory: in .env
python run.py                      # http://localhost:8000
```

To run the full stack in Docker: `docker-compose up -d`. The compose file also starts a Redis container, which the app does not use yet.

### Public demo (Hugging Face Space)

[`deploy/huggingface/`](deploy/huggingface/) contains the Space's Dockerfile and a push script. The Space:
- uses CPU-only torch with both models baked into the image;
- runs Qdrant in memory with the three benchmark PDFs preloaded;
- is read-only (upload and delete return 403);
- allows 6 queries per minute per visitor;
- generates at most 18 Gemini answers per day, to stay under the free tier's 20, then shows retrieved passages only.

The API key is a Space secret. The reranker stays on. A rough local estimate, with torch limited to 2 threads, is about 2.3 s for 20 candidates; it is not in `results.json`, and latency on the Space itself is TBD. Gemini's own latency on the free tier was several seconds per call during the evaluation.

```bash
hf auth login
python deploy/huggingface/push_space.py --space <user>/DocuMindRAG
```

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Qdrant status, demo mode, LLM calls left today |
| POST | `/upload` | Upload a PDF (403 in demo mode) |
| GET | `/documents` | List documents |
| DELETE | `/documents/{id}` | Delete a document (403 in demo mode) |
| POST | `/query` | Answer + citations (with document name and page) + groundedness |
| POST | `/query/stream` | Stream answer tokens as Server-Sent Events |
| POST | `/feedback` | Thumbs up/down on an answer |
| GET | `/analytics` | Query volume, latency, groundedness and feedback stats |
| POST | `/eval/run` | Smoke-test a document with 5 generic questions (not the benchmark) |

## Configuration

Settings come from environment variables or `.env` (see [`.env.example`](.env.example)).

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(empty)* | Gemini key; without it the app tries Ollama |
| `GEMINI_MODEL` / `GEMINI_TEMPERATURE` | `gemini-3.5-flash` / `0.0` | Generator |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | Qdrant server |
| `QDRANT_LOCATION` | *(empty)* | `:memory:` or a folder for in-process Qdrant |
| `EMBEDDING_MODEL` / `RERANKER_MODEL` | `all-MiniLM-L6-v2` / `cross-encoder/ms-marco-MiniLM-L-6-v2` | Models |
| `DENSE_TOP_K` / `SPARSE_TOP_K` / `RRF_K` / `RERANK_TOP_K` | `20` / `20` / `60` / `5` | Retrieval depth |
| `DATABASE_URL` | *(empty = SQLite)* | PostgreSQL connection string |
| `RATE_LIMIT_PER_MINUTE` | `30` | POST/DELETE requests per client per minute |
| `TRUST_PROXY_HEADERS` | `false` | Identify clients by `X-Forwarded-For` (only behind a trusted proxy) |
| `API_KEYS` | *(empty = disabled)* | Comma-separated API keys |
| `DEMO_MODE` / `PRELOAD_DIR` / `LLM_DAILY_LIMIT` | `false` / *(empty)* / `0` | Public demo settings |

## Tests and CI

```bash
python -m pytest -q        # 186 tests
ruff check .
```

The tests cover:
- chunking, page labelling, PDF parsing, BM25 vectors and Qdrant filtering (in-memory);
- RRF, the retrieval and answer metrics, and the benchmark integrity checks;
- the API, middleware (429/401 responses), demo mode and the retrieval-only fallback;
- groundedness, the database and analytics.

LLM calls are mocked. [GitHub Actions](.github/workflows/ci.yml) runs ruff (pinned rule set) and pytest with CPU-only torch, with no API keys and with Hugging Face downloads disabled.

## Project structure

```
app/                  FastAPI app: ingestion/, retrieval/, services/, middleware, DB
static/               Frontend (HTML/CSS/JS)
evaluation/           Benchmark, corpus PDFs, checker, metrics, ablation, e2e + judge, results/
deploy/huggingface/   Space Dockerfile, Space card, push script
tests/                Pytest suite
alembic/              Database migrations
.github/workflows/    CI
```

## Limitations and next steps

- **Small benchmark.** 50 questions over 3 documents, written by one person. Differences of a few points are within noise; see the bootstrap intervals in `ablation.md`.
- **Lexical bias.** The questions reuse document wording, which favours BM25. A paraphrased question set is the next benchmark to add.
- **Heuristic groundedness** checks citations and word overlap, not truth. The LLM judge is a second opinion from the same model family.
- **String-matched correctness** can mark correct paraphrases as wrong.
- **Citations show page numbers only.** The model sees no document names, so `[p.6]` is ambiguous when two documents are retrieved. The UI shows the document name next to each source, but the answer text does not.
- **Model aliases drift.** On 2026-10-19, Google's quota metadata showed `gemini-3.5-flash` served as `gemini-3.6-flash`. The API name is not a frozen checkpoint.
- **English only.** The embedding model, reranker and BM25 tokenizer all assume English.
- **Small open issues.** `MAX_CHUNK_SIZE_TOKENS` is not enforced by the chunker. The Redis container is unused. `google-generativeai` is deprecated in favour of `google-genai`.
- **Next steps:**
  - a held-out, paraphrased question set;
  - weighted RRF or a stronger embedding model, evaluated on that set;
  - document names in citations;
  - a rerank-depth ablation (10 vs 20 candidates);
  - migrating to the `google-genai` SDK.

---

Built by Oussema Gazzeh — [LinkedIn](https://linkedin.com/in/oussemagazzeh) | [GitHub](https://github.com/oussemagazzah) | [Website](https://oussemagazzeh.me)
