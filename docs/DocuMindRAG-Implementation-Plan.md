# DocuMindRAG — MVP Implementation Plan

> [!info] Project Overview
> A PDF Q&A app with real page citations using FastAPI, Qdrant, embeddings, hybrid retrieval, and reranking.

---

## 1. Architecture Overview

```
┌─────────────┐     ┌──────────────────────────────────────────────┐
│  Frontend    │────▶│  FastAPI Backend                             │
│  (Gradio)   │     │                                              │
└─────────────┘     │  ┌────────────┐  ┌──────────┐  ┌──────────┐ │
                    │  │ Ingestion  │  │ Retrieval │  │ Answer   │ │
                    │  │ Pipeline   │  │ Pipeline  │  │ Engine   │ │
                    │  └─────┬──────┘  └────┬─────┘  └────┬─────┘ │
                    └────────┼──────────────┼─────────────┼───────┘
                             ▼              ▼             ▼
                    ┌──────────────┐  ┌──────────┐  ┌──────────┐
                    │  Qdrant      │  │ Sentence │  │ LLM      │
                    │  (2 cols:    │  │ Transf.  │  │ (Ollama/ │
                    │  dense+sparse│  │ Embeddings│  │  OpenAI) │
                    └──────────────┘  └──────────┘  └──────────┘
```

> [!important] Key Decisions
> - **Local-first**: Qdrant runs via Docker, embeddings run locally (`all-MiniLM-L6-v2`), LLM via Ollama (local) with OpenAI as fallback
> - **Frontend**: Gradio (fastest to MVP, portfolio-friendly, no JS work)
> - **Storage**: SQLite for metadata (documents, chunks, citation index)
> - **No auth for MVP** (single-user/local tool)

---

## 2. MVP Scope — What Ships

### ✅ In MVP

- Upload 1-N PDFs
- Page-aware chunking
- Dense + sparse vectors in Qdrant
- RRF fusion
- Cross-encoder reranking
- Citation-linked answers
- Gradio UI
- Single-user local
- SQLite metadata store
- Basic health/retry

### ❌ Out of MVP (Postpone)

- Multi-format (Word, HTML)
- Semantic chunking
- Separate sparse index
- Learnable fusion weights
- Custom reranker fine-tuning
- Highlighted PDF viewer
- Custom React frontend
- Multi-user auth
- PostgreSQL
- Circuit breakers, queues

---

## 3. User Stories

| ID | Story | Acceptance Criteria |
|---|---|---|
| US-01 | As a user, I upload a PDF and see it processed | Upload endpoint returns doc_id; status endpoint shows "ready" within 60s for a 20-page PDF |
| US-02 | As a user, I ask a question and get an answer with page citations | Response contains answer text + array of `{page, chunk_text, score}` citations |
| US-03 | As a user, I see which document(s) the answer came from | Citations include source document name and page number |
| US-04 | As a user, I can upload multiple PDFs and query across them | Answers can cite chunks from different uploaded documents |
| US-05 | As a user, I can see a list of uploaded documents | List endpoint returns all ingested documents with status |
| US-06 | As a user, I can delete a document and its chunks | Delete removes vectors from Qdrant + metadata from SQLite |

---

## 4. Ingestion Design

### Pipeline Stages

```
PDF bytes → Page extraction → Page-aware chunking → Embedding → Qdrant upsert → SQLite metadata
```

### Page-Aware Chunking Rules

> [!tip] Chunking Strategy
> - **Unit**: each page is an atomic unit
> - **Chunk size target**: 512 tokens (~2000 chars)
> - **If page > 512 tokens**: split into overlapping windows (128 token overlap)
> - **If page < 512 tokens**: merge with next page until threshold, never exceed 1024 tokens
> - **Metadata attached to every chunk**: `{doc_id, page_number, chunk_index, char_start, char_end}`
> - **Sparse vectors**: generated via BM25-style `SparseValues` (Qdrant native) at ingestion time

### Libraries

- `pymupdf4llm` (or `pymupdf`) for PDF page extraction — fast, reliable, handles complex layouts
- `sentence-transformers` for dense embeddings
- `qdrant-client` for vector operations

### Ingestion Worker Considerations

> [!note] MVP Scope
> - For MVP: synchronous in the API endpoint (acceptable for single-user)
> - Log ingestion time per document for benchmarking
> - Store raw page text in SQLite for citation retrieval (avoid re-reading PDF at query time)

---

## 5. Retrieval Design

### Hybrid Retrieval Pipeline

```
Query → [Dense search] + [Sparse search] → RRF fusion → Rerank (cross-encoder) → Top-K results
```

### Step-by-Step

1. **Dense retrieval**: Encode query with `all-MiniLM-L6-v2`, search Qdrant dense vector column, retrieve top-20
2. **Sparse retrieval**: Encode query as BM25 sparse vector, search Qdrant sparse vector column, retrieve top-20
3. **RRF fusion**: Combine the two ranked lists using `score = Σ 1/(k + rank_i)` with `k=60` (standard constant). Deduplicate by chunk_id.
4. **Reranking**: Pass fused top-20 through `cross-encoder/ms-marco-MiniLM-L-6-v2`, keep top-5
5. **Answer generation**: Feed top-5 chunk texts + query to LLM with a strict citation prompt

> [!important] Why This Design
> - Dense captures semantic similarity
> - Sparse captures exact keyword matches (technical terms, names, dates)
> - RRF is parameter-free and robust
> - Cross-encoder reranking is the highest-ROI quality lever

---

## 6. Citation Design

### Citation Data Model

```python
{
    "chunk_id": "uuid",
    "doc_id": "uuid",
    "doc_name": "report.pdf",
    "page_number": 3,
    "chunk_text": "The revenue increased by 12%...",
    "char_start": 1024,
    "char_end": 1536,
    "retrieval_score": 0.87
}
```

### Grounded Answer Rules

> [!warning] Citation Rules (enforced via prompt)
> - Every factual claim must reference `[p.X]` or `[p.X, p.Y]`
> - If no retrieved chunk supports the claim, say "I don't have enough information"
> - Never hallucinate page numbers — only cite pages present in the retrieved context
> - Return citations as structured JSON alongside the answer text

### Prompt Template Skeleton

```
You are a document QA assistant. Answer ONLY based on the provided context chunks.
For each claim, cite the page number like [p.X].
If the context doesn't contain enough information, say so explicitly.
Do not make up information.

Context:
{chunk_1} [p.3]
{chunk_2} [p.5]
...

Question: {query}
```

---

## 7. API Plan

| Method | Endpoint | Purpose | Request | Response |
|---|---|---|---|---|
| `POST` | `/upload` | Upload and ingest a PDF | `multipart/form-data` (file) | `{doc_id, status, pages, chunks}` |
| `GET` | `/documents` | List all documents | — | `[{doc_id, name, pages, status, created_at}]` |
| `DELETE` | `/documents/{doc_id}` | Delete a document | — | `{status: "deleted"}` |
| `POST` | `/query` | Ask a question | `{query, doc_ids?: [], top_k?: 5}` | `{answer, citations: [{page, text, score, doc_name}], timing_ms}` |
| `GET` | `/health` | Health check | — | `{status: "ok", qdrant: "connected"}` |

---

## 8. Data Model

### SQLite Tables

#### `documents`

| Column | Type | Notes |
|---|---|---|
| id | TEXT (UUID) | Primary key |
| name | TEXT | Original filename |
| page_count | INTEGER | |
| chunk_count | INTEGER | |
| status | TEXT | `processing`, `ready`, `error` |
| created_at | TIMESTAMP | |
| error_message | TEXT | Nullable |

#### `chunks`

| Column | Type | Notes |
|---|---|---|
| id | TEXT (UUID) | Primary key |
| doc_id | TEXT | FK → documents.id |
| page_number | INTEGER | 1-indexed |
| chunk_index | INTEGER | Within-page order |
| text | TEXT | Raw chunk text (for citation retrieval) |
| char_start | INTEGER | Character offset in page |
| char_end | INTEGER | Character offset in page |

### Qdrant Collection: `documind_chunks`

- Vector config: dense (384-dim, cosine) + sparse (BM25)
- Payload: `{doc_id, page_number, chunk_index}`

---

## 9. Project Structure

```
DocuMindRAG/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app + routes
│   ├── config.py            # Settings (env vars, constants)
│   ├── db.py                # SQLite connection + schema init
│   ├── models.py            # Pydantic schemas (request/response)
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py    # PDF → pages
│   │   ├── chunker.py       # Pages → chunks
│   │   └── embedder.py      # Chunks → vectors → Qdrant
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── search.py        # Dense + sparse + RRF
│   │   ├── reranker.py      # Cross-encoder reranking
│   │   └── answer.py        # LLM answer generation
│   └── services/
│       ├── __init__.py
│       └── document_service.py  # Orchestration logic
├── tests/
│   ├── __init__.py
│   ├── test_chunker.py
│   ├── test_retrieval.py
│   ├── test_citations.py
│   └── conftest.py          # Fixtures (sample PDFs, Qdrant mock)
├── sample_pdfs/             # Test documents
├── gradio_app.py            # Gradio UI
├── docker-compose.yml       # Qdrant
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 10. Milestones

### Milestone 1: Foundation (Days 1-2)

- [ ] Project scaffolding, Docker setup, Qdrant running
- [ ] SQLite schema + connection
- [ ] PDF parsing (PyMuPDF) producing page text
- [ ] Basic FastAPI app with `/health`

### Milestone 2: Ingestion (Days 3-4)

- [ ] Page-aware chunker
- [ ] Embedding generation (sentence-transformers)
- [ ] Qdrant collection creation + upsert
- [ ] `/upload` endpoint working end-to-end
- [ ] `/documents` list endpoint

### Milestone 3: Retrieval (Days 5-7)

- [ ] Dense search working
- [ ] Sparse vector generation + search
- [ ] RRF fusion
- [ ] Cross-encoder reranking
- [ ] `/query` endpoint returning raw chunks

### Milestone 4: Answer + Citations (Days 8-9)

- [ ] LLM integration (Ollama local)
- [ ] Citation-aware prompt engineering
- [ ] Structured answer + citation response
- [ ] End-to-end pipeline tested

### Milestone 5: UI + Polish (Days 10-12)

- [ ] Gradio interface (upload tab + query tab)
- [ ] Error handling, loading states
- [ ] README, demo script
- [ ] Performance benchmarks documented

---

## 11. Testing Strategy

| Layer | What | How | Priority |
|---|---|---|---|
| Unit | Chunker logic, RRF fusion, citation extraction | pytest with fixtures | High |
| Integration | Upload → Qdrant → Query → Answer | pytest + test PDFs | High |
| E2E | Full pipeline through API | httpx + local Qdrant | Medium |
| Manual | Demo quality, answer accuracy | Human evaluation | High |

> [!tip] Key Test Fixtures to Create
> - A 3-page sample PDF with known content
> - A 20+ page PDF with tables and figures
> - A corrupted/empty PDF for error handling
> - Pre-computed expected chunks for regression tests

---

## 12. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| PyMuPDF extraction quality on complex PDFs | Medium | High | Fallback to `pdfplumber`; test with diverse PDFs early |
| Qdrant sparse vector support quirks | Medium | Medium | Use Qdrant's native `SparseValues`; validate with small dataset first |
| Cross-encoder latency (200-500ms) | Low | Low | Acceptable for MVP; document as optimization opportunity |
| LLM hallucination in citations | High | High | Strict prompt + post-validation (check cited pages exist in context) |
| Chunk boundary splits break citation accuracy | Medium | Medium | Store char offsets; overlap windows reduce this |
| Local embedding model quality | Low | Medium | `all-MiniLM-L6-v2` is solid; can swap to `bge-small` later |

---

## 13. Timeline (12 Working Days)

| Day | Focus | Deliverable |
|---|---|---|
| 1 | Scaffold + Docker + Qdrant | `docker-compose up` works, `/health` returns 200 |
| 2 | PDF parsing + SQLite | Can extract pages from any PDF, metadata stored |
| 3 | Chunker + Embeddings | Page-aware chunks generated and embedded |
| 4 | Qdrant ingestion | `/upload` ingests PDF, vectors stored, `/documents` lists them |
| 5 | Dense retrieval | `/query` returns relevant chunks via dense search |
| 6 | Sparse retrieval + RRF | Hybrid retrieval working, results better than dense alone |
| 7 | Reranking | Cross-encoder improves result quality visibly |
| 8 | LLM + Citations | `/query` returns grounded answers with page citations |
| 9 | Error handling + edge cases | Corrupt PDFs, empty queries, missing docs handled |
| 10 | Gradio UI | Upload and query work in browser |
| 11 | Polish + README | Demo-ready, documented, sample PDFs included |
| 12 | Benchmarks + portfolio | Latency numbers, accuracy eval, GitHub README finalized |

---

## 14. Task Checklist (GitHub Issues Ready)

### Epic: Foundation

- [ ] Initialize project structure + `.gitignore` + `requirements.txt`
- [ ] Create `docker-compose.yml` with Qdrant
- [ ] Create `.env.example` with all config vars
- [ ] Set up FastAPI app skeleton in `app/main.py`
- [ ] Implement `/health` endpoint
- [ ] Create SQLite schema + `db.py` initialization

### Epic: Ingestion

- [ ] Implement PDF page extraction (`pdf_parser.py`)
- [ ] Implement page-aware chunker (`chunker.py`)
- [ ] Implement embedding generation (`embedder.py`)
- [ ] Create Qdrant collection with dense + sparse config
- [ ] Implement `/upload` endpoint
- [ ] Implement `/documents` list endpoint
- [ ] Implement `DELETE /documents/{doc_id}` endpoint

### Epic: Retrieval

- [ ] Implement dense vector search
- [ ] Implement sparse vector generation + search
- [ ] Implement RRF fusion
- [ ] Implement cross-encoder reranking
- [ ] Build retrieval orchestrator (`search.py`)

### Epic: Answer Generation

- [ ] Integrate Ollama LLM
- [ ] Design citation prompt template
- [ ] Implement structured answer + citation response
- [ ] Add post-validation for cited pages

### Epic: UI + Demo

- [ ] Build Gradio upload tab
- [ ] Build Gradio query tab
- [ ] Add loading states + error messages
- [ ] Write README with screenshots
- [ ] Add 3 sample PDFs to `sample_pdfs/`
- [ ] Document latency benchmarks

---

## 15. What to Build First

> [!danger] Day 1 Priority — Non-Negotiable
> 1. `docker-compose.yml` with Qdrant
> 2. FastAPI skeleton + `/health`
> 3. PDF parsing that returns page text
>
> **Why**: If PDF extraction fails or Qdrant won't start, nothing else matters. Validate the hardest dependency first.

---

## 16. What to Postpone

| Item | Reason |
|---|---|
| Authentication / multi-user | Not needed for local-first MVP |
| PostgreSQL migration | SQLite is sufficient for single-user |
| Streaming responses | Gradio handles this; add later for API consumers |
| Batch ingestion (100+ PDFs) | Not the MVP use case |
| Document versioning | Complexity with no clear MVP value |
| Custom embedding fine-tuning | `MiniLM` is good enough to ship |

---

## 17. What to Measure

| Metric | Target | How |
|---|---|---|
| Ingestion time (20-page PDF) | < 30 seconds | Timing in `/upload` |
| Query latency (p95) | < 3 seconds | Timing in `/query` response |
| Reranking overhead | < 500ms | Measure before/after reranker |
| Citation accuracy | > 80% correct page references | Manual eval on 20 questions |
| Answer helpfulness | > 75% useful | Manual eval on 20 questions |
| Chunks per page (avg) | 2-4 | Logged during ingestion |

---

## 18. What Would Make the Demo Impressive

> [!tip] Demo Tips
> 1. **Show the citation chain**: Display the answer with highlighted citations, then expand to show the exact source text + page. This is the "wow" moment.
> 2. **Query across multiple documents**: Upload 3 different PDFs, ask a question that requires synthesizing info from 2 of them.
> 3. **Show the "I don't know" case**: Demonstrate that when the docs don't contain the answer, the system says so instead of hallucinating.
> 4. **Latency breakdown in response**: Return `timing_ms` with a breakdown (retrieval: 200ms, reranking: 400ms, generation: 1.2s) — shows engineering rigor.
> 5. **Side-by-side comparison**: Show dense-only results vs hybrid+reranked results for the same query — proves the retrieval stack adds value.
> 6. **Use a real, non-trivial document**: A 30-page technical report, not a toy PDF. Demonstrates production-readiness.

---

## 19. Assumptions

- [ ] Docker Desktop installed and running
- [ ] Python 3.11+ available
- [ ] For local LLM: Ollama installed with `llama3.1:8b` (or `mistral`) pulled
- [ ] For API-based LLM fallback: an OpenAI API key is available
- [ ] Target machine has 8GB+ RAM (for local embeddings + Ollama)
- [ ] Single-user local tool, not multi-tenant SaaS
- [ ] English-language PDFs only for MVP

---

## 20. Recommended Defaults

| Choice | Default | Rationale |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | Fast, 384-dim, good quality, widely used |
| Sparse method | Qdrant native `SparseValues` (BM25) | No extra infra, Qdrant handles it |
| RRF constant k | 60 | Standard value from literature |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Best quality/speed tradeoff |
| Chunk size | 512 tokens | Standard for RAG |
| Chunk overlap | 128 tokens | 25% overlap balances context vs redundancy |
| Top-K before rerank | 20 | Gives reranker enough candidates |
| Top-K after rerank | 5 | Enough context without noise |
| LLM | `llama3.1:8b` via Ollama | Local, fast, good at following instructions |
| Vector DB | Qdrant single-node | Simple, performant, Docker-ready |

---

> [!summary] Next Steps
> 1. Translate the **Task Checklist** (Section 14) into GitHub Issues
> 2. Create a GitHub Projects board with the 5 **Milestones** as columns
> 3. Add the **User Stories** (Section 3) as Notion cards
> 4. Start Day 1: `docker-compose.yml` → FastAPI skeleton → `/health`
