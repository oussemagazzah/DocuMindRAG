# DocuMindRAG

> Context-aware document intelligence platform with verifiable citations. Ask questions about your PDFs and get grounded answers with exact page references.

**Built with:** Python 3.13 | FastAPI | Qdrant | Docker | Ollama | PyMuPDF | Cross-Encoder | Pytest

---

## Demo

```
Upload a PDF → Ask "What was the total revenue in 2025?"
→ Answer: "TechVision reported total revenue of $4.2 billion in 2025..."
→ Source: [p.3] "Revenue grew 12% year-over-year to $4.2B..."
→ Groundedness: 92% | Latency: 1.8s
```

## Key Features

### Retrieval Pipeline
- **Hybrid search** — Dense embeddings (sentence-transformers) + BM25 sparse retrieval with Reciprocal Rank Fusion
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` for precision
- **Page-aware chunking** — PDF pages split into overlapping windows with metadata preserved
- **Streaming SSE** — Token-by-token answer streaming via Server-Sent Events

### Quality & Trust
- **Groundedness scoring** — Citation precision, context coverage, hedging detection
- **Verifiable citations** — Every answer links back to exact source pages with retrieval scores
- **Thumbs up/down feedback** — User feedback loop stored in DB for fine-tuning data

### Production Infrastructure
- **Docker Compose** — One-command deploy: `docker-compose up` (App + Qdrant + Redis)
- **PostgreSQL + SQLite** — Dual database support via async drivers (asyncpg / aiosqlite)
- **Alembic migrations** — Schema versioning for production deployments
- **Structured logging** — JSON-formatted logs with request ID tracking
- **Rate limiting + API key auth** — Sliding window rate limiter with optional API key middleware
- **Input validation** — PDF header checks, file size limits, XSS protection

### Analytics & Evaluation
- **Analytics dashboard** — Query volume, latency stats, groundedness distribution, feedback ratios
- **Evaluation pipeline** — Automated benchmark with 50 questions across 4 categories
- **Admin dashboard** — Recent query history, model usage breakdown, verdict distribution

## Architecture

```
Client (Browser)
    │
    ▼
FastAPI Server (app/main.py)
    │
    ├── Rate Limiting + API Key Auth (middleware.py)
    │
    ├── Upload Pipeline
    │   ├── PDF Parser (PyMuPDF)
    │   ├── Page-Aware Chunker (sliding window)
    │   ├── Embedder (sentence-transformers all-MiniLM-L6-v2)
    │   └── Qdrant Upsert (dense + sparse vectors)
    │
    ├── Query Pipeline
    │   ├── Hybrid Search (dense + BM25 → RRF fusion)
    │   ├── Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
    │   ├── LLM Generation (Ollama qwen2.5 / Gemini fallback)
    │   ├── Groundedness Scoring (heuristic + citation analysis)
    │   └── SSE Streaming Response
    │
    ├── Feedback + Analytics (query_logs + feedback tables)
    │
    └── SQLite / PostgreSQL (dual driver support)
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Pydantic, Uvicorn |
| Vector DB | Qdrant (dense cosine + sparse BM25) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, 384-dim) |
| Reranker | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| PDF Parsing | PyMuPDF |
| LLM | Ollama (`qwen2.5:0.5b`), Gemini API (fallback) |
| Database | SQLite (dev) / PostgreSQL (prod), Alembic |
| Cache | Redis |
| Containerization | Docker, Docker Compose |
| Testing | Pytest, pytest-asyncio, httpx |
| CI/CD | GitHub Actions |
| Frontend | Vanilla JS, HTML5, CSS3 |

## Quick Start

### Prerequisites
- Python 3.11+
- Docker Desktop (for Qdrant)
- Ollama (for local LLM)

### Setup

```bash
# Clone
git clone https://github.com/username/DocuMindRAG.git
cd DocuMindRAG

# Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Dependencies
pip install -r requirements.txt

# Environment config
copy .env.example .env

# Start Qdrant + Redis
docker-compose up -d

# Pull LLM model
ollama pull qwen2.5:0.5b

# Start server
python run.py
```

Open http://localhost:8000

### Docker (Full Stack)

```bash
docker-compose up -d
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check (Qdrant status) |
| POST | `/upload` | Upload a PDF document |
| GET | `/documents` | List all documents |
| DELETE | `/documents/{id}` | Delete a document |
| POST | `/query` | Ask a question (returns answer + citations + groundedness) |
| POST | `/query/stream` | Stream answer tokens as SSE |
| POST | `/feedback` | Submit thumbs up/down feedback |
| GET | `/analytics` | Query analytics and stats |
| POST | `/eval/run` | Run evaluation benchmark |

## Project Structure

```
DocuMindRAG/
├── app/
│   ├── main.py              # FastAPI app, routes, middleware
│   ├── config.py            # Pydantic settings (env-based)
│   ├── db.py                # SQLite + PostgreSQL dual driver
│   ├── models.py            # Pydantic request/response schemas
│   ├── errors.py            # Custom error hierarchy + handlers
│   ├── logging_config.py    # Structured logging (JSON + readable)
│   ├── middleware.py         # Rate limiting + API key auth
│   ├── ingestion/
│   │   ├── pdf_parser.py    # PyMuPDF page extraction
│   │   ├── chunker.py       # Page-aware sliding window chunking
│   │   ├── embedder.py      # Embedding + Qdrant operations
│   │   └── semantic_chunker.py  # Sentence-boundary chunking
│   ├── retrieval/
│   │   ├── search.py        # Hybrid dense + sparse + RRF fusion
│   │   ├── reranker.py      # Cross-encoder reranking
│   │   ├── answer.py        # LLM generation (Gemini + Ollama)
│   │   ├── streaming.py     # SSE streaming answer generation
│   │   └── groundedness.py  # Heuristic groundedness scoring
│   └── services/
│       └── document_service.py  # Pipeline orchestration
├── static/                  # Frontend (HTML + CSS + JS)
├── tests/                   # 122 automated tests
├── evaluation/              # Benchmark dataset + runner
├── alembic/                 # Database migrations
├── .github/workflows/       # CI pipeline
├── Dockerfile               # Multi-stage production build
├── docker-compose.yml       # App + Qdrant + Redis
└── requirements.txt
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=app --cov-report=term-missing
```

**122 tests** across 13 files covering:
- Config validation, chunking logic, PDF parsing
- Hybrid search, reranking, answer generation
- Groundedness scoring, error handling
- API endpoints, middleware, database operations
- Feedback, analytics, evaluation endpoints

## Configuration

All settings via environment variables (`.env`). Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen2.5:0.5b` | LLM model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `DATABASE_URL` | *(empty = SQLite)* | PostgreSQL connection string |
| `RATE_LIMIT_PER_MINUTE` | `30` | Requests per minute limit |
| `API_KEYS` | *(empty = disabled)* | Comma-separated API keys |

## What I Learned

- **RAG pipeline design** — Why hybrid search outperforms single retrieval, and how RRF fusion balances dense + sparse signals
- **Groundedness evaluation** — Building heuristic scoring that catches hedging language, citation precision, and context coverage
- **Production readiness** — Structured logging, input validation, error boundaries, and rate limiting are not optional
- **Database portability** — Abstracting SQLite/PostgreSQL behind a common interface with async drivers
- **Container orchestration** — Docker Compose for multi-service development environments
- **Test-driven development** — 122 tests covering the full stack gives confidence to refactor

## License

MIT

---

Built by Oussema — [LinkedIn](https://linkedin.com/in/username) | [GitHub](https://github.com/username)
