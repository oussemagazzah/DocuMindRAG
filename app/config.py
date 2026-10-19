"""Application configuration for DocuMindRAG."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    # ":memory:" or a local folder runs Qdrant in-process (eval, CI, demo); empty = use host/port
    qdrant_location: str = ""
    collection_name: str = "documind_chunks"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # LLM (Gemini)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"
    gemini_temperature: float = 0.0
    gemini_max_output_tokens: int = 1024

    # LLM (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:0.5b"

    # Chunking
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 128
    max_chunk_size_tokens: int = 1024

    # Retrieval
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 5

    # BM25 (term-frequency part; IDF is applied by Qdrant at query time)
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    bm25_avg_doc_len: float = 300.0

    # Database (PostgreSQL or SQLite)
    database_url: str = ""
    database_path: str = "./documind.db"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Upload
    max_upload_size_mb: int = 50

    # Rate limiting
    rate_limit_per_minute: int = 30
    api_key_header: str = "X-API-Key"
    api_keys: str = ""
    # Behind a reverse proxy (Hugging Face Spaces), rate-limit by X-Forwarded-For.
    trust_proxy_headers: bool = False

    # Public demo: read-only, preloaded documents, capped LLM usage
    demo_mode: bool = False
    preload_dir: str = ""
    llm_daily_limit: int = 0  # 0 = unlimited; counts Gemini calls per UTC day

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: str = "development"
    log_level: str = "INFO"

    @property
    def use_postgres(self) -> bool:
        return bool(self.database_url)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
