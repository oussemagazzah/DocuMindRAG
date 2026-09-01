"""Application configuration for DocuMindRAG."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    collection_name: str = "documind_chunks"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # LLM (Gemini)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

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
