"""Tests for configuration module."""

import os
from unittest.mock import patch

from app.config import Settings, get_settings


def test_settings_defaults():
    s = Settings()
    assert s.qdrant_host == "localhost"
    assert s.qdrant_port == 6333
    assert s.embedding_model == "all-MiniLM-L6-v2"
    assert s.chunk_size_tokens == 512
    assert s.dense_top_k == 20
    assert s.rerank_top_k == 5
    assert s.max_upload_size_mb == 50


def test_settings_env_override():
    with patch.dict(os.environ, {"QDRANT_PORT": "9999", "APP_PORT": "3000"}):
        s = Settings()
        assert s.qdrant_port == 9999
        assert s.app_port == 3000


def test_settings_gemini_defaults():
    s = Settings()
    assert s.gemini_model == "gemini-2.0-flash"
    assert s.ollama_model == "qwen2.5:0.5b"


def test_settings_chunking_params():
    s = Settings()
    assert s.chunk_overlap_tokens < s.chunk_size_tokens
    assert s.max_chunk_size_tokens > s.chunk_size_tokens


def test_get_settings_returns_singleton():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
