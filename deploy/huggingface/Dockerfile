# Hugging Face Docker Space for the DocuMindRAG public demo.
# Built from a staging folder assembled by deploy/huggingface/push_space.py:
# this file becomes the Space's root Dockerfile.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

# Spaces run containers as uid 1000.
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Bake both models into the image so startup needs no download.
RUN python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY app app
COPY static static
COPY run.py .
COPY evaluation/corpus evaluation/corpus
COPY evaluation/sources.md evaluation/sources.md
RUN chown -R user:user /app
USER user

# Demo configuration. GEMINI_API_KEY must be set as a Space *secret*, never here.
# LLM_DAILY_LIMIT stays under the Gemini free tier's 20 requests/day per model.
ENV APP_HOST=0.0.0.0 \
    APP_PORT=7860 \
    APP_ENV=production \
    QDRANT_LOCATION=:memory: \
    DATABASE_PATH=/tmp/documind.db \
    DEMO_MODE=true \
    PRELOAD_DIR=evaluation/corpus \
    TRUST_PROXY_HEADERS=true \
    RATE_LIMIT_PER_MINUTE=6 \
    LLM_DAILY_LIMIT=18 \
    GEMINI_MODEL=gemini-3.5-flash \
    GEMINI_TEMPERATURE=0.0 \
    HF_HUB_OFFLINE=1

EXPOSE 7860
CMD ["python", "run.py"]
