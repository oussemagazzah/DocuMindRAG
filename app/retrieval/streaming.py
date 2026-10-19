"""Streaming answer generation with Gemini + Ollama fallback."""

import json
from typing import AsyncGenerator

import google.generativeai as genai
import ollama as ollama_client

from app.config import get_settings
from app.logging_config import get_logger
from app.retrieval.answer import SYSTEM_PROMPT, _format_context
from app.retrieval.search import SearchResult

settings = get_settings()
logger = get_logger("streaming")

if settings.gemini_api_key:
    genai.configure(api_key=settings.gemini_api_key)


async def stream_answer(query: str, results: list[SearchResult]) -> AsyncGenerator[str, None]:
    """Stream answer tokens as they are generated.

    Yields JSON-encoded events:
    - {"type": "token", "content": "..."}
    - {"type": "done", "model": "...", "timing_ms": ...}
    - {"type": "error", "message": "..."}
    """
    import time
    start = time.time()
    context = _format_context(results)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    if settings.gemini_api_key:
        try:
            async for chunk in _stream_gemini(user_message):
                yield chunk
            elapsed = (time.time() - start) * 1000
            yield json.dumps({"type": "done", "model": settings.gemini_model, "timing_ms": round(elapsed, 2)})
            return
        except Exception as e:
            logger.warning(f"Gemini streaming failed: {type(e).__name__}: {e}")

    try:
        async for chunk in _stream_ollama(user_message):
            yield chunk
        elapsed = (time.time() - start) * 1000
        yield json.dumps({"type": "done", "model": settings.ollama_model, "timing_ms": round(elapsed, 2)})
    except Exception as e:
        logger.error(f"Ollama streaming also failed: {type(e).__name__}: {e}")
        yield json.dumps({"type": "error", "message": "No LLM provider available"})


async def _stream_gemini(user_message: str) -> AsyncGenerator[str, None]:
    model = genai.GenerativeModel(
        model_name=settings.gemini_model,
        system_instruction=SYSTEM_PROMPT,
    )
    response = model.generate_content(
        user_message,
        generation_config=genai.GenerationConfig(
            temperature=settings.gemini_temperature,
            max_output_tokens=settings.gemini_max_output_tokens,
        ),
        stream=True,
    )
    for chunk in response:
        if chunk.text:
            yield json.dumps({"type": "token", "content": chunk.text})


async def _stream_ollama(user_message: str) -> AsyncGenerator[str, None]:
    response = ollama_client.chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        stream=True,
    )
    for chunk in response:
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield json.dumps({"type": "token", "content": content})
