"""LLM-powered answer generation with Gemini + Ollama fallback."""

import google.generativeai as genai
import ollama as ollama_client

from app.config import get_settings
from app.errors import LLMError
from app.logging_config import get_logger
from app.retrieval.search import SearchResult

settings = get_settings()
logger = get_logger("answer")

if settings.gemini_api_key:
    genai.configure(api_key=settings.gemini_api_key)

SYSTEM_PROMPT = """You are a document QA assistant. Answer ONLY based on the provided context chunks.

Rules:
1. For every factual claim, cite the page number like [p.X] or [p.X, p.Y].
2. If the context doesn't contain enough information to answer, say "I don't have enough information in the provided documents to answer this question."
3. Never make up information or cite page numbers that aren't in the context.
4. Be concise and direct.
5. If multiple documents are referenced, mention the document name."""


def _format_context(results: list[SearchResult]) -> str:
    parts = []
    for r in results:
        parts.append(f"{r.text} [p.{r.page_number}]")
    return "\n\n".join(parts)


def extract_citations(answer: str, results: list[SearchResult]) -> list[dict]:
    citations = []
    seen_pages = set()
    for r in results:
        key = (r.doc_id, r.page_number)
        if key not in seen_pages:
            seen_pages.add(key)
            citations.append({
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "doc_name": "",
                "page_number": r.page_number,
                "chunk_text": r.text[:500],
                "char_start": r.char_start,
                "char_end": r.char_end,
                "retrieval_score": round(r.score, 4),
            })
    return citations


def build_user_message(query: str, results: list[SearchResult]) -> str:
    return f"Context:\n{_format_context(results)}\n\nQuestion: {query}"


def generate_answer(query: str, results: list[SearchResult]) -> tuple[str, str]:
    user_message = build_user_message(query, results)

    if settings.gemini_api_key:
        try:
            logger.info("Generating answer with Gemini")
            answer = _generate_with_gemini(user_message)
            logger.info(f"Gemini answer generated ({len(answer)} chars)")
            return answer, settings.gemini_model
        except Exception as e:
            logger.warning(f"Gemini failed: {type(e).__name__}: {e}")

    try:
        logger.info("Generating answer with Ollama")
        answer = _generate_with_ollama(user_message)
        logger.info(f"Ollama answer generated ({len(answer)} chars)")
        return answer, settings.ollama_model
    except Exception as e:
        logger.error(f"Ollama also failed: {type(e).__name__}: {e}")
        raise LLMError()


def gemini_response(user_message: str):
    """Raw Gemini response (the evaluation also reads finish_reason and token usage)."""
    model = genai.GenerativeModel(
        model_name=settings.gemini_model,
        system_instruction=SYSTEM_PROMPT,
    )
    return model.generate_content(
        user_message,
        generation_config=genai.GenerationConfig(
            temperature=settings.gemini_temperature,
            max_output_tokens=settings.gemini_max_output_tokens,
        ),
    )


def _generate_with_gemini(user_message: str) -> str:
    return gemini_response(user_message).text


def _generate_with_ollama(user_message: str) -> str:
    response = ollama_client.chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response["message"]["content"]
