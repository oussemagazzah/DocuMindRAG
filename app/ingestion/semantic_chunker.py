"""Semantic chunking using sentence boundaries and embedding similarity."""

import re
from dataclasses import dataclass

from app.config import get_settings
from app.logging_config import get_logger
from app.ingestion.pdf_parser import PageContent

settings = get_settings()
logger = get_logger("chunker")

CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    page_number: int
    chunk_index: int
    text: str
    char_start: int
    char_end: int


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _split_page_into_windows(text: str, page_number: int, chunk_size: int, overlap: int) -> list[Chunk]:
    chunks = []
    chunk_chars = chunk_size * CHARS_PER_TOKEN
    overlap_chars = overlap * CHARS_PER_TOKEN
    step = chunk_chars - overlap_chars

    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    page_number=page_number,
                    chunk_index=0,
                    text=chunk_text,
                    char_start=start,
                    char_end=end,
                )
            )
        start += step
    return chunks


def chunk_pages_semantic(pages: list[PageContent]) -> list[Chunk]:
    """Semantic chunking: split by sentences, group by token budget.

    This preserves sentence boundaries for better retrieval quality.
    """
    chunk_size = settings.chunk_size_tokens
    overlap = settings.chunk_overlap_tokens
    max_chunk_size = settings.max_chunk_size_tokens

    all_chunks: list[Chunk] = []
    buffer_sentences: list[str] = []
    buffer_tokens = 0
    buffer_page = 1
    buffer_start = 0
    current_page_start = 0

    for page in pages:
        if not page.text:
            continue

        sentences = _split_into_sentences(page.text)
        current_page_start = 0

        for sentence in sentences:
            sent_tokens = _estimate_tokens(sentence)

            if buffer_tokens + sent_tokens > chunk_size and buffer_sentences:
                chunk_text = " ".join(buffer_sentences)
                chunk = Chunk(
                    page_number=buffer_page,
                    chunk_index=len(all_chunks),
                    text=chunk_text.strip(),
                    char_start=buffer_start,
                    char_end=buffer_start + len(chunk_text),
                )
                all_chunks.append(chunk)

                if overlap > 0 and len(buffer_sentences) > 1:
                    overlap_sents = buffer_sentences[-max(1, len(buffer_sentences) // 4):]
                    buffer_sentences = overlap_sents
                    buffer_tokens = sum(_estimate_tokens(s) for s in buffer_sentences)
                    buffer_start += len(chunk_text) - len(" ".join(overlap_sents))
                else:
                    buffer_sentences = []
                    buffer_tokens = 0
                    buffer_start = current_page_start

            buffer_sentences.append(sentence)
            buffer_tokens += sent_tokens
            current_page_start += len(sentence) + 1

            if buffer_page != page.page_number:
                buffer_page = page.page_number

    if buffer_sentences:
        chunk_text = " ".join(buffer_sentences)
        if _estimate_tokens(chunk_text) <= max_chunk_size:
            all_chunks.append(
                Chunk(
                    page_number=buffer_page,
                    chunk_index=len(all_chunks),
                    text=chunk_text.strip(),
                    char_start=buffer_start,
                    char_end=buffer_start + len(chunk_text),
                )
            )

    logger.info(f"Semantic chunking: {len(pages)} pages -> {len(all_chunks)} chunks")
    return all_chunks
