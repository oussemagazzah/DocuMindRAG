from dataclasses import dataclass
from app.ingestion.pdf_parser import PageContent
from app.config import get_settings

settings = get_settings()

# Rough chars-per-token estimate for English text
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


def _split_page_into_windows(text: str, page_number: int, chunk_size: int, overlap: int) -> list[Chunk]:
    chunks = []
    chunk_chars = chunk_size * CHARS_PER_TOKEN
    overlap_chars = overlap * CHARS_PER_TOKEN
    step = chunk_chars - overlap_chars

    start = 0
    chunk_index = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    page_number=page_number,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    char_start=start,
                    char_end=end,
                )
            )
            chunk_index += 1
        start += step
    return chunks


def _assign_pages(chunks: list[Chunk], spans: list[tuple[int, int, int]]) -> None:
    """Label each window with the page that contributes most of its characters.

    Short pages are merged into a buffer with the following page(s); without
    this, every window cut from that buffer would carry the first page's number.
    """
    for c in chunks:
        overlaps = [(min(c.char_end, end) - max(c.char_start, start), page) for start, end, page in spans]
        best = max(o for o, _ in overlaps)
        # On a tie, keep the earlier page.
        c.page_number = next(page for o, page in overlaps if o == best)


def chunk_pages(pages: list[PageContent]) -> list[Chunk]:
    chunk_size = settings.chunk_size_tokens
    overlap = settings.chunk_overlap_tokens
    max_chunk_size = settings.max_chunk_size_tokens

    all_chunks: list[Chunk] = []
    buffer_text = ""
    buffer_start = 0
    buffer_page = 1
    buffer_page_count = 0
    buffer_spans: list[tuple[int, int, int]] = []  # (start, end, page_number) within buffer_text

    for page in pages:
        page_text = page.text
        if not page_text:
            continue

        if buffer_text:
            buffer_text += "\n\n"
        else:
            buffer_start = 0
            buffer_page = page.page_number
            buffer_spans = []
        buffer_spans.append((len(buffer_text), len(buffer_text) + len(page_text), page.page_number))
        buffer_text += page_text

        buffer_page_count += 1
        estimated_tokens = _estimate_tokens(buffer_text)

        if estimated_tokens >= chunk_size:
            sub_chunks = _split_page_into_windows(buffer_text, buffer_page, chunk_size, overlap)
            _assign_pages(sub_chunks, buffer_spans)
            for sc in sub_chunks:
                sc.chunk_index = len(all_chunks)
                all_chunks.append(sc)

            buffer_text = ""
            buffer_start = 0
            buffer_page_count = 0

    if buffer_text.strip():
        sub_chunks = _split_page_into_windows(buffer_text, buffer_page, chunk_size, overlap)
        _assign_pages(sub_chunks, buffer_spans)
        for sc in sub_chunks:
            sc.chunk_index = len(all_chunks)
            all_chunks.append(sc)

    return all_chunks
