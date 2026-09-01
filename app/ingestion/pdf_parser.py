import fitz
from dataclasses import dataclass


@dataclass
class PageContent:
    page_number: int
    text: str
    char_count: int


def extract_pages(pdf_bytes: bytes) -> list[PageContent]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append(
            PageContent(
                page_number=i + 1,
                text=text.strip(),
                char_count=len(text.strip()),
            )
        )
    doc.close()
    return pages


def get_pdf_metadata(pdf_bytes: bytes) -> dict:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    metadata = {
        "page_count": len(doc),
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
    }
    doc.close()
    return metadata
