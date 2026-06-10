import hashlib
import io
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF
import numpy as np
import tiktoken
from PIL import Image

logger = logging.getLogger(__name__)
#  DATA MODELS
@dataclass
class PageContent:
    """Extracted content of one PDF page."""
    page_number: int
    text: str
    method: str  # "pymupdf" | "easyocr"
    char_count: int = 0
    word_count: int = 0
    ocr_confidence: float = 0.0
    has_images: bool = False

    def __post_init__(self):
        self.char_count = len(self.text)
        self.word_count = len(self.text.split()) if self.text.strip() else 0


@dataclass
class TextChunk:
    """One token-aware, overlapping context block."""
    chunk_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    text: str = ""
    token_count: int = 0
    source_pages: List[int] = field(default_factory=list)
    chunk_index: int = 0
    overlap_tokens_prev: int = 0
    content_hash: str = ""

    def __post_init__(self):
        self.char_count = len(self.text)
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.text.encode()).hexdigest()[:16]


@dataclass
class IngestedDocument:
    """Top-level container returned by ingest_pdf()."""
    document_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    filename: str = ""
    file_hash: str = ""
    total_pages: int = 0
    pdf_metadata: Dict[str, Any] = field(default_factory=dict)
    pages: List[PageContent] = field(default_factory=list)
    chunks: List[TextChunk] = field(default_factory=list)
    total_chunks: int = 0
    total_tokens: int = 0
    pages_ocr_count: int = 0
    pages_text_count: int = 0
    elapsed_seconds: float = 0.0
    ingested_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

## PDF Text Extraction

MIN_TEXT_LENGTH = 30  # pages below this will be sent to OCR

# replace hidden control characters 
def _clean_text(text: str) -> str:
    """Normalise whitespace, strip control chars."""
    text = text.replace("\x0c", "\n").replace("\x0b", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines).strip()

# extract text from each page
def _extract_pages(doc: fitz.Document) -> Tuple[List[PageContent], List[int]]:
    """
    Walk every page — extract text via PyMuPDF.
    Returns (pages, list_of_page_indices_needing_ocr).
    """
    pages: List[PageContent] = []
    ocr_needed: List[int] = []

    for i in range(doc.page_count):
        page = doc[i]
        raw = page.get_text("text")
        clean = _clean_text(raw)
        has_images = bool(page.get_images(full=True))

        # id text length less than 30 add it into ocr scanning
        if len(clean) < MIN_TEXT_LENGTH:
            ocr_needed.append(i)
            pages.append(PageContent(
                page_number=i + 1, text="",
                method="easyocr", has_images=has_images,
            ))
        else:
            pages.append(PageContent(
                page_number=i + 1, text=clean,
                method="pymupdf", has_images=has_images,
            ))

    return pages, ocr_needed

def _get_page_image(doc: fitz.Document, page_idx: int, dpi: int = 300) -> bytes:
    """Rasterise one PDF page to PNG bytes."""
    pix = doc[page_idx].get_pixmap(dpi=dpi)
    return pix.tobytes("png")

# find the hidden header and extract basic tags
def _pdf_metadata(doc: fitz.Document) -> Dict[str, Any]:
    meta = doc.metadata or {}
    return {
        "title": meta.get("title", ""),
        "author": meta.get("author", ""),
        "subject": meta.get("subject", ""),
        "page_count": doc.page_count,
    }