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
