"""
Phase 1: Ingestion & Parsing Layer — AI Study Buddy
=====================================================

This module forms the first stage of the AI Study Buddy pipeline.
Its responsibility is to take raw PDF bytes (uploaded by a student),
extract readable text from every page, and split that text into
overlapping chunks that downstream components (embedding, retrieval,
quiz generation) can consume.

Pipeline overview:
    PDF bytes
      │
      ▼
    PyMuPDF text extraction  (fast, native text layer)
      │
      ├─ page has ≥ 50 chars ──► use extracted text
      │
      └─ page has < 50 chars ──► render to image ──► EasyOCR fallback
      │
      ▼
    Recursive token-aware chunker  (overlapping context windows)
      │
      ▼
    List[dict]  — chunks with metadata (page_num, chunk_index, text)

Dependencies:
    - PyMuPDF   (imported as ``fitz``)
    - EasyOCR   (lazy-loaded to avoid slow startup when not needed)
    - numpy     (required by EasyOCR for image array conversion)
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# If PyMuPDF extracts fewer characters than this from a page, we assume
# the page is image-heavy or scanned and fall back to OCR.
_MIN_TEXT_LENGTH_THRESHOLD: int = 50

# Default resolution (DPI) used when rendering a PDF page to a pixmap
# for OCR.  Higher values improve OCR accuracy at the cost of memory.
_OCR_RENDER_DPI: int = 300


class PDFIngestionEngine:
    """Extract and chunk text from PDF documents.

    This engine is the entry-point for Phase 1 of the AI Study Buddy
    pipeline.  It accepts raw PDF bytes, extracts text using PyMuPDF
    (with an EasyOCR fallback for scanned / image-heavy pages), and
    returns a flat list of overlapping text chunks annotated with source
    metadata.

    Parameters
    ----------
    chunk_size : int, optional
        The maximum number of characters per chunk (default ``500``).
    chunk_overlap : int, optional
        The number of characters that consecutive chunks share so that
        context is not lost at chunk boundaries (default ``50``).

    Example
    -------
    >>> engine = PDFIngestionEngine(chunk_size=400, chunk_overlap=40)
    >>> with open("lecture.pdf", "rb") as fh:
    ...     chunks = engine.process_pdf(fh.read())
    >>> chunks[0].keys()
    dict_keys(['text', 'page_num', 'chunk_index'])
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less "
                f"than chunk_size ({chunk_size})."
            )

        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap

        # EasyOCR is heavy (~1 GB models).  We lazy-load it on first use
        # so that the engine starts up quickly when OCR is not needed.
        self._ocr_reader: Optional[object] = None

        logger.info(
            "PDFIngestionEngine initialised (chunk_size=%d, chunk_overlap=%d)",
            self.chunk_size,
            self.chunk_overlap,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_ocr_reader(self):
        """Lazily initialise and return the EasyOCR reader.

        The reader is created once and reused for the lifetime of the
        engine instance.  English is the default language.

        Returns
        -------
        easyocr.Reader
            A ready-to-use EasyOCR reader instance.
        """
        if self._ocr_reader is None:
            logger.info("Lazy-loading EasyOCR reader (this may take a moment)…")
            import easyocr  # heavy import — deferred on purpose

            # gpu=False keeps things portable; set to True if a CUDA GPU
            # is available and you want faster OCR.
            self._ocr_reader = easyocr.Reader(["en"], gpu=False)
            logger.info("EasyOCR reader loaded successfully.")
        return self._ocr_reader

    def _ocr_page(self, page: fitz.Page) -> str:
        """Render a single PDF page to an image and run OCR on it.

        The page is rasterised at ``_OCR_RENDER_DPI`` resolution, then
        passed to EasyOCR which returns a list of detected text blocks.
        The blocks are joined with spaces to form a single string.

        Parameters
        ----------
        page : fitz.Page
            A PyMuPDF page object.

        Returns
        -------
        str
            The OCR-extracted text for the page, or an empty string if
            OCR produces no results.
        """
        import numpy as np  # deferred — only needed when OCR is invoked

        # Render page to a high-resolution pixmap (RGB, no alpha).
        zoom = _OCR_RENDER_DPI / 72  # fitz default is 72 DPI
        matrix = fitz.Matrix(zoom, zoom)
        pixmap: fitz.Pixmap = page.get_pixmap(matrix=matrix, alpha=False)

        # Convert the pixmap to a NumPy array that EasyOCR can consume.
        # Pixmap.samples is a raw bytes buffer in RGB order.
        img_array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, 3
        )

        reader = self._get_ocr_reader()
        results = reader.readtext(img_array, detail=0)  # detail=0 → text only

        ocr_text = " ".join(results).strip()
        logger.debug(
            "OCR on page %d produced %d characters.", page.number + 1, len(ocr_text)
        )
        return ocr_text

    def _recursive_text_chunker(self, text: str, page_num: int) -> list[dict]:
        """Split *text* into overlapping chunks of at most *chunk_size* chars.

        The chunker walks through the text with a sliding window.  Each
        window advances by ``chunk_size - chunk_overlap`` characters so
        that consecutive chunks share ``chunk_overlap`` characters of
        context.  This overlap helps downstream models (embeddings, LLMs)
        maintain coherence across chunk boundaries.

        Parameters
        ----------
        text : str
            The full extracted text for a single page.
        page_num : int
            1-based page number used to tag every resulting chunk.

        Returns
        -------
        list[dict]
            A list of chunk dictionaries, each with keys:
            - ``text``        – the chunk content (str)
            - ``page_num``    – source page number (int)
            - ``chunk_index`` – 0-based index of the chunk within this
              page (int)
        """
        # Nothing to chunk if the text is empty or whitespace-only.
        text = text.strip()
        if not text:
            return []

        chunks: list[dict] = []
        step = self.chunk_size - self.chunk_overlap  # how far the window moves
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()

            # Only emit non-empty chunks.
            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page_num": page_num,
                        "chunk_index": chunk_index,
                    }
                )
                chunk_index += 1

            # If we've reached or passed the end of the text, stop.
            if end >= len(text):
                break

            start += step

        logger.debug(
            "Page %d chunked into %d chunk(s).", page_num, len(chunks)
        )
        return chunks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> list[dict]:
        """Extract text from every page of a PDF document.

        For each page the method first tries PyMuPDF's native text
        extraction.  If that yields fewer than
        ``_MIN_TEXT_LENGTH_THRESHOLD`` characters the page is assumed to
        be image-heavy or scanned, and EasyOCR is used as a fallback.

        Parameters
        ----------
        pdf_bytes : bytes
            The raw bytes of the PDF file.

        Returns
        -------
        list[dict]
            One dictionary per page with keys:
            - ``page_num`` – 1-based page number (int)
            - ``text``     – extracted text (str)

        Raises
        ------
        RuntimeError
            If the PDF cannot be opened by PyMuPDF.
        """
        try:
            doc: fitz.Document = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            logger.error("Failed to open PDF: %s", exc)
            raise RuntimeError(f"Could not open the provided PDF: {exc}") from exc

        pages: list[dict] = []

        for page_index in range(len(doc)):
            page: fitz.Page = doc[page_index]
            page_num = page_index + 1  # 1-based for human readability

            # --- Primary extraction: PyMuPDF native text layer ----------
            text = page.get_text("text").strip()

            if len(text) < _MIN_TEXT_LENGTH_THRESHOLD:
                # The page likely contains scanned images or very little
                # selectable text.  Fall back to OCR.
                logger.info(
                    "Page %d has only %d chars — falling back to OCR.",
                    page_num,
                    len(text),
                )
                ocr_text = self._ocr_page(page)
                # Use OCR text only if it actually found something;
                # otherwise keep whatever PyMuPDF managed to extract.
                if len(ocr_text) > len(text):
                    text = ocr_text

            pages.append({"page_num": page_num, "text": text})
            logger.debug("Page %d: extracted %d characters.", page_num, len(text))

        doc.close()
        logger.info("Extracted text from %d page(s).", len(pages))
        return pages

    def process_pdf(self, pdf_bytes: bytes) -> list[dict]:
        """End-to-end ingestion: extract text then chunk it.

        This is the **main entry-point** for Phase 1.  It orchestrates
        the full pipeline:

        1. Extract text from every PDF page (with OCR fallback).
        2. Split each page's text into overlapping chunks.
        3. Return the flat list of annotated chunks.

        Parameters
        ----------
        pdf_bytes : bytes
            The raw bytes of the PDF file.

        Returns
        -------
        list[dict]
            A flat list of chunk dictionaries, each containing:
            - ``text``        – the chunk content (str)
            - ``page_num``    – 1-based source page number (int)
            - ``chunk_index`` – 0-based index of the chunk within its
              source page (int)
        """
        logger.info("Starting PDF ingestion pipeline…")

        # Step 1 — extract raw text per page.
        pages = self.extract_text_from_pdf(pdf_bytes)

        # Step 2 — chunk each page's text with overlap.
        all_chunks: list[dict] = []
        for page_data in pages:
            page_chunks = self._recursive_text_chunker(
                page_data["text"], page_data["page_num"]
            )
            all_chunks.extend(page_chunks)

        logger.info(
            "Ingestion complete: %d total chunk(s) from %d page(s).",
            len(all_chunks),
            len(pages),
        )
        return all_chunks
