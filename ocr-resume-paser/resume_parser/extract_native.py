"""Native-text extraction via PyMuPDF4LLM.

`page_chunks=True` returns each page as its own dict so cross-page boilerplate
stripping (cleanup step) has page boundaries to work with.

Page-drop guard: pymupdf4llm's per-page image/OCR heuristic occasionally
mis-classifies a dense (e.g. multi-column) page as a scan, runs Tesseract on it,
and returns a near-empty chunk — silently dropping a page whose embedded text
layer is perfectly readable. We cross-check each chunk against PyMuPDF's plain
`get_text("text")` and fall back to the plain text when the markdown is a small
fraction of it. Healthy pages score md/plain ≈ 0.98–1.05; a dropped page scores
~0.0, so the 0.5 threshold sits in a wide gap. The MIN_PLAIN_CHARS floor keeps a
legitimately short page (a one-line title page) from ever tripping the guard.
"""

from __future__ import annotations

import fitz  # PyMuPDF
import pymupdf4llm

# A page's markdown is treated as "dropped" when it is shorter than this fraction
# of the page's plain text layer AND the plain layer has real content.
_DROP_RATIO = 0.5
_MIN_PLAIN_CHARS = 200


def extract_native(pdf_path: str) -> list[str]:
    """Extract a native-text PDF to a list of per-page markdown strings.

    Falls back to PyMuPDF's plain-text extraction for any page pymupdf4llm
    dropped (see module docstring), so a page's embedded text layer is never
    silently lost.
    """
    chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    pages = [chunk["text"] for chunk in chunks]

    doc = fitz.open(pdf_path)
    try:
        for i in range(min(len(pages), doc.page_count)):
            plain = doc[i].get_text("text").strip()
            md = (pages[i] or "").strip()
            if len(plain) > _MIN_PLAIN_CHARS and len(md) < _DROP_RATIO * len(plain):
                pages[i] = plain
    finally:
        doc.close()

    return pages
