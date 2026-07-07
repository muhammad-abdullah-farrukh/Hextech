"""Deterministic cleanup — pure Python, runs before any LLM call.

Order matters and is fixed in `clean_extraction`:
  1. markdown-artifact scrub (strip inline styling native extraction leaves)
  2. boilerplate strip   (page-level: repeated headers/footers/page numbers)
  3. near-duplicate collapse (paragraph-level: dual-read columns, OCR overlaps)
  4. sentence merge      (line-level: repairs column-edge line breaks)
  5. whitespace normalize (final pass; last so it doesn't perturb the similarity
     comparisons above it)
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

# A page number / footer: bare "3", "3/8", "3 of 8", "Page 3", "- 3 -".
_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:-\s*)?(?:page\s+)?\d+(?:\s*(?:/|of|-)\s*\d+)?\s*(?:-\s*)?$", re.I
)
# How many non-empty lines at each page edge form the header/footer band. Real
# headers/footers live here; body content (e.g. a job's date line) does not.
_EDGE_BAND = 3


def strip_repeated_boilerplate(
    pages: list[str], min_page_count: int = 2, repetition_ratio: float = 0.5
) -> list[str]:
    """Drop repeated headers/footers and page numbers — without eating body text.

    A real header/footer repeats at the SAME edge position across pages (e.g. the
    first line of every page). We key candidates by (edge, position, text), so a
    content line (e.g. `Jun 2024 — Aug 2024`) that recurs at *different* positions
    on different pages is never removed. Page-number/footer lines are dropped by
    pattern regardless of position.
    """

    def normalize(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip().lower())

    def edge_tags(rank: int, count: int, norm: str) -> list[tuple]:
        """Position tags for a line at `rank` (0-based from top) of `count` lines."""
        tags = []
        if rank < _EDGE_BAND:
            tags.append(("T", rank, norm))
        if count - 1 - rank < _EDGE_BAND:
            tags.append(("B", count - 1 - rank, norm))
        return tags

    page_nonempty = [[l for l in p.splitlines() if l.strip()] for p in pages]

    boilerplate: set[tuple] = set()
    if len(pages) >= min_page_count:
        counts: Counter[tuple] = Counter()
        for lines in page_nonempty:
            for rank, line in enumerate(lines):
                counts.update(edge_tags(rank, len(lines), normalize(line)))
        threshold = max(2, int(len(pages) * repetition_ratio))
        boilerplate = {tag for tag, c in counts.items() if c >= threshold}

    def kept(lines: list[str]) -> list[str]:
        n = sum(1 for l in lines if l.strip())
        out: list[str] = []
        rank = 0
        for line in lines:
            if not line.strip():
                out.append(line)
                continue
            drop = _PAGE_NUMBER_RE.match(line.strip()) or any(
                t in boilerplate for t in edge_tags(rank, n, normalize(line))
            )
            rank += 1
            if not drop:
                out.append(line)
        return out

    return ["\n".join(kept(p.splitlines())) for p in pages]


# A line-break tag inside a (table) cell -> becomes a space, not a join.
_BR_RE = re.compile(r"<br\s*/?>", re.I)
_INLINE_MARKUP_RE = re.compile(r"~~|</?(?:u|sup|sub|em|strong|b|i|span)>", re.I)
# A "City, Country" line wrongly tagged as a heading (e.g. "### Islamabad, Pakistan").
_LOC_HEADING_RE = re.compile(
    r"^#{1,6}\s+([A-Z][A-Za-z.\-]+(?:\s[A-Za-z.\-]+)*,\s*[A-Z][A-Za-z.\-]+)\s*$", re.M
)


def strip_markdown_artifacts(text: str) -> str:
    """Remove inline styling native extraction leaves that isn't real content.

    pymupdf4llm renders coloured/styled PDF headings as strikethrough (`~~x~~`),
    wraps links/superscripts in `<u>`/`<sup>` tags, uses `<br>` for line wraps
    inside table cells, and sometimes tags a location line as a heading. These are
    noise that confuse the LLM; the wrapped text itself is kept.
    """
    text = _BR_RE.sub(" ", text)
    text = _INLINE_MARKUP_RE.sub("", text)
    text = _LOC_HEADING_RE.sub(r"\1", text)
    return text


def dedupe_near_identical_blocks(text: str, similarity_threshold: float = 0.92) -> str:
    """Collapse paragraph blocks that are near-duplicates of a recent block."""
    blocks = [b for b in text.split("\n\n") if b.strip()]
    kept: list[str] = []
    for block in blocks:
        is_dup = any(
            SequenceMatcher(None, block, k).ratio() >= similarity_threshold
            for k in kept[-5:]
        )
        if not is_dup:
            kept.append(block)
    return "\n\n".join(kept)


# Bullet glyphs seen from either extractor (raw • / ● / ▪ or markdown - / *).
_BULLET_CHARS = "-*•●▪‣"
_BULLET_ONLY = {c for c in _BULLET_CHARS}
# Words that never end a heading but routinely end a column-wrapped prose line —
# used to rejoin a capitalized continuation without swallowing a real heading.
_WRAP_TAIL_WORDS = {
    "and", "or", "with", "the", "a", "an", "of", "in", "to", "for", "on", "at",
    "by", "from", "using", "via", "into", "as",
}


def _starts_bullet(s: str) -> bool:
    """True if `s` begins a new bullet item (glyph, or '- '/'* ' markdown)."""
    return s[:1] in "•●▪‣" or s[:2] in ("- ", "* ") or s in _BULLET_ONLY


def _looks_continued(prev: str) -> bool:
    """True if `prev` was cut off mid-phrase and expects the next line to follow.

    Conservative on purpose: a following *heading* never ends a comma/slash/hyphen,
    leaves a paren open, or dangles on a function word — so this won't merge one in.
    """
    if not prev or prev[-1] in ".!?:;":
        return False
    if prev[-1] in ",/-":
        return True
    if prev.count("(") > prev.count(")"):
        return True
    last = re.split(r"\s+", prev)[-1].strip("*_").lower()
    return last in _WRAP_TAIL_WORDS


def merge_split_sentences(text: str) -> str:
    """Re-join lines broken mid-sentence at column edges.

    Repairs three defects seen when native extraction loses block structure:
      * a bullet glyph stranded alone on its own line, its text on the next line;
      * a bullet/prose line hard-wrapped onto the following line;
    while never merging into a heading (`#`) or table (`|`) row, and never pulling
    up a line that itself starts a new bullet/heading. Fully double-spaced (clean)
    input has a blank line between every block, so this is a no-op there.
    """
    lines = text.split("\n")
    merged: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not merged or not stripped:
            merged.append(line)
            continue

        prev = merged[-1]
        prev_stripped = prev.strip()

        # (a) Previous line is just a stranded bullet marker: pull this line up.
        if prev_stripped in _BULLET_ONLY:
            merged[-1] = prev.rstrip() + " " + stripped
            continue

        # (b) Treat this line as a continuation of an unfinished previous line.
        prev_open = bool(prev_stripped) and not prev_stripped.startswith(("#", "|"))
        cur_new_block = stripped.startswith(("#", "|")) or _starts_bullet(stripped)
        is_continuation = (
            prev_open
            and not cur_new_block
            and (stripped[0].islower() or _looks_continued(prev_stripped))
        )
        if is_continuation:
            merged[-1] = prev.rstrip() + " " + stripped
        else:
            merged.append(line)
    return "\n".join(merged)


def normalize_whitespace(text: str) -> str:
    """Normalize bullets and collapse excess blank lines / trailing spaces."""
    text = re.sub(r"[•●▪]", "-", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def clean_extraction(pages: list[str]) -> str:
    """Run the full cleanup pipeline over per-page text and return one string."""
    pages = [strip_markdown_artifacts(p) for p in pages]
    pages = strip_repeated_boilerplate(pages)
    text = "\n\n".join(pages)
    text = dedupe_near_identical_blocks(text)
    text = merge_split_sentences(text)
    text = normalize_whitespace(text)
    return text
