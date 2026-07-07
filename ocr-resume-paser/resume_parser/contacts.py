"""Deterministic backfill for contact fields (email / phone).

Regular, regex-extractable fields like email and phone are more reliably found by
a pattern than by the LLM (DeepSeek-R1 consistently drops the phone even though
it is in the source). After the LLM step, `backfill_contacts` fills any contact
field the model left empty, matched by field name from the field_spec so the
pipeline stays schema-driven. It never overwrites a value the model did provide.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# A phone-like run: optional +, digits and common separators. Validated by digit
# count afterwards so stray numbers (years, house numbers) don't match.
_PHONE_RE = re.compile(r"\+?\d[\d\s().\-]{6,}\d")
_MIN_PHONE_DIGITS = 7
_MAX_PHONE_DIGITS = 15
# Profile links: capture the canonical "site.com/path" (drop scheme/www/markup).
_LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?(linkedin\.com/[A-Za-z0-9_%/\-]+)", re.I)
_GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?(github\.com/[A-Za-z0-9_.\-]+)", re.I)


def extract_email(text: str) -> str | None:
    m = EMAIL_RE.search(text)
    return m.group(0) if m else None


def extract_linkedin(text: str) -> str | None:
    m = _LINKEDIN_RE.search(text)
    return m.group(1).rstrip("/.,)") if m else None


def extract_github(text: str) -> str | None:
    m = _GITHUB_RE.search(text)
    return m.group(1).rstrip("/.,)") if m else None


# Date of birth: labelled, e.g. "Dob: 22nd Sep 2003" or "D.O.B - 05/09/2003".
_DOB_RE = re.compile(
    r"(?:d\.?o\.?b\.?|date\s+of\s+birth)\s*[:\-]?\s*"
    r"(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,}\.?\s+\d{4}|\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})",
    re.I,
)
# Candidate location, tried in order: an "Address:"/"Location:" label, a street
# address ("House No. ..."), then a "City, Country" at the start of the contact line.
_ADDR_LABEL_RE = re.compile(
    r"(?:address|location)\s*[:\-]\s*(.+?)"
    r"(?:\s+(?:phone|e-?mail|linkedin|github|tel|mobile|dob|date of birth)\b|$)",
    re.I,
)
_STREET_RE = re.compile(r"(house\s*no[.,]?\s*\d+.*?,\s*[A-Z][a-z]+,\s*[A-Z][a-z]+)", re.I)
_CITY_RE = re.compile(r"^([A-Z][A-Za-z.\-]+(?:\s[A-Za-z.\-]+)*,\s*[A-Z][A-Za-z.\-]+)\s*(?:\||$)")


def _demarkup(text: str) -> str:
    """Drop bold/italic markers and collapse spaces (newlines kept)."""
    return re.sub(r"[ \t]+", " ", re.sub(r"[*_]", "", text))


def _tidy(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def extract_dob(text: str) -> str | None:
    m = _DOB_RE.search(_demarkup(text))
    return _tidy(m.group(1)) if m else None


def extract_location(text: str) -> str | None:
    t = _demarkup(text)
    for rx in (_ADDR_LABEL_RE, _STREET_RE):
        m = rx.search(t)
        if m:
            return _tidy(m.group(1))
    for line in t.splitlines():  # first line carrying an email = the contact line
        if "@" in line:
            m = _CITY_RE.match(line.strip())
            return _tidy(m.group(1)) if m else None
    return None


def extract_phone(text: str) -> str | None:
    for m in _PHONE_RE.finditer(text):
        candidate = m.group(0).strip()
        digits = sum(c.isdigit() for c in candidate)
        if _MIN_PHONE_DIGITS <= digits <= _MAX_PHONE_DIGITS:
            return candidate
    return None


def _is_empty(value: object) -> bool:
    return value is None or value == "" or value == []


def _top_level_field_names(field_spec: list[dict]) -> list[str]:
    return [f["name"] for f in field_spec]


def backfill_contacts(
    result: dict, source_text: str, field_spec: list[dict]
) -> dict:
    """Fill empty email/phone top-level fields from regex matches on the source.

    Only acts on fields that exist in `field_spec`, are string-typed, and were
    left empty by the model. Returns the same dict (mutated) for convenience.
    """
    names = _top_level_field_names(field_spec)
    string_fields = {
        f["name"] for f in field_spec if f.get("type", "string") == "string"
    }

    for name in names:
        if name not in string_fields or not _is_empty(result.get(name)):
            continue
        lname = name.lower()
        value = None
        if "email" in lname:
            value = extract_email(source_text)
        elif any(k in lname for k in ("phone", "mobile", "tel", "contact_number")):
            value = extract_phone(source_text)
        elif "linkedin" in lname:
            value = extract_linkedin(source_text)
        elif "github" in lname:
            value = extract_github(source_text)
        elif "location" in lname or "address" in lname:
            value = extract_location(source_text)
        elif "birth" in lname or "dob" in lname:
            value = extract_dob(source_text)
        if value:
            result[name] = value
            logger.info("backfilled contact field %r from source", name)

    return result
