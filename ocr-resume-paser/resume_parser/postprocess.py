"""Deterministic post-extraction cleanup passes (run in ``normalize._finalize``).

Each pass is schema-driven — it locates its target field by name/type in the
``field_spec`` rather than assuming a fixed key, mirroring ``backfill_contacts`` /
``backfill_experience`` — and is individually toggleable via ``Settings`` so a
misbehaving pass can be disabled without a code change. All passes are
deterministic, preserving the pipeline's reproducibility guarantee (embeddings
are deterministic for a fixed local model; RapidFuzz is deterministic).

Passes:
  * ``dedupe_projects``      — merge near-duplicate projects (embedding cosine)   [#3]
  * ``filter_skills``        — drop non-atomic / project-name / boilerplate skills [#6]
  * ``validate_metrics``     — keep only quantitative metrics; reroute the rest    [#4]
  * ``dedupe_cert_activity`` — drop activities that duplicate a certification      [#5]
"""

from __future__ import annotations

import itertools
import logging
import re

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


# --- field-spec locating helpers ------------------------------------------------

def _find_field(field_spec: list[dict], predicate):
    return next((f for f in field_spec if predicate(f)), None)


def _array_object_field(field_spec: list[dict], name_substr: str) -> dict | None:
    """First ``array<object>`` field whose name contains ``name_substr``."""
    return _find_field(
        field_spec,
        lambda f: f.get("type") == "array"
        and f.get("items") == "object"
        and name_substr in f["name"].lower(),
    )


def _prop_key(field: dict | None, substr: str) -> str | None:
    """Name of the first sub-property whose name contains ``substr``, else None."""
    return next(
        (p["name"] for p in (field or {}).get("properties", []) if substr in p["name"].lower()),
        None,
    )


def _dedupe_preserve(values) -> list:
    """De-duplicate while preserving first-seen order."""
    seen: list = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return seen


def prune_empty_strings(obj):
    """Recursively drop dict keys whose value is an empty/whitespace-only string.

    The model sometimes emits "" instead of null for absent optional fields (e.g.
    a certification's issuer/year), which model_dump(exclude_none) doesn't catch.
    Keeps 0/False/empty-list (those are meaningful or handled elsewhere).
    """
    if isinstance(obj, dict):
        return {
            k: prune_empty_strings(v)
            for k, v in obj.items()
            if not (isinstance(v, str) and not v.strip())
        }
    if isinstance(obj, list):
        return [prune_empty_strings(v) for v in obj]
    return obj


# --- #3 project dedup-merge -----------------------------------------------------

_EMBED_CACHE: dict[str, object] = {}


def _get_embedder(model_name: str):
    """Load (and cache) a sentence-transformers model. Lazy: heavy torch import."""
    model = _EMBED_CACHE.get(model_name)
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        _EMBED_CACHE[model_name] = model
    return model


def _cluster_by_similarity(vectors, threshold: float) -> list[list[int]]:
    """Union-find clustering of row-normalized vectors by cosine >= threshold.

    Returns index groups, each ascending, ordered by their first (smallest) index
    so the merged output preserves the original project ordering.
    """
    import numpy as np

    n = len(vectors)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    sim = np.asarray(vectors) @ np.asarray(vectors).T
    for i, j in itertools.combinations(range(n), 2):
        if float(sim[i, j]) >= threshold:
            union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for _, g in sorted(groups.items())]


def _merge_project_group(
    members: list[dict], name_k: str | None, desc_k: str | None, list_keys
) -> dict:
    """Merge a cluster of project dicts: keep the first name, the longest
    description, and the union (order-preserving) of list fields like
    technologies/metrics."""
    base = dict(members[0])  # earliest entry -> keeps its name + scalar fields
    if desc_k:
        best = max(members, key=lambda p: len(p.get(desc_k) or ""))
        if best.get(desc_k):
            base[desc_k] = best[desc_k]
    for key in list_keys:
        if not key:
            continue
        merged = _dedupe_preserve(v for p in members for v in (p.get(key) or []))
        if merged:
            base[key] = merged
    return base


def dedupe_projects(
    result: dict,
    field_spec: list[dict],
    *,
    model_name: str = "BAAI/bge-small-en",
    threshold: float = 0.90,
) -> dict:
    """Merge near-duplicate projects (same work described in two resume sections).

    Embeds ``name + description`` per project and merges clusters whose pairwise
    cosine is >= ``threshold``. No-op if the projects field can't be found, there
    are < 2 entries, or the embedder can't be loaded (logged, then skipped).
    """
    pf = _array_object_field(field_spec, "project")
    if not pf:
        return result
    projects = result.get(pf["name"])
    if not isinstance(projects, list) or len(projects) < 2:
        return result

    name_k = _prop_key(pf, "name")
    desc_k = _prop_key(pf, "description")
    tech_k = _prop_key(pf, "tech")
    metric_k = _prop_key(pf, "metric")

    texts = [
        f"{(p.get(name_k) or '') if name_k else ''}. "
        f"{(p.get(desc_k) or '') if desc_k else ''}".strip()
        for p in projects
    ]
    try:
        vectors = _get_embedder(model_name).encode(texts, normalize_embeddings=True)
    except Exception as exc:  # noqa: BLE001 - embedder missing/unloadable -> skip
        logger.warning("project dedup skipped (embedder unavailable): %s", exc)
        return result

    groups = _cluster_by_similarity(vectors, threshold)
    if len(groups) == len(projects):
        return result  # nothing to merge

    result[pf["name"]] = [
        _merge_project_group([projects[i] for i in g], name_k, desc_k, (tech_k, metric_k))
        for g in groups
    ]
    logger.info("deduped projects: %d -> %d", len(projects), len(groups))
    return result


# --- work-history role/company guard --------------------------------------------

# City, Country / City, ST style — used to catch a location misparsed as a company.
_LOCATION_RE = re.compile(r"^[A-Z][A-Za-z.\- ]+,\s*[A-Z][A-Za-z.\- ]+$")


def _is_self_employed(s: str) -> bool:
    low = s.lower()
    return "self-employ" in low or _norm_skill(s) in {
        "self employed",
        "selfemployed",
        "self",
    }


def _looks_freelance(s: str) -> bool:
    low = s.lower()
    return any(k in low for k in ("freelance", "independent", "consultant", "self-employ"))


def normalize_work_roles(result: dict, field_spec: list[dict]) -> dict:
    """Repair company/title mapping the LLM commonly gets wrong (see #C).

    - role/company reversed for a self-employed line -> put the role in title,
      'Self-employed' in company;
    - company duplicated from title on a freelance role -> set company 'Self-employed';
    - a location misparsed as company -> move it to the location field (if the
      schema has one) and clear company;
    - an entry whose company still EQUALS its title after the above (a project
      role-header like 'Machine Learning Engineer' the LLM misfiled as a job) is
      dropped — a real job never has company == title.
    Only touches clear, safe cases; leaves anything ambiguous untouched.
    """
    wf = _array_object_field(field_spec, "work") or _array_object_field(field_spec, "employ")
    if not wf:
        return result
    entries = result.get(wf["name"])
    if not isinstance(entries, list):
        return result
    title_k = _prop_key(wf, "title")
    company_k = _prop_key(wf, "company")
    loc_k = _prop_key(wf, "location")
    if not title_k or not company_k:
        return result

    for e in entries:
        if not isinstance(e, dict):
            continue
        title = e.get(title_k)
        company = e.get(company_k)
        if not isinstance(title, str) or not isinstance(company, str):
            continue
        # Location misparsed as company (e.g. "Topi, Pakistan").
        if _LOCATION_RE.match(company.strip()) and not _LOCATION_RE.match(title.strip()):
            if loc_k and not e.get(loc_k):
                e[loc_k] = company
            e[company_k] = "Self-employed" if _looks_freelance(title) else None
            if e[company_k] is None:
                e.pop(company_k, None)
            continue
        # Reversed: the self-employed marker landed in title.
        if _is_self_employed(title) and not _is_self_employed(company):
            e[title_k], e[company_k] = company, "Self-employed"
            continue
        # Duplicated: same string in both, and the role reads as freelance.
        if _norm_skill(title) == _norm_skill(company) and _looks_freelance(title):
            e[company_k] = "Self-employed"

    # Drop leftover entries whose company still equals its title (project
    # role-headers the LLM misfiled as jobs, e.g. 'Machine Learning Engineer').
    def _is_spurious(e: dict) -> bool:
        t, c = e.get(title_k), e.get(company_k)
        return isinstance(t, str) and isinstance(c, str) and _norm_skill(t) == _norm_skill(c)

    kept = [e for e in entries if not (isinstance(e, dict) and _is_spurious(e))]
    if len(kept) != len(entries):
        logger.info("dropped %d spurious work entries (company == title)", len(entries) - len(kept))
    result[wf["name"]] = kept
    return result


# --- #6 skills filter -----------------------------------------------------------

# Generic process/outcome nouns that leak in as "skills" from prose. Conservative
# on purpose (near-zero false-positive risk for real skill lists); extend freely.
_SKILL_BLOCKLIST = {
    "measurable business impact",
    "strategic improvements",
    "ongoing maintenance",
    "feasibility scoping",
    "production release",
    "stakeholder requirements",
    "data ingestion",
    "model training",
    "real-time asl predictions",
    "execution costs",
    "time complexity",
    "heap memory usage",
    "python set intersections",
}


def _norm_skill(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s.lower())).strip()


# --- skills backfill from the explicit skills section(s) [#F] --------------------

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
_SKILL_HEADING_RE = re.compile(
    r"technical\s+skills|areas?\s+of\s+expertise|core\s+competenc|\bskills\b|"
    r"technolog|software|\btools\b",
    re.I,
)


def _split_top_level(s: str, seps: str = ",;|") -> list[str]:
    """Split on separators that are NOT inside parentheses/brackets, so
    'Hypothesis Testing (t-test, Chi-squared, ANOVA)' stays one token."""
    parts: list[str] = []
    depth = 0
    cur = ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if ch in seps and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def _skill_section_lines(text: str) -> list[str]:
    """Body lines that sit under a skills-type heading (until the next heading)."""
    out: list[str] = []
    in_section = False
    for line in text.splitlines():
        if _HEADING_RE.match(line):
            heading = re.sub(r"[#*~_`]", "", line).strip()
            in_section = bool(_SKILL_HEADING_RE.search(heading)) and len(heading) < 40
            continue
        if in_section and line.strip():
            out.append(line.strip())
    return out


_LANG_HEADING_RE = re.compile(r"\blanguages?\b", re.I)


def backfill_languages(result: dict, source_text: str, field_spec: list[dict]) -> dict:
    """Populate the (human) languages field from the LANGUAGES section if the model
    left it empty — regression guard for when a large schema makes the LLM drop it.

    Only fires for an empty/missing field, and never touches 'Programming Languages'.
    """
    lf = _find_field(
        field_spec,
        lambda f: f.get("type") == "array"
        and f.get("items") == "object"
        and "language" in f["name"].lower(),
    )
    if not lf:
        return result
    if isinstance(result.get(lf["name"]), list) and result[lf["name"]]:
        return result  # model already populated it
    name_k = _prop_key(lf, "name")
    prof_k = _prop_key(lf, "prof")
    if not name_k:
        return result

    section: list[str] = []
    in_sec = False
    for line in source_text.splitlines():
        if _HEADING_RE.match(line):
            h = re.sub(r"[#*~_`]", "", line).strip()
            in_sec = (
                bool(_LANG_HEADING_RE.search(h))
                and "programming" not in h.lower()
                and len(h) < 40
            )
            continue
        if in_sec and line.strip():
            section.append(line.strip())
    if not section:
        return result

    text = re.sub(r"[*_`\-]", " ", " ".join(section))
    langs: list[dict] = []
    for m in re.finditer(r"([A-Za-z][A-Za-z /&.+-]*?)\s*\(([^)]+)\)", text):
        name = m.group(1).strip(" ,;")
        prof = re.sub(r"\s*level\s*$", "", m.group(2).strip(), flags=re.I).strip()
        if name:
            langs.append({name_k: name, **({prof_k: prof} if prof_k and prof else {})})
    if not langs:  # no "(proficiency)" -> take bare comma-separated names
        for part in re.split(r"[,;]", text):
            name = part.strip(" .")
            if name and re.fullmatch(r"[A-Za-z][A-Za-z /&.+-]*", name):
                langs.append({name_k: name})
    if langs:
        result[lf["name"]] = langs
        logger.info("backfilled %d languages from the LANGUAGES section", len(langs))
    return result


_PRESENTISH = {"present", "current", "ongoing", "now", "to date", "till date"}


def drop_unsupported_project_dates(result: dict, field_spec: list[dict]) -> dict:
    """Null a project's 'present'-style end_date when it has no start_date.

    A lone 'Present' with no start is fabricated precision — the achievement-style
    projects have no dates in the source, so the model shouldn't imply an ongoing
    range. A real end date (e.g. 'May 2024') is left untouched.
    """
    pf = _array_object_field(field_spec, "project")
    if not pf:
        return result
    start_k = _prop_key(pf, "start")
    end_k = _prop_key(pf, "end")
    if not end_k:
        return result
    dropped = 0
    for p in result.get(pf["name"]) or []:
        if not isinstance(p, dict):
            continue
        end = p.get(end_k)
        start = p.get(start_k) if start_k else None
        if (
            isinstance(end, str)
            and end.strip().lower() in _PRESENTISH
            and not (isinstance(start, str) and start.strip())
        ):
            p.pop(end_k, None)
            dropped += 1
    if dropped:
        logger.info("dropped %d unsupported project 'present' end-dates (no start)", dropped)
    return result


def _canon_tech(s: str) -> str:
    """Canonical key folding trivial plurals so 'WebSocket'/'WebSockets' collapse."""
    k = _norm_skill(s)
    return k[:-1] if k.endswith("s") and len(k) > 3 else k


def canonicalize_tech(result: dict, field_spec: list[dict]) -> dict:
    """De-duplicate each project's technologies by a plural-folded key."""
    pf = _array_object_field(field_spec, "project")
    tech_k = _prop_key(pf, "tech") if pf else None
    if not tech_k:
        return result
    for p in result.get(pf["name"]) or []:
        if not isinstance(p, dict) or not isinstance(p.get(tech_k), list):
            continue
        seen: set[str] = set()
        out: list = []
        for t in p[tech_k]:
            if not isinstance(t, str):
                continue
            k = _canon_tech(t)
            if k and k not in seen:
                seen.add(k)
                out.append(t)
        p[tech_k] = out
    return result


def _extract_skill_tokens(line: str) -> list[str]:
    """Pull individual skill tokens from one skills-section line, stripping a
    leading 'Category:' label and bullet/bold markers."""
    s = re.sub(r"^[-*•\s]+", "", line).replace("**", "").strip()
    head, sep, rest = s.partition(":")
    if sep and rest.strip() and len(head) <= 60:
        s = rest.strip()
    return _split_top_level(s)


def backfill_skills(result: dict, source_text: str, field_spec: list[dict]) -> dict:
    """Union skills the model dropped back into its list, from two sources:

      1. the explicit skills section(s) in the source text, and
      2. the technologies already extracted into each project entry (these are
         skills too, and are inconsistently promoted to the top-level list).

    Guarantees a listed/used skill isn't silently dropped. Runs before
    filter_skills, which then removes any noise this introduces.
    """
    sf = _find_field(
        field_spec,
        lambda f: f.get("type") == "array"
        and f.get("items") == "string"
        and "skill" in f["name"].lower(),
    )
    if not sf:
        return result
    existing = result.get(sf["name"])
    existing = existing if isinstance(existing, list) else []
    seen = {_norm_skill(s) for s in existing if isinstance(s, str)}
    added: list[str] = []

    def _consider(tok: object) -> None:
        if not isinstance(tok, str):
            return
        n = _norm_skill(tok)
        if n and n not in seen and 1 <= len(tok.split()) <= 6:
            seen.add(n)
            added.append(tok)

    for line in _skill_section_lines(source_text):
        for tok in _extract_skill_tokens(line):
            _consider(tok)

    pf = _array_object_field(field_spec, "project")
    tech_k = _prop_key(pf, "tech") if pf else None
    if tech_k:
        for p in result.get(pf["name"]) or []:
            if isinstance(p, dict):
                for tok in p.get(tech_k) or []:
                    _consider(tok)

    if added:
        result[sf["name"]] = list(existing) + added
        logger.info("backfilled %d skills (skills section + project technologies)", len(added))
    return result


def filter_skills(
    result: dict, field_spec: list[dict], *, max_words: int = 4, project_match: int = 90
) -> dict:
    """Drop non-atomic skills: blocklisted phrases, over-long runs (> ``max_words``),
    or ones that fuzzy-match an extracted project title. Runs AFTER project dedup so
    it matches against the final titles."""
    sf = _find_field(
        field_spec,
        lambda f: f.get("type") == "array"
        and f.get("items") == "string"
        and "skill" in f["name"].lower(),
    )
    if not sf:
        return result
    skills = result.get(sf["name"])
    if not isinstance(skills, list) or not skills:
        return result

    pf = _array_object_field(field_spec, "project")
    proj_titles: list[str] = []
    if pf:
        nk = _prop_key(pf, "name")
        if nk:
            proj_titles = [
                p.get(nk)
                for p in (result.get(pf["name"]) or [])
                if isinstance(p, dict) and p.get(nk)
            ]

    # Entity names (employers/orgs/roles) that leak into skills from mangled
    # two-column lines — never skills. Exact-normalized match, plus a fuzzy match
    # for multi-word skills (catches concatenations like "Pakistan Lead Media
    # Producer" from a garbled role line).
    entity_norms: set[str] = set()
    entity_names: list[str] = []
    for fname, keys in (("work", ("company", "title")), ("activit", ("name", "organization"))):
        ef = _array_object_field(field_spec, fname)
        if not ef:
            continue
        pkeys = [_prop_key(ef, k) for k in keys]
        for e in result.get(ef["name"]) or []:
            if isinstance(e, dict):
                for pk in pkeys:
                    if pk and isinstance(e.get(pk), str) and e[pk].strip():
                        entity_norms.add(_norm_skill(e[pk]))
                        entity_names.append(e[pk].lower())
    entity_norms.discard("")

    def _is_entity(s: str) -> bool:
        if _norm_skill(s) in entity_norms:
            return True
        return len(s.split()) >= 3 and any(
            fuzz.token_set_ratio(s.lower(), e) >= 80 for e in entity_names
        )

    kept: list[str] = []
    for s in skills:
        if not isinstance(s, str) or not s.strip():
            continue
        if _norm_skill(s) in _SKILL_BLOCKLIST:
            continue
        if _is_entity(s):  # an employer/org/role name, not a skill
            continue
        # Count words outside parentheses, so a legit parenthetical skill like
        # "Hypothesis Testing (t-test, Chi-squared, ANOVA)" isn't judged over-long.
        if len(re.sub(r"\([^)]*\)", "", s).split()) > max_words:
            continue
        # Drop a skill only when it is (nearly) an ENTIRE project title. Use a
        # length-sensitive ratio and require >=3 words, so a short skill that
        # merely appears inside a long title (e.g. "CI/CD" in "WiChat — CI/CD
        # Pipeline...") is NOT mistaken for the title.
        if len(s.split()) >= 3 and any(
            fuzz.ratio(s.lower(), t.lower()) >= project_match for t in proj_titles
        ):
            continue
        kept.append(s)

    kept = _dedupe_preserve(kept)
    if len(kept) != len(skills):
        logger.info("filtered skills: %d -> %d", len(skills), len(kept))
    result[sf["name"]] = kept
    return result


# --- #4 metrics numeric validator -----------------------------------------------

_METRIC_QUANT_RE = re.compile(r"[0-9%]")


def _ensure_sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else text + "."


def validate_metrics(result: dict, field_spec: list[dict]) -> dict:
    """Keep only quantitative project metrics (containing a digit or %). Move
    qualitative phrases into the description (deduped), never drop them. An empty
    metrics list is removed rather than left as ``[]``."""
    pf = _array_object_field(field_spec, "project")
    if not pf:
        return result
    metric_k = _prop_key(pf, "metric")
    desc_k = _prop_key(pf, "description")
    if not metric_k:
        return result
    projects = result.get(pf["name"])
    if not isinstance(projects, list):
        return result

    for p in projects:
        if not isinstance(p, dict):
            continue
        metrics = p.get(metric_k)
        if not isinstance(metrics, list) or not metrics:
            continue
        valid: list[str] = []
        rerouted: list[str] = []
        for m in metrics:
            if not isinstance(m, str) or not m.strip():
                continue
            (valid if _METRIC_QUANT_RE.search(m) else rerouted).append(m.strip())
        if rerouted and desc_k:
            desc = (p.get(desc_k) or "").strip()
            # Skip a phrase already in the description, incl. near-duplicates
            # (e.g. "Optimized solar..." vs "optimizing solar..."), so the reroute
            # doesn't append redundant text.
            add = [
                r
                for r in rerouted
                if r.lower() not in desc.lower()
                and fuzz.partial_ratio(r.lower(), desc.lower()) < 88
            ]
            if add:
                p[desc_k] = " ".join([desc] + [_ensure_sentence(a) for a in add]).strip()
        if valid:
            p[metric_k] = valid
        else:
            p.pop(metric_k, None)
    return result


# --- #5 certification <-> activity dedup ----------------------------------------

def dedupe_cert_activity(result: dict, field_spec: list[dict], *, threshold: int = 88) -> dict:
    """Drop activities that duplicate a certification (fuzzy title match). The
    certification is the more specific bucket, so it wins; the activity is removed."""
    cf = _array_object_field(field_spec, "cert")
    af = _array_object_field(field_spec, "activit")
    if not cf or not af:
        return result
    certs = result.get(cf["name"])
    acts = result.get(af["name"])
    if not isinstance(certs, list) or not isinstance(acts, list) or not certs or not acts:
        return result
    ck = _prop_key(cf, "name")
    ak = _prop_key(af, "name")
    if not ck or not ak:
        return result
    cert_names = [c.get(ck) for c in certs if isinstance(c, dict) and c.get(ck)]

    kept = []
    for a in acts:
        name = a.get(ak) if isinstance(a, dict) else None
        if name and any(fuzz.token_set_ratio(name, cn) >= threshold for cn in cert_names):
            logger.info("dropped activity duplicating a certification: %r", name)
            continue
        kept.append(a)
    result[af["name"]] = kept
    return result
