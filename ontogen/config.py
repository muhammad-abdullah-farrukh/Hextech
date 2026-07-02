import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR          = ROOT / "data"
DOCS_DIR          = DATA_DIR / "documents"
WIKIDATA_DIR      = DATA_DIR / "wikidata"
EMBEDDINGS_DIR    = ROOT / "embeddings"
OUTPUTS_DIR       = ROOT / "outputs"

WIKIDATA_RAW        = WIKIDATA_DIR / "properties_raw.json"
WIKIDATA_FILTERED   = WIKIDATA_DIR / "properties_filtered.json"
WIKIDATA_EMBEDDINGS = EMBEDDINGS_DIR / "wikidata_embeddings.npy"

# Gazetteer JSON files (companies, universities, certifications, skills, job_titles)
GAZETTEER_DIR = DATA_DIR / "gazetteers"
# EDC canonical relation store (persists across documents)
CANON_STORE_DIR = DATA_DIR / "canon_store"
# ── LLM ────────────────────────────────────────────────────────────────────
# OpenAI-compatible endpoint (llama.cpp / llama-server) serving deepseek-r1-32b.
# stages/llm.py talks to {LLM_BASE_URL}/chat/completions and strips deepseek-r1
# <think> blocks. Same model family as ocr_resume_parser; per-call max_tokens is
# set by each stage (server context window is 24576).

LLM_PROVIDER = "openai-compatible"

LLM_BASE_URL = "http://192.168.3.76:8080/v1"

LLM_MODEL = "deepseek-r1-32b-q4"

LLM_TEMPERATURE = 0.0

LLM_API_KEY = "not-needed"  # the server ignores this; the client sends it anyway

# ── Database (shared with ocr_resume_parser) ────────────────────────────────
# Same Postgres instance/URL the parser uses (see ocr-resume-paser/.env).
DATABASE_URL = os.environ.get("DATABASE_URL")
# ── Embedding model (Stage 5) ──────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-small-en"   # verbatim from paper
EMBED_DIM   = 384                   # bge-small-en dimension; matches vector(384) columns

# ── Stage 1: Competency Question generation ───────────────────────────────
# We don't pre-guess how many CQs a document "should" have — no formula
# (word count, entity count, etc.) actually knows that before the model
# reads the document. The LLM decides count based on what's actually in
# the document. CQ_SAFETY_MAX is purely a ceiling to stop a malformed/huge
# doc from generating an unbounded number of CQs (and unbounded downstream
# LLM calls in stage 2/3) — it is not fed to the model as a target.
CQ_SAFETY_MAX = 40

# ── Pipeline modes ─────────────────────────────────────────────────────────
# True  → no-schema-constraint mode  (new props added if no Wikidata match)
# False → target-schema-constrained  (discard unmatched props)
SCHEMA_EXPANSION = True

# ── Wikidata allowed datatypes (Stage 4) ───────────────────────────────────
ALLOWED_DATATYPES = {
    "wikibase-item",
    "quantity",
    "string",
    "monolingualtext",
    "time",
}

# ── Stage 5/6: top-k Wikidata candidate retrieval ──────────────────────────
# Validate top-k nearest Wikidata neighbours in rank order; return first match.
# Set to 3 (default) or 5 (higher recall, more LLM calls) experimentally.
TOP_K_CANDIDATES = 3

# ── Stage 9: context-window guard ──────────────────────────────────────────
# Rough ceiling on the Stage 9 prompt in characters (~4 chars per token).
# At ~6 k tokens input the 8B model still has headroom for 2 k output tokens
# on a typical 8 k context window.
# DeepSeek-R1-70B served through vLLM.
# Maximum context length reported by server: 6144 tokens.
MAX_PROMPT_CHARS = 15000

# ── EDC relation canonicalization ──────────────────────────────────────────
# Top-k canon store candidates to validate before declaring a relation novel.
CANON_TOP_K = 3

# ── Entity resolution ───────────────────────────────────────────────────────
# Enable entity resolution post-processing on generated KG Turtle (Stage 9).
ENTITY_RESOLUTION_ENABLED = True

# Map Wikidata property labels (PascalCase) → entity type for entity resolver.
# Used to infer the entity type of wd: URIs from the predicate that uses them.
PROPERTY_ENTITY_TYPE_MAP: dict[str, str] = {
    # Employment
    "WorkedAt":        "company",
    "Employer":        "company",
    "EmployedBy":      "company",
    "WorksFor":        "company",
    "WorkPlace":       "company",
    # Education
    "EducatedAt":      "university",
    "AlumniOf":        "university",
    "Education":       "university",
    "DegreeFrom":      "university",
    "StudiedAt":       "university",
    "GraduatedFrom":   "university",
    # Skills
    "HasSkill":        "skill",
    "Skill":           "skill",
    "KnowledgeOf":     "skill",
    "TechnicalSkill":  "skill",
    "ProgrammingLanguage": "skill",
    "Technology":      "skill",
    "Tool":            "skill",
    "Framework":       "skill",
    # Certifications
    "Certification":       "certification",
    "HasCertification":    "certification",
    "Certified":           "certification",
    "License":             "certification",
    # Job titles
    "JobTitle":        "job_title",
    "Occupation":      "job_title",
    "Position":        "job_title",
    "Role":            "job_title",
    "Title":           "job_title",
    # Languages / references / activities (structured Path A; kept here so a
    # future Path B predicate of the same name types consistently).
    "SpeaksLanguage":  "language",
    "speaksLanguage":  "language",
    "HasReference":    "person",
    "hasReference":    "person",
    "ParticipatedIn":  "activity",
    "participatedIn":  "activity",
    "MemberOf":        "organization",
}