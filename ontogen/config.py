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
# OpenAI-compatible endpoint (llama.cpp / llama-server), local, serving
# deepseek-r1-32b. stages/llm.py talks to {LLM_BASE_URL}/chat/completions.
# NOTE: this server's context window is 8192 (n_ctx), NOT 24576 — 3x smaller
# than the remote box used earlier. All current per-call max_tokens values
# (up to 3000 for stage7_8_ontology's Turtle generation) plus their prompt
# sizes (measured ~600-700 tokens worst case) stay comfortably under 8192, so
# nothing needed changing — but re-check this comment before raising any
# max_tokens value further while this server is the active endpoint.
LLM_PROVIDER = "openai-compatible"

LLM_BASE_URL = "http://127.0.0.1:9000/v1"

LLM_MODEL = "/home/faryal/models/deepseek-r1/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"

LLM_TEMPERATURE = 0.0

LLM_API_KEY = "not-needed"  # the server ignores this; the client sends it anyway

# ── Database (shared with ocr_resume_parser) ────────────────────────────────
# Same Postgres instance/URL the parser uses (see ocr-resume-paser/.env).
DATABASE_URL = os.environ.get("DATABASE_URL")
# ── Embedding model (Stage 5) ──────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-small-en"   # verbatim from paper
EMBED_DIM   = 384                   # bge-small-en dimension; matches vector(384) columns

# ── EDC verify gate (Step 4) ────────────────────────────────────────────────
# Deterministic pre-filter that replaces most per-candidate reasoning-LLM
# calls. Two tiers, in order (see stages/verify_gate.py):
#
#   1. Token containment + BENIGN_SUFFIXES: catches the "modifier changes
#      meaning" failure mode that dense similarity cannot (e.g. 'employer' vs
#      'current employer' — high cosine, NOT equivalent) — words like
#      'number'/'address' just describe the value's format and don't change
#      what's being asked; anything else extra is treated as meaning-changing.
#      An earlier attempt to solve this with a pretrained NLI cross-encoder
#      failed calibration entirely (merge/reject scores both clustered near
#      zero — entailment is the wrong relationship for "same meaning", and the
#      label:definition input format is out-of-distribution for NLI models).
#   2. bge cosine bands: only reached when containment doesn't apply. Pairs
#      scoring in the ambiguous middle ESCALATE to the LLM (_llm_verify).
#
# VERIFY_TAU_HI/LO below are bootstrap fallbacks only, validated against 70
# backfilled deepseek verdicts (0 false merges, 0 false rejects, 47% coverage
# at these values) — NOT hand-tuned constants to trust indefinitely. The live
# values come from the verify_thresholds table (latest row), recomputed by
# scripts/recompute_thresholds.py only once enough new verdicts accumulate
# (MIN_SAMPLES_FOR_RECOMPUTE) AND the same precision bar is met — see that
# script. These constants are used only if that table is empty (fresh install).
VERIFY_TAU_HI = 0.94                 # cosine ≥ this → MERGE without the LLM
VERIFY_TAU_LO = 0.85                 # cosine ≤ this → REJECT without the LLM
MIN_SAMPLES_FOR_RECOMPUTE = 500       # verify_verdicts rows required before
                                       # recompute_thresholds.py will move
                                       # tau_hi/tau_lo off the current values —
                                       # "hundreds to thousands", not a handful
BENIGN_SUFFIXES = {                   # extra words that don't change meaning
    "address", "number", "name", "id", "code", "type", "value",
}
# Ambiguous-band fallback: "llm" = escalate to _llm_verify (default);
# "reject" = conservative offline mode (never merge on uncertainty) so the
# pipeline stays runnable with the LLM server down.
VERIFY_ESCALATION = "llm"

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
}