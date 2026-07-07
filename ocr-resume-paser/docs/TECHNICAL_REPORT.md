# Technical Report: Resume PDF-to-JSON Extraction Pipeline

---

## 1. System Overview

This pipeline converts arbitrary resume PDFs into structured, schema-conformant JSON. The architecture is a strict sequential pipeline with three major phases: **extraction** (PDF → markdown text), **cleanup** (deterministic text normalization), and **normalization** (LLM-driven structured extraction with deterministic post-processing). All three phases are independently testable; the LLM phase can be skipped entirely for debugging.

The system is designed around a single local LLM backend (DeepSeek-R1-32B via llama.cpp), a runtime-configurable output schema (`field_spec.json`), and a self-verify/refine loop that caps total LLM calls at three per resume. Deterministic fallbacks (regex, date math) handle fields the LLM reliably drops.

---

## 2. Repository Structure

```
ocr-resume-paser/
├── config/
│   └── field_spec.json          # Runtime output schema definition
├── resume_parser/
│   ├── __init__.py
│   ├── triage.py                # PDF classification: native text vs. scanned
│   ├── extract.py               # Extraction router
│   ├── extract_native.py        # PyMuPDF4LLM path (native text layer)
│   ├── extract_marker.py        # Marker/OCR path (scanned images)
│   ├── cleanup.py               # Deterministic text normalization
│   ├── schema_builder.py        # Dynamic Pydantic model + field guide
│   ├── context.py               # Context-window budgeting
│   ├── settings.py              # Configuration (env-driven)
│   ├── llm_client.py            # LLM client factory + retry logic
│   ├── normalize.py             # LLM extraction + self-verify/refine loop
│   ├── contacts.py              # Deterministic email/phone backfill
│   ├── experience.py            # Deterministic years_experience calculator
│   ├── artifacts.py             # Intermediate artifact writer
│   ├── pipeline.py              # Top-level orchestrator
│   └── cli.py                   # argparse CLI entry point
├── tests/
│   ├── test_cleanup.py
│   ├── test_triage.py
│   ├── test_schema_builder.py
│   ├── test_context.py
│   ├── test_normalize.py
│   ├── test_contacts.py
│   └── test_experience.py
├── resumes/                     # Input PDFs (not committed)
├── artifacts/                   # Per-resume output directory (git-ignored)
├── .env                         # Local secrets and tuning (git-ignored)
├── .env.example                 # Template with empty placeholders
├── conftest.py
└── requirements.txt
```

---

## 3. Configuration Layer

**File:** `resume_parser/settings.py`

All runtime configuration is loaded from environment variables (via `python-dotenv`). The `Settings` dataclass is frozen — it is built once at startup and passed through the pipeline immutably.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint (current deployment: llama.cpp on `:8090`) |
| `LLM_MODEL` | *(required)* | Model name the server serves (current: `models/qwen3-8b-bf16.gguf`) |
| `LLM_API_KEY` | `not-needed` | Ignored by local servers |
| `INSTRUCTOR_MODE` | `JSON_SCHEMA` | Structured-output decoding mode |
| `LLM_MAX_RETRIES` | `2` | instructor re-ask attempts on validation failure |
| `LLM_RATELIMIT_ATTEMPTS` | `5` | tenacity retry attempts on transient errors |
| `LLM_HEALTH_TIMEOUT` | `2.0` | Seconds for `/models` health check |
| `LLM_REFINE_PASSES` | `2` | Max self-verify/refine passes after generate |
| `LLM_CONTEXT_WINDOW` | `24576` | Total context window (tokens) — must match server `-c` (current: `32768`) |
| `LLM_CONTEXT_SAFETY_MARGIN` | `256` | Buffer for chat-template overhead |
| `LLM_CHARS_PER_TOKEN` | `3.5` | Token estimate ratio (no tokenizer dependency) |
| `LLM_TEMPERATURE` | `0.0` | Greedy decoding for reproducible extraction (see determinism note) |
| `LLM_SEED` | *(unset)* | Fixed RNG seed for reproducible sampling; passed on every call when set |
| `LLM_DISABLE_THINKING` | `false` | Prepend `/no_think` to the system prompt for hybrid-reasoning models (Qwen3) |
| `LLM_FREQUENCY_PENALTY` | `0.15` | Suppresses repetition loops |
| `LLM_PRESENCE_PENALTY` | `0.0` | |
| `LLM_MAX_TOKENS` | `4096` | Generation cap — must match server `--n-predict` |

**Determinism:** With `LLM_TEMPERATURE=0.0` (greedy) plus a fixed `LLM_SEED`, the same cleaned text produces byte-identical JSON across runs — the fix for the earlier cross-machine/cross-run inconsistency. The seed also pins sampling if temperature is later raised.

**Deterministic post-processing toggles** (all default `true`; each pass can be disabled without a code change — see Stage 7):

| Variable | Purpose |
|---|---|
| `LLM_FIX_WORK_ROLES` | Repair company/title mapping; drop role-header pseudo-jobs |
| `LLM_BACKFILL_LANGUAGES` | Populate `languages` from the LANGUAGES section if the model drops it |
| `LLM_BACKFILL_SKILLS` | Union skills from the skills section(s) + project technologies |
| `LLM_DEDUP_PROJECTS` | Merge near-duplicate projects via embeddings |
| `LLM_FILTER_SKILLS` | Drop non-atomic / entity-name / project-name skills |
| `LLM_VALIDATE_METRICS` | Keep only quantitative project metrics; reroute the rest to description |
| `LLM_DEDUP_CERT_ACTIVITY` | Drop activities duplicating a certification |
| `LLM_EMBEDDING_MODEL` | Sentence-transformers model for project dedup (default `BAAI/bge-small-en`) |
| `LLM_DEDUP_THRESHOLD` | Cosine merge threshold for project dedup (default `0.90`) |

The mode-fallback ladder is `JSON_SCHEMA → TOOLS → JSON → MD_JSON`. If the configured mode fails (due to provider rejection or validation error), the pipeline tries each successive rung before raising.

---

## 4. Runtime Schema: `field_spec.json`

The output schema is defined entirely at runtime as a JSON array of field descriptor objects. This means the output structure can be changed by editing one file — no Python changes needed. The `schema_builder` module compiles this spec into nested Pydantic v2 models at startup.

**Current schema fields:**

| Field | Type | Required | Notes |
|---|---|---|---|
| `candidate_name` | `string` | yes | |
| `email` | `string` | yes | Deterministic regex backfill if LLM drops it |
| `phone` | `string` | no | Deterministic regex backfill if LLM drops it |
| `linkedin` | `string` | no | Deterministic regex backfill |
| `github` | `string` | no | Deterministic regex backfill |
| `location` | `string` | no | Candidate's address/location; deterministic backfill (label → street → city) |
| `date_of_birth` | `string` | no | Deterministic labelled-date backfill |
| `years_experience` | `object {years, months}` | yes | Always overwritten by deterministic date calculator |
| `skills` | `array<string>` | yes | Each element must be atomic; backfilled + filtered deterministically |
| `work_history` | `array<object>` | no | `{company, title, location, start_date, end_date}` |
| `projects` | `array<object>` | no | `{name, description, technologies[], metrics[], start_date, end_date, location}` |
| `education` | `array<object>` | no | `{institution, degree, location, gpa, start_date, end_date, graduation_year}` |
| `languages` | `array<object>` | no | `{name, proficiency}`; deterministic backfill from LANGUAGES section |
| `certifications` | `array<object>` | no | `{name, issuer, year}` |
| `activities` | `array<object>` | no | `{name, organization, date, description}` |
| `references` | `array<object>` | no | `{name, title, contact, relationship}` |

Field descriptor keys: `name`, `type`, `required`, `description`, `items` (for arrays), `properties` (for objects and array-of-objects).

> **Schema-size trade-off:** the schema is intentionally broad to capture everything. On a small (8B) model this raises recall pressure — the model can drop a field it would otherwise fill (e.g. `languages`) when many fields compete. The deterministic backfills (contacts, languages, skills) exist to compensate; the durable cure is a larger model.

---

## 5. Pipeline Stages

### Stage 1 — Triage (`triage.py`)

**Input:** PDF file path  
**Output:** `bool` (True = needs OCR)

Opens the PDF with PyMuPDF (`fitz`) and samples up to 3 pages. For each page it measures:
- Total extractable character count from the text layer
- Total image area as a fraction of page area (using `get_image_info()` which returns dicts with a `"bbox"` key in PyMuPDF ≥1.26)

A page is flagged as scanned if image area exceeds 85% of page area and text characters are below 40. A document is flagged scanned if average characters per sampled page is below 40. Either condition routes the PDF to OCR.

**Cost:** Milliseconds. No model loading.

---

### Stage 2 — Extraction (`extract.py`, `extract_native.py`, `extract_marker.py`)

**Input:** PDF file path  
**Output:** `(engine_name: str, pages: list[str])`

Two sub-paths exist, selected by triage:

**Native path (PyMuPDF4LLM):** Calls `pymupdf4llm.to_markdown(pdf_path, page_chunks=True)`. Returns each page as a separate markdown string, preserving page boundaries for the cleanup boilerplate stripper. Fast, no model weights.

**OCR path (Marker):** Loads Marker's model weights once (lazy singleton, process-scoped) via `create_model_dict()`. Marker's `PdfConverter` renders the full document to markdown in one call, returned as a single-element list. Model weights are expensive to load (~several seconds on first call); subsequent calls reuse the cached converter. A `warm_marker()` function is exposed for eager loading during server startup.

---

### Stage 3 — Cleanup (`cleanup.py`)

**Input:** `pages: list[str]`  
**Output:** `clean_text: str`

Five deterministic transformations applied in fixed order:

1. **Markdown-artifact scrub** — removes inline styling that native extraction leaves but isn't content: strikethrough (`~~x~~`), `<u>`/`<sup>`/`<sub>`/`<em>`/`<strong>` tags (text kept), and `<br>` table-cell wraps (→ space). Also demotes a `City, Country` line wrongly tagged as a heading (`### Islamabad, Pakistan` → plain text). Runs first so downstream passes see clean text.

2. **Boilerplate strip** — *position-aware*. A line is treated as a repeated header/footer only when the same normalized text repeats at the **same page edge position** across pages (keyed by `(top/bottom, rank, text)`); page-number/footer patterns are dropped by regex regardless of position. This prevents a content line that coincidentally recurs (e.g. two roles both dated `Jun 2024 — Aug 2024`) from being deleted — the earlier bug where a real date was stripped as "boilerplate". Digit runs normalize to `#` so `Page 3`/`Page 12` still match.

3. **Near-duplicate collapse** — splits the joined text into paragraph blocks (double-newline separated). Each block is compared to the 5 most recent kept blocks using `difflib.SequenceMatcher`. Blocks with ≥0.92 similarity to any recent block are dropped. Removes column-layout duplicates and OCR overlaps.

4. **Sentence merge** — re-joins lines broken mid-sentence at column edges, and repairs degraded extraction: a stranded bullet marker on its own line is reattached to its text, and a hard-wrapped continuation is rejoined even when it starts capitalized (only when the previous line is clearly cut mid-phrase — trailing comma/slash/hyphen, unclosed paren, or a dangling function word — so headings aren't swallowed).

5. **Whitespace normalize** — standardizes bullet characters to `-`, collapses 3+ consecutive blank lines to 2, and trims trailing spaces from lines.

---

### Stage 4 — Schema Building (`schema_builder.py`)

**Input:** `field_spec: list[dict]`  
**Output:** Pydantic `BaseModel` class (dynamically constructed)

`build_dynamic_model` recursively compiles the field spec to Pydantic v2 models:
- Scalar fields (`string`, `integer`, `number`, `boolean`) map directly via `TYPE_MAP`.
- `object` fields with `properties` become a nested `create_model()` call.
- `array<object>` fields become `List[NestedModel]`.
- `array<scalar>` fields become `List[str/int/...]`.

Required fields use `Field(...)` (no default). Optional fields use `Field(default=None)`. In strict mode, optional fields become `Optional[type]` with `Field(...)` (nullable but still required as a key) and the top-level model sets `extra="forbid"` — producing an OpenAI-strict-compatible schema.

`build_review_model` wraps the resume model in a `ResumeReview` model with two fields: `approved: bool` and `corrected: <resume_model>`. This is the return type for the self-verify pass — one LLM call both judges the extraction and returns the corrected output.

`render_field_guide` converts the field spec to a human-readable block injected into the LLM prompts:
```
- candidate_name (string, required): Full legal name
- years_experience (object, required): Total professional experience ...
    · years (integer, required): Whole years of professional experience
    · months (integer, required): Additional months beyond the whole years (0-11)
```
This is necessary because grammar-constrained decoding (JSON_SCHEMA mode) forces the model to emit the correct field names but strips away the field descriptions — the guide restores that semantic context.

`order_by_spec` reorders a result dict to match the field spec's declaration order. Applied as the final post-processing step so fields added out-of-order by the backfill steps (e.g. `phone`) appear in their schema position rather than appended at the end.

---

### Stage 5 — Context-Window Budgeting (`context.py`)

**Input:** system prompt text, user text, `Settings`  
**Output:** `(user_text: str, BudgetInfo)` — user_text is truncated if needed

Token counts are estimated from character count divided by `chars_per_token` (default 3.5), rounded up. No tokenizer dependency. The safety margin (default 256 tokens) absorbs the estimate's imprecision and chat-template overhead.

**For the generate pass** (`fit_user_content`): available budget = `context_window - max_tokens - safety_margin - system_tokens`. If the resume text exceeds available budget, it is truncated at the character level to fit exactly. The system prompt and generation reserve are never touched. Raises if there is no room for any user text.

**For the verify pass** (`fit_review_content`): the current extracted JSON is protected (never truncated). Available budget excludes the JSON token count. Only the source text is trimmed. Raises if system + JSON + reserve already exceed the window.

The `BudgetInfo` dataclass captures all budget components and is used for the structured log line emitted before each LLM call.

---

### Stage 6 — LLM Normalization (`normalize.py`, `llm_client.py`)

**Overall design:** Up to 3 LLM calls per resume (1 generate + up to 2 verify/refine). The generate pass should be correct enough that verify passes are rarely needed. When verify runs, it must both identify and fix problems in one call.

#### 6a — Generate Pass

System prompt (13 rules) covers:
1. Split all comma/semicolon/pipe/bullet lists into individual array elements.
2. Gather work history from any dated-role section regardless of heading.
3. Skills must be atomic (single technology/tool/language/method); deduplicated; never an outcome phrase, sentence fragment, or project name.
4. Projects scoped strictly to sections whose heading indicates projects or achievements (e.g. `Projects`, `Technical Projects`, `Key Achievements`, `Academic & Work Achievements`). Items under plain work-history sections are excluded.
5. `years_experience` is `{years, months}` computed from work-history dates.
6. Extract both phone and email from contact lines even with irregular formatting.
7. Populate every field where data is present anywhere in the text.
8. Do not invent data. If a role/certification/**project** has no date in the source, its date fields stay null — never borrow a nearby date or infer `Present`. Do not infer an organization from an event/venue name.
9. Preserve original wording for names, titles, companies, and dates.
10. `graduation_year` is the completion year only (null for ongoing/expected programs).
11. A project `metrics` entry must be quantitative (contain a number/%/unit); qualitative phrases belong in the description.
12. Capture a certification/license year when the source states one.
13. `work_history`: keep `title` (role) and `company` (employer) distinct; for self-employed roles use company `Self-employed`; a `City, Country` is a location, never a company; never split one dated role into multiple entries.

The field guide is appended to the system prompt so the model sees field descriptions even though grammar-constrained decoding enforces field names without exposing the schema's description keys. When `LLM_DISABLE_THINKING` is set, a `/no_think` directive is prepended (see backend note).

#### 6b — Self-Verify/Refine Loop

`VERIFY_SYSTEM_PROMPT` defines a strict auditor with an explicit checklist:
- Projects: every item under a qualifying heading is present; no project that merely duplicates a work_history entry.
- Phone and email both present.
- `years_experience` is `{years, months}` with months in range 0–11.
- Every work_history and education entry captured.
- Skills are atomic, deduplicated, and not outcome phrases/project names.
- Every project `metrics` entry is quantitative; qualitative phrases moved to the description.
- Certification/license year captured when stated.

Hard rule: set `approved=true` only when nothing on the checklist needs changing. If anything is wrong, apply the fix in `corrected` — never return `approved=false` with an unchanged `corrected`.

The review model returns `{approved, reason, field, corrected}` — the optional `reason`/`field` name what's still wrong (for observability). `run_refine_loop` calls the verify function up to `refine_passes` times. Stops early on approval or on convergence (output unchanged between passes). If it exhausts passes without approval or converges unapproved, it logs a warning and returns the best-available output; the run's `needs_review` flag and last `reason` are recorded in `03_extraction_metadata.json` (never in the structured JSON).

#### 6c — Mode Fallback Ladder

`_call_structured` walks the mode ladder (`JSON_SCHEMA → TOOLS → JSON → MD_JSON`). On `BadRequestError` (provider rejected the schema) or `ValidationError` (output did not conform after instructor's re-asks), it moves to the next rung. Transient errors (`RateLimitError`, `APIConnectionError`, `APITimeoutError`, `InternalServerError`) are retried with exponential backoff + jitter via tenacity before propagating. If no mode produces conformant output, a `RuntimeError` is raised.

**LLM backend specifics:**  
The current backend is **Qwen3-8B** on llama.cpp. Qwen3 is a hybrid-reasoning model: by default it emits a `<think>` block that llama.cpp routes to `reasoning_content`, which can exhaust the generation budget before any JSON is produced (surfaces as `IncompleteOutputException`). The fix is `LLM_DISABLE_THINKING=true`, which prepends `/no_think` to the system prompt to disable the reasoning phase. With thinking off, greedy decoding (`LLM_TEMPERATURE=0.0`) is safe and minimizes fabrication; `JSON_SCHEMA` mode still enforces the grammar from token 1. (Historical note: the pipeline previously targeted DeepSeek-R1-32B, which required temperature ~0.6; that constraint no longer applies with `/no_think`.)

---

### Stage 7 — Deterministic Post-Processing

Applied in `_finalize()` after the LLM step, in a fixed, order-dependent sequence (each pass is toggleable — see Configuration). All passes are deterministic, preserving reproducibility.

1. **Contact backfill (`contacts.py`):** For each empty string field whose name matches a contact type, a regex on the full cleaned source is tried (never overwriting an LLM value). Covers `email`, `phone`/`mobile`/`tel`, `linkedin`, `github`, `location`/`address` (label → street → `City, Country` on the contact line), and `date_of_birth` (labelled date).

2. **Work-role guard (`postprocess.normalize_work_roles`):** Repairs the company/title mapping the LLM gets wrong: un-reverses a self-employed line, sets `company = Self-employed` for freelance duplicates, moves a `City, Country` misparsed as a company into `location`, and **drops** an entry whose `company == title` (a project role-header the model misfiled as a job).

3. **Experience calculator (`experience.py`):** Parses each work_history entry's dates (`Nov 2025`, `September 2024`, `Present`/`Current`/`Ongoing`, bare year `2023` → **mid-year/June** to avoid over-stating spans), merges overlapping intervals, and sums months → `(years, months)`. Always overwrites the LLM value. Logs the merged intervals (debug) for auditability.

4. **Languages backfill (`postprocess.backfill_languages`):** If the `languages` field is empty, parses the LANGUAGES section (`English (Fluent), Urdu (Native)`) into `{name, proficiency}` objects. Ignores "Programming Languages". Regression guard for when the broad schema makes the model drop the field.

5. **Skills backfill (`postprocess.backfill_skills`):** Unions skills the model dropped from (a) the explicit skills section(s) and (b) each project's `technologies`. Runs before `filter_skills`, which removes any noise this introduces.

6. **Project dedup (`postprocess.dedupe_projects`):** Embeds each project's `name + description` with a sentence-transformers model (`BAAI/bge-small-en`, offline) and merges clusters with cosine ≥ `LLM_DEDUP_THRESHOLD` (0.90 — set high because bge's cosine range is compressed). Keeps the longest description, unions technologies/metrics. No-op if the embedder can't load.

7. **Tech canonicalize (`postprocess.canonicalize_tech`):** De-duplicates each project's technologies by a plural-folded key (`WebSocket`/`WebSockets` → one).

8. **Unsupported-date guard (`postprocess.drop_unsupported_project_dates`):** Nulls a `present`-style project `end_date` that has no `start_date` (fabricated ongoing range on an undated achievement); a real end date like `May 2024` is kept.

9. **Skills filter (`postprocess.filter_skills`):** Drops non-atomic skills: blocklisted phrases, runs > 4 words (parenthetical content excluded), skills that are ~an entire project title (length-sensitive ratio, ≥3 words), and employer/org/role **entity names** (exact-normalized, plus a fuzzy match for multi-word garbled fragments).

10. **Metrics validator (`postprocess.validate_metrics`):** Keeps only quantitative metrics (containing a digit/%); reroutes qualitative phrases into the description (skipping exact and fuzzy near-duplicates so it doesn't append redundant text); removes an emptied metrics list.

11. **Cert⇄activity dedup (`postprocess.dedupe_cert_activity`):** Drops an activity whose title fuzzy-matches a certification (certification wins).

12. **Prune empties (`postprocess.prune_empty_strings`):** Recursively drops dict keys whose value is an empty/whitespace-only string (the model sometimes emits `""` instead of null).

13. **Field ordering (`schema_builder.order_by_spec`):** Reorders keys to field_spec declaration order; extra keys appended after.

---

### Stage 8 — Artifact Writing (`artifacts.py`)

When `--artifacts-dir` is passed, four files are written to `artifacts/<resume_name>/`:

| File | Content |
|---|---|
| `01_raw_<engine>.md` | Raw per-page extraction output, pages separated by `---PAGE BREAK---` |
| `02_cleaned.md` | Full cleaned text after all cleanup passes |
| `03_extraction_metadata.json` | Engine used, page count, character counts, dedup reduction %; after the LLM step, merged with the refine `approved`/`needs_review`/`reason` status |
| `04_structured.json` | Final structured JSON output |

Artifacts 01–03 are written before the LLM call so they are available for debugging even if normalization fails. Artifact 04 is written after; `03` is then updated in place with the refine outcome (`needs_review`) so that flag stays out of the schema-conformant `04`.

---

## 6. Data Flow Diagram

```
PDF file
   │
   ▼
[triage.py]
   │  needs_ocr() — samples 3 pages, checks text density + image area ratio
   │
   ├─ native text ──► [extract_native.py]  PyMuPDF4LLM.to_markdown(page_chunks=True)
   │                        │
   └─ scanned ──────► [extract_marker.py]  Marker PdfConverter (lazy singleton)
                            │
   ◄────────────────────────┘
   │  (engine, pages: list[str])
   │
   ▼
[cleanup.py]  clean_extraction()
   │  1. strip_markdown_artifacts()  — ~~/<u>/<sup>/<br>, demote location headings
   │  2. strip_repeated_boilerplate()  — position-aware header/footer + page numbers
   │  3. dedupe_near_identical_blocks()  — 0.92 similarity threshold, 5-block window
   │  4. merge_split_sentences()  — repair column-edge breaks + degraded bullets
   │  5. normalize_whitespace()  — bullets, blank lines, trailing spaces
   │
   ▼  clean_text: str
   │
   ├──► [artifacts.py]  save_artifacts()  writes 01_raw, 02_cleaned, 03_metadata
   │
   ▼
[schema_builder.py]  build_dynamic_model() + render_field_guide()
   │  Compiles field_spec.json → Pydantic models at runtime
   │
   ▼
[context.py]  fit_user_content()
   │  Budget: ctx_window − max_tokens − margin − sys_tokens = available for resume
   │  Truncates resume text only if it exceeds budget
   │
   ▼
[llm_client.py + normalize.py]  _call_structured() — GENERATE PASS
   │  instructor JSON_SCHEMA mode → grammar-constrained JSON decoding
   │  Fallback ladder: JSON_SCHEMA → TOOLS → JSON → MD_JSON
   │  tenacity backoff on transient errors; instructor re-ask on validation failure
   │
   ▼  initial: dict
   │
   ▼
[normalize.py]  run_refine_loop()  (up to refine_passes iterations)
   │
   │  ┌── refine_fn(current) ───────────────────────────────────────────────────┐
   │  │  [context.py]  fit_review_content()                                    │
   │  │    Budget: json_text is protected; only source_text may be truncated   │
   │  │  [llm_client.py + normalize.py]  _call_structured() — VERIFY PASS     │
   │  │    Returns {approved: bool, corrected: <resume_model>}                 │
   │  │  Stop if approved=True, or corrected==current (converged)              │
   │  └─────────────────────────────────────────────────────────────────────────┘
   │
   ▼  final: dict
   │
   ▼
[_finalize()]  deterministic post-processing (all toggleable)
   │  [contacts.py]  backfill_contacts()  — email/phone/linkedin/github/location/dob
   │  [postprocess]  normalize_work_roles()  — company/title fix, drop role-headers
   │  [experience.py]  backfill_experience()  — union-of-intervals date math
   │  [postprocess]  backfill_languages()  — LANGUAGES section
   │  [postprocess]  backfill_skills()  — skills section + project technologies
   │  [postprocess]  dedupe_projects()  — embedding cosine merge (bge-small-en)
   │  [postprocess]  canonicalize_tech()  — fold plural tech dups
   │  [postprocess]  drop_unsupported_project_dates()  — null lone 'present'
   │  [postprocess]  filter_skills()  — non-atomic / entity / project-name
   │  [postprocess]  validate_metrics()  — quantitative-only, reroute rest
   │  [postprocess]  dedupe_cert_activity()  — cert wins over activity
   │  [postprocess]  prune_empty_strings()
   │  [schema_builder.py]  order_by_spec()  — reorder keys to spec order
   │
   ▼
   ├──► [artifacts.py]  save_structured()  writes 04_structured.json
   │                    + update_metadata() writes needs_review into 03
   └──► stdout  (JSON, pretty-printed)
```

---

## 7. Testing

74 unit tests across 8 test files (+ DB tests skipped when Postgres is absent). The LLM layer is bypassed in unit tests; the project-dedup embedder is monkeypatched (no model load, no network).

| Test file | What it covers |
|---|---|
| `test_cleanup.py` | Markdown scrub (`~~`/`<br>`/location-heading), position-aware boilerplate (keeps recurring body dates, drops page numbers), near-duplicate collapse, sentence merge, whitespace |
| `test_triage.py` | Native-text PDFs stay native; image-only PDFs trigger OCR; mixed cases |
| `test_schema_builder.py` | Dynamic model compilation; field guide rendering; `order_by_spec`; `build_review_model` shape |
| `test_context.py` | Token estimation; fit_user_content under/over budget; fit_review_content JSON-protected truncation |
| `test_normalize.py` | `run_refine_loop`: converges, stop-on-approval, stop-on-stuck |
| `test_contacts.py` | email/phone/linkedin/github/location/dob extraction + backfill; no overwrite; non-contact fields skipped |
| `test_experience.py` | `parse_month_year` (incl. year-only → June); `compute_total_experience` overlaps; `backfill_experience` object/integer |
| `test_postprocess.py` | Project dedup (cluster + merge, injected embedder); skills backfill/filter (entity + project-title + garbled fragments); languages backfill; metrics validate/reroute; tech canonicalize; work-role fix + spurious-drop; unsupported-date guard; cert⇄activity dedup; prune-empties |

---

## 8. Dependency Stack

**Extraction:**
- `pymupdf` 1.28 — triage + native extraction
- `pymupdf4llm` 1.28 — markdown conversion of native PDFs
- `marker-pdf` 1.10 — OCR for scanned PDFs (pulls `torch`, `surya-ocr`, `transformers`)

**LLM interface:**
- `openai` 2.44 — OpenAI-compatible HTTP client
- `instructor` 1.15 — structured output wrapper (Pydantic validation + mode management)
- `tenacity` 9.1 — retry with exponential backoff

**Schema and validation:**
- `pydantic` 2.13 — model compilation, runtime validation, JSON schema emission

**Deterministic post-processing:**
- `sentence-transformers` 5.6 + `BAAI/bge-small-en` (cached offline) — project dedup embeddings
- `rapidfuzz` 3.14 — fuzzy matching (skills filter, cert⇄activity, reroute dedup)
- `scikit-learn` 1.7, `numpy` 2.2 — cosine/clustering support

**Configuration:**
- `python-dotenv` 1.2 — `.env` loading

**Testing:**
- `pytest` 9.1

**LLM server (external, not a Python dependency):**
- llama.cpp / llama-server — serves **Qwen3-8B** at `http://localhost:8090/v1`
- Config: `-ngl 99 -c 32768 -b 512 -ub 512 --flash-attn on --parallel 1 --n-predict 4096`
- Requires `/no_think` (via `LLM_DISABLE_THINKING=true`) to suppress Qwen3's reasoning phase

---

## 9. Known Limitations and Open Issues

### Resolved (kept for history)
- **Cleaning deleted real dates** — the boilerplate stripper removed a date line that recurred across pages (e.g. two items both `Jun 2024 — Aug 2024`), which then made the model fabricate a replacement. Fixed by the position-aware boilerplate strip (Stage 3.2).
- **Fabricated/mismatched work fields** — company/title reversal, location-as-company, and role-header pseudo-jobs are now repaired deterministically (`normalize_work_roles`).
- **Year-only start dates** — now assumed **mid-year (June)** rather than January, so a `2023 – present` span is not over-stated by up to 11 months.
- **Dropped `languages`, garbled `<br>` skills, entity-name skills, duplicate `WebSocket(s)`, fabricated project `Present`, duplicated merged descriptions** — all addressed by the Stage 7 passes.

### Open
**1. Small-model recall vs. schema breadth**  
With the broad schema, the 8B model still occasionally drops content that lives only in prose (e.g. `UnrealBloom`, `MLP`/`GRU`, coursework `HTML`) — it isn't in a skills section or a project's `technologies`, so no deterministic backfill can recover it. The durable fix is a larger model; the backfills only cover regular, locatable fields.

**2. Free-form `location`/`date_of_birth` are best-effort**  
The backfills handle labelled addresses, street addresses, `City, Country` contact lines, and labelled DOBs. Unlabelled or unusual formats may be missed, and the LLM often doesn't fill these on its own.

**3. Degraded native extraction on some pages**  
`pymupdf4llm` (with `pymupdf-layout` installed) still renders some two-column/styled pages as flat text without markdown headings. Cleanup and the backfills compensate, but a section that loses its heading won't be picked up by the section-based backfills.

**4. Scanned path not end-to-end tested**  
Marker's OCR path (`extract_marker.py`) has not been exercised on a real scanned resume; output quality on borderline PDFs is unknown.

**5. Projects count is heading-driven**  
Extraction depends on recognizable achievement-type headings; a resume with none produces an empty projects list even if projects are described under experience.

**6. No batch/parallel processing**  
Single-threaded, one PDF per invocation. Batching is a shell loop over the CLI, not built into the pipeline.

---

## 10. Database Ingestion — Recommendation

### Where to Insert

The natural insertion point is in `pipeline.py`, immediately after `extract_structured()` returns the `structured` dict and before (or alongside) `save_structured()`. The pipeline currently returns the dict to the caller; the caller (CLI or a wrapper script) is where the database write should live. This keeps the pipeline itself database-agnostic — it produces a Python dict, and the caller decides what to do with it.

A thin function with this signature is all that is needed:

```python
def ingest_resume(structured: dict, source_pdf_path: str, metadata: dict) -> str:
    """Write one structured resume dict to the database. Returns a record ID."""
    ...
```

### Schema Mapping Strategy

The structured output is a nested JSON document. Two approaches suit different database families:

**Document store (flat JSON ingestion):**  
Insert the `structured` dict directly as a document. The top-level fields become document fields. Arrays (`skills`, `work_history`, `projects`, `education`) are stored as nested arrays within the document. `years_experience` is stored as a sub-document `{years, months}`. Each resume document should carry:
- The `structured` dict as-is
- `source_file`: the original PDF filename
- `ingested_at`: an ISO-8601 timestamp (stamped by the ingest function, not the pipeline)
- `pipeline_version`: a short string identifying the field_spec and model version used

**Relational store (normalized tables):**  
A normalized design splits the nested arrays into joined tables:

```
resumes (id, candidate_name, email, phone, years, months, ingested_at, source_file)
skills  (resume_id, skill)
work_history (resume_id, company, title, start_date, end_date)
projects (resume_id, name, description)
project_technologies (project_id, technology)
education (resume_id, institution, degree, graduation_year)
```

`years_experience.years` and `years_experience.months` become flat integer columns on the `resumes` table. Skills are one row per element. `project_technologies` is a child of `projects`. This structure supports SQL aggregation (skills frequency, experience distribution, graduation year filtering) without JSON traversal.

### Upsert vs. Insert

Because the same PDF may be reprocessed after prompt or schema changes, the ingest function should support upsert keyed on a stable identifier. A good natural key is a hash of the raw PDF bytes (e.g. SHA-256 of the file), stored alongside the record. On reprocess, the existing record is replaced, not duplicated. The CLI already passes the PDF path to `run_pipeline`, so computing the hash there and threading it through to `save_structured` / the ingest function requires minimal change.

### Integration Pattern

```python
# In pipeline.py — minimal change
def run_pipeline(..., ingest_fn=None) -> dict | None:
    ...
    structured = extract_structured(...)
    if artifacts_dir:
        save_structured(artifacts_dir, structured)
    if ingest_fn is not None:
        ingest_fn(structured, pdf_path)
    return structured
```

The `ingest_fn` parameter is optional and defaults to `None`, so existing callers (CLI, tests) are unaffected. The ingest function is injected by the caller — a CLI flag like `--db-uri` can construct and pass it; a web service can pass its own connection-aware closure. This keeps the pipeline layer free of any database import and makes the ingest function independently testable with a mock.

### Connection Management

For a CLI-driven batch use case, open and close the connection in the CLI's `main()` using a context manager, passing a bound method or closure as `ingest_fn` to `run_pipeline`. For a long-running service, maintain a connection pool at the application level and inject a pool-bound write function.

### Schema Versioning

As `field_spec.json` evolves (new fields, type changes), stored records from different versions will have different shapes. Store `field_spec_hash` (SHA-256 of `field_spec.json` contents) alongside each record. This makes it possible to filter records by schema version, reprocess only old-version records, and avoid silently mixing incompatible shapes in queries.
