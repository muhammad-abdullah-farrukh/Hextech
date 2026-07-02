# HexTech — Résumé → Knowledge Graph Pipeline: Technical Report

## 1. What this system does

HexTech turns résumé PDFs into a queryable **knowledge graph** in Neo4j, using
PostgreSQL as the durable system of record in between. It is two subsystems that
share **one PostgreSQL database**:

| Subsystem | Directory | Job |
|---|---|---|
| **Parser** ("Part 1") | `ocr-resume-paser/` | PDF → structured JSON → `resumes` table (+ projection tables) |
| **Ontogen** ("Part 2") | `ontogen/` | `resumes.structured` → knowledge-graph nodes/edges → Neo4j |

The parser owns the résumé data; Ontogen reads that data, extracts entities and
relationships, stages them in Postgres, then loads them incrementally into Neo4j.

```
PDF ──parser──► Postgres(resumes.structured) ──ontogen──► Postgres(graph_entities/
                                                            graph_relationships) ──► Neo4j
```

---

## 2. Repository layout

```
HexTech/
├── run_pipeline.py                 # top-level orchestrator: parse all PDFs, then build the KG
├── README.md
├── TECHNICAL_REPORT.md             # this file
│
├── ocr-resume-paser/               # Part 1 — the parser (also hosts the shared DB + Alembic)
│   ├── docker-compose.yml          # Postgres (pgvector/pgvector:pg17) on host port 5433
│   ├── .env                        # DATABASE_URL, TEST_DATABASE_URL, LLM_* for the parser
│   ├── alembic/versions/           # 0001 initial · 0002 projects · 0003 ontogen schema
│   ├── config/field_spec.json      # the résumé field schema
│   ├── resume_parser/
│   │   ├── cli.py  pipeline.py  normalize.py  extract*.py  cleanup.py  ...
│   │   ├── llm_lock.py             # shared advisory LLM lock (used by BOTH subsystems)
│   │   └── db/ models.py session.py ingest.py
│   └── docs/commands.md            # copy-paste command reference
│
└── ontogen/                        # Part 2 — the knowledge-graph pipeline
    ├── config.py                   # LLM endpoint/model, DATABASE_URL, embedding model
    ├── render.py                   # render_resume_text() + structured_to_relations()
    ├── pipeline.py                 # process_resume() / main()
    ├── db/                         # models, session, runs, canon, gazetteers, wikidata, kg_staging
    ├── stages/                     # stage1..stage9_10, canonicalize, provenance, llm
    ├── graphdb/                    # load_to_neo4j.py (Postgres → Neo4j), config.py
    ├── scripts/migrate_ontogen_files_to_db.py   # one-time seed of corpus stores
    └── tests/test_ontogen_db.py
```

> **Environment note:** Ontogen runs inside the parser's virtualenv
> (`ocr-resume-paser/.venv`) because `ontogen/db/*` imports `resume_parser` for
> the shared SQLAlchemy models and session factory. The dependency only ever
> points **Ontogen → parser**, never the reverse.

---

## 3. Part 1 — the parser (PDF → Postgres)

**Flow** (`resume_parser/pipeline.py::run_pipeline`):

1. `extract_pdf` — OCR/text extraction (marker / native).
2. `clean_extraction` — deterministic cleanup.
3. `extract_structured` — the **one LLM call**: fills the schema in
   `config/field_spec.json` (candidate_name, email, phone, years_experience,
   skills[], work_history[], projects[], education[]).
4. If `--db-uri` is given, `ingest_fn` upserts into Postgres.

**Storage** (`resume_parser/db/models.py`): one parent table `resumes` holding
the structured JSON verbatim in a `JSONB` column (deduped by `pdf_hash`), plus
four **projection** tables (`skills`, `work_history`, `education`, `projects`)
that are a rebuildable cache for indexed queries — wiped and re-derived on every
re-ingest via `ON DELETE CASCADE`. The upsert is atomic
(`INSERT … ON CONFLICT (pdf_hash) DO UPDATE`).

The parser itself is database-agnostic: it returns a dict; the CLI decides
whether to persist. See `ocr-resume-paser/docs/DATABASE_INTEGRATION.md` and
`DB_REPORT.md`.

---

## 4. Part 2 — Ontogen (Postgres → Knowledge Graph)

Ontogen's input is `resumes.structured` (never files). It extracts facts along
**two paths that converge in the graph tables**:

### Path A — deterministic (no LLM)
`render.structured_to_relations(structured)` turns already-structured fields
directly into concrete triples, so clean data is never sent through an LLM:

- **Person node** (`candidate_name` as label): `email`, `phone`,
  `yearsExperience` become literal properties on the node.
- **work_history[]** → `employer` (entity), `jobTitle`/`startDate`/`endDate`
  (literals).
- **education[]** → `educatedAt` (entity), `degree`/`graduationYear` (literals).
- **skills[]** → `hasSkill` (entity, one per skill).
- **projects[]** → `hasProject` (owner → project); each `technologies[]` →
  `usesTechnology` hung off the **project** node.

These are written straight to `graph_entities` / `graph_relationships` by
`db.kg_staging.stage_structured_relations()`, entity-resolving only the objects
(company/university/skill names) through the 3-tier resolver.

### Path B — CQ-driven (LLM), free text only
`render.render_resume_text(structured)` extracts **only free text** (project
descriptions), and the classic Ontogen stages run over it:

1. **Stage 1 — CQ generation**: competency questions from the text.
2. **Stage 2 — CQ answering**: one LLM call per question.
3. **Stage 3 — relation extraction**: relations from the Q/A pairs.
4. **Stage 3.5 — EDC canonicalization**: unmatched relations checked against the
   cross-document **canon store** (pgvector) for a merge candidate.
5. **Stage 6 — Wikidata match/validate**: each relation's description is embedded
   and matched to the nearest `wikidata_properties` (pgvector HNSW), then an LLM
   confirms the match is correct.
6. **Stage 7/8 — ontology**: builds the OWL/Turtle ontology from the matches;
   genuinely new properties are registered back into the canon store.
7. **Stage 9/10 — KG construction**: the LLM emits facts as JSON (never Turtle),
   validated against the ontology's predicate set, assembled with rdflib, then
   entity-resolved and **staged** to `graph_entities`/`graph_relationships`.

Both paths' *relation dicts* also merge just before Stage 6 to shape the
ontology; both paths' *concrete triples* merge in the graph tables.

### Where the two paths meet
`graph_entities` (nodes: `properties` JSONB incl. `uri`) and
`graph_relationships` (edges: `from_entity`/`to_entity`/`rel_type`). Rows carry
`synced_to_neo4j = FALSE` until pushed. Cross-run/cross-document dedup happens at
Neo4j load time via `MERGE` on `properties->>'uri'`.

### 3-tier entity resolution (`stages/canonicalize.py`)
- **Tier 1 — gazetteer** (`db.gazetteers`): O(1) alias→canonical + QID lookup.
- **Tier 2 — embedding**: bge-small-en similarity against gazetteer canonicals.
- **Tier 3 — LLM**: only for novel/ambiguous mentions.

---

## 5. End-to-end flow

```
                 ┌───────────────────────── run_pipeline.py ─────────────────────────┐
                 │                                                                    │
 resumes/*.pdf ──┤  for each PDF:  python -m resume_parser.cli <pdf> --db-uri …       │
                 │                     │  (holds shared LLM lock while calling LLM)    │
                 │                     ▼                                               │
                 │            Postgres: resumes.structured (+ projection tables)       │
                 │                     │                                               │
                 │  then:  python ontogen/pipeline.py   (holds shared LLM lock)        │
                 │                     │                                               │
                 │      ┌── Path A (no LLM): structured_to_relations ──┐               │
                 │      │                                              ▼               │
                 │  Path B (LLM): render_resume_text → S1→S2→S3→S6→S7/8→S9/10          │
                 │      │                                              │               │
                 │      └──────────────► graph_entities / graph_relationships ◄────────┘
                 │                                   │  (synced_to_neo4j = FALSE)
                 └───────────────────────────────────┼──────────────────────────────
                                                      ▼
                        python ontogen/graphdb/load_to_neo4j.py
                        (batched MERGE on uri; flips synced_to_neo4j = TRUE)
                                                      ▼
                                                   Neo4j
```

---

## 6. Database schema

Shared PostgreSQL database `resume_parser` (host port **5433**), managed by a
single Alembic chain in `ocr-resume-paser/alembic` (`0001 → 0002 → 0003`).

**Part 1 (0001/0002):** `resumes`, `skills`, `work_history`, `education`,
`projects`.

**Part 2 (0003_ontogen_schema):**

| Table | Purpose | Notes |
|---|---|---|
| `pipeline_runs` | per-(document, stage) checkpoint | replaces `outputs/*/*.json` cache; drives `--resume` |
| `canon_store` | EDC cross-document canon store | `embedding vector(384)` + HNSW index |
| `gazetteers` | alias→canonical (+`wikidata_qid`) | 5 files → 1 table |
| `wikidata_properties` | filtered Wikidata props + embeddings | `vector(384)` + HNSW |
| `provenance` | per-triple provenance | (populated only by Path B / Stage 9) |
| `graph_entities` | staged KG nodes | `properties` JSONB incl. `uri`; expression index on `uri` |
| `graph_relationships` | staged KG edges | partial index on `synced_to_neo4j = FALSE` |

**FK delete semantics (deliberate):** `document_id` on `pipeline_runs` /
`provenance` is `ON DELETE CASCADE` (per-document, owned by the résumé).
`source_doc` on `canon_store` / `graph_entities` / `graph_relationships` is
`ON DELETE SET NULL` — it's a provenance pointer to the *originating* résumé, but
the row may still serve other résumés, so deleting a résumé nulls the pointer
rather than destroying shared graph data. Requires the `vector` (pgvector)
extension, which is why the compose image is `pgvector/pgvector:pg17`.

---

## 7. LLM configuration & the shared lock

- **Ontogen LLM** (`ontogen/config.py`): OpenAI-compatible endpoint
  `LLM_BASE_URL = http://192.168.3.76:8080/v1`, `LLM_MODEL = deepseek-r1-32b-q4`.
  `stages/llm.py` posts to `{base}/chat/completions` and strips deepseek-r1
  `<think>` reasoning from the answer.
- **Parser LLM** (`ocr-resume-paser/.env`): its own `LLM_*` — independent.
- **Embeddings**: `BAAI/bge-small-en` (384-dim), matching the `vector(384)` columns.
- **Shared advisory lock** (`resume_parser/llm_lock.py`): both subsystems call one
  llama-server with `--parallel 1`, so each takes a file lock
  (`$TMPDIR/hextech_llm.lock`, `{pid, started_at}`) and pre-flights the endpoint
  (`GET /models`, 5 s) before running. A second concurrent run aborts with
  "LLM appears busy [pid …]"; a lock left by a dead process is auto-cleaned. The
  lock is acquired/released **per parser invocation** so the single inference
  slot isn't held during inter-PDF disk I/O.

---

## 8. How to run

All commands run from the parser repo's venv. See
`ocr-resume-paser/docs/commands.md` for the fuller reference.

### 8.1 One-time setup
```bash
cd ocr-resume-paser
source .venv/bin/activate                 # shared venv (also used by Ontogen)
set -a && . ./.env && set +a              # loads DATABASE_URL etc.

# Start Postgres (pgvector image; keeps data in the pgdata volume).
sudo docker compose up -d

# Create/upgrade the schema (parser + ontogen tables, migration 0003).
alembic upgrade head

# One-time: seed Ontogen's corpus stores (gazetteers, canon store, wikidata
# properties) from the on-disk files, re-embedding with bge-small-en.
python ../ontogen/scripts/migrate_ontogen_files_to_db.py
```

### 8.2 Normal run (parse → build KG)
```bash
# From the HexTech repo root: parse every resumes/*.pdf into Postgres, then
# build the knowledge graph over what's in the database.
python run_pipeline.py
```

### 8.3 Run the pieces individually
```bash
# Parse one PDF into Postgres.
python -m resume_parser.cli "resumes/Riyan Resume.pdf" --db-uri "$DATABASE_URL"

# Ontogen over résumés already in the database.
python ../ontogen/pipeline.py                 # all résumés
python ../ontogen/pipeline.py <resume_uuid>   # one
python ../ontogen/pipeline.py --resume        # skip résumés with a succeeded kg_facts row

# Load the staged graph into Neo4j (incremental — only unsynced rows).
python ../ontogen/graphdb/load_to_neo4j.py
python ../ontogen/graphdb/load_to_neo4j.py --wipe    # clear the graph first
```

### 8.4 Tests
```bash
python -m pytest tests/test_db_ingest.py -q          # parser DB tests
python -m pytest ../ontogen/tests/test_ontogen_db.py -q   # ontogen DB layer (real Postgres)
```

### 8.5 View the graph
Open Neo4j Browser and run:
```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 300
```
(Switch the result frame to the **Graph** tab. The "Database information" side
panel is a connect-time snapshot — click its refresh icon after an external load
to see updated counts.)

---

## 9. Verified run (3 résumés)

Seeded corpus stores: **909** gazetteer rows (113 with QID), **59** canon
entries, **2,936** wikidata properties (all embedded).

Pipeline over the 3 résumés staged **149 entities / 203 relationships**; the
Neo4j load MERGE-deduplicated them to **128 nodes / 179 relationships**:

| Relationship | Count |
|---|---:|
| HAS_SKILL | 91 |
| USES_TECHNOLOGY | 65 |
| HAS_PROJECT | 11 |
| EMPLOYER | 7 |
| EDUCATED_AT | 5 |

All three résumé owners are present as `:Entity` nodes.

---

## 10. Known limitations

- **Path B (LLM) is currently reasoning-starved.** deepseek-r1 on this
  llama-server spends its token budget on hidden reasoning, so stages with small
  `max_tokens` (validation = 10, entity-resolution = 40) return empty answers and
  `provenance` stays at 0 — the graph above is produced entirely by **Path A**
  (deterministic), by design. To enable Path B facts, give those stages enough
  tokens for reasoning + answer, or disable thinking server-side. Not required
  for the graph.
- **Neo4j default password** is hard-coded in `ontogen/graphdb/config.py`
  (`NEO4J_PASSWORD` default). Rotate it out / move to an env var before any
  non-local deployment.
- **Batch/parallel processing and encryption** are intentionally out of scope
  (the schema/ingest are parallel-ready; only the process-pool wiring is
  deferred).

## ACCESSING DATABASE THROUGH ADMINER

Field	Value
System	PostgreSQL (already set)
Server	postgres (already set)
Username	resume_parser
Password	devpassword
Database	resume_parser
Then click Login.