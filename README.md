# HexTech — Résumé Knowledge Graph

Turn résumé PDFs into a queryable **knowledge graph** in Neo4j, backed by
PostgreSQL. Two subsystems share one database:

- **`ocr-resume-paser/`** — parses each PDF into structured JSON and stores it in
  the `resumes` table (+ projection tables).
- **`ontogen/`** — reads those résumés, extracts entities/relationships, stages
  them in Postgres, and loads them into Neo4j.

```
PDF ──parser──► Postgres (resumes.structured) ──ontogen──► Postgres (graph_*) ──► Neo4j
```

For the full design, data flow, schema, and stage-by-stage details, see
**[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)**. For a copy-paste command
reference, see **[ocr-resume-paser/docs/commands.md](ocr-resume-paser/docs/commands.md)**.

## Two extraction paths

- **Path A (deterministic, no LLM):** already-structured fields (employer,
  education, skills, projects, technologies, contact info) become graph triples
  directly — this always produces a graph.
- **Path B (LLM, free text only):** competency-question extraction over project
  descriptions, matched against Wikidata, adding facts the structured fields
  don't capture.

## Prerequisites

- Docker (for Postgres) — `sudo` on this machine.
- The Python virtualenv at `ocr-resume-paser/.venv` (used by **both** subsystems;
  Ontogen imports `resume_parser`). Deps: `sqlalchemy`, `psycopg2`, `alembic`,
  `pgvector`, `sentence-transformers`, `rdflib`, `neo4j`, plus the parser's stack.
- An OpenAI-compatible LLM endpoint (configured in `ocr-resume-paser/.env` for the
  parser and `ontogen/config.py` for Ontogen).
- A running Neo4j (bolt on `:7687`) for the final load.

## Quickstart

```bash
cd ocr-resume-paser
source .venv/bin/activate
set -a && . ./.env && set +a          # DATABASE_URL, LLM_*, ...

# 1. Database (pgvector-enabled Postgres on host port 5433).
sudo docker compose up -d
alembic upgrade head                  # parser + ontogen schema (migration 0003)

# 2. One-time: seed Ontogen's corpus stores from the on-disk files.
python ../ontogen/scripts/migrate_ontogen_files_to_db.py

# 3. Run everything: parse all resumes/*.pdf, then build the KG.
cd ..
python run_pipeline.py

# 4. Load the staged graph into Neo4j.
python ontogen/graphdb/load_to_neo4j.py
```

Then in Neo4j Browser:

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 300
```

## Running the pieces separately

```bash
# Parse one résumé into Postgres.
python -m resume_parser.cli "resumes/Riyan Resume.pdf" --db-uri "$DATABASE_URL"

# Ontogen over résumés already in the database.
python ontogen/pipeline.py               # all
python ontogen/pipeline.py <resume_uuid> # one
python ontogen/pipeline.py --resume      # skip already-processed résumés

# Neo4j load is incremental (only unsynced rows); --wipe clears the graph first.
python ontogen/graphdb/load_to_neo4j.py --wipe
```

## Tests

```bash
cd ocr-resume-paser
python -m pytest tests/test_db_ingest.py -q          # parser DB tests
python -m pytest ../ontogen/tests/test_ontogen_db.py -q   # ontogen DB layer
```

## Notes

- Both subsystems share one llama-server (`--parallel 1`), guarded by an advisory
  lock (`resume_parser/llm_lock.py`) so they never collide on the inference slot.
- The Neo4j default password is currently hard-coded in
  `ontogen/graphdb/config.py` — move it to an env var before any non-local use.
- See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) §10 for known limitations
  (notably: Path B's LLM stages need larger token budgets with the reasoning
  model to contribute facts; Path A carries the graph today).
```
