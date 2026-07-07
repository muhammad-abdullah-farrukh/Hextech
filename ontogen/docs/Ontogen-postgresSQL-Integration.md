# Ontogen → PostgreSQL Integration
## Context
The Ontogen knowledge-graph pipeline (ontogen/) currently runs entirely off the filesystem: it reads data/documents/*.txt, caches every stage under outputs/{cqs,answers,relations,matches,ontology,kg,provenance}/, keeps its corpus-wide stores in JSON/.npy files, and writes final KGs as .ttl that graphdb/load_to_neo4j.py re-parses per file (no cross-document dedup).

Part 1 (ocr-resume-paser/) already persists parsed résumés into PostgreSQL (resumes + skills/work_history/education/projects projection tables, verified in DB_REPORT.md). Part 2 moves Ontogen onto that same database so it reads résumés from resumes.structured, stages entities/relationships in Postgres before Neo4j, and replaces its file stores with tables — using two extraction paths (deterministic from structured fields; CQ-driven from free text) per ontogen/docs/ONTOGEN_DATABASE_INTEGRATION.md.

This plan implements that document. Confirmed decisions (from the user):

Migrations: add 0003_ontogen_schema.py to the existing ocr-resume-paser/alembic tree, chained on 0002_add_projects (one chain, one alembic_version, FK to resumes resolves).
pgvector: switch ocr-resume-paser/docker-compose.yml image from postgres:17 → pgvector/pgvector:pg17.
Env & import: run Ontogen inside ocr-resume-paser/.venv; ontogen/db/* imports resume_parser via a sys.path insert of the parser repo root (matching Ontogen's existing sys.path.insert(0, parent) idiom).
Findings that change the written plan (flag before coding)
No embeddings exist on disk. There is no ontogen/embeddings/wikidata_embeddings.npy and no data/canon_store/embeddings.npy — only the JSON (2,936 wikidata properties, 59 canon entries). So the one-time migration (§6) cannot copy .npy by index as the doc's §6.3 assumes; it must re-embed both sets with bge-small-en (384-dim) using sentence-transformers. (Downloads the model, embeds 2,936 property descriptions — a few minutes.)
field_spec.json has no summary and no work_history[].description. Confirmed shape: top-level candidate_name, email, phone, years_experience, skills[], work_history[]{company,title,start_date,end_date}, projects[]{name, description,technologies[]}, education[]{institution,degree,graduation_year}. The only genuine free text is projects[].description. So render_resume_text() (Path B input) renders projects[].description (plus summary/work_history[].description defensively if present), not the fields the doc's §4.1 hypothesised.
Gazetteer QIDs have nowhere to live. The gazetteers table in §2 has no QID column, but ResumeEntityResolver._get_qid() uses the gazetteer JSON's wikidata_qid map to build canonical wd:Q… URIs. Recommended deviation: add a nullable wikidata_qid TEXT column to gazetteers so that behaviour survives; otherwise entity resolution silently degrades to slug URIs.
Neo4j default password (out of scope — flagging only): graphdb/config.py:13 hardcodes NEO4J_PASSWORD default "5053811238". Per instructions I will not change it; rotate it out before this repo is pushed anywhere public.
Postgres runs on host port 5433 (not 5432); DB/user/pass = resume_parser/resume_parser/devpassword (see ocr-resume-paser/.env).
Path A gets its own deterministic triple write (never touches Stage 9's LLM)
Stage 9 emits graph triples from doc_text + QA pairs via the LLM. Feeding structured facts (owner → WorkedAt → "Google") into Stage 9 as prose to be re-extracted would pay for LLM extraction on facts we already hold cleanly — undoing the entire reason Path A exists. So Path A bypasses Stage 9 entirely:

Stage 9 keeps taking narrative_text only (free text) + qa_pairs, exactly as §4.2 specifies — its LLM only ever handles Path B (CQ-derived) facts.
structured_to_relations(structured) returns richer dicts that carry the concrete triple as well as the ontology-shaping fields: {"property", "description", "subject", "object", "object_type", "source":"structured"}. The property/description still merge before Stage 6 to shape the ontology (as §4.2 says); the subject/object drive the direct write below.
A new stage_structured_relations(session, resume_id, relations) (in ontogen/db/kg_staging.py) writes each Path A relation straight to graph_entities/graph_relationships: subject = the résumé owner's node, predicate = the property (typed via PROPERTY_ENTITY_TYPE_MAP), object = the company/institution/skill string. There is no extraction ambiguity for an LLM here; the only real work is entity resolution on the object — run ResumeEntityResolver's Tier 1/2/3 (canonicalising "Google" vs "Google Inc.") directly on Path A objects, no text-grounding needed.
The two triple sets merge at the graph_entities/graph_relationships level — same converge-at-a-single-point philosophy Path A/B already use before Stage 6, just applied again at the final graph write for Path A.
This is a smaller change than expanding Stage 9's prompt input and leaves stage9_10_kg.py's 839 lines of extraction logic untouched (only its I/O boundary — where the LLM-built graph is written — changes).

## LLM change — Ontogen shares the parser's LLM (explicitly requested)
Ontogen must use the same LLM as ocr_resume_parser: endpoint http://localhost:8080/v1 + model deepseek-r1-32b over the OpenAI-compatible /v1/chat/completions route (the parser's llama.cpp/llama-server). This overrides both §9's "don't change provider/model" and the earlier 192.168.3.74/qwen3 request. Two edits:

ontogen/config.py: LLM_PROVIDER → openai-compatible, LLM_BASE_URL = "http://localhost:8080/v1", LLM_MODEL = "deepseek-r1-32b", LLM_API_KEY = "not-needed" (mirrors the parser's .env values).
ontogen/stages/llm.py: rewrite call_llm() to POST to {LLM_BASE_URL}/chat/completions (OpenAI shape: model, messages, temperature, max_tokens) and read choices[0].message.content / choices[0].finish_reason. Preserve the existing call_llm signature (max_tokens, temperature, presence_penalty, frequency_penalty, guided_json, return_finish_reason) and the retry/backoff loop, so no stage call site changes. deepseek-r1 emits <think>…</think> in content — strip it before returning (the stages parse plain text/JSON, not instructor). guided_json stays best-effort/ignored as today. ocr-resume-paser's LLM config is untouched.
1. Infrastructure
ocr-resume-paser/docker-compose.yml: image: postgres:17 → image: pgvector/pgvector:pg17 (same env/ports/volume). Show + confirm before docker compose up; only against local dev.
ontogen/requirements.txt: add sqlalchemy>=2.0, psycopg2-binary, alembic, pgvector. (neo4j already in graphdb/requirements.txt.)
No new .env; Ontogen reads the same DATABASE_URL the parser uses (port 5433 database resume_parser).
2. Migration — ocr-resume-paser/alembic/versions/0003_ontogen_schema.py
Hand-written in the 0001/0002 style (op.create_table, op.create_index, op.execute for raw SQL), down_revision = "0002_add_projects". Implements §2 verbatim plus the one deviation (finding #3):

op.execute("CREATE EXTENSION IF NOT EXISTS vector") first.
Tables: pipeline_runs, canon_store, gazetteers (+ wikidata_qid col), wikidata_properties, provenance, graph_entities, graph_relationships with the exact columns/PKs/FKs in §2.
vector(384) columns via from pgvector.sqlalchemy import Vector.
HNSW indexes via op.execute(... USING hnsw (embedding vector_cosine_ops)); partial WHERE synced_to_neo4j = FALSE indexes and expression index ((properties->>'uri')) via op.execute.
downgrade() drops all seven tables (reverse order).
ocr-resume-paser/alembic/env.py: also import ontogen.db.models so the shared Base.metadata includes Ontogen tables for future autogenerate (guarded import; hand-written migration doesn't require it).
Show the full migration file before running alembic upgrade — local dev only.

3. Ontogen DB layer — ontogen/db/ (mirrors resume_parser/db/)
All modules start with the existing sys.path.insert(0, <parser repo root>) idiom so import resume_parser… resolves.

ontogen/db/__init__.py — package marker.
ontogen/db/models.py — declares tables on the shared Base imported from resume_parser.db.models (so FKs + optional relationships to Resume resolve in one registry). Models: PipelineRun, CanonStoreEntry, Gazetteer, WikidataProperty, Provenance, GraphEntity, GraphRelationship. vector(384) columns typed via pgvector.sqlalchemy.Vector. Match models.py's docstring/Mapped[...]/mapped_column style.
ontogen/db/session.py — thin re-export: from resume_parser.db.session import make_session_factory (reuse, don't duplicate engine creation) + the fork-safety docstring note.
ontogen/db/runs.py — get_stage_output(session, document_id, stage) -> dict|None, save_stage_output(session, document_id, stage, output, status="succeeded") (upsert on PK (document_id, stage)).
ontogen/db/canon.py — search_similar(session, embedding, top_k) -> list[dict] (pgvector <=> cosine order), add_entry(session, label, definition, turtle, embedding, source_doc).
ontogen/db/gazetteers.py — lookup(session, entity_type, alias) -> str|None (index (entity_type, lower(alias))), get_qid(session, entity_type, canonical), add_alias(session, entity_type, alias, canonical, source="tier3_llm").
ontogen/db/wikidata.py — top_k_candidates(session, embedding, k) -> list[dict] (pgvector), returns {pid,label,description} shape Stage 6 expects.
ontogen/db/kg_staging.py — stage_entity(...) / stage_relationship(...) primitives, plus two writers that both land in graph_entities / graph_relationships (the shared merge point):
stage_graph(session, source_doc, turtle_str) — Path B: parse Stage 9's turtle like load_to_neo4j.parse_ttl (literal-object → node property; uri-object → edge).
stage_structured_relations(session, source_doc, relations) — Path A: for each structured_to_relations() dict, write the concrete triple directly — no Stage 9 LLM. entity objects are entity-resolved via ResumeEntityResolver (Tier 1/2/3) and become nodes/edges; literal objects become properties on the subject node (so the Person node carries email/phone/years, and dates/ grades attach to their subject). Subject is usually the résumé owner, but is the project node for usesTechnology triples. Owner node label = candidate_name.
Match Ontogen's terser docstring/logging voice in these files (not the parser's).

4. New module — ontogen/render.py
render_resume_text(structured) -> str — free-text only (see finding #2): joins projects[].description (+ summary/work_history[].description if present). Excludes skills, dates, company/institution names.
structured_to_relations(structured) -> list[dict] — deterministic, no LLM. Emits dicts carrying both the ontology-shaping fields and the concrete triple: {"property","description","subject","object","object_type","source":"structured"}. object_type = entity for named things (orgs, institutions, skills, projects, technologies), literal for scalar values (dates, grades, email, phone, years). The property/description merge into the Stage 6 input to shape the ontology; the subject/object/object_type feed stage_structured_relations() (§3) for the direct Path A triple write. Coverage — every structured field, not just some:
Person node (subject = candidate_name, all literal): email, phone, and yearsExperience (from years_experience{years,months}, serialised e.g. "5 years 3 months"). These make the owner an actual graph_entities row with properties, not just a dangling subject reference; candidate_name is its node label.
work_history[] (subject = owner): employer (entity), jobTitle (literal), startDate/endDate (literal).
education[] (subject = owner): educatedAt (entity), degree/graduationYear (literal).
skills[] (subject = owner): hasSkill (entity) per skill.
projects[]: hasProject (subject = owner, object = project.name, entity); each technologies[] → usesTechnology (subject = the project node, object = the tech, entity). projects[].description stays free text for Path B (rendered by render_resume_text), so nothing is double-covered.
5. Pipeline rewrite — ontogen/pipeline.py
process_doc(doc_path) → process_resume(resume_id, session_factory): fetch Resume, structured = resume.structured; narrative_text = render_resume_text(structured). All stages that take document text (Stages 1/2/3 and Stage 9) receive narrative_text only — no full rendering anywhere. Each *.exists() checkpoint → get_stage_output(session, resume.id, <stage>) / save_stage_output(...). struct_rels = structured_to_relations(structured); relations = stage3.run(narrative_text, cqs) + struct_rels (shapes the ontology via Stage 6). After Stage 9 stages its Path-B triples, stage_structured_relations(session, resume.id, struct_rels) writes Path A's concrete triples into the same staging tables. _run_edc_canonicalization's direct edc._entries access (pipeline.py:82-88) → canon-store label lookup.
main(): replace DOCS_DIR.glob("*.txt") with SELECT id FROM resumes; --resume = skip ids with a succeeded kg_facts pipeline_runs row.
Drop the top-level from graphdb.load_to_neo4j import parse_ttl, load_into_neo4j (Neo4j load is a separate DB-driven step now, §7).
5b. Stage-by-stage minimal edits (per doc §5 table)
stages/llm.py — rewritten to the OpenAI-compatible /v1/chat/completions route (see "LLM change" above); same call_llm signature + retry loop, strips deepseek-r1 <think> blocks.
stage1_cq_gen.py, stage2_cq_answer.py — receive narrative_text; internals unchanged. Stage output persisted via runs.py at the call site.
stage3_relation_extract.py — internals unchanged; merged with structured_to_relations() at the pipeline call site.
stage6_match_validate.py — _load()/_top_k() stop reading WIKIDATA_EMBEDDINGS/WIKIDATA_FILTERED; query wikidata.top_k_candidates(). Keep embedding the query description + LLM validation logic unchanged.
canonicalize.py — EDCBackend: _entries/_embeddings numpy scan + _load_canon_store/flush → canon.search_similar/add_entry (immediate writes; flush() becomes a no-op). ResumeEntityResolver gazetteer JSON loads → gazetteers.lookup/get_qid. Tier 1/2/3 + resolve_kg_entities logic unchanged. Backends take a session/session_factory.
stage7_8_ontology.py — internals unchanged; final write target → save_stage_output(..., "ontology", {"ttl": ontology_text}) instead of outputs/ontology/{doc}.ttl.
stage9_10_kg.py — extraction/graph-building logic unchanged (§9). Only the I/O boundary: instead of out_ttl.write_text, call kg_staging.stage_graph(session, resume_id, turtle_str); provenance via the provenance table. run() gains session/source_doc params.
provenance.py — ProvenanceStore.save() writes rows to the provenance table (same record shape: doc_id/stage/ts/confidence/model/extra) instead of a JSON path.
stage4_filter_wikidata.py — final WIKIDATA_FILTERED.write_text → insert wikidata_properties(pid,label,description). Fetch/filter unchanged. (One-time; initial seed done by §6 script.)
stage5_embed_wikidata.py — final np.save → UPDATE wikidata_properties SET embedding=…. Embedding logic unchanged. (One-time.)
config.py — change LLM_BASE_URL (above); add DATABASE_URL = os.environ.get("DATABASE_URL") and EMBED_DIM = 384. Keep CANON_STORE_DIR/GAZETTEER_DIR/WIKIDATA_* constants defined (imports still reference them) but mark unused for reads.
6. One-time seed script — ontogen/scripts/migrate_ontogen_files_to_db.py
Run once, then discarded. Seeds tables from the on-disk files:

data/gazetteers/*.json → gazetteers rows (source='static'), including wikidata_qid where the JSON's wikidata_qid map has it.
data/canon_store/entries.json → canon_store rows, re-embedding each definition with bge-small-en (no .npy exists — finding #1).
data/wikidata/properties_filtered.json → wikidata_properties rows, re-embedding each description (finding #1; not index-copied from .npy).
Show this script in full and get approval before running it against any database (local dev only).

7. Neo4j loader — graphdb/load_to_neo4j.py
Replace file-based parse_ttl with a DB-driven batched loader: read graph_entities/graph_relationships WHERE synced_to_neo4j = FALSE in batches (~500–1000), CREATE CONSTRAINT entity_uri … REQUIRE n.uri IS UNIQUE, UNWIND … MERGE keyed on properties->>'uri' (nodes) and matched endpoints (edges, grouped by rel type as today), then UPDATE … SET synced_to_neo4j = TRUE on success. graphdb/config.py unchanged (password flagged, not fixed).

8. Tests — ontogen/tests/test_ontogen_db.py (+ ontogen/conftest.py)
Real-Postgres pattern mirroring tests/test_db_ingest.py, reusing the ensure_test_db approach from ocr-resume-paser/conftest.py. ontogen/conftest.py adds an ontogen_session_factory fixture: reuse _ensure_test_db, CREATE EXTENSION IF NOT EXISTS vector, Base.metadata.create_all (shared Base incl. Ontogen tables), truncate Ontogen tables. Cases (§8):

structured_to_relations() → expected dicts from a known structured fixture, asserting full coverage: Person literals (email/phone/yearsExperience), work_history, education, skills, and projects (hasProject + per-tech usesTechnology off the project node).
render_resume_text() includes only description/summary text, excludes structured fields.
pipeline_runs round-trip + resume-mode skip logic.
gazetteer / canon-store / wikidata-properties lookups against seeded rows (incl. pgvector search_similar/top_k_candidates).
graph_entities/graph_relationships staging from a known Turtle fragment.
9b. Orchestration + shared LLM advisory lock
Both the parser and Ontogen hit one llama-server running --parallel 1, so they must not run LLM work concurrently. Add a lightweight guard that makes the parser → Ontogen sequence the default while keeping standalone Ontogen runs safe.

run_pipeline.py (HexTech repo root) — the documented "just run this" entry point. Loops over ocr-resume-paser/resumes/*.pdf, invoking the parser (python -m resume_parser.cli <pdf> --field-spec config/field_spec.json --db-uri $DATABASE_URL --artifacts-dir …, one PDF per call, via subprocess); on success, runs python ontogen/pipeline.py. Fails loudly if any parser step errors before starting Ontogen.
resume_parser/llm_lock.py (shared, in the parser package) — imported by both sides (Ontogen already imports resume_parser; the parser never imports Ontogen, so the lock lives here to keep the dependency one-directional). A context manager llm_lock(base_url) that:
Reachability pre-flight: GET {base_url}/models with a ~5s timeout; abort with a clear message if the endpoint isn't reachable.
File-based advisory lock at a fixed shared path (Path(tempfile.gettempdir()) / "hextech_llm.lock", location-independent so both processes agree), holding {pid, started_at}.
On acquire: if a lock exists and its PID is alive, print "LLM appears busy — parser or another Ontogen run may be in progress [pid X, started Y]" and exit; if the PID is dead (stale), auto-clean and proceed. Remove the lock in a finally on exit.
Symmetric wiring: wrap the LLM-calling span on both sides in llm_lock(...): parser in cli.py:main around the run_pipeline call when not --no-llm (base_url from settings.base_url); Ontogen in pipeline.py's main() around processing (base_url from config.LLM_BASE_URL). Lock is advisory + fail-safe (stale locks self-heal).
Docs: update ocr-resume-paser/docs/commands.md with both modes — python run_pipeline.py (normal two-step) and python ontogen/pipeline.py (standalone against already-parsed résumés), noting the lock check.
Lock granularity (deliberate): run_pipeline.py invokes the parser once per PDF, so the lock is acquired/released per résumé, not held across the batch — intentionally, so the single --parallel 1 slot isn't held during inter-PDF disk I/O. The only per-call overhead is the ~5s reachability GET + a tiny lock-file write, negligible at this volume (3 résumés today, fine into the dozens). If batch volume ever grows enough to matter, the orchestrator can hold one lock for the whole run instead — noted as a conscious non-issue, not a change to make now.
9. Out of scope (unchanged)
Batch/parallel processing; encryption; rewriting Stage 7/8 or 9/10 extraction/ontology logic (only I/O boundaries change). Note: §9's original "no LLM provider/model change" is deliberately overridden per the user's instruction — Ontogen now shares the parser's endpoint+model (see "LLM change").

## Verification
Migration: show 0003 + edited docker-compose.yml; on approval, docker compose up -d (local dev) then alembic upgrade head; confirm 7 tables + vector extension + HNSW indexes exist (introspect like DB_REPORT.md).
Seed: show the seed script; on approval run it; confirm row counts (gazetteers ≈ sum of 5 files, canon_store = 59, wikidata_properties = 2,936 all with non-null embedding).
Tests: pytest ontogen/tests/test_ontogen_db.py against TEST_DATABASE_URL (skips cleanly if Postgres is down, matching Part 1).
End-to-end (local dev): run python run_pipeline.py (parser → Ontogen, shared LLM http://localhost:8080/v1 / deepseek-r1-32b); also verify standalone python ontogen/pipeline.py works against the already-parsed résumés. Confirm pipeline_runs rows per stage and graph_entities/graph_relationships populated with synced_to_neo4j = FALSE. Verify Path A triples exist independently of Path B — e.g. the Person node carries email/phone/yearsExperience properties, and employer/educatedAt/ hasSkill/hasProject/usesTechnology edges are present, even where there was little/no free-text description.
LLM lock: start one run, then launch a second concurrent run — confirm the second aborts with the "LLM appears busy [pid …]" message; kill a run mid-flight and confirm the next run auto-cleans the stale lock and proceeds.
Neo4j: run the new load_to_neo4j.py; confirm nodes MERGE on uri and rows flip to synced_to_neo4j = TRUE; re-run pushes only deltas.