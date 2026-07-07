# CLAUDE.md

## Known Gotchas / Environment Notes

- CUDA builds on this machine must pin **both** `-DCMAKE_CUDA_COMPILER` and
  `-DCUDAToolkit_ROOT` explicitly (e.g. to `/usr/local/cuda-12.4`). The
  `/usr/local/cuda` symlink points to a different, mismatched toolkit
  version — pinning only the compiler still lets CMake resolve the toolkit
  root through that symlink and pull in an incompatible target.
- This GPU (RTX 4500 Ada, compute capability 8.9) only needs
  `-DCMAKE_CUDA_ARCHITECTURES=89` — no need to build for other architectures.
- `pipeline.py` mirrors every stage's output to
  `outputs/stages/<resume_uuid>/stageN_*.json|.ttl` in addition to Postgres.
  `python pipeline.py <uuid> --from-stage=<stage>` resumes from it, loading
  everything before `<stage>` from disk instead of recomputing — valid keys:
  `cq_gen, cq_answer, relation_extract, match_validate, edc_canon, ontology,
  stage9_10`. Must be `--from-stage=x` (single token) — the CLI parsing has
  no `argparse` and a space-separated value gets swallowed as the résumé UUID.
- `db/kg_staging.py`'s writers (`stage_entity`, `stage_relationship`) are
  plain `INSERT`s with **no dedup/upsert**. Re-running `pipeline.py` for an
  already-processed résumé duplicates its `graph_entities`/
  `graph_relationships` rows rather than updating them in place. Clear a
  résumé's rows first (`DELETE FROM graph_relationships/graph_entities WHERE
  source_doc = '<uuid>'`) before re-staging it if you want a clean set.
- `graphdb/load_to_neo4j.py --wipe` clears Neo4j but does **not** reset
  Postgres's `synced_to_neo4j` flags. After any manual Neo4j wipe
  (`cypher-shell ... MATCH (n) DETACH DELETE n`), reloading only pushes rows
  still marked `synced_to_neo4j = FALSE` — any résumé whose rows were already
  synced before the wipe silently disappears from Neo4j and won't come back
  without resetting its flags too
  (`UPDATE graph_entities/graph_relationships SET synced_to_neo4j = FALSE
  WHERE source_doc = '<uuid>'`).
- Entity resolution (`ResumeEntityResolver` in `stages/canonicalize.py`) has
  3 tiers: gazetteer exact-match, gazetteer embedding match, LLM normalization.
  Tier 2 (embedding) now requires an LLM to confirm a match before accepting
  it (mirroring `stage6_match_validate.py`'s design for relationship/property
  matching) — cosine similarity alone was observed accepting wrong entities
  (e.g. "Air University" → "Brown University" at 0.884) because generic
  shared words inflate similarity between short, otherwise-unrelated names.
  Wikidata's own embedding space (`wikidata_properties`) is used only for
  *relationship* matching in `stage6_match_validate.py`, never for entities —
  entity Tier 2 matches against the local gazetteer's own canonical values,
  not Wikidata directly.
- No `psql` client on the host. Postgres runs in Docker
  (`ocr-resume-paser-postgres-1`) — query it via `sudo docker exec
  ocr-resume-paser-postgres-1 psql -U resume_parser -d resume_parser -c
  "..."`. Neo4j: `cypher-shell -a bolt://127.0.0.1:7687 -u neo4j -p
  5053811238 "..."`.
- When root-causing a resolver/embedding-matching bug, reproduce against the
  exact production embedding model name (`config.EMBED_MODEL`) and the full
  real candidate set/gazetteer — a smaller hand-picked candidate list or a
  slightly different model variant can show a similarity score safely under
  threshold when the real path is actually over it.
