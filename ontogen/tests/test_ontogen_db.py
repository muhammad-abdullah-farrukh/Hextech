"""Real-Postgres tests for the Ontogen DB layer (Part 2).

Same pattern as ocr_resume_parser's test_db_ingest.py: exercised against a real
test database (pgvector-enabled), skipped cleanly when Postgres is down. The two
pure-Python tests (render / structured_to_relations) run without a database.
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ontogen root

from render import render_resume_text, structured_to_relations

STRUCTURED = {
    "candidate_name": "Jane Doe",
    "email": "jane@example.com",
    "phone": "555-1234",
    "years_experience": {"years": 5, "months": 3},
    "skills": ["Python", "SQL"],
    "work_history": [
        {"company": "Acme", "title": "Engineer", "start_date": "2020", "end_date": "2022"},
    ],
    "education": [
        {"institution": "MIT", "degree": "BSc", "graduation_year": "2019"},
        {"institution": "GIKI", "degree": "BS AI", "start_date": "09/2023", "end_date": "present"},
    ],
    "projects": [
        {"name": "Widget", "description": "A widget that widgets",
         "technologies": ["Python", "FastAPI"], "metrics": ["78.58% accuracy", "F1 0.91"]},
    ],
    "languages": [
        {"name": "English", "proficiency": "C2"},
        {"name": "Arabic", "proficiency": "A1"},
    ],
    "certifications": [
        {"name": "AWS Certified", "issuer": "Amazon", "year": "2022"},
    ],
    "activities": [
        {"name": "IEEE Power & Energy", "organization": "IEEE"},
    ],
    "references": [
        {"name": "Dr. Ali Sarosh", "title": "Professor", "contact": "ali@example.com"},
    ],
}


def _vec(i: int) -> list[float]:
    """A 384-dim unit vector with a 1.0 at position i (orthogonal across i)."""
    v = [0.0] * 384
    v[i % 384] = 1.0
    return v


# ── Pure-Python (no DB) ──────────────────────────────────────────────────────

def test_structured_to_relations_covers_all_fields():
    rels = structured_to_relations(STRUCTURED)
    by_prop = {}
    for r in rels:
        assert r["source"] == "structured"
        by_prop.setdefault(r["property"], []).append(r)

    # Person literals live on the owner as literal objects.
    for prop in ("email", "phone", "yearsExperience"):
        assert by_prop[prop][0]["object_type"] == "literal"
        assert by_prop[prop][0]["subject"] == "Jane Doe"
    assert by_prop["yearsExperience"][0]["object"] == "5 years 3 months"

    # Work history / education entity + literal facts.
    assert by_prop["employer"][0]["object"] == "Acme"
    assert by_prop["employer"][0]["object_entity_type"] == "company"
    assert by_prop["educatedAt"][0]["object"] == "MIT"
    assert by_prop["educatedAt"][0]["object_entity_type"] == "university"

    # Skills — one entity relation each.
    assert {r["object"] for r in by_prop["hasSkill"]} == {"Python", "SQL"}

    # Projects: hasProject off the owner; usesTechnology off the PROJECT node.
    assert by_prop["hasProject"][0]["object"] == "Widget"
    tech = by_prop["usesTechnology"]
    assert all(t["subject"] == "Widget" and t["subject_type"] == "project" for t in tech)
    assert {t["object"] for t in tech} == {"Python", "FastAPI"}

    # Project metrics attach as literals on the PROJECT node.
    metrics = by_prop["achievesMetric"]
    assert all(m["subject"] == "Widget" and m["subject_type"] == "project" for m in metrics)
    assert {m["object"] for m in metrics} == {"78.58% accuracy", "F1 0.91"}

    # Education ongoing entry keeps start/end and no invented graduation year.
    assert by_prop["educationStartDate"][0]["object"] == "09/2023"
    assert by_prop["educationEndDate"][0]["object"] == "present"
    assert {g["object"] for g in by_prop["graduationYear"]} == {"2019"}  # only the completed degree

    # Languages: entity + proficiency on the edge.
    langs = {l["object"]: l for l in by_prop["speaksLanguage"]}
    assert set(langs) == {"English", "Arabic"}
    assert langs["English"]["edge_props"] == {"proficiency": "C2"}
    assert all(l["object_entity_type"] == "language" for l in langs.values())

    # Certifications: entity off owner; issuer/year on the cert node.
    assert by_prop["hasCertification"][0]["object"] == "AWS Certified"
    assert by_prop["issuer"][0]["subject"] == "AWS Certified"
    assert by_prop["certificationYear"][0]["object"] == "2022"

    # Activities: entity off owner; organization on the activity node.
    assert by_prop["participatedIn"][0]["object"] == "IEEE Power & Energy"
    assert by_prop["activityOrganization"][0]["subject"] == "IEEE Power & Energy"

    # References: entity off owner; title/contact on the reference node.
    assert by_prop["hasReference"][0]["object"] == "Dr. Ali Sarosh"
    assert by_prop["referenceTitle"][0]["subject"] == "Dr. Ali Sarosh"
    assert by_prop["referenceContact"][0]["object"] == "ali@example.com"


def test_render_resume_text_only_free_text():
    text = render_resume_text(STRUCTURED)
    assert "A widget that widgets" in text  # project description (free text)
    assert "Acme" not in text and "MIT" not in text  # structured names excluded
    assert "555-1234" not in text  # structured scalar excluded


# ── DB-backed ────────────────────────────────────────────────────────────────

def _make_resume(session, structured=None) -> uuid.UUID:
    from resume_parser.db.models import Resume

    r = Resume(
        pdf_hash=uuid.uuid4().hex,
        source_file="test.pdf",
        structured=structured or STRUCTURED,
        field_spec_hash="deadbeef",
    )
    session.add(r)
    session.commit()
    return r.id


def test_pipeline_runs_roundtrip_and_resume_skip(ontogen_session_factory):
    from db import runs

    with ontogen_session_factory() as session:
        rid = _make_resume(session)
        assert runs.get_stage_output(session, rid, runs.CQ_GEN) is None
        assert runs.has_succeeded(session, rid, runs.KG_FACTS) is False

        runs.save_stage_output(session, rid, runs.CQ_GEN, [{"subject": "person", "question": "Q?"}])
        out = runs.get_stage_output(session, rid, runs.CQ_GEN)
        assert out == [{"subject": "person", "question": "Q?"}]

        # Upsert overwrites, not appends.
        runs.save_stage_output(session, rid, runs.CQ_GEN, [{"subject": "x", "question": "Y?"}])
        assert runs.get_stage_output(session, rid, runs.CQ_GEN)[0]["subject"] == "x"

        runs.save_stage_output(session, rid, runs.KG_FACTS, {"staged": True})
        assert runs.has_succeeded(session, rid, runs.KG_FACTS) is True


def test_gazetteer_lookup(ontogen_session_factory):
    from db import gazetteers
    from db.models import Gazetteer

    with ontogen_session_factory() as session:
        session.add_all([
            Gazetteer(entity_type="company", alias="google llc", canonical="Google",
                      wikidata_qid="Q95", source="static"),
            Gazetteer(entity_type="company", alias="msft", canonical="Microsoft", source="static"),
            Gazetteer(entity_type="skill", alias="py", canonical="Python", source="static"),
        ])
        session.commit()

        assert gazetteers.lookup(session, "company", "Google LLC") == "Google"  # case-insensitive
        assert gazetteers.lookup(session, "company", "unknown") is None
        assert gazetteers.get_qid(session, "company", "Google") == "Q95"
        assert gazetteers.get_qid(session, "company", "Microsoft") is None
        vals = gazetteers.canonical_values(session, "company")
        assert set(vals) == {"Google", "Microsoft"}


def test_canon_search_similar(ontogen_session_factory):
    from db import canon
    from db.models import CanonStoreEntry

    with ontogen_session_factory() as session:
        session.add_all([
            CanonStoreEntry(label="employer", definition="works at", turtle="t1", embedding=_vec(0)),
            CanonStoreEntry(label="skill", definition="has skill", turtle="t2", embedding=_vec(1)),
        ])
        session.commit()

        top = canon.search_similar(session, _vec(0), top_k=2)
        assert top[0]["label"] == "employer"  # nearest to _vec(0)
        assert top[0]["cos_score"] > top[1]["cos_score"]
        assert canon.find_by_label(session, "SKILL")["definition"] == "has skill"
        assert canon.find_by_label(session, "nope") is None

        canon.add_entry(session, "novel", "a new prop", "t3", _vec(2), source_doc=None)
        assert canon.find_by_label(session, "novel") is not None
        # Dedup by label — second add is a no-op.
        canon.add_entry(session, "novel", "dup", "t4", _vec(3), source_doc=None)
        assert canon.find_by_label(session, "novel")["definition"] == "a new prop"


def test_wikidata_top_k(ontogen_session_factory):
    from db import wikidata
    from db.models import WikidataProperty

    with ontogen_session_factory() as session:
        session.add_all([
            WikidataProperty(pid="P108", label="Employer", description="employer", embedding=_vec(0)),
            WikidataProperty(pid="P69", label="EducatedAt", description="educated at", embedding=_vec(1)),
            WikidataProperty(pid="P1", label="Other", description="other", embedding=_vec(2)),
        ])
        session.commit()

        top = wikidata.top_k_candidates(session, _vec(1), k=2)
        assert top[0]["pid"] == "P69"
        assert {"pid", "label", "description", "cos_score"} <= set(top[0])


def test_stage_graph_from_turtle(ontogen_session_factory):
    from db import kg_staging
    from db.models import GraphEntity, GraphRelationship
    from sqlalchemy import func, select

    turtle = (
        "@prefix wd: <http://www.wikidata.org/entity/> .\n"
        "@prefix wdt: <http://www.wikidata.org/prop/direct/> .\n"
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
        'wd:Jane_Doe rdfs:label "Jane Doe"@en .\n'
        'wd:Acme rdfs:label "Acme"@en .\n'
        "wd:Jane_Doe wdt:employer wd:Acme .\n"
        'wd:Jane_Doe wdt:jobTitle "Engineer" .\n'
    )
    with ontogen_session_factory() as session:
        rid = _make_resume(session)
        kg_staging.stage_graph(session, rid, turtle)

        ents = session.execute(select(GraphEntity)).scalars().all()
        by_uri = {e.properties["uri"]: e for e in ents}
        assert set(by_uri) == {
            "http://www.wikidata.org/entity/Jane_Doe",
            "http://www.wikidata.org/entity/Acme",
        }
        jane = by_uri["http://www.wikidata.org/entity/Jane_Doe"]
        assert jane.properties["jobTitle"] == "Engineer"  # literal → node property

        rels = session.execute(select(GraphRelationship)).scalars().all()
        assert len(rels) == 1
        assert rels[0].rel_type == "EMPLOYER"


def test_resolver_gates_fuzzy_by_type(ontogen_session_factory):
    """Fix 4: skills/technologies/persons are kept verbatim (no Tier 2/3), so a
    skill mention absent from the gazetteer is NOT snapped to a neighbour or
    rewritten by the LLM. Only CANONICALIZE_TYPES reach the fuzzy tiers."""
    from stages.canonicalize import ResumeEntityResolver, CANONICALIZE_TYPES
    from db.models import Gazetteer
    from config import EMBED_MODEL

    assert "skill" not in CANONICALIZE_TYPES and "company" in CANONICALIZE_TYPES

    with ontogen_session_factory() as session:
        # A gazetteer canonical that a naive embedding match might snap onto.
        session.add(Gazetteer(entity_type="skill", alias="aws lambda",
                              canonical="AWS Lambda", source="static"))
        session.commit()

    resolver = ResumeEntityResolver(ontogen_session_factory, EMBED_MODEL)

    # Skill with no exact alias → verbatim (never touches embeddings/LLM).
    res = resolver.resolve("AWS (EC2, ECR, IAM)", "skill")
    assert res.resolution_tier == "verbatim"
    assert res.canonical_form == "AWS (EC2, ECR, IAM)"

    # Exact gazetteer alias still resolves via Tier 1, even for a gated type.
    res2 = resolver.resolve("AWS Lambda", "skill")
    assert res2.resolution_tier == "gazetteer"
    assert res2.canonical_form == "AWS Lambda"

    # A reference name (person, no gazetteer) is preserved, not LLM-rewritten.
    res3 = resolver.resolve("Dr. Ali Sarosh", "person")
    assert res3.resolution_tier == "verbatim"
    assert res3.canonical_form == "Dr. Ali Sarosh"


def test_stage_structured_relations(ontogen_session_factory):
    from types import SimpleNamespace

    from db import kg_staging
    from db.models import GraphEntity, GraphRelationship
    from sqlalchemy import select

    class FakeResolver:
        def resolve(self, mention, entity_type):
            # Canonicalize "Acme" → "Acme Corp" with a QID; pass others through.
            if mention == "Acme":
                return SimpleNamespace(canonical_form="Acme Corp", wikidata_qid="Q1")
            return SimpleNamespace(canonical_form=mention, wikidata_qid=None)

    rels = structured_to_relations(STRUCTURED)
    with ontogen_session_factory() as session:
        rid = _make_resume(session)
        kg_staging.stage_structured_relations(session, rid, rels, resolver=FakeResolver())

        ents = {e.properties["uri"]: e for e in session.execute(select(GraphEntity)).scalars()}
        # Person node exists with its literal properties.
        person = ents["http://www.wikidata.org/entity/Jane_Doe"]
        assert person.properties["email"] == "jane@example.com"
        assert person.properties["yearsExperience"] == "5 years 3 months"

        # Resolved employer got the QID-based URI.
        assert "http://www.wikidata.org/entity/Q1" in ents

        # Project node carries its metrics as literal properties.
        widget = ents["http://www.wikidata.org/entity/Widget"]
        assert set(widget.properties.get("achievesMetric")) == {"78.58% accuracy", "F1 0.91"}

        rels_rows = session.execute(select(GraphRelationship)).scalars().all()
        rel_types = {r.rel_type for r in rels_rows}
        assert {
            "EMPLOYER", "EDUCATED_AT", "HAS_SKILL", "HAS_PROJECT", "USES_TECHNOLOGY",
            "SPEAKS_LANGUAGE", "HAS_CERTIFICATION", "PARTICIPATED_IN", "HAS_REFERENCE",
        } <= rel_types

        # Language proficiency rode through as an edge property.
        lang_rels = [r for r in rels_rows if r.rel_type == "SPEAKS_LANGUAGE"]
        profs = {(r.properties or {}).get("proficiency") for r in lang_rels}
        assert profs == {"C2", "A1"}
