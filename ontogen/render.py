"""Turn a stored résumé's `structured` dict into pipeline inputs.

Two functions, one per extraction path:

  render_resume_text()     — Path B input: the résumé's *free text* only
                             (project descriptions, and defensively a summary /
                             work-history descriptions if a future field spec
                             adds them). Everything already broken into named
                             fields is deliberately excluded — those go to Path A.

  structured_to_relations() — Path A: every already-structured field turned into
                             a relation dict, deterministically, with no LLM.
                             Each dict carries both the ontology-shaping fields
                             (property/description, merged before Stage 6) and the
                             concrete triple (subject/object/object_type) that
                             db.kg_staging.stage_structured_relations() writes
                             straight into the graph tables.

The confirmed field spec has free text only in projects[].description, so that's
what render_resume_text() emits today; the summary / work_history description
handling is defensive for forward-compatibility.
"""
from __future__ import annotations


def render_resume_text(structured: dict) -> str:
    """Free-text fields only, for CQ-driven (Path B) extraction."""
    parts: list[str] = []

    summary = (structured.get("summary") or "").strip()
    if summary:
        parts.append(summary)

    for job in structured.get("work_history") or []:
        desc = (job.get("description") or "").strip()
        if desc:
            parts.append(desc)

    for proj in structured.get("projects") or []:
        desc = (proj.get("description") or "").strip()
        if desc:
            name = (proj.get("name") or "").strip()
            parts.append(f"{name}: {desc}" if name else desc)

    return "\n\n".join(parts)


def _rel(property: str, description: str, subject: str, obj: str, object_type: str,
         subject_type: str = "person", object_entity_type: str | None = None,
         edge_props: dict | None = None) -> dict:
    return {
        "property": property,
        "description": description,
        "subject": subject,
        "subject_type": subject_type,
        "object": obj,
        "object_type": object_type,
        "object_entity_type": object_entity_type,
        # Optional properties to hang on the edge itself (entity objects only),
        # e.g. {"proficiency": "C2"} on a speaksLanguage edge. kg_staging passes
        # these through to graph_relationships.properties.
        "edge_props": edge_props or None,
        "source": "structured",
    }


def _years_experience_text(ye: dict | None) -> str | None:
    if not isinstance(ye, dict):
        return None
    years = ye.get("years")
    months = ye.get("months")
    bits = []
    if years:
        bits.append(f"{years} year{'s' if years != 1 else ''}")
    if months:
        bits.append(f"{months} month{'s' if months != 1 else ''}")
    return " ".join(bits) or None


def structured_to_relations(structured: dict) -> list[dict]:
    """Deterministic relation extraction (Path A) — no LLM call.

    Covers *every* structured field: the Person node's own literals, work
    history, education, skills, and projects (incl. per-technology edges hung off
    the project node). projects[].description is intentionally left to Path B.
    """
    owner = (structured.get("candidate_name") or "").strip() or "Unknown"
    rels: list[dict] = []

    # ── Person node literals (make the owner an actual node with properties) ──
    email = (structured.get("email") or "").strip()
    if email:
        rels.append(_rel("email", "The email address of the person.", owner, email, "literal"))
    phone = (structured.get("phone") or "").strip()
    if phone:
        rels.append(_rel("phone", "The contact phone number of the person.", owner, phone, "literal"))
    ye_text = _years_experience_text(structured.get("years_experience"))
    if ye_text:
        rels.append(_rel(
            "yearsExperience",
            "Total professional experience of the person.",
            owner, ye_text, "literal",
        ))

    # ── Work history ──────────────────────────────────────────────────────────
    for job in structured.get("work_history") or []:
        company = (job.get("company") or "").strip()
        if company:
            rels.append(_rel(
                "employer", "The organization the person worked at.",
                owner, company, "entity", object_entity_type="company",
            ))
        title = (job.get("title") or "").strip()
        if title:
            rels.append(_rel("jobTitle", "A job title the person held.", owner, title, "literal"))
        start = (job.get("start_date") or "").strip()
        if start:
            rels.append(_rel("startDate", "The start date of a role.", owner, start, "literal"))
        end = (job.get("end_date") or "").strip()
        if end:
            rels.append(_rel("endDate", "The end date of a role.", owner, end, "literal"))

    # ── Education ─────────────────────────────────────────────────────────────
    for edu in structured.get("education") or []:
        institution = (edu.get("institution") or "").strip()
        if institution:
            rels.append(_rel(
                "educatedAt", "An institution the person was educated at.",
                owner, institution, "entity", object_entity_type="university",
            ))
        degree = (edu.get("degree") or "").strip()
        if degree:
            rels.append(_rel("degree", "A degree the person earned.", owner, degree, "literal"))
        edu_start = (edu.get("start_date") or "").strip()
        if edu_start:
            rels.append(_rel("educationStartDate", "The start date of a program of study.", owner, edu_start, "literal"))
        edu_end = (edu.get("end_date") or "").strip()
        if edu_end:
            rels.append(_rel("educationEndDate", "The end date of a program of study ('present' if ongoing).", owner, edu_end, "literal"))
        grad = (edu.get("graduation_year") or "").strip()
        if grad:
            rels.append(_rel("graduationYear", "The graduation year of a degree.", owner, grad, "literal"))

    # ── Skills ────────────────────────────────────────────────────────────────
    for skill in structured.get("skills") or []:
        skill = (skill or "").strip()
        if skill:
            rels.append(_rel(
                "hasSkill", "A skill the person has.",
                owner, skill, "entity", object_entity_type="skill",
            ))

    # ── Projects (name → owner; technologies → the project node) ─────────────
    for proj in structured.get("projects") or []:
        name = (proj.get("name") or "").strip()
        if not name:
            continue
        rels.append(_rel(
            "hasProject", "A project the person built or contributed to.",
            owner, name, "entity", object_entity_type="project",
        ))
        for tech in proj.get("technologies") or []:
            tech = (tech or "").strip()
            if tech:
                rels.append(_rel(
                    "usesTechnology", "A technology used in a project.",
                    name, tech, "entity",
                    subject_type="project", object_entity_type="technology",
                ))
        # Quantitative results attach as literal properties on the PROJECT node,
        # so "which project hit >90% accuracy" is answerable off that node.
        for metric in proj.get("metrics") or []:
            metric = (metric or "").strip()
            if metric:
                rels.append(_rel(
                    "achievesMetric", "A quantitative result reported for a project.",
                    name, metric, "literal", subject_type="project",
                ))

    # ── Languages (spoken/written) — proficiency rides on the edge ───────────
    for lang in structured.get("languages") or []:
        lname = (lang.get("name") or "").strip()
        if not lname:
            continue
        prof = (lang.get("proficiency") or "").strip()
        rels.append(_rel(
            "speaksLanguage", "A human language the person knows.",
            owner, lname, "entity", object_entity_type="language",
            edge_props={"proficiency": prof} if prof else None,
        ))

    # ── Certifications (issuer/year hang on the certification node) ──────────
    for cert in structured.get("certifications") or []:
        cname = (cert.get("name") or "").strip()
        if not cname:
            continue
        rels.append(_rel(
            "hasCertification", "A certification or credential the person earned.",
            owner, cname, "entity", object_entity_type="certification",
        ))
        issuer = (cert.get("issuer") or "").strip()
        if issuer:
            rels.append(_rel("issuer", "The organization that issued a certification.",
                             cname, issuer, "literal", subject_type="certification"))
        cyear = (cert.get("year") or "").strip()
        if cyear:
            rels.append(_rel("certificationYear", "The year a certification was obtained.",
                             cname, cyear, "literal", subject_type="certification"))

    # ── Activities / memberships (organization/description on the node) ──────
    for act in structured.get("activities") or []:
        aname = (act.get("name") or "").strip()
        if not aname:
            continue
        rels.append(_rel(
            "participatedIn", "An activity, membership, or extracurricular of the person.",
            owner, aname, "entity", object_entity_type="activity",
        ))
        org = (act.get("organization") or "").strip()
        if org:
            rels.append(_rel("activityOrganization", "The organization associated with an activity.",
                             aname, org, "literal", subject_type="activity"))
        adesc = (act.get("description") or "").strip()
        if adesc:
            rels.append(_rel("activityDescription", "What an activity involved.",
                             aname, adesc, "literal", subject_type="activity"))

    # ── References (title/contact hang on the reference person node) ─────────
    for ref in structured.get("references") or []:
        rname = (ref.get("name") or "").strip()
        if not rname:
            continue
        rels.append(_rel(
            "hasReference", "A person listed as a reference for the candidate.",
            owner, rname, "entity", object_entity_type="person",
        ))
        rtitle = (ref.get("title") or "").strip()
        if rtitle:
            rels.append(_rel("referenceTitle", "The title/affiliation of a reference.",
                             rname, rtitle, "literal", subject_type="person"))
        rcontact = (ref.get("contact") or "").strip()
        if rcontact:
            rels.append(_rel("referenceContact", "Contact detail of a reference.",
                             rname, rcontact, "literal", subject_type="person"))

    return rels
