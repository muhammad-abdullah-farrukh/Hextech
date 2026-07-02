"""Gazetteer lookups — replaces loading the 5 data/gazetteers/*.json files into
memory at startup.

lookup() is the Tier-1 alias→canonical resolution; get_qid() recovers the
canonical→Wikidata-QID mapping (stored in the gazetteers.wikidata_qid column)
so entity URIs can be built as wd:Q… when known.
"""
from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import Gazetteer


def lookup(session: Session, entity_type: str, alias: str) -> str | None:
    """Return the canonical form for an alias of `entity_type`, or None."""
    return session.execute(
        select(Gazetteer.canonical).where(
            Gazetteer.entity_type == entity_type,
            func.lower(Gazetteer.alias) == alias.lower().strip(),
        )
    ).scalars().first()


def canonical_values(session: Session, entity_type: str) -> list[str]:
    """Distinct canonical forms for an entity type — the Tier-2 embedding
    targets that used to come from the in-memory gazetteer's values."""
    return list(
        session.execute(
            select(Gazetteer.canonical)
            .where(Gazetteer.entity_type == entity_type)
            .distinct()
        ).scalars()
    )


def get_qid(session: Session, entity_type: str, canonical: str) -> str | None:
    """Return the Wikidata QID for a canonical value, if one is recorded."""
    return session.execute(
        select(Gazetteer.wikidata_qid)
        .where(
            Gazetteer.entity_type == entity_type,
            Gazetteer.canonical == canonical,
            Gazetteer.wikidata_qid.isnot(None),
        )
        .limit(1)
    ).scalars().first()


def add_alias(
    session: Session,
    entity_type: str,
    alias: str,
    canonical: str,
    source: str = "tier3_llm",
    wikidata_qid: str | None = None,
) -> None:
    """Insert a learned/seeded alias→canonical row and commit.

    Dedups on (entity_type, lower(alias)) via a partial-safe check so re-runs and
    Tier-3 learning don't pile up duplicates.
    """
    exists = session.execute(
        select(Gazetteer.id).where(
            Gazetteer.entity_type == entity_type,
            func.lower(Gazetteer.alias) == alias.lower().strip(),
        )
    ).first()
    if exists is not None:
        return
    session.add(
        Gazetteer(
            entity_type=entity_type,
            alias=alias,
            canonical=canonical,
            wikidata_qid=wikidata_qid,
            source=source,
        )
    )
    session.commit()
