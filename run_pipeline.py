"""Top-level entry point: parse résumés into Postgres, then build the KG.

The normal "just run this" command for the two-step flow:

    python run_pipeline.py

It runs the parser over every PDF in ocr-resume-paser/resumes/ (one process per
PDF, persisting to Postgres via --db-uri), and on success runs the Ontogen
pipeline over the résumés now in the database.

Both steps take the shared advisory LLM lock (resume_parser.llm_lock) around
their LLM work, so they never collide on the single --parallel 1 inference slot.
The lock is acquired/released per parser invocation — deliberately, so the slot
isn't held during inter-PDF disk I/O (see docs/commands.md).

Ontogen can also be run on its own (python ontogen/pipeline.py) against résumés
already in the database; this orchestrator is just the convenient default.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
PARSER_DIR = ROOT / "ocr-resume-paser"
ONTOGEN_DIR = ROOT / "ontogen"
RESUMES_DIR = PARSER_DIR / "resumes"


def _database_url() -> str:
    load_dotenv(PARSER_DIR / ".env", override=False)
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit(
            "DATABASE_URL is not set. Add it to ocr-resume-paser/.env (matches "
            "docker-compose.yml) or export it before running."
        )
    return url


def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
    import subprocess

    print(f"\n$ (cd {cwd.name} && {' '.join(cmd)})", flush=True)
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        sys.exit(f"✗ step failed (exit {result.returncode}): {' '.join(cmd)}")


def main() -> None:
    db_uri = _database_url()

    pdfs = sorted(RESUMES_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {RESUMES_DIR}; skipping parse step.")
    else:
        print(f"Parsing {len(pdfs)} résumé(s) → {db_uri}")
        for pdf in pdfs:
            _run(
                [
                    sys.executable, "-m", "resume_parser.cli", str(pdf),
                    "--field-spec", "config/field_spec.json",
                    "--db-uri", db_uri,
                    "--artifacts-dir", f"artifacts/{pdf.stem}",
                ],
                cwd=PARSER_DIR,
            )

    # Ontogen reads DATABASE_URL from the environment; pass it through.
    env = dict(os.environ)
    env["DATABASE_URL"] = db_uri
    print("\nBuilding the knowledge graph (Ontogen) …")
    _run([sys.executable, "pipeline.py"], cwd=ONTOGEN_DIR, env=env)

    print("\n✓ Pipeline complete. Load into Neo4j with: "
          "python ontogen/graphdb/load_to_neo4j.py")


if __name__ == "__main__":
    main()
