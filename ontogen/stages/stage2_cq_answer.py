"""Stage 2 — CQ Answering (verbatim prompt from paper, one CQ per call).

Fix log:
- SUBJECT TAGGING passthrough (root cause #1 follow-up): Stage 1 now emits
  CQs as list[dict] ({"subject": ..., "question": ...}) instead of
  list[str], so that Stage 9/10 can group QA pairs by the entity each
  question is actually about instead of guessing from question phrasing.
  This stage is the pass-through point — it must carry that "subject"
  field into each QA pair unchanged, or the tag never reaches Stage
  9/10 and the whole point of tagging at the source is lost.
- Backward compatibility: if this is ever run against an older CQ file
  that's still a flat list[str] (e.g. output/cqs/*.json generated before
  this change), each item is treated as an untagged question with
  subject="_unlabeled" rather than crashing. This is intentionally
  degraded (not grouped downstream) rather than silently guessed — see
  stage9_10_kg.py's fix log for why guessing was rejected as the
  approach here.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from stages.llm import call_llm
from config import OUTPUTS_DIR

PROMPT_TEMPLATE = """\
Use the provided document to answer user query. If you don't
know the answer, just say that you don't know, don't try to
make up an answer.

Passage: {doc}

Query: {query}
"""


def _normalize_cq(cq) -> dict:
    """Accept either the current {"subject": ..., "question": ...} shape
    or a legacy plain string, and always return the dict shape. Legacy
    strings get subject="_unlabeled" so they're visibly ungrouped
    downstream instead of silently mis-grouped."""
    if isinstance(cq, dict):
        subject = str(cq.get("subject") or "_unlabeled").strip() or "_unlabeled"
        question = str(cq.get("question", "")).strip()
        return {"subject": subject, "question": question}
    # legacy: plain string
    return {"subject": "_unlabeled", "question": str(cq).strip()}


def answer_cqs(doc_text: str, cqs: list) -> list[dict]:
    qa_pairs = []
    for raw_cq in cqs:
        cq = _normalize_cq(raw_cq)
        if not cq["question"]:
            continue
        prompt = PROMPT_TEMPLATE.format(doc=doc_text, query=cq["question"])
        answer = call_llm(prompt)
        qa_pairs.append({
            "question": cq["question"],
            "answer": answer,
            "subject": cq["subject"],
        })
    return qa_pairs


def run(doc_name: str, doc_text: str, cqs: list) -> list[dict]:
    qa_pairs = answer_cqs(doc_text, cqs)
    out = OUTPUTS_DIR / "answers" / f"{doc_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(qa_pairs, indent=2))
    n_subjects = len({p["subject"] for p in qa_pairs})
    print(f"[Stage 2] {len(qa_pairs)} QA pairs across {n_subjects} subject(s) → {out}")
    return qa_pairs