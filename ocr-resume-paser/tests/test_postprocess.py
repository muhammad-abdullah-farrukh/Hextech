import numpy as np

from resume_parser import postprocess
from resume_parser.postprocess import (
    _cluster_by_similarity,
    _merge_project_group,
    backfill_languages,
    backfill_skills,
    canonicalize_tech,
    dedupe_cert_activity,
    dedupe_projects,
    drop_unsupported_project_dates,
    filter_skills,
    normalize_work_roles,
    validate_metrics,
)

PROJECT_SPEC = [
    {
        "name": "projects",
        "type": "array",
        "items": "object",
        "properties": [
            {"name": "name", "type": "string"},
            {"name": "description", "type": "string"},
            {"name": "technologies", "type": "array", "items": "string"},
            {"name": "metrics", "type": "array", "items": "string"},
            {"name": "start_date", "type": "string"},
            {"name": "end_date", "type": "string"},
        ],
    }
]

SKILL_SPEC = [{"name": "skills", "type": "array", "items": "string"}] + PROJECT_SPEC

WORK_SPEC = [
    {
        "name": "work_history",
        "type": "array",
        "items": "object",
        "properties": [
            {"name": "company", "type": "string"},
            {"name": "title", "type": "string"},
            {"name": "location", "type": "string"},
        ],
    }
]

CERT_ACT_SPEC = [
    {
        "name": "certifications",
        "type": "array",
        "items": "object",
        "properties": [{"name": "name", "type": "string"}],
    },
    {
        "name": "activities",
        "type": "array",
        "items": "object",
        "properties": [
            {"name": "name", "type": "string"},
            {"name": "description", "type": "string"},
        ],
    },
]


# --- #3 clustering + merge -------------------------------------------------------

def test_cluster_merges_near_and_keeps_distinct():
    v = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]])
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    groups = _cluster_by_similarity(v, 0.9)
    assert [0, 1] in groups
    assert [2] in groups


def test_merge_project_group_first_name_longest_desc_union_lists():
    a = {"name": "P1", "description": "short", "technologies": ["x"], "metrics": ["1%"]}
    b = {
        "name": "P1-variant",
        "description": "a much longer, more detailed description",
        "technologies": ["y", "x"],
        "metrics": ["2x"],
    }
    out = _merge_project_group([a, b], "name", "description", ("technologies", "metrics"))
    assert out["name"] == "P1"  # first entry's name
    assert out["description"] == "a much longer, more detailed description"  # longest
    assert out["technologies"] == ["x", "y"]  # union, order preserved
    assert out["metrics"] == ["1%", "2x"]


def test_dedupe_projects_merges_with_injected_embedder(monkeypatch):
    projs = [
        {"name": "1v1 Pursuit-Evasion", "description": "short", "technologies": ["JSBSim"], "metrics": ["1%"]},
        {"name": "Multi-Agent Pursuit-Evasion", "description": "a longer detailed description", "technologies": ["OpenAI Gym"], "metrics": ["1%", "2x"]},
        {"name": "Acoustic Comms", "description": "totally different work", "technologies": ["MATLAB"]},
    ]

    class _Fake:
        def encode(self, texts, normalize_embeddings=True):
            return np.array([[1.0, 0.0], [0.999, 0.0447], [0.0, 1.0]])

    monkeypatch.setattr(postprocess, "_get_embedder", lambda name: _Fake())
    out = dedupe_projects({"projects": projs}, PROJECT_SPEC, threshold=0.9)
    assert len(out["projects"]) == 2
    merged = out["projects"][0]
    assert merged["name"] == "1v1 Pursuit-Evasion"
    assert merged["description"] == "a longer detailed description"
    assert merged["technologies"] == ["JSBSim", "OpenAI Gym"]
    assert merged["metrics"] == ["1%", "2x"]


def test_dedupe_projects_noop_when_embedder_unavailable(monkeypatch):
    def _boom(name):
        raise RuntimeError("no model")

    monkeypatch.setattr(postprocess, "_get_embedder", _boom)
    projs = [{"name": "A", "description": "x"}, {"name": "B", "description": "y"}]
    out = dedupe_projects({"projects": list(projs)}, PROJECT_SPEC)
    assert out["projects"] == projs  # unchanged, no crash


# --- #6 skills filter -----------------------------------------------------------

def test_filter_skills_drops_noise_keeps_atomic():
    result = {
        "skills": [
            "Python",
            "Retrieval-Augmented Generation (RAG)",
            "Measurable Business Impact",             # blocklist
            "Hardware Aware Machine Learning Systems",  # > 4 words
            "ASL Translation Project",                # ~= a project title
            "Python",                                 # dup
        ],
        "projects": [{"name": "ASL Translation Project", "description": "d"}],
    }
    out = filter_skills(result, SKILL_SPEC)
    assert out["skills"] == ["Python", "Retrieval-Augmented Generation (RAG)"]


def test_filter_skills_keeps_short_skill_that_is_substring_of_project_title():
    # Regression: "CI/CD"/"GitHub"/"WebGL" appear INSIDE long project titles but
    # are real skills — they must survive (token_set_ratio used to drop them).
    result = {
        "skills": ["CI/CD", "GitHub Actions", "WebGL", "Data Visualization"],
        "projects": [
            {"name": "WiChat DevOps — CI/CD Pipeline with Docker, AWS ECR & GitHub Actions"},
            {"name": "The Chase — WebGL Scrollytelling Data Visualization Experience"},
        ],
    }
    out = filter_skills(result, SKILL_SPEC)
    assert out["skills"] == ["CI/CD", "GitHub Actions", "WebGL", "Data Visualization"]


# --- #4 metrics validator -------------------------------------------------------

def test_validate_metrics_keeps_quant_reroutes_qualitative():
    result = {"projects": [{"name": "P", "description": "Did stuff", "metrics": ["78.58% accuracy", "optimal positioning"]}]}
    out = validate_metrics(result, PROJECT_SPEC)
    p = out["projects"][0]
    assert p["metrics"] == ["78.58% accuracy"]
    assert "optimal positioning" in p["description"].lower()


def test_validate_metrics_removes_empty_metrics_field():
    result = {"projects": [{"name": "P", "description": "d", "metrics": ["in-progress"]}]}
    out = validate_metrics(result, PROJECT_SPEC)
    assert "metrics" not in out["projects"][0]
    assert "in-progress" in out["projects"][0]["description"].lower()


def test_validate_metrics_no_double_write_when_already_in_desc():
    result = {"projects": [{"name": "P", "description": "Achieved optimal positioning.", "metrics": ["optimal positioning"]}]}
    out = validate_metrics(result, PROJECT_SPEC)
    assert out["projects"][0]["description"].lower().count("optimal positioning") == 1


# --- #5 cert <-> activity dedup -------------------------------------------------

def test_normalize_work_roles_fixes_reversed_duplicated_and_location():
    result = {
        "work_history": [
            {"company": "Freelance Software & Data Engineer", "title": "Self-Employed"},  # reversed
            {"company": "Freelance ML Engineer", "title": "Freelance ML Engineer"},       # duplicated
            {"company": "Topi, Pakistan", "title": "Lead Videographer & Media Producer"}, # location-as-company
            {"company": "RST Moto", "title": "Operations & Data Intern"},                 # correct -> untouched
        ]
    }
    out = normalize_work_roles(result, WORK_SPEC)["work_history"]
    assert out[0] == {"company": "Self-employed", "title": "Freelance Software & Data Engineer"}
    assert out[1]["company"] == "Self-employed" and out[1]["title"] == "Freelance ML Engineer"
    assert out[2]["location"] == "Topi, Pakistan" and out[2].get("company") is None
    assert out[3] == {"company": "RST Moto", "title": "Operations & Data Intern"}


def test_normalize_work_roles_drops_spurious_company_equals_title():
    # Project role-headers the LLM misfiled as jobs (company == title) are dropped;
    # a real job with distinct company/title is kept.
    result = {
        "work_history": [
            {"company": "GIKI Administration", "title": "Lead Videographer"},
            {"company": "Machine Learning Engineer", "title": "Machine Learning Engineer"},
            {"company": "Algorithm Researcher", "title": "Algorithm Researcher"},
        ]
    }
    out = normalize_work_roles(result, WORK_SPEC)["work_history"]
    assert len(out) == 1
    assert out[0]["company"] == "GIKI Administration"


def test_prune_empty_strings_removes_blank_values_only():
    from resume_parser.postprocess import prune_empty_strings

    obj = {"a": "x", "b": "", "c": "   ", "d": 0, "e": [], "g": [{"h": "", "i": "y"}]}
    assert prune_empty_strings(obj) == {"a": "x", "d": 0, "e": [], "g": [{"i": "y"}]}


def test_backfill_skills_recovers_missing_keeps_parenthetical():
    source = (
        "## Technical Skills\n"
        "**Machine Learning:** Python, Algorithmic Trading, "
        "Hypothesis Testing (t-test, Chi-squared, ANOVA), Data Visualization\n"
        "## Education\nSome school\n"
    )
    result = {"skills": ["Python"]}
    out = backfill_skills(result, source, SKILL_SPEC)
    assert "Algorithmic Trading" in out["skills"]
    assert "Hypothesis Testing (t-test, Chi-squared, ANOVA)" in out["skills"]  # not split
    assert "Data Visualization" in out["skills"]
    assert out["skills"].count("Python") == 1  # not duplicated
    assert "Some school" not in out["skills"]  # non-skills section ignored


def test_backfill_skills_unions_project_technologies():
    result = {
        "skills": ["Python"],
        "projects": [{"name": "P", "technologies": ["KNN", "SVM", "Python"]}],
    }
    out = backfill_skills(result, "", SKILL_SPEC)
    assert "KNN" in out["skills"] and "SVM" in out["skills"]  # promoted from project tech
    assert out["skills"].count("Python") == 1  # not duplicated


def test_filter_skills_drops_entity_names():
    spec = [
        {"name": "skills", "type": "array", "items": "string"},
        {
            "name": "work_history",
            "type": "array",
            "items": "object",
            "properties": [{"name": "company", "type": "string"}, {"name": "title", "type": "string"}],
        },
    ]
    result = {
        "skills": ["Python", "GIKI Administration", "React"],
        "work_history": [{"company": "GIKI Administration", "title": "Lead Videographer"}],
    }
    out = filter_skills(result, spec)
    assert out["skills"] == ["Python", "React"]  # employer name dropped


def test_filter_skills_drops_garbled_role_fragment():
    spec = [
        {"name": "skills", "type": "array", "items": "string"},
        {
            "name": "work_history",
            "type": "array",
            "items": "object",
            "properties": [{"name": "company", "type": "string"}, {"name": "title", "type": "string"}],
        },
    ]
    result = {
        "skills": ["Python", "Pakistan Lead Media Producer"],
        "work_history": [{"company": "GIKI Administration", "title": "Lead Videographer & Media Producer"}],
    }
    out = filter_skills(result, spec)
    assert out["skills"] == ["Python"]  # garbled role fragment fuzzy-matched -> dropped


def test_drop_unsupported_project_dates():
    result = {
        "projects": [
            {"name": "A", "end_date": "Present"},                       # no start -> drop
            {"name": "B", "start_date": "Nov 2025", "end_date": "Present"},  # has start -> keep
            {"name": "C", "end_date": "May 2024"},                      # real date -> keep
        ]
    }
    out = drop_unsupported_project_dates(result, PROJECT_SPEC)["projects"]
    assert "end_date" not in out[0]
    assert out[1]["end_date"] == "Present"
    assert out[2]["end_date"] == "May 2024"


LANG_SPEC = [
    {
        "name": "languages",
        "type": "array",
        "items": "object",
        "properties": [{"name": "name", "type": "string"}, {"name": "proficiency", "type": "string"}],
    }
]


def test_backfill_languages_parses_section():
    src = "## LANGUAGES\n\nEnglish (Fluent), Urdu (Native)\n\n## INTERESTS\nchess"
    out = backfill_languages({}, src, LANG_SPEC)
    assert out["languages"] == [
        {"name": "English", "proficiency": "Fluent"},
        {"name": "Urdu", "proficiency": "Native"},
    ]


def test_backfill_languages_ignores_programming_and_populated():
    assert "languages" not in backfill_languages({}, "## Programming Languages\n\nPython, C++", LANG_SPEC)
    pre = {"languages": [{"name": "French"}]}
    assert backfill_languages(pre, "## Languages\n\nEnglish (Fluent)", LANG_SPEC)["languages"] == [{"name": "French"}]


def test_canonicalize_tech_folds_plurals():
    result = {"projects": [{"name": "P", "technologies": ["WebSocket", "WebSockets", "FastAPI"]}]}
    out = canonicalize_tech(result, PROJECT_SPEC)
    assert out["projects"][0]["technologies"] == ["WebSocket", "FastAPI"]


def test_validate_metrics_reroute_skips_near_duplicate():
    result = {
        "projects": [
            {
                "name": "P",
                "description": "Work optimizing solar energy harvesting and thermal management.",
                "metrics": ["Optimized solar energy harvesting and thermal management"],
            }
        ]
    }
    out = validate_metrics(result, PROJECT_SPEC)
    assert "metrics" not in out["projects"][0]  # non-quant, removed
    assert out["projects"][0]["description"].lower().count("solar energy harvesting") == 1  # not re-appended


def test_dedupe_cert_activity_cert_wins():
    result = {
        "certifications": [{"name": "Amazon Assistant Online Marketing"}],
        "activities": [
            {"name": "Amazon Assistant Online Marketing – Certified", "description": "x"},
            {"name": "IEEE Power & Energy Society – Student Member", "description": "y"},
        ],
    }
    out = dedupe_cert_activity(result, CERT_ACT_SPEC)
    names = [a["name"] for a in out["activities"]]
    assert not any("Amazon" in n for n in names)  # duplicate dropped from activities
    assert any("IEEE" in n for n in names)         # unrelated activity kept
    assert len(out["certifications"]) == 1         # certification untouched
