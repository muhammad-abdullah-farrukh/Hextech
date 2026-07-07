from resume_parser.contacts import (
    backfill_contacts,
    extract_dob,
    extract_email,
    extract_github,
    extract_linkedin,
    extract_location,
    extract_phone,
)

SOURCE = (
    "** +92-3125722830  abdullahzahid6555@gmail.com "
    "www.linkedin.com/in/abdullahzahid655 House No. 993, street 40, "
    "Islamabad, Pakistan Dob: 22 Sep 2003"
)

FIELD_SPEC = [
    {"name": "candidate_name", "type": "string", "required": True},
    {"name": "email", "type": "string", "required": True},
    {"name": "phone", "type": "string", "required": False},
    {"name": "linkedin", "type": "string", "required": False},
    {"name": "github", "type": "string", "required": False},
    {"name": "location", "type": "string", "required": False},
    {"name": "date_of_birth", "type": "string", "required": False},
    {"name": "skills", "type": "array", "items": "string", "required": True},
]


def test_extract_email():
    assert extract_email(SOURCE) == "abdullahzahid6555@gmail.com"
    assert extract_email("no contact here") is None


def test_extract_phone():
    assert extract_phone(SOURCE) == "+92-3125722830"
    # A bare year / short number must not be treated as a phone.
    assert extract_phone("graduated in 2025, house 40") is None


def test_backfill_fills_only_empty_contact_fields():
    result = {"candidate_name": "Abdullah Zahid", "skills": ["Python"]}
    out = backfill_contacts(result, SOURCE, FIELD_SPEC)
    assert out["email"] == "abdullahzahid6555@gmail.com"
    assert out["phone"] == "+92-3125722830"


def test_extract_linkedin_and_github():
    assert extract_linkedin(SOURCE) == "linkedin.com/in/abdullahzahid655"
    assert extract_github(SOURCE) is None  # not present in SOURCE
    gh = "profile: https://github.com/abdullahzahid655 and more"
    assert extract_github(gh) == "github.com/abdullahzahid655"


def test_backfill_fills_linkedin_from_source():
    result = {"candidate_name": "Abdullah Zahid", "skills": ["Python"]}
    out = backfill_contacts(result, SOURCE, FIELD_SPEC)
    assert out["linkedin"] == "linkedin.com/in/abdullahzahid655"
    assert "github" not in out or out.get("github") is None  # none in source


def test_extract_dob():
    assert extract_dob(SOURCE) == "22 Sep 2003"
    assert extract_dob("Date of Birth - 05/09/2003") == "05/09/2003"
    assert extract_dob("no birth date here") is None


def test_extract_location_street_labeled_city():
    assert extract_location(SOURCE) == "House No. 993, street 40, Islamabad, Pakistan"
    assert extract_location("Address: 12 Main St, Lahore, Pakistan Phone: 1") == "12 Main St, Lahore, Pakistan"
    assert extract_location("Karachi, Pakistan | a@b.com | +92 300") == "Karachi, Pakistan"


def test_backfill_fills_location_and_dob():
    out = backfill_contacts({"skills": []}, SOURCE, FIELD_SPEC)
    assert out["location"] == "House No. 993, street 40, Islamabad, Pakistan"
    assert out["date_of_birth"] == "22 Sep 2003"


def test_backfill_does_not_overwrite_model_values():
    result = {"email": "kept@example.com", "phone": "000"}
    out = backfill_contacts(result, SOURCE, FIELD_SPEC)
    assert out["email"] == "kept@example.com"  # model value preserved
    assert out["phone"] == "000"


def test_backfill_ignores_non_contact_and_non_spec_fields():
    # 'skills' is an array, not a contact string field -> untouched.
    result = {"skills": []}
    out = backfill_contacts(result, SOURCE, FIELD_SPEC)
    assert out["skills"] == []
    # A field name not in the spec is never added.
    assert "fax" not in out
