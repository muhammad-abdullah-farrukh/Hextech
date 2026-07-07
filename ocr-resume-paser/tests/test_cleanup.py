from resume_parser.cleanup import (
    clean_extraction,
    dedupe_near_identical_blocks,
    merge_split_sentences,
    normalize_whitespace,
    strip_markdown_artifacts,
    strip_repeated_boilerplate,
)


def test_strip_boilerplate_removes_repeated_header_across_pages():
    pages = [
        "ACME RESUME\nPage 1\nJohn Doe\nEngineer",
        "ACME RESUME\nPage 2\nExperience\nAcme Corp",
    ]
    out = strip_repeated_boilerplate(pages)
    assert all("ACME RESUME" not in p for p in out)
    # Page numbers normalize to the same token and get stripped too.
    assert all("Page" not in p for p in out)
    # Unique content survives.
    assert "John Doe" in out[0]
    assert "Acme Corp" in out[1]


def test_strip_boilerplate_noop_for_single_page():
    pages = ["only one page\nwith content"]
    assert strip_repeated_boilerplate(pages) == pages


def test_strip_boilerplate_keeps_body_date_recurring_across_pages():
    # The RST Moto / Unity bug: the same date range appears mid-page on both pages.
    # It is NOT an edge header/footer, so it must survive (previously deleted).
    pages = [
        "Operations & Data Intern, RST Moto\nJun 2024 - Aug 2024\nSialkot, Pakistan\n1",
        "Certifications\nUnity Game Development Program, MLabs\nJun 2024 - Aug 2024\n2",
    ]
    out = strip_repeated_boilerplate(pages)
    assert all("Jun 2024 - Aug 2024" in p for p in out)  # body content preserved
    # Bare page-number lines still dropped.
    assert out[0].splitlines()[-1] != "1"


def test_strip_boilerplate_drops_page_numbers_and_edge_headers():
    pages = [
        "ACME RESUME\nJohn Doe\nProfile text\n1 / 2",
        "ACME RESUME\nExperience\nAcme Corp\n2 / 2",
    ]
    out = strip_repeated_boilerplate(pages)
    assert all("ACME RESUME" not in p for p in out)  # repeated edge header
    assert all("/ 2" not in p for p in out)          # page numbers
    assert "John Doe" in out[0] and "Acme Corp" in out[1]


def test_strip_markdown_artifacts_removes_styling_keeps_text():
    assert strip_markdown_artifacts("### **~~Profile~~**") == "### **Profile**"
    assert strip_markdown_artifacts("Dob: 22<sup>nd</sup> Sep") == "Dob: 22nd Sep"
    assert strip_markdown_artifacts("<u>linkedin.com/in/x</u>") == "linkedin.com/in/x"


def test_strip_markdown_artifacts_demotes_location_heading():
    # A location wrongly tagged as a heading is demoted to plain text...
    assert strip_markdown_artifacts("### Islamabad, Pakistan") == "Islamabad, Pakistan"
    # ...but a real section heading (no City, Country shape) is left alone.
    assert strip_markdown_artifacts("## **PROFILE**") == "## **PROFILE**"


def test_strip_markdown_artifacts_br_becomes_space():
    # <br> inside a table cell is a line-wrap -> space (not a join, not removal).
    assert strip_markdown_artifacts("Deep Learning & Neural<br>Networks") == "Deep Learning & Neural Networks"
    assert "<br" not in strip_markdown_artifacts("A<br/>B <BR> C").lower()


def test_dedupe_near_identical_blocks():
    text = "Led the migration project.\n\nLed the migration project!\n\nDistinct block."
    out = dedupe_near_identical_blocks(text)
    assert out.count("Led the migration project") == 1
    assert "Distinct block." in out


def test_merge_split_sentences_rejoins_lowercase_continuation():
    text = "Built a service that handled\nmillions of requests per day"
    out = merge_split_sentences(text)
    assert "handled millions" in out


def test_merge_split_sentences_keeps_list_items_separate():
    text = "- first bullet\n- second bullet"
    out = merge_split_sentences(text)
    assert out == text


def test_normalize_whitespace_bullets_and_blank_lines():
    text = "• item\n\n\n\nnext"
    out = normalize_whitespace(text)
    assert "- item" in out
    assert "\n\n\n" not in out


def test_clean_extraction_end_to_end():
    pages = [
        "HEADER\nJohn Doe\n• Python developer with experience in\nbuilding APIs",
        "HEADER\nMore unique content here",
    ]
    out = clean_extraction(pages)
    assert "HEADER" not in out
    assert "- Python developer with experience in building APIs" in out
