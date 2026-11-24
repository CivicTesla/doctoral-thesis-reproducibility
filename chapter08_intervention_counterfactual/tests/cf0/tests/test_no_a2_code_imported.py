from pathlib import Path

CF0 = Path(__file__).resolve().parents[1]

FORBIDDEN_TERMS = (
    "future_intention", "semantic_admission", "token_ranking",
    "queue_scheduling", "utility_per_byte", "prefix_credibility",
    "soft_diversity", "valid_future", "from a2", "import a2",
    "a2 project", "/A2/", "../A2",
)


def test_no_a2_terms_anywhere_in_cf0():
    for path in list(CF0.glob("*.py")) + list((CF0 / "src").glob("*.py")) \
            + list((CF0 / "scripts").glob("*.py")):
        text = path.read_text()
        low = text.lower()
        for term in FORBIDDEN_TERMS:
            assert term.lower() not in low, f"{path.name}: {term}"


def test_no_a2_directory_referenced_in_docs():
    # "A2" legitimately appears once, in the "not touching A1/A2/A3/C1/D1"
    # scope disclaimer -- what must never appear is an actual path/import
    # reference *into* the A2 project.
    for path in CF0.glob("*.md"):
        text = path.read_text()
        assert "../A2" not in text
        assert "A2/scripts" not in text
        assert "A2/config" not in text
