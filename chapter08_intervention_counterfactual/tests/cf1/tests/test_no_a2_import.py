from pathlib import Path

CF1 = Path(__file__).resolve().parents[1]

FORBIDDEN_TERMS = (
    "future_intention", "semantic_admission", "token_ranking",
    "queue_scheduling", "utility_per_byte", "prefix_credibility",
    "soft_diversity", "valid_future", "from a2", "import a2",
    "a2 project", "../A2", "A2/scripts", "A2/config",
)


def test_no_a2_terms_anywhere_in_cf1():
    for path in list(CF1.glob("*.py")) + list((CF1 / "scripts").glob("*.py")):
        low = path.read_text().lower()
        for term in FORBIDDEN_TERMS:
            assert term.lower() not in low, f"{path.name}: {term}"


def test_no_a2_terms_in_cf1_docs():
    for path in CF1.glob("*.md"):
        text = path.read_text()
        assert "../A2" not in text
