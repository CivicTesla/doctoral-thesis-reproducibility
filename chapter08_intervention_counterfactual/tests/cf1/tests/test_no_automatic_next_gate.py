from pathlib import Path

CF1 = Path(__file__).resolve().parents[1]


def test_no_forbidden_conclusions_declared():
    for path in list(CF1.glob("*.md")) + list(
            (CF1 / "results/CONCLUSIONS/CF1_STRICT_REGRET").glob("*.md")):
        text = path.read_text()
        assert "B1_REVIVED" not in text
        assert "ROUTE3_APPROVED" not in text
        assert "AS1_APPROVED" not in text
        assert "PAPER_SUBMITTED" not in text


def test_report_limits_next_step_to_a_single_recommendation():
    text = " ".join((CF1 / "21_cf1_report.md").read_text().split())
    assert "does not authorize Route 3, AS1, model retraining, or paper writing" in text


def test_no_route3_or_as1_code_invoked():
    for path in (CF1 / "scripts").glob("*.py"):
        text = path.read_text().lower()
        assert "run_as1" not in text
        assert "run_route3" not in text
