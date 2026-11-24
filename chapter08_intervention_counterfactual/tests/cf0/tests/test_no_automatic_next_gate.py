from pathlib import Path

CF0 = Path(__file__).resolve().parents[1]


def test_no_forbidden_conclusions_declared():
    for path in list(CF0.glob("*.md")) + list(
            (CF0 / "results/CONCLUSIONS/CF0_STRICT_COUNTERFACTUAL").glob("*.md")):
        text = path.read_text()
        assert "B1_REVIVED" not in text
        assert "AS1_APPROVED" not in text
        assert "POMACS_ESTABLISHED" not in text
        assert "CCR_ESTABLISHED" not in text


def test_report_states_as1_not_authorized():
    text = (CF0 / "16_cf0_report.md").read_text()
    assert "does not authorize AS1" in text


def test_pilot_script_never_invokes_as1_or_route3_code():
    script = (CF0 / "scripts/run_cf0_pilot.py").read_text()
    assert "AS1_action_identifying" not in script.lower().replace("_", "")
    assert "run_as1" not in script.lower()
