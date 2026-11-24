from pathlib import Path

CF0 = Path(__file__).resolve().parents[1]


def test_thresholds_hardcoded_in_script():
    script = (CF0 / "scripts/run_cf0_pilot.py").read_text()
    assert "NONTRIVIAL_FRACTION = 0.10" in script
    assert "NONTRIVIAL_MIN_SEEDS = 5" in script
    assert ">= 0.80" in script
    assert ">= 10" in script


def test_thresholds_match_the_frozen_protocol_document():
    protocol = (CF0 / "02_preregistered_protocol.md").read_text()
    normalized = " ".join(protocol.split())
    assert ">= 0.80" in normalized
    assert ">= 10 independent seeds" in normalized
    assert "at least two non-tie best actions each occupying >= 10%" in normalized


def test_decision_is_one_of_the_five_preregistered_values():
    decision = (CF0 / "17_cf0_decision.txt").read_text().strip()
    assert decision in {
        "CF0_FIXED_ACTION_SEPARATION_SURVIVES",
        "CF0_STATE_DEPENDENT_ACTION_STRUCTURE_SURVIVES",
        "CF0_ACTION_SEPARATION_FAILS",
        "CF0_IMPLEMENTATION_INVALID",
        "CF0_DATA_INSUFFICIENT",
    }
