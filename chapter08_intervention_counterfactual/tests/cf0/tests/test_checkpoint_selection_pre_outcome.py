from pathlib import Path

import pandas as pd

CF0 = Path(__file__).resolve().parents[1]


def test_checkpoint_rule_is_fixed_and_not_outcome_dependent():
    text = (CF0 / "02_preregistered_protocol.md").read_text()
    assert "before any CF0 action branch is executed" in text
    assert "decision_tick = 0" in text or "decision_tick=0" in text
    assert "it cannot depend" in text and "outcome, regret" in text


def test_all_checkpoints_use_decision_tick_zero():
    frame = pd.read_csv(CF0 / "06_checkpoint_manifest.csv")
    assert (frame.decision_tick == 0).all()


def test_no_outcome_based_filtering_in_pilot_script():
    script = (CF0 / "scripts/run_cf0_pilot.py").read_text()
    # eligibility is unconditional (True) in the manifest-building loop,
    # not computed from any utility/regret/advantage value
    assert '"eligible": True' in script
