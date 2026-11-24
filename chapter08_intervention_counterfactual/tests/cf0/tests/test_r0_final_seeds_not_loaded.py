from pathlib import Path

import pandas as pd

CF0 = Path(__file__).resolve().parents[1]


def test_no_forbidden_r0_seeds():
    frame = pd.read_csv(CF0 / "08_per_checkpoint_action_outcomes.csv")
    assert not frame.seed.between(120, 149).any()


def test_script_asserts_forbidden_range_absent():
    script = (CF0 / "scripts/run_cf0_pilot.py").read_text()
    assert "range(120, 150)" in script
