from pathlib import Path

import pandas as pd

CF0 = Path(__file__).resolve().parents[1]


def test_no_sealed_r1_seeds():
    frame = pd.read_csv(CF0 / "08_per_checkpoint_action_outcomes.csv")
    assert not frame.seed.between(2000, 2029).any()


def test_script_asserts_sealed_range_absent():
    script = (CF0 / "scripts/run_cf0_pilot.py").read_text()
    assert "range(2000, 2030)" in script
