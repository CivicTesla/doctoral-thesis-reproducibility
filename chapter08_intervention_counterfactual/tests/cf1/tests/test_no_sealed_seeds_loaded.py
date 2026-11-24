from pathlib import Path

import pandas as pd

CF1 = Path(__file__).resolve().parents[1]


def test_no_sealed_r1_seeds_in_checkpoints():
    frame = pd.read_csv(CF1 / "07_counterfactual_branch_manifest.csv")
    assert not frame.seed.between(2000, 2029).any()


def test_no_forbidden_r0_seeds_in_checkpoints():
    frame = pd.read_csv(CF1 / "07_counterfactual_branch_manifest.csv")
    assert not frame.seed.between(120, 149).any()


def test_checkpoints_script_asserts_seed_safety():
    script = (CF1 / "scripts/run_cf1_checkpoints.py").read_text()
    assert "range(2000, 2030)" in script
    assert "range(120, 150)" in script
