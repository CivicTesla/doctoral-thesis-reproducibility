from pathlib import Path

import pandas as pd

CF0 = Path(__file__).resolve().parents[1]


def test_exogenous_schedule_hash_identical_across_the_three_branches():
    frame = pd.read_csv(CF0 / "07_branch_hash_validation.csv")
    per_checkpoint = frame.groupby(
        "checkpoint_id").exogenous_schedule_hash_this_branch.nunique()
    assert (per_checkpoint == 1).all()


def test_matches_other_branches_flag_all_true():
    frame = pd.read_csv(CF0 / "07_branch_hash_validation.csv")
    assert frame.matches_other_branches.all()


def test_simulator_raises_on_mismatch_by_construction():
    text = (CF0 / "src/cf0_simulator.py").read_text()
    assert "exogenous schedule hash mismatch across branches" in text
    assert "raise RuntimeError" in text
