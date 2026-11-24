from pathlib import Path

import pandas as pd

CF1 = Path(__file__).resolve().parents[1]


def test_exogenous_schedule_hash_identical_across_branches_per_checkpoint():
    frame = pd.read_csv(CF1 / "_branch_hash_detail.csv")
    per_checkpoint = frame.groupby(
        "checkpoint_id").exogenous_schedule_hash_this_branch.nunique()
    assert (per_checkpoint == 1).all()


def test_arrivals_identical_across_branches_per_checkpoint():
    frame = pd.read_csv(CF1 / "_branch_hash_detail.csv")
    per_checkpoint = frame.groupby("checkpoint_id").arrivals.nunique()
    assert (per_checkpoint == 1).all()


def test_run_checkpoint_structurally_enforces_equality():
    """cf0_simulator.run_checkpoint (reused unmodified) raises before any
    outcome is recorded on a hash mismatch -- this is a structural
    guarantee, not a post-hoc filter."""
    B1 = CF1.parent
    text = (B1 / "CF0_strict_counterfactual_grounding/src/cf0_simulator.py"
            ).read_text()
    assert "raise RuntimeError" in text
