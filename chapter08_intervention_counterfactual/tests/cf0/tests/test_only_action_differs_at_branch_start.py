from pathlib import Path
import sys

import pandas as pd

CF0 = Path(__file__).resolve().parents[1]
B1 = CF0.parent
sys.path[:0] = [str(CF0 / "src"), str(B1 / "R0_hidden_mechanism_audit")]


def test_arrivals_identical_across_action_branches_same_checkpoint():
    """The empirical property AS0 showed was broken for H2 -- confirmed
    fixed here: arrivals (an exogenous, action-independent process) must
    be identical across NO_ACTION/CAPACITY_ACTION/SYNC_ACTION for the same
    checkpoint."""
    frame = pd.read_csv(CF0 / "07_branch_hash_validation.csv")
    per_checkpoint = frame.groupby("checkpoint_id").arrivals.nunique()
    assert (per_checkpoint == 1).all()


def test_h1_and_h2_both_pass_the_arrivals_consistency_check():
    frame = pd.read_csv(CF0 / "07_branch_hash_validation.csv")
    for mech in ("H1_CAPACITY_SHORTAGE", "H2_DISTRIBUTED_CAPACITY_SUPPRESSION"):
        subset = frame[frame.mechanism == mech]
        per_checkpoint = subset.groupby("checkpoint_id").arrivals.nunique()
        assert (per_checkpoint == 1).all(), mech


def test_run_checkpoint_asserts_exogenous_equality_before_returning():
    text = (CF0 / "src/cf0_simulator.py").read_text()
    assert "if len(set(hashes.values())) != 1:" in text
