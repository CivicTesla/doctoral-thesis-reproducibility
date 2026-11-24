from pathlib import Path

import pandas as pd

CF0 = Path(__file__).resolve().parents[1]


def test_pre_action_state_hash_present_and_consistent_per_checkpoint():
    frame = pd.read_csv(CF0 / "06_checkpoint_manifest.csv")
    assert frame.pre_action_state_hash.notna().all()
    # one hash per checkpoint row (already the unit of the manifest); the
    # real cross-branch identity check lives in 07_branch_hash_validation
    assert frame.pre_action_state_hash.nunique() == len(frame)  # seed-derived, unique per checkpoint


def test_config_hash_identical_within_mechanism():
    frame = pd.read_csv(CF0 / "06_checkpoint_manifest.csv")
    per_mech = frame.groupby("mechanism").config_hash.nunique()
    assert (per_mech == 1).all()
