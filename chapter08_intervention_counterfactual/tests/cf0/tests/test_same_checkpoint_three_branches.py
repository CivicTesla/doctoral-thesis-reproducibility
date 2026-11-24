from pathlib import Path

import pandas as pd

CF0 = Path(__file__).resolve().parents[1]


def test_every_checkpoint_has_all_three_action_branches():
    frame = pd.read_csv(CF0 / "07_branch_hash_validation.csv")
    counts = frame.groupby("checkpoint_id").action.nunique()
    assert (counts == 3).all()
    assert set(frame.action.unique()) == {
        "NO_ACTION", "CAPACITY_ACTION", "SYNC_ACTION"}


def test_checkpoint_count_matches_manifest():
    manifest = pd.read_csv(CF0 / "06_checkpoint_manifest.csv")
    hashes = pd.read_csv(CF0 / "07_branch_hash_validation.csv")
    assert len(manifest) == 60  # 30 seeds x 2 mechanisms
    assert len(hashes) == 180   # 60 checkpoints x 3 branches
