from pathlib import Path

import pandas as pd

CF1 = Path(__file__).resolve().parents[1]


def test_every_checkpoint_has_all_three_action_branches():
    frame = pd.read_csv(CF1 / "_branch_hash_detail.csv")
    counts = frame.groupby("checkpoint_id").action.nunique()
    assert (counts == 3).all()


def test_checkpoint_and_branch_counts_match_800_seed_mechanism_pairs():
    manifest = pd.read_csv(CF1 / "07_counterfactual_branch_manifest.csv")
    hashes = pd.read_csv(CF1 / "_branch_hash_detail.csv")
    assert len(manifest) == 800  # 400 seeds x 2 mechanisms
    assert len(hashes) == 2400   # 800 checkpoints x 3 branches
