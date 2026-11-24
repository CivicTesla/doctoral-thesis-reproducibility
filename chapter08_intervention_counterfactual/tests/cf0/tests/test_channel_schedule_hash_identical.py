from pathlib import Path

import pandas as pd

CF0 = Path(__file__).resolve().parents[1]


def test_channel_schedule_hash_identical_across_branches_per_checkpoint():
    frame = pd.read_csv(CF0 / "07_branch_hash_validation.csv")
    per_checkpoint = frame.groupby(
        "checkpoint_id").channel_schedule_hash_this_branch.nunique()
    assert (per_checkpoint == 1).all()


def test_h1_channel_hash_honestly_marked_not_applicable():
    frame = pd.read_csv(CF0 / "06_checkpoint_manifest.csv")
    h1 = frame[frame.mechanism == "H1_CAPACITY_SHORTAGE"]
    assert set(h1.channel_schedule_hash) == {
        "NOT_APPLICABLE_NO_CHANNEL_AT_ACTION_LAYER"}


def test_h2_channel_hash_repurposed_and_documented():
    frame = pd.read_csv(CF0 / "06_checkpoint_manifest.csv")
    h2 = frame[frame.mechanism == "H2_DISTRIBUTED_CAPACITY_SUPPRESSION"]
    assert (h2.channel_schedule_hash != "NOT_APPLICABLE_NO_CHANNEL_AT_ACTION_LAYER").all()
    text = (CF0 / "04_exogenous_endogenous_event_inventory.md").read_text()
    assert "repurposed" in text.lower()
