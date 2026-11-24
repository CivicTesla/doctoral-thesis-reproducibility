from pathlib import Path

import pandas as pd

CF0 = Path(__file__).resolve().parents[1]


def test_service_schedule_hash_identical_across_branches_per_checkpoint():
    frame = pd.read_csv(CF0 / "07_branch_hash_validation.csv")
    per_checkpoint = frame.groupby(
        "checkpoint_id").service_schedule_hash_this_branch.nunique()
    assert (per_checkpoint == 1).all()


def test_service_schedule_hash_present_in_manifest():
    frame = pd.read_csv(CF0 / "06_checkpoint_manifest.csv")
    assert frame.service_schedule_hash.notna().all()
    assert frame.service_schedule_hash.str.len().gt(0).all()
