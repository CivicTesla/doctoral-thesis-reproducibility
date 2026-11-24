from pathlib import Path

import pandas as pd

CF1 = Path(__file__).resolve().parents[1]


def test_each_old_row_maps_to_exactly_one_checkpoint():
    # NOTE: `run_id` alone is not a unique row key in the original
    # decision_state_manifest -- the same run_id appears once per delay
    # (1 and 2), since it is built as f"{split}_{mechanism}_seed{seed}"
    # with no delay component. The true row key is (run_id, delay).
    frame = pd.read_csv(CF1 / "05_old_to_cf_checkpoint_mapping.csv")
    assert not frame.duplicated(["old_prediction_row_id", "delay"]).any()
    assert frame.groupby(
        ["old_prediction_row_id", "delay"]
    ).strict_cf_checkpoint_id.nunique().max() == 1


def test_checkpoint_reused_by_exactly_the_two_delay_rows_of_its_seed_mechanism():
    frame = pd.read_csv(CF1 / "05_old_to_cf_checkpoint_mapping.csv")
    per_checkpoint = frame.groupby("strict_cf_checkpoint_id").size()
    assert set(per_checkpoint.unique()) == {2}
