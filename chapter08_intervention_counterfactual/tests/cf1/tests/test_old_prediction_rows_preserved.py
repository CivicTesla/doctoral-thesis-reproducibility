from pathlib import Path

import pandas as pd

CF1 = Path(__file__).resolve().parents[1]


def test_row_count_matches_original_decision_state_manifest():
    frame = pd.read_csv(CF1 / "05_old_to_cf_checkpoint_mapping.csv")
    assert len(frame) == 1600


def test_no_row_dropped_or_added():
    B1 = CF1.parent
    original = pd.read_csv(
        B1 / "R2_mechanism_conditioned_risk_rescue/G1A_instrumented_replay_data/"
        "03_decision_state_manifest.csv")
    original = original[original.delay.isin([1, 2])]
    mapping = pd.read_csv(CF1 / "05_old_to_cf_checkpoint_mapping.csv")
    assert set(original.run_id) == set(mapping.old_prediction_row_id)


def test_seed_and_mechanism_columns_unchanged_in_mapping():
    frame = pd.read_csv(CF1 / "05_old_to_cf_checkpoint_mapping.csv")
    assert frame.seed.between(3000, 3399).all()
    assert set(frame.mechanism.unique()) == {
        "H1_CAPACITY_SHORTAGE", "H2_DISTRIBUTED_CAPACITY_SUPPRESSION"}
