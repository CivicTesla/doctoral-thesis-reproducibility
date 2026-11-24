from pathlib import Path

import pandas as pd

CF0 = Path(__file__).resolve().parents[1]


def test_bootstrap_replicate_count_and_unit():
    frame = pd.read_csv(CF0 / "13_seed_bootstrap_results.csv")
    assert set(frame.replicates) == {2000}
    assert set(frame.resampling_unit) == {"seed"}


def test_bootstrap_covers_all_four_required_comparisons():
    frame = pd.read_csv(CF0 / "13_seed_bootstrap_results.csv")
    required = {
        "H1_CAPACITY_minus_SYNC", "H1_CAPACITY_minus_NO_ACTION",
        "H2_SYNC_minus_CAPACITY", "H2_SYNC_minus_NO_ACTION"}
    assert required <= set(frame.comparison)


def test_resamples_by_seed_not_by_checkpoint_when_seed_has_both_mechanisms():
    """A seed's H1 and H2 checkpoints are resampled as one unit."""
    script = (CF0 / "scripts/run_cf0_pilot.py").read_text()
    assert "rng.choice(unique_seeds" in script
    assert "outcomes.seed.to_numpy() == s" in script


def test_leave_one_seed_out_covers_all_thirty_seeds():
    frame = pd.read_csv(CF0 / "14_leave_one_seed_out.csv")
    assert frame.excluded_seed.nunique() == 30
