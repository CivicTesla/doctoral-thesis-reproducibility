from pathlib import Path

import pandas as pd

CF1 = Path(__file__).resolve().parents[1]


def test_bootstrap_replicate_count_and_unit():
    frame = pd.read_csv(CF1 / "17_seed_bootstrap_results.csv")
    assert set(frame.replicates) == {2000}
    assert set(frame.resampling_unit) == {"seed"}


def test_bootstrap_covers_required_comparisons():
    frame = pd.read_csv(CF1 / "17_seed_bootstrap_results.csv")
    required = {
        ("T4-T0", "balanced_accuracy"), ("T4-T0", "total_regret"),
        ("T4-T0", "h1_regret"), ("T4-T0", "h2_regret"),
        ("R4-T4", "total_regret"), ("R4-T4", "h1_regret"),
        ("R4-T4", "h2_regret"),
        ("R4-T0", "total_regret"), ("R4-T0", "h1_regret"),
        ("R4-T0", "h2_regret")}
    present = set(zip(frame.comparison, frame.metric))
    assert required <= present


def test_a_seeds_multiple_mechanisms_resampled_together():
    script = (CF1 / "scripts/run_cf1_regret.py").read_text()
    assert "rng.choice(unique_seeds" in script
    assert "combined.seed.to_numpy() == s" in script


def test_leave_one_seed_out_covers_all_seeds():
    frame = pd.read_csv(CF1 / "18_leave_one_seed_out.csv")
    assert frame.excluded_seed.nunique() == 400
