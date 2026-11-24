from pathlib import Path

import pandas as pd

CF1 = Path(__file__).resolve().parents[1]


def test_regret_script_never_reads_old_oracle_action_or_potential_regret():
    """The cached old oracle_action_old / potential_regret_old fields exist
    only for provenance comparison (19_old_vs_new_regret_comparison.csv);
    they must never feed into the strict regret computation itself."""
    text = (CF1 / "scripts/run_cf1_regret.py").read_text()
    assert "oracle_action_old" not in text
    assert "potential_regret_old" not in text


def test_strict_rows_use_checkpoint_utilities_only():
    text = (CF1 / "scripts/run_cf1_regret.py").read_text()
    assert "checkpoint.oracle_best_action_2way" in text
    assert "checkpoint.utility_capacity" in text
    assert "checkpoint.utility_sync" in text


def test_old_vs_new_file_is_transparency_only_not_an_input():
    frame = pd.read_csv(CF1 / "19_old_vs_new_regret_comparison.csv")
    assert "old_total_regret_HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY" in frame.columns
