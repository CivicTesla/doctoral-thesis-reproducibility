from pathlib import Path

import pandas as pd

CF1 = Path(__file__).resolve().parents[1]


def test_ba_matches_archived_for_all_methods():
    frame = pd.read_csv(CF1 / "09_per_method_strict_regret.csv")
    assert frame.ba_matches_archived.all()


def test_ba_values_match_known_archived_numbers():
    frame = pd.read_csv(CF1 / "09_per_method_strict_regret.csv").set_index("method")
    assert abs(frame.loc["T0", "balanced_accuracy"] - 0.813125) < 1e-9
    assert abs(frame.loc["T4", "balanced_accuracy"] - 0.8675) < 1e-9
    assert abs(frame.loc["R4", "balanced_accuracy"] - 0.83) < 1e-9
