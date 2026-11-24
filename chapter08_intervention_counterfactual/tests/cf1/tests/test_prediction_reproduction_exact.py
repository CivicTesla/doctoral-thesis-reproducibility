from pathlib import Path

import pandas as pd

CF1 = Path(__file__).resolve().parents[1]


def test_all_reproduction_checks_pass():
    frame = pd.read_csv(CF1 / "04_prediction_reproduction.csv")
    assert frame.reproduced_exactly.all()


def test_reproduction_covers_all_three_methods_and_three_metrics():
    frame = pd.read_csv(CF1 / "04_prediction_reproduction.csv")
    assert set(frame.method) == {"T0", "T4", "R4"}
    assert set(frame.metric) == {
        "balanced_accuracy", "coverage", "total_regret"}
    assert len(frame) == 9


def test_differences_within_tight_tolerance():
    frame = pd.read_csv(CF1 / "04_prediction_reproduction.csv")
    assert (frame.abs_difference < 1e-9).all()
