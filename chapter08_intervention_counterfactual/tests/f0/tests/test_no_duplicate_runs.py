import csv
from pathlib import Path

import pytest

F0 = Path(__file__).resolve().parents[1]


def test_no_duplicate_seed_mechanism_rows_in_pair1_main():
    path = F0 / "runs" / "main_pair1" / "per_sample_strict_action_outcomes.csv"
    if not path.exists():
        pytest.skip("main pair1 run not yet present")
    rows = list(csv.DictReader(path.open()))
    keys = [(r["method"], r["seed"], r["mechanism"], r["delay"]) for r in rows]
    assert len(keys) == len(set(keys))


def test_no_duplicate_seed_mechanism_rows_in_pair2_main():
    path = F0 / "runs" / "main_pair2" / "out_of_fold_predictions.csv"
    if not path.exists():
        pytest.skip("main pair2 run not yet present")
    rows = list(csv.DictReader(path.open()))
    keys = [(r["seed"], r["mechanism"]) for r in rows]
    assert len(keys) == len(set(keys))
