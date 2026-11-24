from pathlib import Path

import pandas as pd

CF0 = Path(__file__).resolve().parents[1]


def test_tie_definition_is_exact_equality():
    script = (CF0 / "scripts/run_cf0_pilot.py").read_text()
    assert "is_tie = best_val == second_val" in script


def test_tie_flag_matches_recorded_utilities():
    frame = pd.read_csv(CF0 / "08_per_checkpoint_action_outcomes.csv")
    utilities = frame[["utility_no_action", "utility_capacity", "utility_sync"]]
    sorted_vals = utilities.apply(lambda row: sorted(row, reverse=True), axis=1)
    expected_tie = sorted_vals.apply(lambda v: v[0] == v[1])
    assert (frame.tie == expected_tie).all()


def test_no_action_ordering_used_to_break_ties():
    script = (CF0 / "scripts/run_cf0_pilot.py").read_text()
    assert "ACTIONS[0]" not in script.split("def best_action")[1].split(
        "def main")[0]
