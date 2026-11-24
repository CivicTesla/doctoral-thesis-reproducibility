from pathlib import Path

import numpy as np
import pandas as pd

CF1 = Path(__file__).resolve().parents[1]


def test_wrong_accepted_rows_have_nonzero_regret_matching_advantage():
    frame = pd.read_csv(CF1 / "08_per_sample_strict_action_outcomes.csv")
    wrong_accepted = frame[frame.accepted & frame.wrong]
    assert len(wrong_accepted) > 0
    assert (wrong_accepted.accepted_only_regret > 0).all()


def test_correct_accepted_rows_have_zero_regret():
    frame = pd.read_csv(CF1 / "08_per_sample_strict_action_outcomes.csv")
    correct_accepted = frame[frame.accepted & ~frame.wrong]
    assert (correct_accepted.accepted_only_regret == 0).all()


def test_rejected_rows_have_flat_reject_cost():
    frame = pd.read_csv(CF1 / "08_per_sample_strict_action_outcomes.csv")
    rejected = frame[~frame.accepted]
    assert (rejected.system_regret == 5.0).all()
