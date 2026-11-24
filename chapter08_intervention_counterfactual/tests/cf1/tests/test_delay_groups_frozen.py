from pathlib import Path

import pandas as pd

CF1 = Path(__file__).resolve().parents[1]


def test_delay_0_reported_as_not_present_not_fabricated():
    frame = pd.read_csv(CF1 / "14_delay_results.csv")
    delay0 = frame[frame.delay == 0]
    assert (delay0.status == "NOT_PRESENT_IN_OLD_EVALUATION_SAMPLE").all()
    assert (delay0.n == 0).all()


def test_delay_1_and_2_present_for_all_methods():
    frame = pd.read_csv(CF1 / "14_delay_results.csv")
    for delay in (1, 2):
        subset = frame[frame.delay == delay]
        assert (subset.status == "present").all()
        assert set(subset.method) == {"T0", "T4", "R4"}


def test_no_model_adaptation_per_delay():
    script = (CF1 / "scripts/run_cf1_regret.py").read_text()
    assert "fit_classifier" not in script  # T4/R4 predictions are loaded, not refit here
