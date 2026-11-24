from pathlib import Path

import pandas as pd

CF1 = Path(__file__).resolve().parents[1]


def test_matched_coverage_uses_certainty_ranking_of_t4_own_probability():
    script = (CF1 / "scripts/run_cf1_regret.py").read_text()
    assert "certainty = np.abs(t4_df.p_h2.to_numpy() - .5)" in script
    assert "order = np.argsort(-certainty)" in script


def test_matched_coverage_excludes_reject_option_for_kept_rows_like_original():
    script = (CF1 / "scripts/run_cf1_regret.py").read_text()
    assert 'costs = {"CAPACITY_ACTION": p * h2_cost, "SYNC_ACTION": (1 - p) * h1_cost}' in script


def test_matched_coverage_target_equals_r4_actual_coverage():
    frame = pd.read_csv(CF1 / "16_matched_coverage_results.csv")
    t4_matched = frame[frame.candidate == "T4_matched_to_R4_coverage"].iloc[0]
    r4_actual = frame[frame.candidate == "R4_actual"].iloc[0]
    assert abs(t4_matched.coverage - r4_actual.coverage) < 1e-6
