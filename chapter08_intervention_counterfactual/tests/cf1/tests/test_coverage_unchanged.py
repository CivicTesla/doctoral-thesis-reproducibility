from pathlib import Path

import pandas as pd

CF1 = Path(__file__).resolve().parents[1]


def test_coverage_matches_archived_for_all_methods():
    frame = pd.read_csv(CF1 / "09_per_method_strict_regret.csv")
    assert frame.coverage_matches_archived.all()


def test_coverage_values_match_known_archived_numbers():
    frame = pd.read_csv(CF1 / "09_per_method_strict_regret.csv").set_index("method")
    assert abs(frame.loc["T0", "coverage"] - 0.679375) < 1e-9
    assert abs(frame.loc["T4", "coverage"] - 0.87875) < 1e-9
    assert abs(frame.loc["R4", "coverage"] - 0.811875) < 1e-9


def test_pipeline_halts_if_ba_or_coverage_drift():
    script = (CF1 / "scripts/run_cf1_regret.py").read_text()
    assert "if not ba_coverage_unchanged:" in script
    assert "raise SystemExit" in script
