from pathlib import Path
import subprocess

import pandas as pd

CF1 = Path(__file__).resolve().parents[1]
B1 = CF1.parent


def test_t4_script_hash_recorded():
    frame = pd.read_csv(CF1 / "02_frozen_model_manifest.csv")
    row = frame[frame.method == "T4"].iloc[0]
    assert row.sha256 and len(row.sha256) == 64


def test_t4_script_untouched_by_this_branch():
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--",
         "R3_temporal_mechanism_evidence/scripts/run_g0_temporal_audit.py"],
        cwd=B1, capture_output=True, text=True)
    assert result.stdout.strip() == ""


def test_t4_uses_frozen_window_selection_not_reselected():
    frame = pd.read_csv(CF1 / "03_old_prediction_inventory.csv")
    row = frame[frame.method == "T4"].iloc[0]
    assert "02_window_manifest.csv" in row.reconstruction_source
