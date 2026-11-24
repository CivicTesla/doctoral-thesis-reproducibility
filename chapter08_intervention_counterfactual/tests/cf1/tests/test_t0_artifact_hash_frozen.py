from pathlib import Path

import pandas as pd

CF1 = Path(__file__).resolve().parents[1]
B1 = CF1.parent


def test_t0_artifact_hash_recorded():
    frame = pd.read_csv(CF1 / "02_frozen_model_manifest.csv")
    row = frame[frame.method == "T0"].iloc[0]
    assert row.sha256 and len(row.sha256) == 64


def test_t0_artifact_hash_matches_current_file():
    import hashlib
    frame = pd.read_csv(CF1 / "02_frozen_model_manifest.csv")
    row = frame[frame.method == "T0"].iloc[0]
    path = B1 / row.artifact_path
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    assert h.hexdigest() == row.sha256


def test_t0_never_refit():
    frame = pd.read_csv(CF1 / "03_old_prediction_inventory.csv")
    row = frame[frame.method == "T0"].iloc[0]
    assert not bool(row.reconstruction_needed)
    assert bool(row.has_stored_per_row_predictions)
