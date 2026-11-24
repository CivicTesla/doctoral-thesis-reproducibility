from pathlib import Path
import subprocess

import pandas as pd

CF0 = Path(__file__).resolve().parents[1]
B1 = CF0.parent


def test_seeds_are_the_frozen_4000_4029_range():
    frame = pd.read_csv(CF0 / "08_per_checkpoint_action_outcomes.csv")
    assert frame.seed.min() == 4000
    assert frame.seed.max() == 4029
    assert frame.seed.nunique() == 30


def test_no_other_csv_in_repo_used_these_seeds_before_cf0():
    """A best-effort repo-wide check mirroring the manual audit recorded in
    01_git_and_seed_provenance.md: no *other* R-folder's seed column
    contains a value in [4000, 4029]."""
    other_dirs = [
        "R0_hidden_mechanism_audit", "R1_age_conditioned_decision_recovery",
        "R2_mechanism_conditioned_risk_rescue",
        "R3_temporal_mechanism_evidence", "AS0_action_nondegeneracy_audit"]
    for d in other_dirs:
        for path in (B1 / d).rglob("*.csv"):
            try:
                cols = pd.read_csv(path, nrows=0).columns
            except Exception:
                continue
            seed_cols = [c for c in cols if "seed" in c.lower()]
            if not seed_cols:
                continue
            try:
                df = pd.read_csv(path, usecols=seed_cols)
            except Exception:
                continue
            for c in seed_cols:
                vals = pd.to_numeric(df[c], errors="coerce").dropna()
                assert not ((vals >= 4000) & (vals <= 4029)).any(), path


def test_provenance_document_records_the_freeze():
    text = (CF0 / "01_git_and_seed_provenance.md").read_text()
    assert "4000" in text and "4029" in text
    assert "before freezing" in text or "before any CF0 simulation" in text
