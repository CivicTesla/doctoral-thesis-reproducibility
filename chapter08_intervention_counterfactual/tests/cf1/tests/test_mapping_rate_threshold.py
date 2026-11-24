from pathlib import Path

import pandas as pd

CF1 = Path(__file__).resolve().parents[1]


def test_mapping_rate_at_least_95_percent():
    frame = pd.read_csv(CF1 / "05_old_to_cf_checkpoint_mapping.csv")
    rate = frame.mapped.mean()
    assert rate >= 0.95


def test_mapping_audit_reports_rate_and_breakdown():
    text = (CF1 / "06_mapping_audit.md").read_text()
    assert "Mapping rate" in text
    assert "By mechanism" in text
    assert "By delay" in text


def test_no_systematic_exclusion_by_mechanism_or_delay():
    frame = pd.read_csv(CF1 / "05_old_to_cf_checkpoint_mapping.csv")
    by_mech = frame.groupby("mechanism").mapped.mean()
    by_delay = frame.groupby("delay").mapped.mean()
    assert (by_mech >= 0.95).all()
    assert (by_delay >= 0.95).all()
