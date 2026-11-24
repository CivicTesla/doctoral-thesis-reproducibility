import csv
from pathlib import Path

F0 = Path(__file__).resolve().parents[1]


def test_harm_mass_reconstruction_error_below_tolerance():
    path = F0 / "18_harm_mass_results.csv"
    if not path.exists():
        import pytest
        pytest.skip("harm mass results not yet generated")
    rows = list(csv.DictReader(path.open()))
    assert rows
    max_err = max(abs(float(r["absolute_error"])) for r in rows)
    assert max_err < 1e-9, f"max reconstruction error {max_err} >= 1e-9"
