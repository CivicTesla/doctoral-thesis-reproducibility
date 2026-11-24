import csv
from pathlib import Path

import pytest

F0 = Path(__file__).resolve().parents[1]


def test_pair2_all_branch_hashes_equal():
    path = F0 / "runs" / "main_pair2" / "branch_hash_validation.csv"
    if not path.exists():
        pytest.skip("main pair2 run not yet present")
    rows = list(csv.DictReader(path.open()))
    assert rows
    bad = [r for r in rows if r["status"] != "VALID"]
    assert not bad, f"invalid strict-fork hashes: {bad}"


def test_pair1_checkpoint_manifest_present():
    path = F0 / "runs" / "main_pair1" / "checkpoint_hash_manifest.csv"
    if not path.exists():
        pytest.skip("main pair1 run not yet present")
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 800  # 400 seeds x 2 mechanisms
