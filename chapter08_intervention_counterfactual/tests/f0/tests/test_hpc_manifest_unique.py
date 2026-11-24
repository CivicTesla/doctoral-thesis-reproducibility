import csv
from pathlib import Path

import pytest

F0 = Path(__file__).resolve().parents[1]


def test_manifest_task_ids_unique():
    path = F0 / "06_main_run_manifest.csv"
    if not path.exists():
        pytest.skip("manifest not yet generated")
    rows = list(csv.DictReader(path.open()))
    ids = [r["task_id"] for r in rows]
    assert len(ids) == len(set(ids))
