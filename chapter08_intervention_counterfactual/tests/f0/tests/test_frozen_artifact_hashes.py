import csv
import hashlib
from pathlib import Path

F0 = Path(__file__).resolve().parents[1]
B1 = F0.parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_all_frozen_artifacts_match_manifest():
    rows = list(csv.DictReader((F0 / "01_frozen_artifact_manifest.csv").open()))
    assert rows
    for row in rows:
        path = B1 / row["path"]
        assert path.exists(), f"missing frozen artifact: {row['path']}"
        assert sha256(path) == row["sha256"], f"hash drift: {row['path']}"
