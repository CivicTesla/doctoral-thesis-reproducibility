#!/usr/bin/env python3
"""Recompute sha256 of every entry in 01_frozen_artifact_manifest.csv and
confirm it still matches -- run before trusting any analysis output."""
import csv
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
F0 = HERE.parents[1]
B1 = F0.parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest = list(csv.DictReader((F0 / "01_frozen_artifact_manifest.csv").open()))
    mismatches = []
    for row in manifest:
        path = B1 / row["path"]
        if not path.exists() or sha256(path) != row["sha256"]:
            mismatches.append(row["artifact"])
    if mismatches:
        print("F0_FROZEN_ARTIFACT_INVALID:", mismatches)
        raise SystemExit(1)
    print(f"all {len(manifest)} frozen artifacts verified unchanged")


if __name__ == "__main__":
    main()
