#!/usr/bin/env python3
"""Aggregate + hash-verify holdout outputs, run the same analysis as main
(read-only comparison against main-scale results, no threshold/claim
changes permitted regardless of what holdout shows -- Section 6)."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
F0 = HERE.parents[1]

if __name__ == "__main__":
    runs = Path(sys.argv[1]) if len(sys.argv) > 1 else F0 / "runs"
    for pair in ("pair1", "pair2"):
        meta = runs / f"holdout_{pair}" / "_run_metadata.json"
        if not meta.exists():
            print(f"F0_HPC_EXECUTION_INCOMPLETE: holdout_{pair} missing")
            raise SystemExit(1)
    subprocess.run([
        sys.executable, str(F0 / "analyze_f0.py"),
        str(runs / "holdout_pair1"), str(runs / "holdout_pair2"),
        str(F0 / "holdout_analysis"),
    ], check=True)
    print("holdout aggregation OK")
