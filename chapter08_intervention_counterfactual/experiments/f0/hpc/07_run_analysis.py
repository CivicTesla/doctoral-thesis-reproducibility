#!/usr/bin/env python3
"""Thin wrapper: run the unified analysis (../analyze_f0.py) over the
main-phase run outputs."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
F0 = HERE.parents[1]

if __name__ == "__main__":
    runs = Path(sys.argv[1]) if len(sys.argv) > 1 else F0 / "runs"
    subprocess.run([
        sys.executable, str(F0 / "analyze_f0.py"),
        str(runs / "main_pair1"), str(runs / "main_pair2"), str(F0),
    ], check=True)
