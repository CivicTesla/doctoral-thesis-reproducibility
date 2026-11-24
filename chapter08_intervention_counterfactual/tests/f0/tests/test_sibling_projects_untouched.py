import subprocess
from pathlib import Path

TOPLEVEL = Path(__file__).resolve().parents[4]  # .../codexlab
SIBLINGS = ("A2", "A3", "C1", "Pheromone")


def test_no_diff_against_main_outside_b1():
    out = subprocess.run(
        ["git", "diff", "--stat", "main...HEAD"],
        cwd=TOPLEVEL, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        for sib in SIBLINGS:
            assert not line.strip().startswith(sib + "/"), line
