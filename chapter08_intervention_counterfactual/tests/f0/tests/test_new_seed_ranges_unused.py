import subprocess
from pathlib import Path

B1 = Path(__file__).resolve().parents[2]
NEW_RANGES = [range(6990, 6995), range(7000, 7400), range(7400, 7500)]
OLD_RANGES = [
    range(120, 150), range(2000, 2030), range(3000, 3400),
    range(4000, 4030), range(5000, 5030), range(5100, 5200),
    range(6000, 6010), range(6100, 6160), range(6200, 6220),
]


def test_new_ranges_disjoint_from_old():
    for new in NEW_RANGES:
        for old in OLD_RANGES:
            assert not (set(new) & set(old)), f"{new} overlaps {old}"


def test_new_ranges_disjoint_from_each_other():
    for i, a in enumerate(NEW_RANGES):
        for b in NEW_RANGES[i + 1:]:
            assert not (set(a) & set(b))


def test_no_seed_grep_hits_outside_f0():
    out = subprocess.run(
        ["git", "grep", "-l", "-E", "range\\(7[0-4][0-9]{2}, ?7[0-4][0-9]{2}\\)"],
        cwd=B1.parent, capture_output=True, text=True)
    hits = [l for l in out.stdout.splitlines() if "F0_final_hpc_validation" not in l]
    assert not hits, f"unexpected new-seed range usage outside F0: {hits}"
