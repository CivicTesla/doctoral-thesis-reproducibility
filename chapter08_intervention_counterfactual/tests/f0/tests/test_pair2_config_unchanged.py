import subprocess
from pathlib import Path

F0 = Path(__file__).resolve().parents[1]
B1 = F0.parent
FILES = (
    "P2_radio_compute_pair_pilot/src/pilot.py",
    "P2_radio_compute_pair_pilot/01_frozen_config.yaml",
    "P3_independent_risk_validation/src/run_p3.py",
)


def test_pair2_frozen_files_match_git_head():
    out = subprocess.run(["git", "status", "--porcelain"] + list(FILES),
                         cwd=B1, capture_output=True, text=True)
    assert out.stdout.strip() == "", f"uncommitted changes to frozen Pair 2 files: {out.stdout}"
