import hashlib
import subprocess
from pathlib import Path

F0 = Path(__file__).resolve().parents[1]
B1 = F0.parent
FILES = (
    "R0_hidden_mechanism_audit/src/simulator.py",
    "R0_hidden_mechanism_audit/src/suppression_simulator.py",
    "R0_hidden_mechanism_audit/results/phase3c_redesign/matched_config_h1.yaml",
    "R0_hidden_mechanism_audit/results/phase3c_redesign/matched_config_h2.yaml",
    "CF0_strict_counterfactual_grounding/src/cf0_simulator.py",
    "R3_temporal_mechanism_evidence/scripts/run_g0_temporal_audit.py",
    "R3_temporal_mechanism_evidence/scripts/run_g0b_risk_constrained_convolution.py",
)


def test_pair1_frozen_files_match_git_head():
    out = subprocess.run(["git", "status", "--porcelain"] + list(FILES),
                         cwd=B1, capture_output=True, text=True)
    assert out.stdout.strip() == "", f"uncommitted changes to frozen Pair 1 files: {out.stdout}"
