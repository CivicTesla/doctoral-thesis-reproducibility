from pathlib import Path
import subprocess

CF1 = Path(__file__).resolve().parents[1]
B1 = CF1.parent


def test_matched_configs_untouched():
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--",
         "R0_hidden_mechanism_audit/results/phase3c_redesign/"
         "matched_config_h1.yaml",
         "R0_hidden_mechanism_audit/results/phase3c_redesign/"
         "matched_config_h2.yaml"],
        cwd=B1, capture_output=True, text=True)
    assert result.stdout.strip() == ""


def test_simulator_sources_untouched():
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--",
         "R0_hidden_mechanism_audit/src/simulator.py",
         "R0_hidden_mechanism_audit/src/suppression_simulator.py"],
        cwd=B1, capture_output=True, text=True)
    assert result.stdout.strip() == ""


def test_action_space_matches_frozen_definition():
    script = (CF1 / "scripts/run_cf1_checkpoints.py").read_text()
    assert "cf0.ACTIONS" in script  # imported, not redefined
