from pathlib import Path
import subprocess

CF0 = Path(__file__).resolve().parents[1]
B1 = CF0.parent


def test_matched_configs_untouched_by_git():
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--",
         "R0_hidden_mechanism_audit/results/phase3c_redesign/"
         "matched_config_h1.yaml",
         "R0_hidden_mechanism_audit/results/phase3c_redesign/"
         "matched_config_h2.yaml"],
        cwd=B1, capture_output=True, text=True)
    assert result.stdout.strip() == ""


def test_simulator_source_files_untouched_by_git():
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--",
         "R0_hidden_mechanism_audit/src/simulator.py",
         "R0_hidden_mechanism_audit/src/suppression_simulator.py"],
        cwd=B1, capture_output=True, text=True)
    assert result.stdout.strip() == ""


def test_cf0_reads_config_values_rather_than_hardcoding_them():
    text = (CF0 / "src/cf0_simulator.py").read_text()
    for key in ("capacity_extra_slots", "capacity_action_cost",
                "sync_action_cost", "sync_bytes_per_step"):
        assert f'config["actions"]["{key}"]' in text or (
            f"config['actions']['{key}']" in text)
