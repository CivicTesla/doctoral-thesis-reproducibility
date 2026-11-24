from pathlib import Path

CF1 = Path(__file__).resolve().parents[1]
B1 = CF1.parent


def test_cf1_imports_cf0_simulator_not_a_copy():
    script = (CF1 / "scripts/run_cf1_checkpoints.py").read_text()
    assert "import cf0_simulator as cf0" in script
    assert str(B1 / "CF0_strict_counterfactual_grounding" / "src") in script or \
        'CF0 / "src"' in script


def test_cf0_source_untouched_by_this_branch():
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--",
         "CF0_strict_counterfactual_grounding/src/cf0_simulator.py"],
        cwd=B1, capture_output=True, text=True)
    assert result.stdout.strip() == ""


def test_no_duplicate_simulator_logic_in_cf1():
    script = (CF1 / "scripts/run_cf1_checkpoints.py").read_text()
    assert "def run_h1_branch" not in script
    assert "def run_h2_branch" not in script
    assert "def build_h1_exogenous" not in script
