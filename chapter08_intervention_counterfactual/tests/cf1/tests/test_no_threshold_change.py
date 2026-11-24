from pathlib import Path
import json

CF1 = Path(__file__).resolve().parents[1]
B1 = CF1.parent


def test_cost_constants_read_from_frozen_json_not_hardcoded():
    script = (CF1 / "scripts/run_cf1_regret.py").read_text()
    assert 'frozen["h1_wrong_regret_estimate"]' in script
    assert 'frozen["h2_wrong_regret_estimate"]' in script


def test_cost_constants_match_frozen_file_values():
    frozen = json.loads((
        B1 / "R1_age_conditioned_decision_recovery/results/"
        "gate1_formal_age_necessity/gate1_frozen_receiver_models.json"
    ).read_text())["cost_sensitive_reject"]
    assert abs(frozen["h1_wrong_regret_estimate"] - 10.214756305262583) < 1e-9
    assert abs(frozen["h2_wrong_regret_estimate"] - 39.48683653673746) < 1e-9


def test_no_new_gate_threshold_introduced_ad_hoc():
    script = (CF1 / "scripts/run_cf1_regret.py").read_text()
    assert "0.80" not in script  # no coverage threshold redefinition here
