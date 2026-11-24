from pathlib import Path

CF1 = Path(__file__).resolve().parents[1]


def test_decision_is_one_of_the_seven_preregistered_values():
    decision = (CF1 / "22_cf1_decision.txt").read_text().strip()
    assert decision in {
        "CF1_ACCURACY_CONTROL_INVERSION_SURVIVES",
        "CF1_RISK_TRANSFER_ONLY_SURVIVES",
        "CF1_CLASSIFICATION_GAIN_ALIGNS_WITH_CONTROL",
        "CF1_NO_STABLE_CONTROL_RISK_DIFFERENCE",
        "CF1_MAPPING_INSUFFICIENT",
        "CF1_PREDICTION_REPRODUCTION_FAILED",
        "CF1_COUNTERFACTUAL_IMPLEMENTATION_INVALID",
    }


def test_inversion_criteria_hardcoded_not_tuned_post_hoc():
    script = (CF1 / "scripts/run_cf1_regret.py").read_text()
    assert 'ba_gain = results["T4"]["balanced_accuracy"] > results["T0"]["balanced_accuracy"]' in script
    assert "ci_excludes_zero_positive" in script


def test_mapping_threshold_hardcoded_at_95_percent():
    script = (CF1 / "scripts/run_cf1_checkpoints.py").read_text()
    assert "mapping_rate < 0.95" in script
