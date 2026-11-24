from pathlib import Path

CF0 = Path(__file__).resolve().parents[1]


def test_finalize_utility_formula_matches_frozen_source():
    cf0_text = (CF0 / "src/cf0_simulator.py").read_text()
    frozen_h1 = (CF0.parent / "R0_hidden_mechanism_audit/src/simulator.py").read_text()
    formula = (
        "utility = -(\n"
        "        weights[\"w_delay\"] * mean_delay\n"
        "        + weights[\"w_deadline_miss\"] * miss_rate\n"
        "        + weights[\"w_failure\"] * failure_rate\n"
        "        + weights[\"w_action_cost\"] * action_cost\n"
        "        + weights[\"w_communication_byte\"] * communication_bytes)"
    )
    assert formula in cf0_text
    # same weight keys read from the same config section in the frozen source
    for key in ("w_delay", "w_deadline_miss", "w_failure", "w_action_cost",
                "w_communication_byte"):
        assert key in frozen_h1
        assert key in cf0_text


def test_argmax_convention_used_not_argmin():
    text = (CF0 / "scripts/run_cf0_pilot.py").read_text()
    assert "def best_action(utilities):" in text
    assert "sorted(utilities.items(), key=lambda kv: -kv[1])" in text
    assert "argmin" not in text.lower()
