from pathlib import Path

CF1 = Path(__file__).resolve().parents[1]


def test_reject_cost_constant_reused_unchanged():
    script = (CF1 / "scripts/run_cf1_regret.py").read_text()
    assert "REJECT_COST = 5.0" in script
    assert 'assert float(frozen["reject_cost"]) == REJECT_COST' in script


def test_decide_function_matches_frozen_g0_action_rows_costs():
    script = (CF1 / "scripts/run_cf1_regret.py").read_text()
    assert '"CAPACITY_ACTION": p * h2_cost' in script
    assert '"SYNC_ACTION": (1 - p) * h1_cost' in script
    assert '"REJECT": REJECT_COST' in script


def test_reject_never_assigned_a_strict_counterfactual_cost():
    """Reject semantics (flat administrative cost, no modeled real action)
    are audited and reused, never redefined to a strict NO_ACTION cost."""
    script = (CF1 / "scripts/run_cf1_regret.py").read_text()
    assert "system_regret = strict_regret if accepted else REJECT_COST" in script
