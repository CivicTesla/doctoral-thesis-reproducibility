from pathlib import Path
import sys

CF0 = Path(__file__).resolve().parents[1]
B1 = CF0.parent
sys.path[:0] = [str(CF0 / "src"), str(B1 / "R0_hidden_mechanism_audit")]


def test_exogenous_table_built_before_action_is_known_per_branch():
    """The exogenous table generator never receives `action` as a
    parameter -- it cannot condition on it, structurally."""
    import inspect
    import cf0_simulator as cf0
    for fn in (cf0.build_h1_exogenous, cf0.build_h2_exogenous):
        params = list(inspect.signature(fn).parameters)
        assert "action" not in params


def test_run_checkpoint_rebuilds_exogenous_independently_per_action():
    text = (CF0 / "src/cf0_simulator.py").read_text()
    assert "for action in ACTIONS:" in text
    assert "build_h1_exogenous(config, seed)" in text
    assert "build_h2_exogenous(config, seed)" in text
