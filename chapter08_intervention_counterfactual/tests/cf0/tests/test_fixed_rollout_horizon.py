from pathlib import Path
import sys

CF0 = Path(__file__).resolve().parents[1]
B1 = CF0.parent
sys.path[:0] = [str(CF0 / "src"), str(B1 / "R0_hidden_mechanism_audit")]


def test_horizon_matches_frozen_matched_configs():
    from src.config import load_config
    h1 = load_config(str(
        B1 / "R0_hidden_mechanism_audit/results/phase3c_redesign/"
        "matched_config_h1.yaml"))
    h2 = load_config(str(
        B1 / "R0_hidden_mechanism_audit/results/phase3c_redesign/"
        "matched_config_h2.yaml"))
    assert int(h1["horizon"]) == 500
    assert int(h2["horizon"]) == 500


def test_protocol_documents_horizon_reuse_not_invention():
    text = (CF0 / "02_preregistered_protocol.md").read_text()
    assert "horizon = 500" in text
    assert "no new horizon is invented" in text.lower()


def test_exogenous_arrays_shaped_to_horizon_500():
    import cf0_simulator as cf0
    from src.config import load_config
    h1 = load_config(str(
        B1 / "R0_hidden_mechanism_audit/results/phase3c_redesign/"
        "matched_config_h1.yaml"))
    exo = cf0.build_h1_exogenous(h1, 4000)
    assert exo.arrival.shape[0] == 500
