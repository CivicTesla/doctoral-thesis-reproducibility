from pathlib import Path

CF0 = Path(__file__).resolve().parents[1]


def test_cf0_simulator_does_not_use_bare_random_random_seed():
    text = (CF0 / "src/cf0_simulator.py").read_text()
    assert "random.Random(seed)" not in text


def test_cf0_simulator_uses_keyed_rng_derivation():
    text = (CF0 / "src/cf0_simulator.py").read_text()
    assert "def keyed_rng(" in text
    assert "def derived_seed(" in text
    assert "hashlib.sha256" in text


def test_exogenous_and_endogenous_streams_are_separately_keyed():
    text = (CF0 / "src/cf0_simulator.py").read_text()
    assert 'keyed_rng(seed, mechanism, "exogenous")' in text or \
        'keyed_rng(seed, H1, "exogenous")' in text
    assert 'keyed_rng(seed, mechanism, action, "endogenous")' in text
