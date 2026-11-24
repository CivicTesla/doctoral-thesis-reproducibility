from pathlib import Path

PAIR1 = Path(__file__).resolve().parents[1] / "pair1" / "run_f0_pair1.py"
PAIR2 = Path(__file__).resolve().parents[1] / "pair2" / "run_f0_pair2.py"


def test_pair1_driver_reuses_frozen_functions_not_redefine():
    text = PAIR1.read_text()
    for forbidden in ("def fit_classifier(", "def fit_classifier_costsensitive(",
                      "def convolution_features(", "def static_features(",
                      "def is_safe(", "def tiebreak_key("):
        assert forbidden not in text, f"F0 driver redefines a frozen function: {forbidden}"
    for required in ("g0.fit_classifier", "g0b.fit_classifier_costsensitive",
                     "g0.convolution_features", "g0.static_features",
                     "g0b.is_safe", "cf1r.decide", "cf1r.REJECT_COST"):
        assert required in text, f"F0 driver missing expected reuse of {required}"


def test_pair2_driver_reuses_frozen_functions_not_redefine():
    text = PAIR2.read_text()
    for forbidden in ("def fit_logistic(", "def outcome(", "def step(", "def features("):
        assert forbidden not in text
    for required in ("p2.fit_logistic", "p2.outcome", "p2.run_until", "p2.features"):
        assert required in text
