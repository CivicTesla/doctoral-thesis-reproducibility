from pathlib import Path

CF1 = Path(__file__).resolve().parents[1]

FORBIDDEN = (
    "import torch", "tensorflow", "keras", "lstm", "gru(", "attention(",
    "transformer(", "gnn(", "graph_neural", "reinforcement", "q_learning",
    "sklearn.neural_network", "MLPClassifier", "GradientBoosting",
    "RandomForest",
)


def test_no_forbidden_model_keywords_in_cf1_scripts():
    for path in (CF1 / "scripts").glob("*.py"):
        text = path.read_text().lower()
        for token in FORBIDDEN:
            assert token.lower() not in text, f"{path.name}: {token}"


def test_cf1_only_reuses_existing_frozen_classifier_functions():
    text = (CF1 / "scripts/reproduce_predictions.py").read_text()
    assert "g0.fit_classifier(" in text
    assert "g0b.fit_classifier_costsensitive(" in text
    assert "def fit_classifier" not in text
