from pathlib import Path

CF0 = Path(__file__).resolve().parents[1]

FORBIDDEN = (
    "logisticregression", "logistic_model", "randomforest", "gradientboosting",
    "decisiontree", "sklearn", "import torch", "tensorflow", "keras",
    "nn.module", "fit(", ".predict(", "cross_val", "gridsearch",
)


def test_no_classifier_training_in_cf0_code():
    for path in (CF0 / "src/cf0_simulator.py", CF0 / "scripts/run_cf0_pilot.py",
                 CF0 / "scripts/make_cf0_figures.py"):
        text = path.read_text().lower()
        for token in FORBIDDEN:
            assert token not in text, f"{path.name}: forbidden token {token}"


def test_oracle_best_action_is_argmax_not_a_learned_model():
    script = (CF0 / "scripts/run_cf0_pilot.py").read_text()
    assert "def best_action(utilities):" in script
    assert "class " not in script  # no model class defined anywhere
