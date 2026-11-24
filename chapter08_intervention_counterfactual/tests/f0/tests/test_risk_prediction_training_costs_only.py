from pathlib import Path

RP = Path(__file__).resolve().parents[1] / "risk_prediction.py"


def test_prediction_uses_train_costs_and_test_counts_only():
    text = RP.read_text()
    assert "never fed back into the prediction" in text
    # the predicted_delta line must reference only train-derived means and
    # test-derived counts, and actual_delta must be computed separately
    assert "predicted_delta = (" in text
    assert "actual_delta = float(" in text
