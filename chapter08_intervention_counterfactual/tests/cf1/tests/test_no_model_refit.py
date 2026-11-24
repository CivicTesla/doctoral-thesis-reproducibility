from pathlib import Path

CF1 = Path(__file__).resolve().parents[1]

FORBIDDEN = ("gridsearch", "cross_val", "hyperparameter search",
             "select_window(", "new candidate grid")


def test_reproduce_predictions_does_not_search_hyperparameters():
    text = (CF1 / "scripts/reproduce_predictions.py").read_text().lower()
    for token in FORBIDDEN:
        assert token not in text, token


def test_reproduce_predictions_reads_frozen_selection_files_only():
    text = (CF1 / "scripts/reproduce_predictions.py").read_text()
    assert 'pd.read_csv(G0_DIR / "02_window_manifest.csv")' in text
    assert ('pd.read_csv(G0B_DIR / "05_selected_candidate_by_outer_fold.csv")'
            in text)
