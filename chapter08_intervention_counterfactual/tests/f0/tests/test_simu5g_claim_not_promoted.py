from pathlib import Path

F0 = Path(__file__).resolve().parents[1]


def test_closure_doc_declares_incomplete():
    text = (F0 / "00_external_validation_closure.md").read_text()
    assert "SIMU5G_EXTERNAL_VALIDATION_INCOMPLETE" in text
    assert "not established" in text.lower()


def test_survival_matrix_marks_simu5g_not_surviving():
    path = F0 / "24_final_claim_survival_matrix.csv"
    if not path.exists():
        import pytest
        pytest.skip("survival matrix not yet generated")
    import csv
    rows = list(csv.DictReader(path.open()))
    simu5g_rows = [r for r in rows if "simu5g" in r["claim"].lower()]
    assert simu5g_rows
    for r in simu5g_rows:
        assert r["final_status"] in ("FAILS", "NOT_EVALUATED")
