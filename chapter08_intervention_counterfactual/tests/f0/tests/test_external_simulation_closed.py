from pathlib import Path

F0 = Path(__file__).resolve().parents[1]


def test_closure_doc_lists_forbidden_actions():
    text = (F0 / "00_external_validation_closure.md").read_text()
    for forbidden in ("Fixing the RADIO action", "Installing ns-3",
                     "Installing 5G-LENA", "any other external simulator"):
        assert forbidden in text
