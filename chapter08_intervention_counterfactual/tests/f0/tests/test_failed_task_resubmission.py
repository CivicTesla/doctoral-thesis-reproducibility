from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "hpc" / "05_resubmit_failed.sh"


def test_resubmit_only_targets_missing_outputs():
    text = SCRIPT.read_text()
    assert "FAILED=" in text
    assert 'nothing to resubmit' in text
