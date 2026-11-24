from pathlib import Path

F0 = Path(__file__).resolve().parents[1]


def test_freeze_script_refuses_if_holdout_already_run():
    text = (F0 / "hpc" / "08_freeze_pre_holdout.sh").read_text()
    assert "REFUSING" in text and "holdout output already exists" in text


def test_holdout_submit_requires_freeze_commit():
    text = (F0 / "hpc" / "09_submit_holdout_array.sh").read_text()
    assert "no pre-holdout freeze commit found" in text
