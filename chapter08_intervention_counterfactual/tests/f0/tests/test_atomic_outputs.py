from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "hpc" / "02_run_array_task.sh"


def test_array_task_writes_to_tmp_then_renames():
    text = SCRIPT.read_text()
    assert "OUT_TMP" in text and 'mv "$OUT_TMP" "$OUT_FINAL"' in text
    assert 'rm -rf "$OUT_TMP"' in text  # clears any stale partial tmp first


def test_array_task_skips_already_complete():
    text = SCRIPT.read_text()
    assert "already complete" in text and "skip -- idempotent rerun" in text
