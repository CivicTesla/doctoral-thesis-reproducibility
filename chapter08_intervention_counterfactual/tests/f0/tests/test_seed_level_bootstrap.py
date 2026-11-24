from pathlib import Path

ANALYZE = Path(__file__).resolve().parents[1] / "analyze_f0.py"


def test_bootstrap_resamples_seeds_not_rows():
    text = ANALYZE.read_text()
    assert "BOOTS = 2000" in text
    # Vectorized seed-block bootstrap: each seed contributes a fixed-size
    # row block (asserted in p1_blocks/p2_blocks), and resampling draws
    # whole block INDICES with replacement (rng.integers over n_seeds),
    # which is exactly a seed-level bootstrap, just without per-replicate
    # DataFrame filtering.
    assert "rng.integers(0, n_seeds, n_seeds)" in text
    assert "ragged per-seed row count" in text  # asserts fixed block size
    # every resample draws whole per-seed blocks, never row-level .sample(
    assert ".sample(" not in text
