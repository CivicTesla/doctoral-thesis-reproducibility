def test_main_holdout_seed_ranges_disjoint():
    main = set(range(7000, 7400))
    holdout = set(range(7400, 7500))
    assert not (main & holdout)
    assert len(main) == 400
    assert len(holdout) == 100
