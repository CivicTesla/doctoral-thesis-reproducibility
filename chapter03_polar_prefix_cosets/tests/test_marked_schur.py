"""Independent bounded checks of the marked quotient-Schur four-moment theorem."""
from __future__ import annotations

import random

from census_core import polar_row
from marked_schur import (direct_t_branches_n, t_equation_counts,
                          marked_pencil_signature, polar_dual_functional,
                          polar_effective_signature, w_from_t)


def test_random_small_codes_four_formulae() -> None:
    rng = random.Random(0x4D534330)
    for n in range(3, 9):
        for _ in range(20):
            candidates = list(range(1, 1 << n))
            rng.shuffle(candidates)
            rows = []
            for word in candidates:
                if len(rows) == min(3, n - 1):
                    break
                # Independence test without importing a second implementation.
                from psi2_trellis import gf2_rank
                if gf2_rank(rows + [word]) > len(rows):
                    rows.append(word)
            a = rng.randrange(1 << n)
            direct = direct_t_branches_n(tuple(rows), a, n)
            assert direct == t_equation_counts(tuple(rows), a)
            signature = marked_pencil_signature(tuple(rows), a, n)
            assert direct == signature["T_from_characters"]
            assert w_from_t(direct, n - len(rows))


def test_polar_row_controls() -> None:
    # Nontrivial upper-ideal-style row sets at N=4 and N=8.
    for n, indices, prefix in ((4, (2, 3), 1), (8, (4, 5, 6, 7), 3)):
        rows = tuple(polar_row(index, n) for index in indices)
        a = polar_row(prefix, n)
        direct = direct_t_branches_n(rows, a, n)
        assert direct == t_equation_counts(rows, a)
        assert direct == marked_pencil_signature(rows, a, n)["T_from_characters"]
        effective = polar_effective_signature(n, prefix, indices)
        assert direct == effective["T_from_characters"]
        for j in range(n):
            for k in range(n):
                assert ((polar_dual_functional(j, n) & polar_row(k, n)).bit_count() & 1) == (j == k)
