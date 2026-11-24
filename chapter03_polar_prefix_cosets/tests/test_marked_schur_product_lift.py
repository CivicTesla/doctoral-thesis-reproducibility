"""Bounded direct checks for the C1 product-lift T000 square law."""
from __future__ import annotations

from census_core import polar_row, span_words
from marked_schur import direct_t_branches_n
from n64_hierarchical_controls import product_lift


def _split(word: int, n: int) -> tuple[int, int]:
    return (sum(((word >> (2 * j)) & 1) << j for j in range(n)),
            sum(((word >> (2 * j + 1)) & 1) << j for j in range(n)))


def test_c1_lift_is_coordinate_product_and_squares_t000() -> None:
    for n, i, js in ((4, 1, (2, 3)), (8, 3, (5, 6, 7))):
        parent_rows = tuple(polar_row(j, n) for j in js)
        child_i, child_js = product_lift(i, js)
        child_rows = tuple(polar_row(j, 2 * n) for j in child_js)
        parent_words = set(span_words(parent_rows))
        child_words = set(span_words(child_rows))
        assert {_split(word, n) for word in child_words} == {(left, right)
                                                               for left in parent_words for right in parent_words}
        parent_t0 = direct_t_branches_n(parent_rows, polar_row(i, n), n)[0]
        child_t0 = direct_t_branches_n(child_rows, polar_row(child_i, 2 * n), 2 * n)[0]
        assert child_t0 == parent_t0 * parent_t0
