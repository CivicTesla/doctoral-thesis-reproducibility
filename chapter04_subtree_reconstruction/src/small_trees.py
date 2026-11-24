"""Named small unlabeled tree representatives used by KT R0."""
from __future__ import annotations


def path(n: int) -> tuple[tuple[int, int], ...]:
    return tuple((i, i + 1) for i in range(n - 1))


SMALL_TREES = {
    "P1": (1, ()),
    "P2": (2, path(2)),
    "P3": (3, path(3)),
    "P4": (4, path(4)),
    "K1,3": (4, ((0, 1), (0, 2), (0, 3))),
    "P5": (5, path(5)),
    "F3": (5, ((0, 1), (0, 2), (0, 3), (3, 4))),
    "K1,4": (5, ((0, 1), (0, 2), (0, 3), (0, 4))),
}


def star(leaves: int) -> tuple[int, tuple[tuple[int, int], ...]]:
    return leaves + 1, tuple((0, leaf) for leaf in range(1, leaves + 1))
