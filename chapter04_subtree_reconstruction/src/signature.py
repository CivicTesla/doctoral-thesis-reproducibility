"""Exact C2 higher-order signatures and active-row reconstruction keys."""
from __future__ import annotations

from itertools import combinations


def signature(family: tuple[int, ...], k: int) -> tuple[int, ...]:
    """Labelled squarefree intersection moments in lexicographic subset order."""
    answer = []
    r = len(family)
    for degree in range(1, min(k, r) + 1):
        for indices in combinations(range(r), degree):
            common = family[indices[0]]
            for index in indices[1:]:
                common &= family[index]
            answer.append(common.bit_count())
    return tuple(answer)


def reconstruction_key(family: tuple[int, ...], n: int) -> tuple[int, ...]:
    """Sorted multiset of nonzero labelled incidence rows."""
    rows = []
    for vertex in range(n):
        row = sum((1 << column) for column, subset in enumerate(family)
                  if subset >> vertex & 1)
        if row:
            rows.append(row)
    return tuple(sorted(rows))


def host_orbit_representative(
    family: tuple[int, ...], automorphisms: tuple[tuple[int, ...], ...], permute
) -> tuple[int, ...]:
    """Canonicalize only host vertices; column labels remain fixed."""
    return min(tuple(permute(mask, automorphism) for mask in family)
               for automorphism in automorphisms)
