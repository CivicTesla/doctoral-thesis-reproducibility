"""Exact connected-subset and host-automorphism primitives for KT R0."""
from __future__ import annotations

from itertools import permutations


def connected_subsets(n: int, edges: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    """All nonempty connected vertex subsets, represented by n-bit masks."""
    adjacent = [0] * n
    for u, v in edges:
        adjacent[u] |= 1 << v
        adjacent[v] |= 1 << u
    out = []
    for mask in range(1, 1 << n):
        reached = mask & -mask
        frontier = reached
        while frontier:
            bit = frontier & -frontier
            frontier ^= bit
            u = bit.bit_length() - 1
            new = adjacent[u] & mask & ~reached
            reached |= new
            frontier |= new
        if reached == mask:
            out.append(mask)
    return tuple(out)


def automorphisms(n: int, edges: tuple[tuple[int, int], ...]) -> tuple[tuple[int, ...], ...]:
    """Brute-force host automorphisms; only used on hosts with at most 7 vertices."""
    edge_set = {tuple(sorted(edge)) for edge in edges}
    answer = []
    for perm in permutations(range(n)):
        image = {tuple(sorted((perm[u], perm[v]))) for u, v in edges}
        if image == edge_set:
            answer.append(perm)
    return tuple(answer)


def permute_mask(mask: int, perm: tuple[int, ...]) -> int:
    out = 0
    for old, new in enumerate(perm):
        if mask >> old & 1:
            out |= 1 << new
    return out
