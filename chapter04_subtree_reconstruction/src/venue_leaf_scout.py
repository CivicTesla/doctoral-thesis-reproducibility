"""Exact scout for padded affine [3]-trade realizability on fixed-leaf hosts.

For an n-vertex tree this tests the *restricted* collision construction with
an all-one connected core of n-8 vertices and the eight even, respectively
odd, rows of the four-coordinate parity trade on the complement.  It is not
an enumeration of arbitrary subtree families and does not decide k(T).
"""
from __future__ import annotations

from itertools import permutations

import numpy as np

from enumerate_connected_sets import connected_subsets


EVEN = tuple(x for x in range(16) if x.bit_count() % 2 == 0)
ODD = tuple(x for x in range(16) if x.bit_count() % 2 == 1)


def connected(mask: int, adjacent: list[int]) -> bool:
    if not mask:
        return False
    reached = mask & -mask
    frontier = reached
    while frontier:
        bit = frontier & -frontier
        frontier ^= bit
        vertex = bit.bit_length() - 1
        new = adjacent[vertex] & mask & ~reached
        reached |= new
        frontier |= new
    return reached == mask


def parity_realization(
    n: int, edges: tuple[tuple[int, int], ...], parity: tuple[int, ...]
) -> tuple[int, tuple[int, ...]] | None:
    """Return one all-one-core/affine-leg realization, if one exists."""
    core_size = n - 8
    if core_size < 1:
        raise ValueError("the padded-core scout requires n >= 9")
    adjacent = [0] * n
    for u, v in edges:
        adjacent[u] |= 1 << v
        adjacent[v] |= 1 << u
    assignments = np.array(tuple(permutations(parity)), dtype=np.uint8)
    for core in connected_subsets(n, edges):
        if core.bit_count() != core_size:
            continue
        slots = tuple(v for v in range(n) if not (core >> v & 1))
        index = {vertex: position for position, vertex in enumerate(slots)}
        parent: dict[int, int] = {}
        frontier = [vertex for vertex in slots if adjacent[vertex] & core]
        for vertex in frontier:
            parent[vertex] = -1
        while frontier:
            vertex = frontier.pop()
            for child in range(n):
                if child not in index or child in parent:
                    continue
                if adjacent[vertex] >> child & 1:
                    parent[child] = vertex
                    frontier.append(child)
        if len(parent) != 8:
            raise AssertionError("connected core did not root its complement")
        valid = np.ones(len(assignments), dtype=bool)
        for child, ancestor in parent.items():
            if ancestor >= 0:
                child_rows = assignments[:, index[child]]
                parent_rows = assignments[:, index[ancestor]]
                valid &= (child_rows & ~parent_rows) == 0
        if valid.any():
            return core, tuple(int(x) for x in assignments[valid.argmax()])
    return None


def spider(legs: tuple[int, ...]) -> tuple[int, tuple[tuple[int, int], ...]]:
    edges: list[tuple[int, int]] = []
    next_vertex = 1
    for length in legs:
        parent = 0
        for _ in range(length):
            edges.append((parent, next_vertex))
            parent = next_vertex
            next_vertex += 1
    return next_vertex, tuple(edges)


def double_spider(
    left_leaves: int, right_leaves: int, bridge_edges: int
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Two branching centres joined by a path with ``bridge_edges`` edges."""
    if bridge_edges < 1:
        raise ValueError("need distinct centres")
    edges: list[tuple[int, int]] = []
    for vertex in range(bridge_edges):
        edges.append((vertex, vertex + 1))
    next_vertex = bridge_edges + 1
    for _ in range(left_leaves):
        edges.append((0, next_vertex))
        next_vertex += 1
    for _ in range(right_leaves):
        edges.append((bridge_edges, next_vertex))
        next_vertex += 1
    return next_vertex, tuple(edges)


HOSTS = {
    "Q2=S7^(1)": (9, ((0, 1), (0, 2), (1, 3), (1, 4), (1, 5),
                          (1, 6), (1, 7), (1, 8))),
    "Y(3,3,3)": spider((3, 3, 3)),
    "Y(4,4,3)": spider((4, 4, 3)),
    "Y(3,2,2,2)": spider((3, 2, 2, 2)),
    "Y(3,3,3,2)": spider((3, 3, 3, 2)),
    "DS(2,2;5)": double_spider(2, 2, 5),
    "DS(2,2;7)": double_spider(2, 2, 7),
}


def main() -> None:
    for name, (n, edges) in HOSTS.items():
        even = parity_realization(n, edges, EVEN)
        odd = parity_realization(n, edges, ODD)
        print(f"{name}: n={n}, even={even is not None}, odd={odd is not None}, "
              f"padded-affine-collision={even is not None and odd is not None}")


if __name__ == "__main__":
    main()
