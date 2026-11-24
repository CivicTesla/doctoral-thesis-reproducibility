"""Exact constrained minimum-[3]-trade realizability audit for R2.

This is not an enumeration of arbitrary subtree families.  It tests only the
eight rows of the even/odd parity minimum trade, with an optional one common
row, on NetworkX's complete list of unlabeled trees on 8 or 9 vertices.
"""
from __future__ import annotations

from itertools import permutations

import networkx as nx
import numpy as np


def connected_table(tree: nx.Graph) -> np.ndarray:
    """Boolean table indexed by a vertex mask for connected induced subsets."""
    n = tree.number_of_nodes()
    adjacent = [0] * n
    for u, v in tree.edges():
        adjacent[u] |= 1 << v
        adjacent[v] |= 1 << u
    answer = np.zeros(1 << n, dtype=bool)
    for mask in range(1, 1 << n):
        reached = mask & -mask
        frontier = reached
        while frontier:
            bit = frontier & -frontier
            frontier ^= bit
            vertex = bit.bit_length() - 1
            new = adjacent[vertex] & mask & ~reached
            reached |= new
            frontier |= new
        answer[mask] = reached == mask
    return answer


def parity_supports(parity: int) -> np.ndarray:
    """All labelled assignments of one parity class to eight ordered slots."""
    rows = [x for x in range(16) if x.bit_count() % 2 == parity]
    answer = np.empty((40320, 4), dtype=np.uint8)
    for index, assignment in enumerate(permutations(rows)):
        for coordinate in range(4):
            answer[index, coordinate] = sum(
                ((row >> coordinate) & 1) << slot
                for slot, row in enumerate(assignment)
            )
    return answer


PARITY_SUPPORTS = (parity_supports(0), parity_supports(1))


def realizable_paddings(
    connected: np.ndarray, parity: int, common_vertex: int | None
) -> int:
    """Return a 16-bit mask of common-row patterns that are realizable.

    The eight parity blocks are assigned bijectively to all vertices except
    ``common_vertex``.  A one in bit i of the returned pattern adds that
    common vertex to coordinate support i.  Passing None is the unpadded
    eight-vertex test and returns either bit 0 or zero.
    """
    n = (len(connected)).bit_length() - 1
    if common_vertex is None:
        slots = list(range(n))
        translate = np.arange(256, dtype=np.uint16)
        candidates = PARITY_SUPPORTS[parity]
        base = connected[translate[candidates]]
        return int(np.all(base, axis=1).any())

    slots = [v for v in range(n) if v != common_vertex]
    translate = np.zeros(256, dtype=np.uint16)
    for slot_mask in range(256):
        translate[slot_mask] = sum(
            ((slot_mask >> slot) & 1) << vertex
            for slot, vertex in enumerate(slots)
        )
    supports = translate[PARITY_SUPPORTS[parity]]
    without_common = connected[supports]
    with_common = connected[supports | (1 << common_vertex)]
    result = 0
    for common_row in range(16):
        valid = np.ones(40320, dtype=bool)
        for coordinate in range(4):
            valid &= (
                with_common[:, coordinate]
                if common_row >> coordinate & 1
                else without_common[:, coordinate]
            )
        if valid.any():
            result |= 1 << common_row
    return result


def audit(n: int) -> None:
    """Print exact host indices and common-row patterns for the constrained test."""
    for index, tree in enumerate(nx.generators.nonisomorphic_trees(n), start=1):
        tree = nx.convert_node_labels_to_integers(tree)
        connected = connected_table(tree)
        even = odd = 0
        if n == 8:
            even = realizable_paddings(connected, 0, None)
            odd = realizable_paddings(connected, 1, None)
        elif n == 9:
            for vertex in range(n):
                even |= realizable_paddings(connected, 0, vertex)
                odd |= realizable_paddings(connected, 1, vertex)
        else:
            raise ValueError("R2 audit is only for n=8 or n=9")
        common = even & odd
        if common:
            print(index, sorted(dict(tree.degree()).values()), list(tree.edges()),
                  [q for q in range(16) if common >> q & 1])


if __name__ == "__main__":
    audit(8)
    audit(9)
