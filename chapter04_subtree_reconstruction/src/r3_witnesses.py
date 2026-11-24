"""Verify compact R3 padded parity-trade witnesses and Q-free status."""
from __future__ import annotations

from itertools import combinations

import networkx as nx


Q_EDGES = (
    # Q1,...,Q7 in the canonical R3 order.
    ((0, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8)),
    ((0, 1), (0, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8)),
    ((0, 1), (0, 2), (1, 3), (1, 5), (1, 6), (1, 7), (1, 8), (3, 4)),
    ((0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8)),
    ((0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7), (1, 8)),
    ((0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 7), (1, 8), (5, 6)),
    ((0, 1), (0, 2), (0, 3), (1, 4), (1, 6), (1, 7), (1, 8), (4, 5)),
)
Q_GRAPHS = tuple(nx.Graph(edges) for edges in Q_EDGES)
EVEN = (0, 3, 5, 6, 9, 10, 12, 15)
ODD = (1, 2, 4, 7, 8, 11, 13, 14)


def signature(rows: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        sum((row & mask) == mask for row in rows)
        for size in range(1, 4)
        for columns in combinations(range(4), size)
        for mask in (sum(1 << column for column in columns),)
    )


def connected_columns(tree: nx.Graph, rows: tuple[int, ...]) -> bool:
    for column in range(4):
        vertices = [v for v, row in enumerate(rows) if row >> column & 1]
        if not vertices or not nx.is_connected(tree.subgraph(vertices)):
            return False
    return True


def avoids_q(tree: nx.Graph) -> bool:
    return not any(nx.is_isomorphic(tree.subgraph(nodes), q)
                   for q in Q_GRAPHS
                   for nodes in combinations(tree.nodes(), 9))


def check(name: str, edges: tuple[tuple[int, int], ...], a: tuple[int, ...], b: tuple[int, ...]) -> None:
    tree = nx.Graph(edges)
    assert len(tree) == len(a) == len(b)
    assert signature(a) == signature(b)
    assert connected_columns(tree, a) and connected_columns(tree, b)
    assert sorted(row for row in a if row) != sorted(row for row in b if row)
    assert avoids_q(tree)
    print(name, "verified; degree sequence", sorted(dict(tree.degree()).values()))


def core_corona(core_edges: tuple[tuple[int, int], ...], core: tuple[int, ...], leaves: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """All core vertices have row 1111; leaves receive parity rows."""
    n = len(core) + len(leaves)
    a = [15] * n
    b = [15] * n
    for vertex, row_a, row_b in zip(leaves, EVEN, ODD):
        a[vertex] = row_a
        b[vertex] = row_b
    return tuple(a), tuple(b)


if __name__ == "__main__":
    # Smallest Q-free counterexample: common rows at 0,1; all rows are
    # ordered by vertex 0,...,9.
    e10 = ((0, 1), (0, 2), (1, 5), (1, 7), (1, 8), (1, 9),
           (2, 3), (2, 4), (5, 6))
    a10 = (15, 15, 15, 3, 5, 6, 0, 9, 10, 12)
    b10 = (15, 15, 7, 1, 2, 11, 8, 4, 13, 14)
    check("n=10", e10, a10, b10)

    # Connected all-one core plus eight pendant parity vertices.
    e11 = ((0, 1), (1, 2), (0, 3), (0, 4), (0, 5),
           (2, 6), (2, 7), (2, 8), (2, 9), (2, 10))
    a11, b11 = core_corona(e11, (0, 1, 2), tuple(range(3, 11)))
    check("n=11", e11, a11, b11)

    e12 = ((0, 1), (1, 2), (2, 3), (1, 4), (1, 5), (1, 6),
           (3, 7), (3, 8), (3, 9), (3, 10), (3, 11))
    a12, b12 = core_corona(e12, (0, 1, 2, 3), tuple(range(4, 12)))
    check("n=12", e12, a12, b12)
