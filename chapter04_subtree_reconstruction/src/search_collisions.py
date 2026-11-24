"""Finite labelled-family collision reconnaissance for KT R0.

The search is exact over its declared range.  It quotients only by a host
automorphism, never by a column permutation.  Use --max-families to make a
deterministic prefix inspection rather than silently calling a partial run
exhaustive.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from enumerate_connected_sets import automorphisms, connected_subsets, permute_mask
from signature import reconstruction_key, signature
from small_trees import SMALL_TREES, star


def host(name: str) -> tuple[int, tuple[tuple[int, int], ...]]:
    if name in SMALL_TREES:
        return SMALL_TREES[name]
    if name.startswith("K1,"):
        return star(int(name.split(",", 1)[1]))
    raise ValueError(f"unknown host {name}")


def search(name: str, r: int, k: int, max_families: int | None = None) -> dict:
    n, edges = host(name)
    choices = connected_subsets(n, edges)
    autos = automorphisms(n, edges)
    mask_maps = [tuple(permute_mask(mask, automorphism) for mask in range(1 << n))
                 for automorphism in autos]
    seen: dict[tuple[int, ...], tuple[int, ...]] = {}
    checked = orbit_representatives = 0
    collision = None
    complete = True
    for family in itertools.product(choices, repeat=r):
        checked += 1
        if max_families is not None and checked > max_families:
            complete = False
            break
        if family != min(tuple(mapping[mask] for mask in family) for mapping in mask_maps):
            continue
        orbit_representatives += 1
        sig = signature(family, k)
        key = reconstruction_key(family, n)
        previous = seen.setdefault(sig, key)
        if previous != key:
            collision = {"family": list(family), "signature": list(sig),
                         "key": list(key), "previous_key": list(previous)}
            break
    return {
        "host": name, "n": n, "edges": list(edges), "r": r, "k": k,
        "connected_set_count": len(choices), "host_automorphism_count": len(autos),
        "raw_families_examined": checked, "orbit_representatives_examined": orbit_representatives,
        "complete": complete and collision is None, "collision": collision,
    }


def search_star_orbits(leaves: int, r: int, k: int, max_orbits: int | None = None) -> dict:
    """Exact star search modulo leaf permutations, with column labels fixed.

    A star family is encoded by its centre incidence row and a sorted multiset
    of leaf incidence rows.  For a column absent from the centre, connectedness
    permits that column in at most one leaf row (a leaf singleton).
    """
    n, edges = star(leaves)
    patterns = range(1 << r)
    seen: dict[tuple[int, ...], tuple[int, ...]] = {}
    examined = 0
    collision = None
    complete = True
    for centre in patterns:
        absent = [j for j in range(r) if not (centre >> j & 1)]
        for leaf_rows in itertools.combinations_with_replacement(patterns, leaves):
            if any(sum(row >> j & 1 for row in leaf_rows) > 1 for j in absent):
                continue
            examined += 1
            if max_orbits is not None and examined > max_orbits:
                complete = False
                break
            family = tuple(
                ((1 if centre >> j & 1 else 0) |
                 sum((1 << (leaf + 1)) for leaf, row in enumerate(leaf_rows)
                     if row >> j & 1))
                for j in range(r)
            )
            sig = signature(family, k)
            key = tuple(sorted(row for row in (centre, *leaf_rows) if row))
            previous = seen.setdefault(sig, key)
            if previous != key:
                collision = {"family": list(family), "signature": list(sig),
                             "key": list(key), "previous_key": list(previous)}
                break
        if collision:
            break
        if not complete:
            break
    return {
        "host": f"K1,{leaves}", "n": n, "edges": list(edges), "r": r, "k": k,
        "method": "exact leaf-permutation orbit enumeration",
        "orbit_representatives_examined": examined,
        "complete": complete and collision is None, "collision": collision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("host")
    parser.add_argument("r", type=int)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-families", type=int)
    parser.add_argument("--max-orbits", type=int)
    parser.add_argument("--star-orbits", action="store_true")
    args = parser.parse_args()
    if args.star_orbits:
        if not args.host.startswith("K1,"):
            raise ValueError("--star-orbits requires a K1,d host")
        result = search_star_orbits(int(args.host.split(",", 1)[1]), args.r, args.k, args.max_orbits)
    else:
        result = search(args.host, args.r, args.k, args.max_families)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
