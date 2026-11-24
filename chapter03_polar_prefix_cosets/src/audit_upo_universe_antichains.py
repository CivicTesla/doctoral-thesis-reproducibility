#!/usr/bin/env python3
"""Independent antichain reconstruction audit for the finite UPO universe.

The primary census enumerates upper sets by an included/excluded closure
recursion.  This audit instead enumerates antichains of the UPO poset and
reconstructs each upper set as the upward closure of its unique minimal
antichain.  It intentionally does not call ``poset_closures`` or
``iter_upo_closed_masks`` on the antichain side.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from census_core import iter_upo_closed_masks, make_object, object_id


def direct_upo_successors(vertex: int, bits: int) -> tuple[int, ...]:
    """Independent literal implementation of the UPO cover definition.

    A cover either changes one zero coordinate to one, or moves a one from a
    lower coordinate to a strictly higher zero coordinate.  This deliberately
    does not call the primary enumerator's successor helper.
    """
    successors: list[int] = []
    for coordinate in range(bits):
        if (vertex >> coordinate) & 1 == 0:
            successors.append(vertex | (1 << coordinate))
    for low in range(bits):
        if (vertex >> low) & 1:
            for high in range(low + 1, bits):
                if (vertex >> high) & 1 == 0:
                    successors.append((vertex & ~(1 << low)) | (1 << high))
    return tuple(successors)


def independent_upper_closure(bits: int) -> tuple[int, ...]:
    """Compute reachability from covers by descending dynamic programming."""
    n = 1 << bits
    direct = [direct_upo_successors(vertex, bits) for vertex in range(n)]
    # Every listed successor has a larger integer label: an addition sets a
    # zero bit; a left swap moves one from a lower to a higher bit.
    upper = [0] * n
    for vertex in range(n - 1, -1, -1):
        mask = 1 << vertex
        for successor in direct[vertex]:
            if successor <= vertex:
                raise AssertionError("UPO cover order is not increasing")
            mask |= upper[successor]
        upper[vertex] = mask
    return tuple(upper)


def antichain_upper_sets(bits: int):
    upper = independent_upper_closure(bits)
    n = 1 << bits
    lower = [0] * n
    for low, mask in enumerate(upper):
        for high in range(n):
            if mask >> high & 1:
                lower[high] |= 1 << low
    comparable = tuple(upper[x] | lower[x] for x in range(n))

    def visit(candidates: int, generated_upper: int):
        if not candidates:
            yield generated_upper
            return
        vertex = (candidates & -candidates).bit_length() - 1
        # Exclude this vertex from the antichain.
        yield from visit(candidates & ~(1 << vertex), generated_upper)
        # Include it and exclude every comparable vertex from later choices.
        yield from visit(candidates & ~comparable[vertex], generated_upper | upper[vertex])

    yield from visit((1 << n) - 1, 0)


def object_ids(masks, n: int) -> set[str]:
    return {object_id(n, *make_object(mask, n)) for mask in masks if mask}


def audit(bits: int) -> dict:
    n = 1 << bits
    primary_masks = tuple(iter_upo_closed_masks(bits))
    antichain_masks = tuple(antichain_upper_sets(bits))
    if len(set(antichain_masks)) != len(antichain_masks):
        raise AssertionError("antichain route produced duplicate upper sets")
    primary_ids, antichain_ids = object_ids(primary_masks, n), object_ids(antichain_masks, n)
    return {"bits": bits, "N": n, "status": "PASS" if set(primary_masks) == set(antichain_masks) and primary_ids == antichain_ids else "FAIL",
            "primary_upper_set_count": len(primary_masks), "antichain_upper_set_count": len(antichain_masks),
            "primary_nonempty_object_count": len(primary_ids), "antichain_nonempty_object_count": len(antichain_ids),
            "upper_set_sets_equal": set(primary_masks) == set(antichain_masks),
            "canonical_object_id_sets_equal": primary_ids == antichain_ids,
            "primary_object_id_set_sha256": hashlib.sha256("\n".join(sorted(primary_ids)).encode("ascii")).hexdigest(),
            "antichain_object_id_set_sha256": hashlib.sha256("\n".join(sorted(antichain_ids)).encode("ascii")).hexdigest(),
            "method_A": "included/excluded upper-closure recursion", "method_B": "antichain enumeration followed by upward-closure reconstruction"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.bits)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
