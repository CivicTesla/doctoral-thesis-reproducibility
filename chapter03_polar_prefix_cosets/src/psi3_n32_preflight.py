#!/usr/bin/env python3
"""Exact N=32 Psi3 preflight on the six certified Psi2 survivors only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from census_core import canonical_bytes, object_id
from hierarchical_census import run as hierarchical_run
from n32_recursive import signature as c1_signature
from polar_recursive import signature
from reconstruct_n32 import EXPECTED


def run() -> dict:
    census = hierarchical_run(32)
    if (census["candidate_objects"], census["psi2_collision_bucket_count"],
            census["psi3_evaluated_objects"], census["psi3_collision_bucket_count"]) != (118, 3, 6, 0):
        raise AssertionError("N=32 collision-first preflight census changed")
    observed_ks = [bucket["K"] for bucket in census["psi2_collision_buckets"]]
    if observed_ks != [9, 18, 19]:
        raise AssertionError("N=32 collision K classes changed")
    pairs = []
    for k, pair in sorted(EXPECTED.items()):
        values = []
        for i, js in pair:
            generic, _ = signature(32, i, js, 3)
            c1 = c1_signature(i, js)
            if generic != c1:
                raise AssertionError(f"generic Psi3 differs from C1 recursion at K={k}")
            # A second exact calculation establishes deterministic serialization.
            repeated, _ = signature(32, i, js, 3)
            canonical = canonical_bytes(generic).decode("ascii")
            if generic != repeated or canonical != canonical_bytes(repeated).decode("ascii"):
                raise AssertionError(f"nondeterministic Psi3 at K={k}")
            values.append({"object": object_id(32, i, js), "psi3_canonical": canonical,
                           "psi3_sha256": hashlib.sha256(canonical.encode("ascii")).hexdigest()})
        if values[0]["psi3_canonical"] == values[1]["psi3_canonical"]:
            raise AssertionError(f"Psi3 fails to separate the K={k} pair")
        pairs.append({"K": k, "objects": values, "psi3_equal": False,
                      "separated": True, "engine": "exact C1 N=8-base Polar parity recursion"})
    return {"status": "PASS", "N": 32, "nonempty_ordered_objects": 118,
            "psi2_collision_K": [9, 18, 19], "psi2_collision_pairs": 3,
            "psi3_evaluated_objects": 6, "psi3_collision_pairs": 0,
            "pairs": pairs,
            "definition_check": "generic exact integer Psi3 equals the authoritative C1 N=32 recursion for all six survivors"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "pairs": len(result["pairs"])}, sort_keys=True))


if __name__ == "__main__":
    main()
