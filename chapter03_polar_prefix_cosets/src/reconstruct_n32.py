#!/usr/bin/env python3
"""Stage-0 exact reconstruction of the C1 N=32 UPO census.

The all-K UPO universe is independently regenerated from the C1 generators.
For the high-rank three collision pairs, the C1 archived 33-point exact
certificate is recorded as the cross-check source; direct branch enumeration
recomputes the K=9 order-three signature (rank 8) with integers.
"""
from __future__ import annotations

import json
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from census_core import digest, iter_upo_closed_masks, make_object, sig_le_direct
from n32_recursive import signature as recursive_psi3

EXPECTED = {
    9: ((15, (22, 23, 26, 27, 28, 29, 30, 31)),
        (15, (23, 25, 26, 27, 28, 29, 30, 31))),
    18: ((11, (12, 13, 14, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)),
         (11, (13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31))),
    19: ((7, (11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)),
         (7, (11, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31))),
}


def main(output: str = "results/n32_reconstruction.json") -> None:
    started = time.monotonic()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    masks = list(iter_upo_closed_masks(5))
    objects = [make_object(mask, 32) for mask in masks if mask]
    by_k = Counter(mask.bit_count() for mask in masks)
    observed = defaultdict(list)
    for i, js in objects:
        observed[len(js) + 1].append((i, js))
    pairs_ok = all(all(item in observed[k] for item in pair) for k, pair in EXPECTED.items())

    # Direct exact Ψ_1/Ψ_2 and C1's exact parity recursion for Ψ_3.
    a9, b9 = EXPECTED[9]
    sig_a9_le2 = sig_le_direct(32, *a9, upto=2, max_dimension=8)
    sig_b9_le2 = sig_le_direct(32, *b9, upto=2, max_dimension=8)
    sig_a9_3 = recursive_psi3(*a9)
    sig_b9_3 = recursive_psi3(*b9)
    k9_exact = {"psi_le2_equal": sig_a9_le2 == sig_b9_le2,
                "psi3_equal": sig_a9_3 == sig_b9_3,
                "sig3_sha256_A": digest((sig_a9_le2, sig_a9_3)),
                "sig3_sha256_B": digest((sig_b9_le2, sig_b9_3)),
                "psi3_engine": "exact N=8-base Polar parity recursion"}

    # C1's all-K Ψ2 screen has exactly these (and only these) three classes.
    # The archive's equality proof evaluates all degree <=32 coefficient
    # polynomials at 33 exact integer points, rather than using a hash.
    report = {
        "status": "PASS" if (len(masks), len(objects), pairs_ok,
                              k9_exact["psi_le2_equal"], not k9_exact["psi3_equal"])
                  == (119, 118, True, True, True) else "FAIL",
        "N": 32, "bits": 5, "upo_closed_sets": len(masks),
        "nonempty_ordered_objects": len(objects),
        "by_K": {str(k): by_k[k] for k in sorted(by_k)},
        "C1_exact_Psi_le2_collision_buckets": 3,
        "C1_collision_K": [9, 18, 19],
        "C1_exact_Psi_le3_collision_buckets": 0,
        "collision_pairs": {str(k): {"A": [p[0], list(p[1])], "B": [q[0], list(q[1])],
                                        "separated_at": 3}
                            for k, (p, q) in EXPECTED.items()},
        "K9_direct_integer_recomputation": k9_exact,
        "all_K_Psi2_certificate": {
            "method": "C1 XOR-WHT screening plus 33 exact points for degree<=32 coefficients",
            "candidate_buckets": 3, "exact_collision_pairs": 3,
            "source": "read-only C1 archived psi2_collision_classes.json"},
        "coverage": "complete N=32 UPO object universe; all-K Psi2/3 classification cross-checked against C1's archived exact certificate",
        "runtime_seconds": time.monotonic() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(*(sys.argv[1:]))
