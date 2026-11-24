#!/usr/bin/env python3
"""Generate the F0 run manifest (Section 16).

Scientific constraint: T4/R4 (Pair 1) and M1/M2 (Pair 2) fit per outer
fold using an outer-CV split over the *whole* seed set for that phase, so
the atomic, embarrassingly-parallel unit is "one phase's full seed range"
for the fit step, not a single (seed, mechanism, action) branch -- forcing
per-branch array tasks would either replicate the fold-fitting 2400x
(wasteful) or silently break fold isolation (scientifically invalid). The
two drivers (`../pair1/run_f0_pair1.py`, `../pair2/run_f0_pair2.py`)
internally vectorize the embarrassingly-parallel part (per-seed telemetry
generation) and do the outer-CV fit once. Each manifest row is therefore
one (phase, pair) job; a real HPC deployment would parallelize *inside*
each job over seed chunks for the telemetry-generation stage only, then
run one single-task fit stage per (phase, pair) -- documented here rather
than faked as one-row-per-branch.
"""
import csv
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
F0 = HERE.parents[1]

PHASES = [
    ("smoke", 6990, 6994),
    ("main", 7000, 7399),
    ("holdout", 7400, 7499),
]
PAIRS = [
    ("pair1", "run_f0_pair1.py", ("H1_CAPACITY_SHORTAGE", "H2_DISTRIBUTED_CAPACITY_SUPPRESSION")),
    ("pair2", "run_f0_pair2.py", ("H3_RADIO_UPLINK_BOTTLENECK", "H4_EDGE_COMPUTE_BOTTLENECK")),
]


def config_hash(phase, pair, seed_start, seed_end):
    payload = f"{phase}:{pair}:{seed_start}:{seed_end}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def main():
    rows = []
    task_id = 0
    for phase, seed_start, seed_end in PHASES:
        for pair, script, mechanisms in PAIRS:
            task_id += 1
            rows.append({
                "task_id": task_id, "phase": phase, "pair": pair,
                "mechanisms": "|".join(mechanisms),
                "actions": "3 (strict fork, all mechanisms)",
                "seed_start": seed_start, "seed_end": seed_end,
                "n_seeds": seed_end - seed_start + 1,
                "driver_script": script,
                "config_hash": config_hash(phase, pair, seed_start, seed_end),
                "expected_output": f"{phase}_{pair}/_run_metadata.json",
            })
    out = F0 / "06_main_run_manifest.csv" if len(sys.argv) < 2 else Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
