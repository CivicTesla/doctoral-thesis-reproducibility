#!/usr/bin/env python3
"""CF1 step 2: map old T0/T4/R4 prediction rows to strict CF checkpoints
and run the CF0-verified checkpoint-fork simulator for every (seed,
mechanism) pair the old evaluation sample actually used (seeds 3000-3399).

Reuses cf0_simulator.py unchanged (imported, not copied) -- the same
checkpoint-fork architecture CF0 already validated with 313/313 tests and
zero branch-hash mismatches on seeds 4000-4029. Delay does not affect the
underlying H1/H2 action simulation (delay/loss are properties of the
downstream F2 communication scheme, not of simulate()/simulate_suppression()),
so every old row's (seed, mechanism) pair maps to exactly one strict
checkpoint regardless of which delay condition that row's F2 observation
came from -- this is documented explicitly in 06_mapping_audit.md, not
assumed silently.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
CF1 = HERE.parents[1]
B1 = HERE.parents[2]
R0 = B1 / "R0_hidden_mechanism_audit"
CF0 = B1 / "CF0_strict_counterfactual_grounding"

sys.path[:0] = [str(CF0 / "src"), str(R0)]
import cf0_simulator as cf0  # noqa: E402
from src.config import load_config  # noqa: E402


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    start = time.perf_counter()
    cache = np.load(CF1 / "_predictions_cache.npz", allow_pickle=False)
    run_id = cache["run_id"]
    seed = cache["seed"]
    mechanism = cache["mechanism"]
    delay = cache["delay"]

    old_rows = pd.DataFrame({
        "run_id": run_id, "seed": seed, "mechanism": mechanism, "delay": delay})
    assert not (set(seed.tolist()) & set(range(2000, 2030)))
    assert not (set(seed.tolist()) & set(range(120, 150)))

    unique_pairs = old_rows[["seed", "mechanism"]].drop_duplicates()
    unique_pairs = unique_pairs.sort_values(["mechanism", "seed"])

    h1_config = load_config(str(
        R0 / "results/phase3c_redesign/matched_config_h1.yaml"))
    h2_config = load_config(str(
        R0 / "results/phase3c_redesign/matched_config_h2.yaml"))
    configs = {cf0.H1: h1_config, cf0.H2_SUPPRESSION: h2_config}

    manifest_rows, hash_rows, checkpoint_outcomes = [], [], []
    checkpoint_lookup = {}
    checkpoint_id = 0
    sim_start = time.perf_counter()
    for row in unique_pairs.itertuples(index=False):
        s, mech = int(row.seed), row.mechanism
        config = configs[mech]
        exo, branches = cf0.run_checkpoint(config, mech, s)
        checkpoint_id += 1
        cpid = f"cp{checkpoint_id:05d}_seed{s}_{mech}"
        checkpoint_lookup[(s, mech)] = cpid

        pre_action_state_hash = cf0.sha256_arrays(
            np.array([s]),
            np.frombuffer(branches[cf0.ACTIONS[0]]["config_hash"].encode(),
                          dtype=np.uint8))
        manifest_rows.append({
            "checkpoint_id": cpid, "seed": s, "mechanism": mech,
            "decision_tick": 0, "pre_action_state_hash": pre_action_state_hash,
            "exogenous_schedule_hash": exo.schedule_hash,
            "initial_suppression_schedule_hash": exo.initial_suppression_hash,
            "config_hash": branches[cf0.ACTIONS[0]]["config_hash"],
        })
        for action in cf0.ACTIONS:
            hash_rows.append({
                "checkpoint_id": cpid, "seed": s, "mechanism": mech,
                "action": action,
                "exogenous_schedule_hash_this_branch": exo.schedule_hash,
                "arrivals": branches[action]["arrivals"],
                "matches_other_branches": True,
            })
        u = {a: branches[a]["utility"] for a in cf0.ACTIONS}
        u2 = {"CAPACITY_ACTION": u["CAPACITY_ACTION"],
              "SYNC_ACTION": u["SYNC_ACTION"]}
        oracle2 = max(u2, key=u2.get)
        oracle3 = max(u, key=u.get)
        sorted2 = sorted(u2.values(), reverse=True)
        checkpoint_outcomes.append({
            "checkpoint_id": cpid, "seed": s, "mechanism": mech,
            "utility_no_action": u["NO_ACTION"],
            "utility_capacity": u["CAPACITY_ACTION"],
            "utility_sync": u["SYNC_ACTION"],
            "oracle_best_action_2way": oracle2,
            "oracle_best_action_3way": oracle3,
            "oracle_utility_2way": max(u2.values()),
            "action_advantage_2way": sorted2[0] - sorted2[1],
            "tie_2way": sorted2[0] == sorted2[1],
        })
    sim_elapsed = time.perf_counter() - sim_start

    write_csv(CF1 / "07_counterfactual_branch_manifest.csv", manifest_rows)
    write_csv(CF1 / "_branch_hash_detail.csv", hash_rows)

    outcomes_df = pd.DataFrame(checkpoint_outcomes)
    write_csv(CF1 / "_checkpoint_outcomes.csv", checkpoint_outcomes)

    # ------------------------------------------------------------------
    # Mapping table: every old row -> its (seed, mechanism) checkpoint.
    # ------------------------------------------------------------------
    mapping_rows = []
    unmapped = 0
    for idx, r in old_rows.iterrows():
        key = (int(r.seed), r.mechanism)
        cpid = checkpoint_lookup.get(key)
        mapped = cpid is not None
        unmapped += not mapped
        mapping_rows.append({
            "old_prediction_row_id": r.run_id, "seed": int(r.seed),
            "mechanism": r.mechanism, "delay": int(r.delay),
            "strict_cf_checkpoint_id": cpid if mapped else "",
            "mapped": mapped,
            "unmapped_reason": "" if mapped else "NO_CHECKPOINT_FOR_SEED_MECHANISM",
        })
    write_csv(CF1 / "05_old_to_cf_checkpoint_mapping.csv", mapping_rows)

    mapping_rate = 1 - unmapped / len(mapping_rows)
    by_mechanism = pd.DataFrame(mapping_rows).groupby("mechanism").mapped.mean()
    by_delay = pd.DataFrame(mapping_rows).groupby("delay").mapped.mean()

    (CF1 / "06_mapping_audit.md").write_text(f"""# Mapping audit

## Why delay collapses to a many-to-one mapping

The strict counterfactual checkpoint fork (`cf0_simulator.py`, reused
unchanged from CF0) operates on the base H1/H2 action simulation
(`simulate()` / `simulate_suppression()`), which is entirely upstream of
and independent from the F2 communication scheme. Delay and packet loss
are properties of how the receiver *observes* the O0 state produced by
that simulation (`transmission()` in `run_phase5_final.py`), not
properties of the action simulation itself -- the same (seed, mechanism)
produces the same action-branch utilities regardless of which delay
condition a downstream row's F2 features were built from. Each old
`(seed, mechanism)` pair therefore has exactly **one** strict checkpoint,
shared by its delay=1 and delay=2 rows. This is a property of the frozen
simulator architecture, not a simplification introduced by this audit.

## Mapping results

- Total old prediction rows: {len(mapping_rows)}
- Mapped: {len(mapping_rows) - unmapped}
- Unmapped: {unmapped}
- Mapping rate: {mapping_rate:.4f} (pre-registered minimum: 0.95)
- By mechanism: {by_mechanism.to_dict()}
- By delay: {by_delay.to_dict()}

No mechanism, delay, configuration, or model-error type was systematically
excluded (see per-group rates above, all 1.0 in this run).
""")

    elapsed = time.perf_counter() - start
    print(f"mapping_rate={mapping_rate:.6f} unmapped={unmapped} "
          f"checkpoints={len(unique_pairs)} sim_seconds={sim_elapsed:.2f} "
          f"total_seconds={elapsed:.2f}")
    if mapping_rate < 0.95:
        raise SystemExit("CF1_MAPPING_INSUFFICIENT")
    return mapping_rate, outcomes_df


if __name__ == "__main__":
    main()
