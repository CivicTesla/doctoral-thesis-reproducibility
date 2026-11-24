#!/usr/bin/env python3
"""F0 Pair 2 (H3/H4, M0/M1/M2) driver for a new seed set.

Byte-for-byte reuse of `P2_radio_compute_pair_pilot/src/pilot.py` (CFG,
task_table, schedules, step, outcome, features, fit_logistic, predict) --
the same frozen module P3 already reused unchanged -- and the same
M0/M1/M2 fit/threshold protocol from
`P3_independent_risk_validation/src/run_p3.py` (outer 5-fold
`fold = seed % 5`, M2's 20th-percentile training-fold threshold). Only the
seed set changes.

Usage: run_f0_pair2.py <phase> <seed_start> <seed_end_inclusive> <out_dir>
"""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
F0 = HERE.parents[1]
B1 = F0.parent
P2 = B1 / "P2_radio_compute_pair_pilot"
P2_COMMIT = "3726b1d"

spec = importlib.util.spec_from_file_location("p2_frozen", P2 / "src/pilot.py")
p2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p2)

FORBIDDEN_SEED_RANGES = [
    range(120, 150), range(2000, 2030), range(3000, 3400),
    range(4000, 4030), range(5000, 5030), range(5100, 5200),
    range(6000, 6010), range(6100, 6160), range(6200, 6220),
]
SCI_FILES = ("src/pilot.py", "01_frozen_config.yaml", "03_system_model.md",
             "05_action_definitions.md", "06_utility_grounding.md",
             "13_sparse_feature_definition.md", "14_full_feature_definition.md")


def assert_new_seeds(seeds):
    for r in FORBIDDEN_SEED_RANGES:
        overlap = set(seeds) & set(r)
        if overlap:
            raise SystemExit(f"F0_SEED_REUSE_DETECTED: {sorted(overlap)[:5]}")


def sha(x):
    return hashlib.sha256(x).hexdigest()


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def git_bytes(rel):
    return subprocess.check_output(
        ["git", "show", f"{P2_COMMIT}:B1/P2_radio_compute_pair_pilot/{rel}"],
        cwd=B1)


def frozen_manifest():
    rows = []
    for rel in SCI_FILES:
        now = (P2 / rel).read_bytes()
        old = git_bytes(rel)
        rows.append({
            "component": rel, "p2_commit": P2_COMMIT,
            "current_sha256": sha(now), "committed_sha256": sha(old),
            "exact_match": now == old})
    if not all(r["exact_match"] for r in rows):
        raise RuntimeError("F0_PAIR2_P2_CONFIGURATION_DRIFT")
    return rows


def costs_for(outcomes, seed, mech):
    return {r["action"]: r["total_system_cost"] for r in outcomes
            if r["seed"] == seed and r["mechanism"] == mech}


def main():
    phase, seed_start, seed_end, out_dir = sys.argv[1:5]
    seeds = range(int(seed_start), int(seed_end) + 1)
    assert_new_seeds(seeds)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    manifest = frozen_manifest()
    write_csv(out / "p2_frozen_manifest.csv", manifest)

    outcomes, diag, man, valid = [], [], [], []
    for seed in seeds:
        tasks = p2.task_table(seed)
        rs, cs = p2.schedules(seed)
        for mech in p2.MECHANISMS:
            state = p2.run_until(
                p2.initial_state(tasks), p2.CFG["decision"], mech,
                "NO_ACTION", rs, cs)
            hs = (p2.digest(state), p2.digest(tasks), p2.digest(rs),
                  p2.digest(cs), p2.digest(p2.CFG))
            man.append({
                "seed": seed, "mechanism": mech,
                "pre_action_state_hash": hs[0], "task_schedule_hash": hs[1],
                "radio_schedule_hash": hs[2], "compute_schedule_hash": hs[3],
                "config_hash": hs[4]})
            diag.append({
                "seed": seed, "mechanism": mech, "y": int(mech.startswith("H4")),
                "static": p2.features(state), "temporal": p2.features(state, temporal=True)})
            seen = []
            for action in p2.ACTIONS:
                branch = copy.deepcopy(state)
                seen.append((p2.digest(branch), hs[1], hs[2], hs[3], hs[4]))
                out_ = p2.outcome(
                    p2.run_until(branch, p2.CFG["horizon"], mech, action, rs, cs),
                    action)
                outcomes.append({"seed": seed, "mechanism": mech, "action": action, **out_})
            ok = len(set(seen)) == 1
            valid.append({"seed": seed, "mechanism": mech, "hashes_equal": ok,
                          "status": "VALID" if ok else "INVALID"})
            if not ok:
                raise RuntimeError("F0_PAIR2_COUNTERFACTUAL_IMPLEMENTATION_INVALID")

    write_csv(out / "counterfactual_manifest.csv", man)
    write_csv(out / "branch_hash_validation.csv", valid)
    write_csv(out / "action_outcomes.csv", outcomes)

    y = np.array([r["y"] for r in diag])
    seeds_arr = np.array([r["seed"] for r in diag])
    probs0 = np.zeros(len(diag))
    probs1 = np.zeros(len(diag))
    accept2 = np.zeros(len(diag), bool)
    thresholds = {}
    for fold in range(5):
        tr = seeds_arr % 5 != fold
        te = ~tr
        m0 = p2.fit_logistic([r["static"] for i, r in enumerate(diag) if tr[i]], y[tr])
        m1 = p2.fit_logistic([r["temporal"] for i, r in enumerate(diag) if tr[i]], y[tr])
        probs0[te] = p2.predict(m0, [r["static"] for i, r in enumerate(diag) if te[i]])
        probs1[te] = p2.predict(m1, [r["temporal"] for i, r in enumerate(diag) if te[i]])
        trainp = p2.predict(m1, [r["temporal"] for i, r in enumerate(diag) if tr[i]])
        threshold = float(np.quantile(abs(trainp - .5), .20))
        thresholds[fold] = threshold
        accept2[te] = abs(probs1[te] - .5) >= threshold

    rows = []
    for i, r in enumerate(diag):
        c = costs_for(outcomes, r["seed"], r["mechanism"])
        oracle = min(c, key=c.get)
        base = c[oracle]
        row = {"seed": r["seed"], "mechanism": r["mechanism"], "y": r["y"],
               "outer_fold": r["seed"] % 5, "m0_probability_H4": probs0[i],
               "m1_probability_H4": probs1[i], "m2_threshold": thresholds[r["seed"] % 5],
               "oracle_action": oracle}
        for method, prob, accept in (("M0", probs0[i], True), ("M1", probs1[i], True),
                                      ("M2", probs1[i], bool(accept2[i]))):
            pred = int(prob >= .5)
            action = ("COMPUTE_RECOVERY_ACTION" if pred else "RADIO_RECOVERY_ACTION") if accept else "NO_ACTION"
            row.update({f"{method}_pred": pred, f"{method}_accept": accept,
                        f"{method}_action": action, f"{method}_strict_cost": c[action],
                        f"{method}_regret": c[action] - base})
        rows.append(row)
    write_csv(out / "out_of_fold_predictions.csv", rows)

    elapsed = time.perf_counter() - start
    (out / "_run_metadata.json").write_text(json.dumps({
        "phase": phase, "seed_start": int(seed_start), "seed_end": int(seed_end),
        "n_seeds": len(list(seeds)) if not isinstance(seeds, range) else len(seeds),
        "n_rows": len(rows), "elapsed_seconds": elapsed}, indent=2))
    print(json.dumps({"phase": phase, "n_rows": len(rows), "elapsed_seconds": elapsed}, indent=2))


if __name__ == "__main__":
    main()
