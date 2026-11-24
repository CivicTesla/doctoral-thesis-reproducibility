#!/usr/bin/env python3
"""F0 Pair 1 (H1/H2, T0/T4/R4) driver for a new seed set.

Every scientifically load-bearing function here is imported, unchanged,
from the already-frozen B1 modules (see
`../01_frozen_artifact_manifest.csv`): mechanism dynamics
(`src.simulator`/`src.suppression_simulator`), O0 generation
(`run_gate0_pilot.simulate_records`), the F2 communication scheme
(`run_phase5_final.transmission`), the frozen T0 model
(`src.phase5_final.predict_frozen` / `gate1_frozen_receiver_models.json`),
T4's feature/window-selection/fit protocol (`run_g0_temporal_audit`), R4's
cost-sensitive/candidate-selection protocol
(`run_g0b_risk_constrained_convolution`), the strict checkpoint-fork action
simulator (`cf0_simulator`), and CF1's strict-regret decision rule
(`run_cf1_regret.decide`). This script only substitutes a new seed set for
the historical 3000-3399 sample and re-drives the exact same functions --
it defines no new feature, threshold, model family, or cost.

Usage: run_f0_pair1.py <phase> <seed_start> <seed_end_inclusive> <out_dir>
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
F0 = HERE.parents[1]
B1 = F0.parent
R0 = B1 / "R0_hidden_mechanism_audit"
R1 = B1 / "R1_age_conditioned_decision_recovery"
R2 = B1 / "R2_mechanism_conditioned_risk_rescue"
R3 = B1 / "R3_temporal_mechanism_evidence"
CF0 = B1 / "CF0_strict_counterfactual_grounding"
CF1 = B1 / "CF1_strict_regret_regeneration"

sys.path[:0] = [
    str(R1 / "src"), str(R1 / "scripts"),
    str(R0), str(R0 / "scripts"),
    str(R2 / "src"), str(R2 / "scripts"),
    str(R3 / "scripts"),
    str(CF0 / "src"),
    str(CF1 / "scripts"),
]

from run_gate0_pilot import simulate_records  # noqa: E402
from run_phase5_final import transmission  # noqa: E402
from src.config import load_config  # noqa: E402
from src.phase5_final import frozen_scheme, predict_frozen  # noqa: E402
from src.simulator import H1  # noqa: E402
from src.suppression_simulator import H2_SUPPRESSION  # noqa: E402
from receiver_features import selected_receiver_features  # noqa: E402
from protocol import assert_fixed_schemes  # noqa: E402
import run_g0_temporal_audit as g0  # noqa: E402
import run_g0b_risk_constrained_convolution as g0b  # noqa: E402
import cf0_simulator as cf0  # noqa: E402
import run_cf1_regret as cf1r  # noqa: E402

H2 = H2_SUPPRESSION
PRIMARY_CONDITIONS = ((1, 0.0), (2, 0.0))  # delay, loss -- see 03_preregistered_protocol.md
FORBIDDEN_SEED_RANGES = [
    range(120, 150), range(2000, 2030), range(3000, 3400),
    range(4000, 4030), range(5000, 5030), range(5100, 5200),
    range(6000, 6010), range(6100, 6160), range(6200, 6220),
]


def assert_new_seeds(seeds):
    for r in FORBIDDEN_SEED_RANGES:
        overlap = set(seeds) & set(r)
        if overlap:
            raise SystemExit(f"F0_SEED_REUSE_DETECTED: {sorted(overlap)[:5]}")


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def serialized_probability(artifact, feature_map):
    order = artifact["feature_order"]
    x = np.asarray([float(feature_map[name]) for name in order])
    mean = np.asarray(artifact["scaler_mean"], float)
    scale = np.asarray(artifact["scaler_scale"], float)
    weights = np.asarray(artifact["coefficients_intercept_first"], float)
    score = np.clip(weights[0] + ((x - mean) / scale) @ weights[1:], -30, 30)
    return float(1 / (1 + np.exp(-score)))


def build_g1a_style_data(seeds, out):
    """Reproduces run_g1a_instrumented_replay.py's own pipeline (O0
    generation -> F2 transmission -> T0 frozen probability -> decision
    state manifest) for a new seed set, primary conditions only."""
    traces_dir = out / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    frozen_gate1 = json.loads(
        (R1 / "results/gate1_formal_age_necessity/"
         "gate1_frozen_receiver_models.json").read_text())
    frozen_phase5 = assert_fixed_schemes(json.loads(
        (R0 / "results/phase5_minimal_distributed_evidence/"
         "development_frozen_schemes.json").read_text()))
    f2 = frozen_phase5["F2"]
    scheme = frozen_scheme(f2)
    include_o1 = f2["scheme_definition"]["include_derived_o1"]
    ranges = json.loads(
        (R0 / "results/phase5_minimal_distributed_evidence/"
         "frozen_scheme_manifest.json").read_text())["quantization_ranges"]
    h1_cfg = load_config(R0 / "results/phase3c_redesign/matched_config_h1.yaml")
    h2_cfg = load_config(R0 / "results/phase3c_redesign/matched_config_h2.yaml")

    records = simulate_records(
        h1_cfg, h2_cfg,
        {"train": set(seeds), "validation": set(), "pilot_test": set()})
    records.sort(key=lambda r: (int(r["seed"]), r["mechanism"]))

    o0_path = traces_dir / "f0_o0.npz"
    np.savez_compressed(
        o0_path,
        values=np.asarray([r["o0"] for r in records], dtype=np.int32),
        run_id=np.asarray([r["run_id"] for r in records]),
        seed=np.asarray([r["seed"] for r in records], dtype=np.int32),
        mechanism=np.asarray([r["mechanism"] for r in records]))

    seed_base = min(seeds)
    states = []
    trace_manifest = []
    for delay, loss in PRIMARY_CONDITIONS:
        received_rows, arrival_rows, run_ids = [], [], []
        for index, record in enumerate(records):
            ordinary, cost, sent = transmission(
                record["o0"], scheme, ranges, include_o1, delay, loss,
                seed=(record["seed"] * 100003 + delay * 1009
                      + int(loss * 1000) + index))
            groups = {
                name_: selected_receiver_features(
                    sent, (name_,), initial_arrival_age=delay)
                for name_ in ("age", "mask", "silence")}
            m4_features = dict(ordinary)
            for values in groups.values():
                m4_features.update(values)
            p4 = serialized_probability(
                frozen_gate1["age_aware_model"], m4_features)
            received_rows.append(np.asarray(sent["received"], np.float32))
            arrival_rows.append(np.asarray(sent["arrival_mask"], bool))
            run_ids.append(record["run_id"])

            fold = (int(record["seed"]) - seed_base) % 5
            states.append({
                "run_id": record["run_id"], "seed": record["seed"],
                "mechanism": record["mechanism"], "delay": delay,
                "packet_loss": loss, "outer_fold": fold,
                "mechanism_probability_H1": 1.0 - p4,
                "mechanism_probability_H2": p4,
                "predicted_mechanism": H2 if p4 >= .5 else H1,
                "oracle_action": record["oracle_action"],
                "potential_wrong_transfer_regret":
                    record["wrong_transfer_regret"],
            })
        trace_path = traces_dir / f"f0_f2_delay_{delay}_loss_{loss}.npz"
        np.savez_compressed(
            trace_path,
            received=np.asarray(received_rows, np.float32),
            arrival_mask=np.asarray(arrival_rows, bool),
            run_id=np.asarray(run_ids))
        trace_manifest.append({
            "condition": f"delay_{delay}_loss_{loss}", "delay": delay,
            "packet_loss": loss, "runs": len(records),
            "trace_reference": str(trace_path.relative_to(out)),
            "o0_reference": str(o0_path.relative_to(out))})

    write_csv(out / "01_replay_manifest.csv", trace_manifest)
    write_csv(out / "03_decision_state_manifest.csv", states)
    return records


def load_features(out, seeds):
    states = pd.read_csv(out / "03_decision_state_manifest.csv")
    assert states.delay.isin([1, 2]).all()
    assert set(states.seed).issubset(set(seeds))
    manifest = pd.read_csv(out / "01_replay_manifest.csv")
    traces, trace_index = {}, {}
    for delay in (1, 2):
        row = manifest[manifest.delay == delay].iloc[0]
        with np.load(out / row.trace_reference, allow_pickle=False) as z:
            traces[delay] = {
                "received": z["received"], "arrival": z["arrival_mask"]}
            trace_index[delay] = {
                str(r): i for i, r in enumerate(z["run_id"])}

    base, received, arrival = [], [], []
    for row in states.itertuples():
        idx = trace_index[row.delay][row.run_id]
        received.append(traces[row.delay]["received"][idx])
        arrival.append(traces[row.delay]["arrival"][idx])
        base.append({
            "run_id": row.run_id, "seed": int(row.seed),
            "mechanism": row.mechanism, "delay": int(row.delay),
            "outer_fold": int(row.outer_fold),
            "frozen_probability_h2": float(row.mechanism_probability_H2),
            "oracle_action": row.oracle_action,
            "potential_regret": float(row.potential_wrong_transfer_regret)})
    labels = np.asarray([b["mechanism"] == H2 for b in base], int)
    seeds_arr = np.asarray([b["seed"] for b in base])
    static = np.asarray([
        g0.static_features(r, a, b["delay"])
        for r, a, b in zip(received, arrival, base)])
    return base, received, arrival, labels, seeds_arr, static


def fit_t4_and_r4(base, received, labels, seeds_arr, static):
    """Exact reuse of g0's window-selection + fit and g0b's
    candidate-selection + fit, per outer fold, on the new seed set."""
    n = len(base)
    t4_pred = np.full(n, np.nan)
    r4_pred = np.full(n, np.nan)
    window_rows, selection_rows = [], []

    # Convolution features per window length, computed once (not per fold):
    # `convolution_features`/`visible_series` are pure functions of the
    # (already fold-independent) received trace, so precomputing them
    # outside the fold loop changes nothing about what gets fit -- it only
    # avoids recomputing the same fixed-kernel convolution response
    # 5 times. T4's classifier (`g0.fit_classifier`) is still refit fresh
    # per outer fold, on that fold's own training rows only.
    conv_by_length = {
        length: np.asarray([
            g0.convolution_features(g0.visible_series(r, length))
            for r in received])
        for length in g0.WINDOWS}

    for fold in range(5):
        train = np.asarray([r["outer_fold"] != fold for r in base])
        test = ~train
        train_idx = np.flatnonzero(train)

        # T4-only window selection: exact reuse of run_g0_temporal_audit's
        # T4 branch of select_window() (x = [static, conv], inner 4-fold BA
        # via g0.fit_classifier/g0.probabilities, tie-break smaller window)
        # -- skips T1/T2/T3/T5's AR-model fitting, which CF1/F0 never use
        # for T4/R4, instead of paying for it via the shared select_window
        # entry point.
        inner_group = g0b.inner_groups(train_idx, seeds_arr)
        scores = {}
        for length in g0.WINDOWS:
            x_len = np.column_stack([static, conv_by_length[length]])
            fold_scores = []
            for inner in range(4):
                fit = np.asarray([
                    i for i in train_idx if inner_group[seeds_arr[i]] != inner])
                val = np.asarray([
                    i for i in train_idx if inner_group[seeds_arr[i]] == inner])
                model = g0.fit_classifier(x_len[fit], labels[fit])
                fold_scores.append(g0.balanced_accuracy_score(
                    labels[val], g0.probabilities(model, x_len[val]) >= .5))
            scores[length] = np.mean(fold_scores)
        t4_window = max(g0.WINDOWS, key=lambda w: (scores[w], -w))
        window_rows.append({"outer_fold": fold, "selected_window": t4_window,
                            **{f"score_len{k}": v for k, v in scores.items()}})

        x = np.column_stack([static, conv_by_length[t4_window]])
        t4_model = g0.fit_classifier(x[train], labels[train])
        t4_pred[test] = g0.probabilities(t4_model, x[test])

        # ---- R4: same candidates/selection procedure as g0b, inner
        # GroupKFold over this fold's own training seeds. ----
        groups = g0b.inner_groups(train_idx, seeds_arr)
        inner_fold_of = np.array([
            groups.get(seeds_arr[i], -1) for i in range(n)])
        frozen_p = np.asarray([b["frozen_probability_h2"] for b in base])
        t0_m, _ = g0b.pooled_metrics(base, train_idx, frozen_p[train_idx], "T0")

        candidate_rows = {}
        for w, b in g0b.ALL_CANDIDATES:
            inner_pred = np.full(n, np.nan)
            for inner in range(4):
                inner_val = np.asarray([
                    i for i in train_idx if inner_fold_of[i] == inner])
                inner_fit = np.asarray([
                    i for i in train_idx if inner_fold_of[i] != inner])
                model = g0b.fit_classifier_costsensitive(
                    x[inner_fit], labels[inner_fit], w)
                inner_pred[inner_val] = g0b.probabilities_biased(
                    model, x[inner_val], b)
            m, _ = g0b.pooled_metrics(base, train_idx, inner_pred[train_idx], "candidate")
            safe, collapse = g0b.is_safe(m, t0_m)
            candidate_rows[(w, b)] = {**m, "safe": safe, "collapse": collapse}

        safe_candidates = [
            (w, b) for (w, b), r in candidate_rows.items() if r["safe"]]
        if safe_candidates:
            r4_w, r4_b = min(
                safe_candidates,
                key=lambda t: g0b.tiebreak_key(t[0], t[1], candidate_rows[t]))
            fallback = False
        else:
            r4_w, r4_b, fallback = None, None, True
        selection_rows.append({
            "outer_fold": fold, "selected_w_H2": r4_w if r4_w else "",
            "selected_b_H2": r4_b if r4_b else "", "fallback_to_T0": fallback})

        if fallback:
            r4_pred[test] = frozen_p[test]
        else:
            full_model = g0b.fit_classifier_costsensitive(
                x[train_idx], labels[train_idx], r4_w)
            r4_pred[test] = g0b.probabilities_biased(full_model, x[test], r4_b)

    return t4_pred, r4_pred, window_rows, selection_rows


def strict_checkpoints(seeds, out):
    """Exact reuse of CF0's run_checkpoint + CF1's checkpoint-fork mapping
    logic (run_cf1_checkpoints.py lines 73-116) for the new seed set."""
    h1_config = load_config(str(
        R0 / "results/phase3c_redesign/matched_config_h1.yaml"))
    h2_config = load_config(str(
        R0 / "results/phase3c_redesign/matched_config_h2.yaml"))
    configs = {cf0.H1: h1_config, cf0.H2_SUPPRESSION: h2_config}

    hash_rows, outcomes = [], {}
    for mech in (cf0.H1, cf0.H2_SUPPRESSION):
        for s in seeds:
            exo, branches = cf0.run_checkpoint(configs[mech], mech, s)
            u = {a: branches[a]["utility"] for a in cf0.ACTIONS}
            u2 = {"CAPACITY_ACTION": u["CAPACITY_ACTION"],
                  "SYNC_ACTION": u["SYNC_ACTION"]}
            oracle2 = max(u2, key=u2.get)
            sorted2 = sorted(u2.values(), reverse=True)
            outcomes[(s, mech)] = {
                "utility_capacity": u["CAPACITY_ACTION"],
                "utility_sync": u["SYNC_ACTION"],
                "oracle_best_action_2way": oracle2,
                "oracle_utility_2way": max(u2.values()),
                "tie_2way": sorted2[0] == sorted2[1],
            }
            hash_rows.append({
                "seed": s, "mechanism": mech,
                "exogenous_schedule_hash": exo.schedule_hash,
                "initial_suppression_schedule_hash":
                    exo.initial_suppression_hash})
    write_csv(out / "checkpoint_hash_manifest.csv", hash_rows)
    return outcomes


def main():
    phase, seed_start, seed_end, out_dir = sys.argv[1:5]
    seeds = list(range(int(seed_start), int(seed_end) + 1))
    assert_new_seeds(seeds)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    build_g1a_style_data(seeds, out)
    base, received, arrival, labels, seeds_arr, static = load_features(out, seeds)
    t4_pred, r4_pred, window_rows, selection_rows = fit_t4_and_r4(
        base, received, labels, seeds_arr, static)
    write_csv(out / "t4_window_selection.csv", window_rows)
    write_csv(out / "r4_candidate_selection.csv", selection_rows)

    checkpoint_outcomes = strict_checkpoints(seeds, out)

    frozen_gate1 = json.loads(
        (R1 / "results/gate1_formal_age_necessity/"
         "gate1_frozen_receiver_models.json").read_text())["cost_sensitive_reject"]
    h1_cost = float(frozen_gate1["h1_wrong_regret_estimate"])
    h2_cost = float(frozen_gate1["h2_wrong_regret_estimate"])
    predictions = {
        "T0": np.asarray([b["frozen_probability_h2"] for b in base]),
        "T4": t4_pred, "R4": r4_pred}

    rows = []
    for method, p in predictions.items():
        for i, b in enumerate(base):
            action, accepted = cf1r.decide(p[i], h1_cost, h2_cost)
            checkpoint = checkpoint_outcomes[(b["seed"], b["mechanism"])]
            oracle_best = checkpoint["oracle_best_action_2way"]
            wrong = accepted and action != oracle_best
            if wrong:
                executed_u = (checkpoint["utility_capacity"]
                              if action == "CAPACITY_ACTION"
                              else checkpoint["utility_sync"])
                strict_regret = checkpoint["oracle_utility_2way"] - executed_u
            else:
                strict_regret = 0.0
            system_regret = strict_regret if accepted else cf1r.REJECT_COST
            rows.append({
                "method": method, "run_id": b["run_id"], "seed": b["seed"],
                "mechanism": b["mechanism"], "delay": b["delay"],
                "outer_fold": b["outer_fold"], "p_h2": float(p[i]),
                "predicted_mechanism": H2 if p[i] >= .5 else H1,
                "action": action, "accepted": accepted, "wrong": wrong,
                "oracle_best_action": oracle_best,
                "system_regret": system_regret,
                "accepted_only_regret": strict_regret if accepted else np.nan,
            })
    write_csv(out / "per_sample_strict_action_outcomes.csv", rows)

    elapsed = time.perf_counter() - start
    (out / "_run_metadata.json").write_text(json.dumps({
        "phase": phase, "seed_start": int(seed_start),
        "seed_end": int(seed_end), "n_seeds": len(seeds),
        "n_rows": len(rows), "elapsed_seconds": elapsed}, indent=2))
    print(json.dumps({
        "phase": phase, "n_seeds": len(seeds), "n_rows": len(rows),
        "elapsed_seconds": elapsed}, indent=2))


if __name__ == "__main__":
    main()
