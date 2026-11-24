#!/usr/bin/env python3
"""B1 R3 Gate 0B: risk-constrained temporal convolution audit.

Reuses the frozen T4 fixed-convolution feature pipeline from Gate 0
(``run_g0_temporal_audit.py``) unchanged. The only things this script is
allowed to vary are (1) an H2 sample-weight multiplier applied to the T4
classifier's training loss and (2) a fixed additive bias on the H2 logit
applied after training, both selected from pre-registered grids using
inner (within outer-training) GroupKFold splits only. F2 history, the
communication budget, the convolution kernels, the outer folds, the seed
partition, and the downstream expected-cost/reject decision rule are all
imported from the frozen modules and never redefined here.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
R3 = HERE.parents[1]
B1 = HERE.parents[2]
R2 = B1 / "R2_mechanism_conditioned_risk_rescue"
DATA = R2 / "G1A_instrumented_replay_data"
G0 = R3 / "G0_temporal_consistency_and_convolution"
OUT = R3 / "G0B_risk_constrained_convolution"
CONCLUSION = R3 / "results/CONCLUSIONS/G0B_RISK_CONSTRAINED_CONVOLUTION"

sys.path[:0] = [str(R3 / "scripts")]
import run_g0_temporal_audit as g0  # noqa: E402  frozen T4 pipeline, read-only reuse

frozen = g0.frozen  # R2's frozen decision/cost module, imported by g0
H1, H2 = g0.H1, g0.H2
COMMUNICATION = g0.COMMUNICATION
REJECT_COST = g0.REJECT_COST
BOOTSTRAPS = 2000
IRLS_ITERS = 80

# Pre-registered candidate grids (Section VI). These are the ONLY values
# ever tried; none are searched continuously and none are chosen using the
# outer held-out fold.
W_H2_GRID = (1, 2, 4, 8, 16)
B_H2_GRID = (0.0, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00)
R1_FIXED = (4.0, 0.0)  # "natural" main setting: H2 error costs ~4x H1 error
R2_FAMILY_WEIGHT = 1.0
R3_W_GRID = (2, 4, 8)
R3_B_GRID = (0.25, 0.50, 0.75, 1.00)

R1_GRID = tuple((float(w), 0.0) for w in W_H2_GRID)
R2_GRID = tuple((1.0, float(b)) for b in B_H2_GRID)
R3_GRID = tuple((float(w), float(b)) for w in R3_W_GRID for b in R3_B_GRID)
ALL_CANDIDATES = sorted(set(R1_GRID) | set(R2_GRID) | set(R3_GRID))

COVERAGE_TARGETS = (0.75, 0.80, 0.85, 0.90)


# ---------------------------------------------------------------------
# Section II change #1 and #2: cost-sensitive training loss and a fixed
# H2 logit bias. Structurally identical to g0.fit_classifier / g0.probabilities
# (same IRLS routine, same regularization); w_h2=1.0, b_h2=0.0 reproduces the
# frozen T4 baseline bit-for-bit.
# ---------------------------------------------------------------------
def fit_classifier_costsensitive(x, y, w_h2=1.0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    weights = np.zeros(design.shape[1])
    base_pos = .5 / max(np.mean(y == 1), 1e-9)
    base_neg = .5 / max(np.mean(y == 0), 1e-9)
    sample_weight = np.where(y == 1, w_h2 * base_pos, base_neg)
    for _ in range(IRLS_ITERS):
        p = 1 / (1 + np.exp(-np.clip(design @ weights, -30, 30)))
        curvature = sample_weight * p * (1 - p)
        hessian = design.T @ (curvature[:, None] * design)
        hessian[1:, 1:] += np.eye(len(weights) - 1)
        gradient = design.T @ (sample_weight * (p - y))
        gradient[1:] += weights[1:]
        step = np.linalg.solve(hessian + np.eye(len(weights)) * 1e-9, gradient)
        weights -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return mean, scale, weights


def raw_logits(model, x):
    mean, scale, weights = model
    design = np.column_stack([np.ones(len(x)), (np.asarray(x, float) - mean) / scale])
    return design @ weights


def probabilities_biased(model, x, b_h2=0.0):
    z = np.clip(raw_logits(model, x) + b_h2, -30, 30)
    return 1 / (1 + np.exp(-z))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_data():
    """Frozen F2 receiver history load, identical to run_g0_temporal_audit.main()."""
    states = pd.read_csv(DATA / "03_decision_state_manifest.csv")
    states = states[states.delay.isin([1, 2])].copy()
    assert states.seed.between(3000, 3399).all()
    assert not states.seed.between(2000, 2029).any()
    assert not states.seed.between(120, 149).any()
    manifest = pd.read_csv(DATA / "01_replay_manifest.csv")
    traces, trace_index = {}, {}
    for delay in (1, 2):
        row = manifest[
            (manifest.delay == delay) & (manifest.packet_loss == 0)].iloc[0]
        with np.load(R2 / row.trace_reference, allow_pickle=False) as z:
            traces[delay] = {
                "received": z["received"], "arrival": z["arrival_mask"]}
            trace_index[delay] = {str(r): i for i, r in enumerate(z["run_id"])}

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
    seeds = np.asarray([b["seed"] for b in base])
    static = np.asarray([
        g0.static_features(r, a, b["delay"])
        for r, a, b in zip(received, arrival, base)])
    return base, received, labels, seeds, static


def inner_groups(train_indices, seeds):
    unique = np.unique(seeds[train_indices])
    return {seed: i % 4 for i, seed in enumerate(sorted(unique))}


def pooled_metrics(base, indices, p_h2, method):
    rows = [base[i] for i in indices]
    action_rows = g0.action_rows(rows, p_h2)
    m = g0.metrics(action_rows, method)
    pred = np.asarray([r["predicted"] == H2 for r in action_rows])
    m["predicted_h2_fraction"] = float(np.mean(pred))
    m["predicted_h1_fraction"] = float(1 - np.mean(pred))
    return m, action_rows


def is_safe(m, t0_ref):
    collapse = m["h1_recall"] < .65 or m["predicted_h2_fraction"] > .85
    return (
        m["coverage"] >= .80
        and m["h2_regret"] <= t0_ref["h2_regret"]
        and m["h2_to_h1_error"] <= t0_ref["h2_to_h1_error"]
        and not collapse), collapse


def tiebreak_key(w, b, m):
    return (m["total_regret"], m["h2_regret"], -m["balanced_accuracy"], w, b)


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    CONCLUSION.mkdir(parents=True, exist_ok=True)

    base, received, labels, seeds, static = load_data()
    n = len(base)
    frozen_p = np.asarray([b["frozen_probability_h2"] for b in base])

    window_manifest = pd.read_csv(G0 / "02_window_manifest.csv")
    t4_window = {
        int(r.outer_fold): int(r.selected_window)
        for r in window_manifest[window_manifest.method == "T4"].itertuples()}

    extraction_start = time.perf_counter()
    conv_by_length = {}
    for length in sorted(set(t4_window.values())):
        conv_by_length[length] = np.asarray([
            g0.convolution_features(g0.visible_series(r, length))
            for r in received])
    extraction_seconds = time.perf_counter() - extraction_start

    x_by_fold = {
        f: np.column_stack([static, conv_by_length[t4_window[f]]])
        for f in range(5)}

    data_rows = [{
        "seed_min": 3000, "seed_max": 3399, "seed_count": 400, "runs": n,
        "delays": "1|2", "packet_loss": 0,
        "deployable_source": "F2_RECEIVED_AND_ARRIVAL_TENSORS_ONLY",
        "decision_timestamp": 499, "sealed_seeds_loaded": False,
        "r0_final_seeds_loaded": False, "simulation_rerun": False,
        "new_hand_features": False, "new_convolution_kernels": False,
        "t4_feature_extraction_source": "run_g0_temporal_audit.py (unmodified)",
        "trace_manifest_sha256": g0.sha256(DATA / "01_replay_manifest.csv")}]
    write_csv(OUT / "01_data_and_fold_manifest.csv", data_rows)

    grid_rows = []
    for w, b in ALL_CANDIDATES:
        grid_rows.append({
            "w_h2": w, "b_h2": b,
            "in_r0": (w, b) == (1.0, 0.0),
            "in_r1_grid": (w, b) in R1_GRID,
            "in_r2_grid": (w, b) in R2_GRID,
            "in_r3_grid": (w, b) in R3_GRID,
            "in_r4_pool": True})
    write_csv(OUT / "03_candidate_weight_and_bias_grid.csv", grid_rows)

    # -----------------------------------------------------------------
    # Per outer fold: (a) inner GroupKFold evaluation of every candidate,
    # pooled over the 4 inner validation folds; (b) one full-outer-train
    # fit per candidate for held-out outer prediction. Same outer folds
    # as R3 Gate 0 (outer_fold column from the frozen decision-state
    # manifest); the outer held-out fold is never touched by selection.
    # -----------------------------------------------------------------
    inner_grid_rows = []
    selection_rows = []
    full_predictions = {c: np.full(n, np.nan) for c in ALL_CANDIDATES}
    t0_full_by_fold = {}
    training_seconds = 0.0
    # per-fold, per-candidate row-level inner-val predictions, kept for
    # leave-one-seed-out selection-stability influence analysis
    inner_val_pred_store = {f: {} for f in range(5)}

    for fold in range(5):
        train = np.asarray([r["outer_fold"] != fold for r in base])
        test = ~train
        train_idx = np.flatnonzero(train)
        test_idx = np.flatnonzero(test)
        x = x_by_fold[fold]
        groups = inner_groups(train_idx, seeds)
        inner_fold_of = np.array([
            groups.get(seeds[i], -1) for i in range(n)])

        t0_m, _ = pooled_metrics(base, train_idx, frozen_p[train_idx], "T0")
        t0_full_by_fold[fold] = t0_m

        for w, b in ALL_CANDIDATES:
            inner_pred = np.full(n, np.nan)
            fit_start = time.perf_counter()
            for inner in range(4):
                inner_val = np.asarray([
                    i for i in train_idx if inner_fold_of[i] == inner])
                inner_fit = np.asarray([
                    i for i in train_idx if inner_fold_of[i] != inner])
                model = fit_classifier_costsensitive(x[inner_fit], labels[inner_fit], w)
                inner_pred[inner_val] = probabilities_biased(model, x[inner_val], b)
            training_seconds += time.perf_counter() - fit_start
            inner_val_pred_store[fold][(w, b)] = inner_pred
            m, _ = pooled_metrics(base, train_idx, inner_pred[train_idx], "candidate")
            safe, collapse = is_safe(m, t0_m)
            inner_grid_rows.append({
                "outer_fold": fold, "w_h2": w, "b_h2": b,
                "coverage": m["coverage"], "balanced_accuracy": m["balanced_accuracy"],
                "total_regret": m["total_regret"], "h1_regret": m["h1_regret"],
                "h2_regret": m["h2_regret"], "h1_recall": m["h1_recall"],
                "h2_recall": m["h2_recall"], "h2_to_h1_error": m["h2_to_h1_error"],
                "h1_to_h2_error": m["h1_to_h2_error"],
                "predicted_h2_fraction": m["predicted_h2_fraction"],
                "t0_inner_h2_regret": t0_m["h2_regret"],
                "t0_inner_h2_to_h1_error": t0_m["h2_to_h1_error"],
                "collapse_detected": collapse,
                "safe_constraint_met": safe,
                "outer_test_used_for_selection": False,
                "inner_group_folds": 4})

            # full outer-training fit for held-out outer prediction
            fit_start = time.perf_counter()
            full_model = fit_classifier_costsensitive(x[train_idx], labels[train_idx], w)
            training_seconds += time.perf_counter() - fit_start
            full_predictions[(w, b)][test_idx] = probabilities_biased(
                full_model, x[test_idx], b)

        def select(family_grid, fallback_to_t0):
            scored = []
            for w, b in family_grid:
                row = next(
                    r for r in inner_grid_rows
                    if r["outer_fold"] == fold and r["w_h2"] == w and r["b_h2"] == b)
                scored.append((w, b, row))
            safe_rows = [(w, b, r) for w, b, r in scored if r["safe_constraint_met"]]
            if safe_rows:
                w, b, r = min(
                    safe_rows,
                    key=lambda t: tiebreak_key(t[0], t[1], t[2]))
                return w, b, True
            if fallback_to_t0:
                return None, None, False
            w, b, r = min(scored, key=lambda t: tiebreak_key(t[0], t[1], t[2]))
            return w, b, False

        r2_w, r2_b, r2_safe = select(R2_GRID, fallback_to_t0=False)
        r3_w, r3_b, r3_safe = select(R3_GRID, fallback_to_t0=False)
        r4_w, r4_b, r4_safe = select(ALL_CANDIDATES, fallback_to_t0=True)

        r1_row = next(
            r for r in inner_grid_rows
            if r["outer_fold"] == fold and r["w_h2"] == R1_FIXED[0]
            and r["b_h2"] == R1_FIXED[1])

        for family, w, b, safe in (
                ("R1", R1_FIXED[0], R1_FIXED[1], r1_row["safe_constraint_met"]),
                ("R2", r2_w, r2_b, r2_safe),
                ("R3", r3_w, r3_b, r3_safe),
                ("R4", r4_w, r4_b, r4_safe)):
            selection_rows.append({
                "outer_fold": fold, "method_family": family,
                "selected_w_H2": w if w is not None else "",
                "selected_b_H2": b if b is not None else "",
                "whether_safe_constraint_met": safe,
                "fallback_to_T0": w is None})

    write_csv(OUT / "04_inner_selection_results.csv", inner_grid_rows)
    write_csv(OUT / "05_selected_candidate_by_outer_fold.csv", selection_rows)

    # -----------------------------------------------------------------
    # Assemble final out-of-outer-fold predictions per method.
    # -----------------------------------------------------------------
    predictions = {"T0": frozen_p, "T4": full_predictions[(1.0, 0.0)]}
    predictions["R1"] = full_predictions[R1_FIXED]
    for family in ("R2", "R3", "R4"):
        pred = np.full(n, np.nan)
        for fold in range(5):
            test_idx = np.flatnonzero(np.asarray(
                [r["outer_fold"] == fold for r in base]))
            row = next(
                r for r in selection_rows
                if r["outer_fold"] == fold and r["method_family"] == family)
            if row["fallback_to_T0"]:
                pred[test_idx] = frozen_p[test_idx]
            else:
                w, b = float(row["selected_w_H2"]), float(row["selected_b_H2"])
                pred[test_idx] = full_predictions[(w, b)][test_idx]
        predictions[family] = pred

    METHODS = ("T0", "T4", "R1", "R2", "R3", "R4")
    all_rows = {}
    summaries, delays, errors, fold_rows = [], [], [], []
    for method in METHODS:
        rows = g0.action_rows(base, predictions[method])
        all_rows[method] = rows
        for scope in ("all", "delay_1", "delay_2"):
            m = g0.metrics(rows, method, scope)
            pred = np.asarray([r["predicted"] == H2 for r in rows
                                if scope == "all" or r["delay"] == int(scope[-1])])
            m["predicted_h2_fraction"] = float(np.mean(pred))
            m["predicted_h1_fraction"] = float(1 - np.mean(pred))
            (summaries if scope == "all" else delays).append(m)
        for mechanism in (H1, H2):
            subset = [r for r in rows if r["mechanism"] == mechanism]
            n_h2_to_h1 = sum(
                1 for r in subset if mechanism == H2 and r["predicted"] == H1)
            mean_regret_h2_to_h1 = (
                np.mean([r["regret"] for r in subset
                          if mechanism == H2 and r["predicted"] == H1])
                if mechanism == H2 and n_h2_to_h1 else 0.0)
            errors.append({
                "method": method, "true_mechanism": mechanism,
                "predicted_H1": sum(r["predicted"] == H1 for r in subset),
                "predicted_H2": sum(r["predicted"] == H2 for r in subset),
                "runs": len(subset),
                "n_h2_to_h1_errors": n_h2_to_h1 if mechanism == H2 else "",
                "mean_regret_per_h2_to_h1_error": (
                    mean_regret_h2_to_h1 if mechanism == H2 else ""),
                "fraction_rejected": np.mean(
                    [not r["accepted"] for r in subset])})
        for fold in range(5):
            subset = [r for r in rows if r["outer_fold"] == fold]
            fold_rows.append(g0.metrics(subset, method))
        # per seed

    # -----------------------------------------------------------------
    # Baseline reproduction check (Section IV). Must match the frozen R3
    # Gate 0 archived T0/T4 numbers within tolerance before anything else
    # is trusted.
    # -----------------------------------------------------------------
    summary_lookup = {r["method"]: r for r in summaries}
    ARCHIVED = {
        "T0": {"balanced_accuracy": 0.8131, "coverage": 0.6794,
               "total_regret": 2.8499, "h2_regret": 1.4515},
        "T4": {"balanced_accuracy": 0.8675, "coverage": 0.8788,
               "total_regret": 2.5853, "h2_regret": 3.0223}}
    TOLERANCE = 5e-3
    baseline_rows = []
    baseline_ok = True
    for method, targets in ARCHIVED.items():
        observed = summary_lookup[method]
        for metric_name, archived_value in targets.items():
            observed_value = observed[metric_name]
            diff = abs(observed_value - archived_value)
            passed = diff <= TOLERANCE
            baseline_ok = baseline_ok and passed
            baseline_rows.append({
                "method": method, "metric": metric_name,
                "archived_value": archived_value,
                "observed_value": observed_value,
                "abs_difference": diff, "tolerance": TOLERANCE,
                "reproduced": passed})
    write_csv(OUT / "02_baseline_reproduction.csv", baseline_rows)
    if not baseline_ok:
        (OUT / "17_gate0b_decision.txt").write_text(
            "G0B_BASELINE_REPRODUCTION_FAILED\n")
        print("G0B_BASELINE_REPRODUCTION_FAILED")
        return "G0B_BASELINE_REPRODUCTION_FAILED"

    write_csv(OUT / "06_outer_fold_results.csv", fold_rows)
    write_csv(OUT / "07_method_summary.csv", summaries)
    write_csv(OUT / "08_delay_specific_results.csv", delays)
    write_csv(OUT / "09_mechanism_confusion_and_regret.csv", errors)

    seed_rows = []
    for method in METHODS:
        for seed in np.unique(seeds):
            subset = [r for r in all_rows[method] if r["seed"] == seed]
            m = g0.metrics(subset, method)
            seed_rows.append({"method": method, "seed": int(seed), **{
                k: v for k, v in m.items() if k not in ("method",)}})
    write_csv(OUT / "09b_per_seed_results.csv", seed_rows)

    # -----------------------------------------------------------------
    # Coverage-risk frontier: every pre-registered (w_h2, b_h2) candidate,
    # applied uniformly across all outer folds (not adaptively selected),
    # plus T0 and plain T4.
    # -----------------------------------------------------------------
    frontier_rows = []
    for label, pred in (("T0", frozen_p), ("T4", full_predictions[(1.0, 0.0)])):
        m, _ = pooled_metrics(base, np.arange(n), pred, label)
        frontier_rows.append({
            "candidate": label, "w_h2": "" if label == "T0" else 1.0,
            "b_h2": "" if label == "T0" else 0.0, **{
                k: v for k, v in m.items() if k != "method"}})
    for w, b in ALL_CANDIDATES:
        m, _ = pooled_metrics(base, np.arange(n), full_predictions[(w, b)], "candidate")
        frontier_rows.append({
            "candidate": f"w{w}_b{b}", "w_h2": w, "b_h2": b, **{
                k: v for k, v in m.items() if k != "method"}})
    nearest_rows = []
    for target in COVERAGE_TARGETS:
        best = min(frontier_rows, key=lambda r: abs(r["coverage"] - target))
        nearest_rows.append({
            "coverage_target": target, "nearest_candidate": best["candidate"],
            "coverage": best["coverage"], "total_regret": best["total_regret"],
            "h2_regret": best["h2_regret"], "h2_to_h1_error": best["h2_to_h1_error"],
            "balanced_accuracy": best["balanced_accuracy"]})
    write_csv(OUT / "10_coverage_risk_curves.csv", frontier_rows)
    write_csv(OUT / "10b_coverage_risk_nearest_targets.csv", nearest_rows)

    # -----------------------------------------------------------------
    # H2 error diagnostics: which convolution responses are largest on
    # H2-true rows that R0(T4, w=1,b=0) mispredicts as H1, versus overall.
    # -----------------------------------------------------------------
    conv_lengths_used = sorted(set(t4_window.values()))
    fn_diag_rows = []
    t4_rows = all_rows["T4"]
    fn_h2_idx = np.asarray([
        i for i, r in enumerate(t4_rows)
        if r["mechanism"] == H2 and r["predicted"] == H1])
    for length in conv_lengths_used:
        conv = conv_by_length[length]
        n_signals = conv.shape[1] // (len(g0.KERNELS) * 4)
        for k_index, k_name in enumerate(g0.KERNELS):
            cols = np.asarray([
                s * len(g0.KERNELS) * 4 + k_index * 4 + stat
                for s in range(n_signals) for stat in range(4)])
            fn_mean = (
                np.mean(np.abs(conv[fn_h2_idx][:, cols]))
                if len(fn_h2_idx) else float("nan"))
            all_mean = np.mean(np.abs(conv[:, cols]))
            fn_diag_rows.append({
                "window": length, "kernel": k_name,
                "n_h2_to_h1_errors": len(fn_h2_idx),
                "mean_abs_response_h2_to_h1_errors": fn_mean,
                "mean_abs_response_overall": all_mean,
                "ratio": (fn_mean / all_mean) if all_mean else float("nan")})
    write_csv(OUT / "11_h2_error_diagnostics.csv", fn_diag_rows)

    # -----------------------------------------------------------------
    # Coefficient change analysis: R0 (w=1,b=0) vs R1 fixed (w=4,b=0),
    # refit once on fold 0's full outer-training data, grouped by kernel.
    # -----------------------------------------------------------------
    coef_rows = []
    fold0_train = np.flatnonzero(np.asarray([r["outer_fold"] != 0 for r in base]))
    x0 = x_by_fold[0]
    static_len = static.shape[1]
    model_r0 = fit_classifier_costsensitive(x0[fold0_train], labels[fold0_train], 1.0)
    model_r1 = fit_classifier_costsensitive(x0[fold0_train], labels[fold0_train], R1_FIXED[0])
    coef_r0, coef_r1 = model_r0[2], model_r1[2]
    static_coef_r0 = coef_r0[1:1 + static_len]
    static_coef_r1 = coef_r1[1:1 + static_len]
    coef_rows.append({
        "group": "static_S0", "mean_abs_coef_R0": np.mean(np.abs(static_coef_r0)),
        "mean_abs_coef_R1_w4": np.mean(np.abs(static_coef_r1)),
        "mean_abs_change": np.mean(np.abs(static_coef_r1 - static_coef_r0))})
    conv_coef_r0 = coef_r0[1 + static_len:]
    conv_coef_r1 = coef_r1[1 + static_len:]
    length0 = t4_window[0]
    n_signals = conv_by_length[length0].shape[1] // (len(g0.KERNELS) * 4)
    for k_index, k_name in enumerate(g0.KERNELS):
        cols = np.asarray([
            s * len(g0.KERNELS) * 4 + k_index * 4 + stat
            for s in range(n_signals) for stat in range(4)])
        coef_rows.append({
            "group": f"conv_{k_name}",
            "mean_abs_coef_R0": np.mean(np.abs(conv_coef_r0[cols])),
            "mean_abs_coef_R1_w4": np.mean(np.abs(conv_coef_r1[cols])),
            "mean_abs_change": np.mean(np.abs(conv_coef_r1[cols] - conv_coef_r0[cols]))})
    write_csv(OUT / "12_coefficient_change_analysis.csv", coef_rows)

    # -----------------------------------------------------------------
    # Runtime and complexity
    # -----------------------------------------------------------------
    elapsed = time.perf_counter() - start
    runtime_rows = [{
        "method": "T4_feature_extractor",
        "feature_extraction_seconds_per_run": extraction_seconds / n,
        "training_time_per_fold_seconds": training_seconds / 5,
        "inference_time_per_run_seconds": elapsed / n / len(METHODS),
        "inference_time_per_tick_seconds": elapsed / n / len(METHODS) / 500,
        "parameter_count_deployed": 1 + static.shape[1] + conv_by_length[
            t4_window[0]].shape[1],
        "memory_footprint_bytes_estimate": 8 * (
            1 + static.shape[1] + conv_by_length[t4_window[0]].shape[1]) * 2,
        "communication_bytes_per_tick": COMMUNICATION,
        "candidates_enumerated_during_selection": len(ALL_CANDIDATES),
        "deployed_model_components": (
            "T4_feature_extractor+one_logistic_classifier"
            "+optional_one_fixed_logit_bias")}]
    write_csv(OUT / "13_runtime_and_complexity.csv", runtime_rows)

    # -----------------------------------------------------------------
    # Bootstrap over seeds (>=2000 replicates)
    # -----------------------------------------------------------------
    rng = np.random.default_rng(191307)
    unique_seeds = np.unique(seeds)
    comparisons = (
        ("R4", "T4", "total_regret"), ("R4", "T4", "h2_regret"),
        ("R4", "T4", "h2_to_h1_error"), ("R4", "T4", "balanced_accuracy"),
        ("R4", "T4", "coverage"),
        ("R4", "T0", "total_regret"), ("R4", "T0", "h2_regret"),
        ("R4", "T0", "balanced_accuracy"))
    bootstrap = []
    for replicate in range(BOOTSTRAPS):
        draw = rng.choice(unique_seeds, len(unique_seeds), replace=True)
        indices = np.concatenate([np.flatnonzero(seeds == s) for s in draw])
        for method, baseline, metric_name in comparisons:
            left = g0.metrics([all_rows[method][i] for i in indices], method)
            right = g0.metrics([all_rows[baseline][i] for i in indices], baseline)
            bootstrap.append({
                "replicate": replicate, "comparison": f"{method}-{baseline}",
                "metric": metric_name,
                "difference": left[metric_name] - right[metric_name],
                "resampling_unit": "seed"})
    boot = pd.DataFrame(bootstrap)
    boot_summary = []
    for (comparison, metric_name), group in boot.groupby(["comparison", "metric"]):
        boot_summary.append({
            "comparison": comparison, "metric": metric_name,
            "replicates": len(group), "mean": group.difference.mean(),
            "ci95_low": group.difference.quantile(.025),
            "ci95_high": group.difference.quantile(.975),
            "resampling_unit": "seed"})
    write_csv(OUT / "14_bootstrap_results.csv", boot_summary)

    # -----------------------------------------------------------------
    # Leave-one-seed-out influence: final-metric influence (all methods)
    # plus selection-stability influence for R4 (post-hoc row filtering
    # of already-computed inner predictions; no refitting).
    # -----------------------------------------------------------------
    influence = []
    for seed in unique_seeds:
        keep = np.flatnonzero(seeds != seed)
        for method in ("T4", "R1", "R2", "R3", "R4"):
            m = g0.metrics([all_rows[method][i] for i in keep], method)
            influence.append({
                "excluded_seed": int(seed), "method": method,
                "influence_type": "final_metric",
                "balanced_accuracy": m["balanced_accuracy"],
                "coverage": m["coverage"], "total_regret": m["total_regret"],
                "h2_regret": m["h2_regret"], "selected_w_H2": "",
                "selected_b_H2": ""})

    selection_influence = []
    for fold in range(5):
        train = np.asarray([r["outer_fold"] != fold for r in base])
        train_idx = np.flatnonzero(train)
        fold_seeds = np.unique(seeds[train_idx])
        t0_m = t0_full_by_fold[fold]
        actual = next(
            r for r in selection_rows
            if r["outer_fold"] == fold and r["method_family"] == "R4")
        for seed in fold_seeds:
            keep = np.asarray([i for i in train_idx if seeds[i] != seed])
            scored = []
            for w, b in ALL_CANDIDATES:
                pred = inner_val_pred_store[fold][(w, b)]
                m, _ = pooled_metrics(base, keep, pred[keep], "candidate")
                safe, _ = is_safe(m, t0_m)
                scored.append((w, b, m, safe))
            safe_rows = [(w, b, m) for w, b, m, safe in scored if safe]
            if safe_rows:
                w, b, m = min(safe_rows, key=lambda t: tiebreak_key(t[0], t[1], t[2]))
                fallback = False
            else:
                w, b, fallback = None, None, True
            changed = (
                str(w) != str(actual["selected_w_H2"])
                or str(b) != str(actual["selected_b_H2"]))
            selection_influence.append({
                "excluded_seed": int(seed), "outer_fold": fold,
                "loo_selected_w_H2": w if w is not None else "",
                "loo_selected_b_H2": b if b is not None else "",
                "loo_fallback_to_T0": fallback,
                "actual_selected_w_H2": actual["selected_w_H2"],
                "actual_selected_b_H2": actual["selected_b_H2"],
                "selection_changed": changed})
    selection_influence_df = pd.DataFrame(selection_influence)
    influence_summary = [{
        "metric": "R4_selected_w_H2_influence",
        "fraction_of_seed_exclusions_changing_selection": float(
            selection_influence_df.selection_changed.mean())}]
    write_csv(OUT / "15_seed_influence.csv", influence)
    write_csv(OUT / "15b_selection_stability_influence.csv", selection_influence)
    write_csv(OUT / "15c_selection_influence_summary.csv", influence_summary)

    # -----------------------------------------------------------------
    # Gate decision
    # -----------------------------------------------------------------
    summary = {r["method"]: r for r in summaries}
    delay_lookup = {
        (r["method"], r["scope"]): r for r in delays}
    t0, t4, r4 = summary["T0"], summary["T4"], summary["R4"]

    def delay_ok(method, metric, compare_to, cmp):
        return all(
            cmp(delay_lookup[(method, f"delay_{d}")][metric],
                delay_lookup[(compare_to, f"delay_{d}")][metric])
            for d in (1, 2))

    def full_survives(method):
        m = summary[method]
        boot_row = next(
            (r for r in boot_summary
             if r["comparison"] == f"{method}-T4" and r["metric"] == "total_regret"),
            None)
        ci_ok = boot_row is None or boot_row["ci95_high"] < 0
        seed_flip = False
        if method == "R4":
            diffs = []
            for seed in unique_seeds:
                keep = np.flatnonzero(seeds != seed)
                left = g0.metrics([all_rows["R4"][i] for i in keep], "R4")
                right = g0.metrics([all_rows["T4"][i] for i in keep], "T4")
                diffs.append(left["total_regret"] - right["total_regret"])
            base_diff = m["total_regret"] - t4["total_regret"]
            seed_flip = any(
                (d > 0) != (base_diff > 0) for d in diffs) if base_diff != 0 else False
        return (
            m["coverage"] >= .80 and m["balanced_accuracy"] >= .85
            and m["total_regret"] <= .8 * t0["total_regret"]
            and m["h2_regret"] <= t0["h2_regret"]
            and m["h1_recall"] >= .65
            and delay_ok(method, "total_regret", "T4", lambda a, b: a < b)
            and delay_ok(method, "h2_regret", "T0", lambda a, b: a <= b)
            and ci_ok and not seed_flip)

    r1_survives = full_survives("R1")
    r2_survives = full_survives("R2")
    r3_survives = full_survives("R3")
    r4_survives = full_survives("R4")

    fallback_folds = sum(
        1 for r in selection_rows
        if r["method_family"] == "R4" and r["fallback_to_T0"])
    no_safe_candidate_frequent = fallback_folds >= 3

    if no_safe_candidate_frequent and not r4_survives:
        decision = "G0B_NO_SAFE_CONVOLUTION_CANDIDATE"
    elif r4_survives:
        decision = "G0B_RISK_CONSTRAINED_CONVOLUTION_SURVIVES"
    elif r2_survives and not r1_survives and not r3_survives:
        decision = "G0B_ASYMMETRIC_DECISION_ONLY"
    elif r1_survives and not r2_survives:
        decision = "G0B_COST_WEIGHTING_ONLY"
    else:
        decision = "G0B_CONVOLUTION_CANNOT_BE_MADE_SAFE"

    # -----------------------------------------------------------------
    # Narrative outputs
    # -----------------------------------------------------------------
    (OUT / "00_scope_and_nonclaims.md").write_text(f"""# B1 R3 Gate 0B scope and nonclaims

Development-only candidate audit using seeds 3000--3399 (identical to R3
Gate 0). Sealed seeds 2000--2029 and R0 final seeds 120--149 are never
loaded (asserted at data-load time). No simulation is rerun. F2 received
history, the 5.10 B/tick communication budget, the window definitions, the
five fixed convolution kernels, the static feature set, the convolution
feature set, the outer GroupKFold-by-seed partition, the historical
H1/H2 error-cost matrix, and the downstream expected-cost/reject decision
rule are all imported unchanged from `run_g0_temporal_audit.py` and
`run_g1a_query_replay.py` (R2's frozen module) and are never modified here.

The only degrees of freedom exercised in this audit are, per fold, chosen
using inner (within outer-training) GroupKFold splits only:

1. an H2 sample-weight multiplier `w_H2` applied to the T4 classifier's
   training loss, drawn from the pre-registered grid {{1, 2, 4, 8, 16}};
2. a fixed additive bias `b_H2` on the H2 logit, applied after training
   and before the frozen decision rule, drawn from the pre-registered grid
   {{0, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00}};
3. combinations of the two from a pre-registered 3x4 grid.

Method-family selection rules (documented here since the task prompt left
R2/R3 selection unspecified beyond "select from pre-registered candidates
within inner folds"):

- **R1** reports the fixed `w_H2=4` setting only (the historical H2:H1
  cost ratio is ~3.9), with no inner search, per the "natural main
  setting" instruction. Its safety status is still evaluated per fold.
- **R2** selects `b_H2` from its 7-candidate grid (`w_H2` fixed at 1) via
  inner-fold-constrained, then total-regret-minimizing, tie-break
  selection; if no candidate in the family satisfies the coverage/H2-risk
  safety constraint, the least-unsafe (lowest total regret) candidate in
  the family is still reported, flagged `whether_safe_constraint_met =
  False`.
- **R3** selects `(w_H2, b_H2)` from its 12-candidate grid the same way.
- **R4** selects from the union of all 23 unique R1/R2/R3 grid points
  (the candidate pool referenced in Section VI); if none of the 23
  satisfies the safety constraint, R4 falls back to T0 (no convolution)
  for that outer fold and records `NO_SAFE_CONVOLUTION_CANDIDATE`-style
  provenance in `05_selected_candidate_by_outer_fold.csv`.

No hand feature, convolution kernel, dynamic model, or trainable deep
model is added. B1 remains frozen; this audit does not authorize the next
Gate and does not declare B1 revived.
""")

    report_lines = [
        "# B1 R3 Gate 0B risk-constrained temporal convolution audit",
        "",
        "## Formal decision",
        "",
        f"`{decision}`",
        "",
        "| Method | Balanced accuracy | Coverage | Total regret | "
        "H2 regret | H1 recall | Pred. H2 frac. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in METHODS:
        s = summary[m]
        report_lines.append(
            f"| {m} | {s['balanced_accuracy']:.4f} | {s['coverage']:.4f} | "
            f"{s['total_regret']:.4f} | {s['h2_regret']:.4f} | "
            f"{s['h1_recall']:.4f} | {s['predicted_h2_fraction']:.4f} |")
    report_lines += [
        "",
        "`T0` = B0 static baseline; `T4` = B1 plain fixed-convolution baseline "
        "(ordinary training, `w_H2=1`, `b_H2=0`).",
        "",
        "## Hard gate values",
        "",
        f"- T0 total regret (gate reference): {t0['total_regret']:.4f}; "
        f"20% reduction target: {.8 * t0['total_regret']:.4f}",
        f"- T0 H2 regret (gate ceiling): {t0['h2_regret']:.4f}",
        f"- R4 total regret: {r4['total_regret']:.4f}, "
        f"H2 regret: {r4['h2_regret']:.4f}, "
        f"coverage: {r4['coverage']:.4f}, "
        f"balanced accuracy: {r4['balanced_accuracy']:.4f}, "
        f"H1 recall: {r4['h1_recall']:.4f}",
        f"- R4 outer folds falling back to T0 (no safe candidate): "
        f"{fallback_folds}/5",
        "",
        "## Per-family survival against the full R4 hard criteria",
        "",
        f"- R1 (fixed w_H2=4): {r1_survives}",
        f"- R2 (logit bias only): {r2_survives}",
        f"- R3 (weight + bias): {r3_survives}",
        f"- R4 (constrained selector): {r4_survives}",
        "",
        "## Diagnostics (Section XV)",
        "",
    ]

    fn_summary = pd.DataFrame(fn_diag_rows)
    if len(fn_h2_idx):
        top_kernel = fn_summary.loc[fn_summary.ratio.idxmax()]
        q1 = (
            f"1. Plain-T4 H2-to-H1 errors ({len(fn_h2_idx)} runs) show the "
            f"largest relative response concentration on kernel "
            f"`{top_kernel['kernel']}` at window {int(top_kernel['window'])} "
            f"(ratio to overall mean |response| = {top_kernel['ratio']:.3f}); "
            "full per-kernel/window ratios are in `11_h2_error_diagnostics.csv`."
        )
    else:
        q1 = "1. Plain-T4 has zero H2-to-H1 errors in this run; no concentration to report."
    report_lines.append(q1)

    coef_df = pd.DataFrame(coef_rows)
    conv_change = coef_df[coef_df.group.str.startswith("conv_")]
    static_change = coef_df[coef_df.group == "static_S0"].iloc[0]
    report_lines.append(
        f"2. Moving from `w_H2=1` to the fixed `w_H2=4` R1 setting (fold 0 "
        f"refit), mean |coefficient| across convolution groups changes by "
        f"{conv_change.mean_abs_change.mean():.4f} on average versus "
        f"{static_change.mean_abs_change:.4f} for static S0 features; "
        "per-kernel detail is in `12_coefficient_change_analysis.csv`.")

    h1_recall_delta = r4["h1_recall"] - t4["h1_recall"]
    report_lines.append(
        f"3. R4 H1 recall is {r4['h1_recall']:.4f} versus plain T4's "
        f"{t4['h1_recall']:.4f} (delta {h1_recall_delta:+.4f}); H1 regret "
        f"moves from {t4['h1_regret']:.4f} to {r4['h1_regret']:.4f}. "
        "A negative delta larger than noise indicates part of the H2-risk "
        "reduction is paid for with H1 cases (more H1 rejects or H1-wrong "
        "actions); see `06_outer_fold_results.csv` and "
        "`09_mechanism_confusion_and_regret.csv` for the fold/mechanism split.")

    w_selected = [
        r["selected_w_H2"] for r in selection_rows if r["method_family"] == "R4"
        and r["selected_w_H2"] != ""]
    w_mode_is_4 = (
        len(w_selected) > 0
        and sum(1 for w in w_selected if float(w) == 4.0) >= 3)
    report_lines.append(
        f"4. R4-selected `w_H2` across the 5 outer folds: {w_selected}; "
        f"{'stable at 4 in a majority of folds' if w_mode_is_4 else 'not stable at 4 across folds'}.")

    b_selected = [
        float(r["selected_b_H2"]) for r in selection_rows
        if r["method_family"] == "R2" and r["selected_b_H2"] != ""]
    w_r3 = [
        float(r["selected_w_H2"]) for r in selection_rows
        if r["method_family"] == "R3" and r["selected_w_H2"] != ""]
    b_std = float(np.std(b_selected)) if b_selected else float("nan")
    w_std = float(np.std(w_r3)) if w_r3 else float("nan")
    report_lines.append(
        f"5. Across outer folds, R2's selected `b_H2` has std "
        f"{b_std:.4f} versus R3's selected `w_H2` std {w_std:.4f}; "
        f"{'logit bias selection is more stable' if b_std <= w_std else 'weighted-training selection is more stable'}.")

    # matched-coverage comparison: reject the most-uncertain plain-T4 cases
    # down to R4's coverage and recompute
    t4_p = predictions["T4"]
    r4_coverage = r4["coverage"]
    n_accept_target = int(round(r4_coverage * n))
    certainty = np.abs(t4_p - .5)
    order = np.argsort(-certainty)
    keep_idx = set(order[:n_accept_target].tolist())
    h1_cost = frozen.FROZEN["cost_sensitive_reject"]["h1_wrong_regret_estimate"]
    h2_cost = frozen.FROZEN["cost_sensitive_reject"]["h2_wrong_regret_estimate"]
    matched_rows = []
    for i in range(n):
        p = t4_p[i]
        row = base[i]
        if i in keep_idx:
            costs = {"CAPACITY_ACTION": p * h2_cost, "SYNC_ACTION": (1 - p) * h1_cost}
            action = min(costs, key=costs.get)
            wrong = action != row["oracle_action"]
            regret = row["potential_regret"] if wrong else 0.0
        else:
            action, wrong, regret = "REJECT", False, REJECT_COST
        matched_rows.append({**row, "action": action, "accepted": action != "REJECT",
                              "wrong": wrong, "regret": regret})
    matched_total_regret = float(np.mean([r["regret"] for r in matched_rows]))
    matched_h2_regret = float(np.mean([
        r["regret"] for r in matched_rows if r["mechanism"] == H2]))
    matched_coverage = float(np.mean([r["accepted"] for r in matched_rows]))
    report_lines.append(
        f"6/7. Matching plain-T4 to R4's coverage ({r4_coverage:.4f}) by "
        f"rejecting T4's most-uncertain cases (not adaptively re-decided) "
        f"gives total regret {matched_total_regret:.4f} and H2 regret "
        f"{matched_h2_regret:.4f} at coverage {matched_coverage:.4f}, versus "
        f"R4's actual total regret {r4['total_regret']:.4f} and H2 regret "
        f"{r4['h2_regret']:.4f}. "
        + ("R4 still improves on matched-coverage T4, so the gain is not "
           "purely a coverage effect."
           if r4["total_regret"] < matched_total_regret
           else "R4 does not improve on matched-coverage T4, so part or all "
                "of R4's apparent gain is attributable to lower coverage "
                "rather than better risk-weighted classification."))
    report_lines.append("")
    report_lines.append(
        "The result does not revive B1 and does not authorize the next Gate.")
    report = "\n".join(report_lines) + "\n"
    (OUT / "16_gate0b_report.md").write_text(report)
    (OUT / "17_gate0b_decision.txt").write_text(decision + "\n")
    (CONCLUSION / "G0B_FORMAL_DECISION.md").write_text(report)
    print(report)
    return decision


if __name__ == "__main__":
    main()
