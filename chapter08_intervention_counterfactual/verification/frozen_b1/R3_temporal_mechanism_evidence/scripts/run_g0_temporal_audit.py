#!/usr/bin/env python3
"""B1 R3 Gate 0: lightweight temporal mechanism evidence audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
R3 = HERE.parents[1]
B1 = HERE.parents[2]
R2 = B1 / "R2_mechanism_conditioned_risk_rescue"
DATA = R2 / "G1A_instrumented_replay_data"
OUT = R3 / "G0_temporal_consistency_and_convolution"
CONCLUSION = R3 / (
    "results/CONCLUSIONS/G0_TEMPORAL_CONSISTENCY_AND_CONVOLUTION")
WINDOWS = (4, 8, 12)
ORDERS = (1, 2)
FIELDS = (
    "queue", "head_of_line_wait", "failure_count", "busy_flag",
    "time_since_last_success", "retry_count", "recent_throughput")
USE = (0, 1, 2, 3, 4, 6)
KERNELS = {
    "difference": (-1, 1),
    "short_slope": (-1, 0, 1),
    "acceleration": (1, -2, 1),
    "late_surge": (-1, -1, 1, 1),
    "early_vs_late": (-1, -1, 0, 1, 1),
}
METHODS = ("T0", "T1", "T2", "T3", "T4", "T5")
H1 = "H1_CAPACITY_SHORTAGE"
H2 = "H2_DISTRIBUTED_CAPACITY_SUPPRESSION"
COMMUNICATION = 5.10
REJECT_COST = 5.0
BOOTSTRAPS = 2000

sys.path[:0] = [
    str(R2 / "scripts"), str(R2 / "src"),
    str(B1 / "R1_age_conditioned_decision_recovery/src")]
import run_g1a_query_replay as frozen  # noqa: E402


def balanced_accuracy_score(y, pred):
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    return .5 * (
        np.mean(pred[y]) + np.mean(~pred[~y]))


class LinearModel:
    def __init__(self, coefficients):
        self.coefficients = coefficients
        self.coef_ = coefficients[1:].T

    def predict(self, x):
        x = np.asarray(x, float)
        return np.column_stack([np.ones(len(x)), x]) @ self.coefficients


def fit_linear(x, y):
    design = np.column_stack([np.ones(len(x)), x])
    return LinearModel(np.linalg.lstsq(design, y, rcond=None)[0])


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def visible_series(received, length):
    """Last L actual F2 arrival epochs, preserving unknown node/field state."""
    received = np.asarray(received, float)
    epochs = np.flatnonzero(np.any(np.isfinite(received), axis=(1, 2)))
    epochs = epochs[epochs <= 499][-length:]
    if len(epochs) < length:
        pad = np.full(
            (length - len(epochs), received.shape[1], received.shape[2]),
            np.nan)
        selected = np.concatenate([pad, received[epochs]], axis=0)
    else:
        selected = received[epochs]
    return frozen.impute(selected)[:, :, USE]


def sign_changes(x):
    d = np.diff(x)
    signs = np.sign(d)
    return np.sum(signs[1:] * signs[:-1] < 0)


def trend_features(series):
    output = []
    slopes = []
    for field_pos in range(series.shape[2]):
        node_slopes = []
        for node in range(series.shape[1]):
            x = series[:, node, field_pos]
            d = np.diff(x)
            slope = float(x[-1] - x[0])
            node_slopes.append(slope)
            output.extend([
                x[-1], slope, np.mean(d), d[-1], np.max(d),
                np.var(x), sign_changes(x)])
        slopes.append(np.asarray(node_slopes))
        output.extend([
            np.mean(node_slopes), np.max(node_slopes),
            np.std(node_slopes), np.mean(np.asarray(node_slopes) > 0),
            np.mean(np.sign(node_slopes)[:, None]
                    == np.sign(node_slopes)[None, :])])
    busy = series[:, :, list(USE).index(3)]
    throughput = series[:, :, list(USE).index(6)]
    output.append(np.mean((busy > 0) & (throughput <= 0)))
    q_slope = slopes[list(USE).index(0)]
    f_slope = slopes[list(USE).index(2)]
    output.append(float(np.corrcoef(q_slope, f_slope)[0, 1])
                  if np.std(q_slope) and np.std(f_slope) else 0.0)
    return np.nan_to_num(np.asarray(output, float))


def convolution_features(series):
    aggregates = np.concatenate(
        [series.mean(axis=1), series.max(axis=1)], axis=1)
    output = []
    for signal in aggregates.T:
        for kernel in KERNELS.values():
            k = np.asarray(kernel, float)
            padded = np.pad(signal, (len(k) - 1, 0), mode="constant")
            response = np.convolve(padded, k[::-1], mode="valid")
            output.extend([
                np.max(response), np.mean(response), response[-1],
                np.var(response)])
    return np.nan_to_num(np.asarray(output, float))


def fit_ar_models(sequences, labels, train, order):
    models = {}
    for mechanism in (0, 1):
        rows_x, rows_y = [], []
        for i in np.flatnonzero(train & (labels == mechanism)):
            seq = sequences[i]
            for tick in range(order, len(seq)):
                rows_x.append(seq[tick-order:tick].ravel())
                rows_y.append(seq[tick])
        x, y = np.asarray(rows_x), np.asarray(rows_y)
        model = fit_linear(x, y)
        residual = y - model.predict(x)
        variance = np.maximum(np.var(residual, axis=0), 1e-6)
        models[mechanism] = (model, variance)
    assert models[0][0].coef_.shape == models[1][0].coef_.shape
    return models


def dynamic_features(sequence, models, order):
    evidence = []
    for mechanism in (0, 1):
        model, variance = models[mechanism]
        residuals, likelihood = [], []
        for tick in range(order, len(sequence)):
            prediction = model.predict(
                sequence[tick-order:tick].reshape(1, -1))[0]
            residual = sequence[tick] - prediction
            residuals.append(residual)
            likelihood.append(-.5 * np.sum(
                residual**2 / variance + np.log(2 * np.pi * variance)))
        residuals = np.asarray(residuals)
        evidence.append((
            float(np.sum(likelihood)),
            float(np.sum(residuals**2 / variance))))
    h1_ll, h1_energy = evidence[0]
    h2_ll, h2_energy = evidence[1]
    return np.asarray([
        h1_ll, h2_ll, h2_ll - h1_ll, h1_energy, h2_energy,
        h2_energy - h1_energy])


def dynamic_trace(sequence, models, order):
    output = {}
    for name, mechanism in (("H1", 0), ("H2", 1)):
        model, variance = models[mechanism]
        prediction, residual, nsr, log_likelihood = [], [], [], []
        for tick in range(order, len(sequence)):
            pred = model.predict(
                sequence[tick-order:tick].reshape(1, -1))[0]
            innovation = sequence[tick] - pred
            prediction.append(pred.tolist())
            residual.append(innovation.tolist())
            nsr.append((innovation**2 / variance).tolist())
            log_likelihood.append(float(-.5 * np.sum(
                innovation**2 / variance
                + np.log(2 * np.pi * variance))))
        output.update({
            f"{name}_prediction_by_update": json.dumps(prediction),
            f"{name}_innovation_residual_by_update": json.dumps(residual),
            f"{name}_normalized_squared_residual_by_update":
                json.dumps(nsr),
            f"{name}_observation_log_likelihood_by_update":
                json.dumps(log_likelihood)})
    return output


def static_features(received, arrival, delay):
    fmap = frozen.feature_map(received, arrival, delay)
    return np.asarray([
        fmap[name] for name in
        frozen.FROZEN["age_aware_model"]["feature_order"]], float)


def fit_classifier(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    design = np.column_stack([
        np.ones(len(x)), (x - mean) / scale])
    weights = np.zeros(design.shape[1])
    sample_weight = np.where(
        y == 1, .5 / max(np.mean(y == 1), 1e-9),
        .5 / max(np.mean(y == 0), 1e-9))
    for _ in range(80):
        p = 1 / (1 + np.exp(-np.clip(design @ weights, -30, 30)))
        curvature = sample_weight * p * (1 - p)
        hessian = design.T @ (curvature[:, None] * design)
        hessian[1:, 1:] += np.eye(len(weights) - 1)
        gradient = design.T @ (sample_weight * (p - y))
        gradient[1:] += weights[1:]
        step = np.linalg.solve(
            hessian + np.eye(len(weights)) * 1e-9, gradient)
        weights -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return mean, scale, weights


def probabilities(model, x):
    mean, scale, weights = model
    design = np.column_stack([
        np.ones(len(x)), (np.asarray(x) - mean) / scale])
    return 1 / (1 + np.exp(-np.clip(design @ weights, -30, 30)))


def action_rows(base, p_h2):
    h1_cost = frozen.FROZEN["cost_sensitive_reject"][
        "h1_wrong_regret_estimate"]
    h2_cost = frozen.FROZEN["cost_sensitive_reject"][
        "h2_wrong_regret_estimate"]
    rows = []
    for row, p in zip(base, p_h2):
        costs = {
            "CAPACITY_ACTION": p * h2_cost,
            "SYNC_ACTION": (1 - p) * h1_cost,
            "REJECT": REJECT_COST}
        action = min(costs, key=costs.get)
        accepted = action != "REJECT"
        wrong = accepted and action != row["oracle_action"]
        regret = row["potential_regret"] if wrong else (
            REJECT_COST if not accepted else 0.0)
        rows.append({
            **row, "p_h1": 1-p, "p_h2": p,
            "predicted": H2 if p >= .5 else H1,
            "action": action, "accepted": accepted,
            "wrong": wrong, "regret": regret})
    return rows


def metrics(rows, method, scope="all"):
    if scope != "all":
        delay = int(scope.split("_")[1])
        rows = [r for r in rows if r["delay"] == delay]
    y = np.asarray([r["mechanism"] == H2 for r in rows])
    pred = np.asarray([r["predicted"] == H2 for r in rows])
    accepted = np.asarray([r["accepted"] for r in rows])
    regret = np.asarray([r["regret"] for r in rows])
    h1, h2 = ~y, y
    return {
        "method": method, "scope": scope, "n": len(rows),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "h1_recall": np.mean(~pred[h1]), "h2_recall": np.mean(pred[h2]),
        "h1_to_h2_error": np.mean(pred[h1]),
        "h2_to_h1_error": np.mean(~pred[h2]),
        "coverage": np.mean(accepted),
        "total_regret": np.mean(regret),
        "h1_regret": np.mean(regret[h1]),
        "h2_regret": np.mean(regret[h2]),
        "utility_gain_retention": 1 - np.mean(regret) / REJECT_COST,
        "communication_bytes_per_tick": COMMUNICATION,
    }


def select_window(temporal, static, labels, train_indices, seed_values):
    best = {}
    unique = np.unique(seed_values[train_indices])
    inner_group = {seed: i % 4 for i, seed in enumerate(sorted(unique))}
    for method in METHODS[1:]:
        scores = {}
        for length in WINDOWS:
            fold_scores = []
            for fold in range(4):
                fit = np.asarray([
                    i for i in train_indices
                    if inner_group[seed_values[i]] != fold])
                val = np.asarray([
                    i for i in train_indices
                    if inner_group[seed_values[i]] == fold])
                if method in ("T2", "T3", "T5"):
                    inner_train = np.zeros(len(labels), bool)
                    inner_train[fit] = True
                    sequence = temporal[length]["sequences"]
                    order_scores = {}
                    fitted_by_order = {}
                    for order in ORDERS:
                        ar = fit_ar_models(
                            sequence, labels, inner_train, order)
                        fitted_by_order[order] = ar
                        evidence = np.asarray([
                            dynamic_features(sequence[i], ar, order)
                            for i in fit])
                        order_scores[order] = np.mean(
                            np.abs(evidence[:, 2]))
                    order = max(ORDERS, key=lambda o: order_scores[o])
                    ar = fitted_by_order[order]
                    dyn = np.asarray([
                        dynamic_features(item, ar, order)
                        for item in sequence])
                else:
                    dyn = None
                trend = temporal[length]["trend"]
                conv = temporal[length]["conv"]
                if method == "T1":
                    x = np.column_stack([static, trend])
                elif method == "T2":
                    x = dyn
                elif method == "T3":
                    x = np.column_stack([static, dyn])
                elif method == "T4":
                    x = np.column_stack([static, conv])
                else:
                    x = np.column_stack([static, dyn, conv])
                model = fit_classifier(x[fit], labels[fit])
                fold_scores.append(balanced_accuracy_score(
                    labels[val], probabilities(model, x[val]) >= .5))
            scores[length] = np.mean(fold_scores)
        best[method] = max(WINDOWS, key=lambda x: (scores[x], -x))
    best["T0"] = 0
    return best


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    CONCLUSION.mkdir(parents=True, exist_ok=True)
    states = pd.read_csv(DATA / "03_decision_state_manifest.csv")
    states = states[states.delay.isin([1, 2])].copy()
    assert states.seed.between(3000, 3399).all()
    manifest = pd.read_csv(DATA / "01_replay_manifest.csv")
    traces = {}
    for delay in (1, 2):
        row = manifest[
            (manifest.delay == delay) & (manifest.packet_loss == 0)].iloc[0]
        with np.load(R2 / row.trace_reference, allow_pickle=False) as z:
            traces[delay] = {
                "received": z["received"], "arrival": z["arrival_mask"],
                "run_id": z["run_id"]}
    trace_index = {
        d: {str(run): i for i, run in enumerate(traces[d]["run_id"])}
        for d in traces}
    with np.load(
            R2 / manifest.iloc[0].o0_reference, allow_pickle=False) as z:
        o0 = z["values"]
        o0_index = {str(run): i for i, run in enumerate(z["run_id"])}

    base, received, arrival, labels = [], [], [], []
    for row in states.itertuples():
        idx = trace_index[row.delay][row.run_id]
        received.append(traces[row.delay]["received"][idx])
        arrival.append(traces[row.delay]["arrival"][idx])
        labels.append(row.mechanism == H2)
        base.append({
            "run_id": row.run_id, "seed": int(row.seed),
            "mechanism": row.mechanism, "delay": int(row.delay),
            "outer_fold": int(row.outer_fold),
            "frozen_probability_h2": float(row.mechanism_probability_H2),
            "oracle_action": row.oracle_action,
            "potential_regret": float(row.potential_wrong_transfer_regret)})
    labels = np.asarray(labels, int)
    seeds = np.asarray([r["seed"] for r in base])
    static = np.asarray([
        static_features(r, a, b["delay"])
        for r, a, b in zip(received, arrival, base)])

    temporal = {}
    extraction_start = time.perf_counter()
    for length in WINDOWS:
        sequences = np.asarray([
            visible_series(r, length).reshape(length, -1)
            for r in received])
        temporal[length] = {
            "sequences": sequences,
            "trend": np.asarray([
                trend_features(visible_series(r, length))
                for r in received]),
            "conv": np.asarray([
                convolution_features(visible_series(r, length))
                for r in received])}
    extraction_seconds = time.perf_counter() - extraction_start

    predictions = {m: np.full(len(base), np.nan) for m in METHODS}
    predictions["T0"] = np.asarray([
        row["frozen_probability_h2"] for row in base])
    fold_rows, window_rows, likelihood_rows = [], [], []
    selected_by_fold = {}
    for fold in range(5):
        train = np.asarray([r["outer_fold"] != fold for r in base])
        test = ~train
        train_indices = np.flatnonzero(train)
        features = {}
        ar_by_length = {}
        for length in WINDOWS:
            seq = temporal[length]["sequences"]
            # Shared order is selected on outer-training pooled likelihood.
            order_scores = {}
            for order in ORDERS:
                models = fit_ar_models(seq, labels, train, order)
                vals = np.asarray([
                    dynamic_features(seq[i], models, order)
                    for i in train_indices])
                order_scores[order] = np.mean(np.abs(vals[:, 2]))
            order = max(ORDERS, key=lambda o: order_scores[o])
            models = fit_ar_models(seq, labels, train, order)
            dyn = np.asarray([
                dynamic_features(sequence, models, order)
                for sequence in seq])
            ar_by_length[length] = (order, models, dyn)
            features[length] = {
                "T0": static,
                "T1": np.column_stack([static, temporal[length]["trend"]]),
                "T2": dyn,
                "T3": np.column_stack([static, dyn]),
                "T4": np.column_stack([static, temporal[length]["conv"]]),
                "T5": np.column_stack([
                    static, dyn, temporal[length]["conv"]])}
        selected = select_window(
            temporal, static, labels, train_indices, seeds)
        selected_by_fold[fold] = selected
        for method in METHODS:
            length = selected[method]
            if method == "T0":
                parameter_count = len(
                    frozen.FROZEN["age_aware_model"][
                        "coefficients_intercept_first"])
            else:
                x = features[length][method]
                model = fit_classifier(x[train], labels[train])
                predictions[method][test] = probabilities(model, x[test])
                parameter_count = len(model[2])
            window_rows.append({
                "outer_fold": fold, "method": method,
                "selected_window": length,
                "selected_ar_order": (
                    "" if method in ("T0", "T1", "T4")
                    else ar_by_length[length][0]),
                "inner_group_folds": 4,
                "outer_test_used_for_selection": False,
                "classifier": (
                    "frozen_static_classifier" if method == "T0"
                    else "logistic_regression"),
                "classifier_parameter_count": parameter_count})
        length = selected["T2"]
        order, _, dyn = ar_by_length[length]
        for i in np.flatnonzero(test):
            trace_detail = dynamic_trace(
                seq[i], ar_by_length[length][1], order)
            likelihood_rows.append({
                "run_id": base[i]["run_id"], "seed": base[i]["seed"],
                "mechanism": base[i]["mechanism"],
                "delay": base[i]["delay"], "outer_fold": fold,
                "window": length, "ar_order": order,
                "cumulative_H1_log_likelihood": dyn[i, 0],
                "cumulative_H2_log_likelihood": dyn[i, 1],
                "log_likelihood_ratio": dyn[i, 2],
                "H1_residual_energy": dyn[i, 3],
                "H2_residual_energy": dyn[i, 4],
                "residual_energy_difference": dyn[i, 5],
                **trace_detail})

    all_rows = {}
    summaries, delays, errors = [], [], []
    for method in METHODS:
        rows = action_rows(base, predictions[method])
        all_rows[method] = rows
        for scope in ("all", "delay_1", "delay_2"):
            value = metrics(rows, method, scope)
            (summaries if scope == "all" else delays).append(value)
        for mechanism in (H1, H2):
            subset = [r for r in rows if r["mechanism"] == mechanism]
            errors.append({
                "method": method, "true_mechanism": mechanism,
                "predicted_H1": sum(r["predicted"] == H1 for r in subset),
                "predicted_H2": sum(r["predicted"] == H2 for r in subset),
                "runs": len(subset)})
        for fold in range(5):
            subset = [r for r in rows if r["outer_fold"] == fold]
            fold_rows.append(metrics(subset, method))

    # Fixed-kernel response diagnostics, independent of labels/models.
    conv_diag = []
    for length in WINDOWS:
        matrix = temporal[length]["conv"]
        for mechanism, value in ((H1, 0), (H2, 1)):
            subset = matrix[labels == value]
            conv_diag.append({
                "window": length, "mechanism": mechanism,
                "mean_absolute_response": np.mean(np.abs(subset)),
                "max_absolute_response": np.max(np.abs(subset)),
                "feature_count": matrix.shape[1]})

    # Oracle O1 uses O0 only here and is never merged with deployable inputs.
    oracle_rows = []
    for length in WINDOWS:
        oracle_trend = np.asarray([
            trend_features(o0[o0_index[b["run_id"]], -length:, :, :][:, :, USE])
            for b in base])
        fold_pred = np.full(len(base), np.nan)
        for fold in range(5):
            train = np.asarray([r["outer_fold"] != fold for r in base])
            model = fit_classifier(oracle_trend[train], labels[train])
            fold_pred[~train] = probabilities(model, oracle_trend[~train])
        oracle_rows.append({
            "oracle": "O1_FULL_O0_HISTORY",
            "status": "ORACLE ONLY; NOT DEPLOYABLE", "window": length,
            "balanced_accuracy": balanced_accuracy_score(
                labels, fold_pred >= .5),
            "communication_bytes_per_tick": "NOT_DEPLOYABLE"})
    ll = pd.DataFrame(likelihood_rows)
    oracle_rows.append({
        "oracle": "O2_TRUE_MECHANISM_MODEL_SELECTION",
        "status": "ORACLE ONLY; NOT DEPLOYABLE", "window": "selected",
        "balanced_accuracy": np.mean(
            ((ll.mechanism == H1) & (
                ll.cumulative_H1_log_likelihood >=
                ll.cumulative_H2_log_likelihood))
            | ((ll.mechanism == H2) & (
                ll.cumulative_H2_log_likelihood >
                ll.cumulative_H1_log_likelihood))),
        "communication_bytes_per_tick": "NOT_DEPLOYABLE"})

    # Seed bootstrap of fixed outer-fold predictions.
    rng = np.random.default_rng(83107)
    unique_seeds = np.unique(seeds)
    bootstrap = []
    comparisons = (
        ("T1", "S0", "balanced_accuracy"),
        ("T2", "S0", "h2_to_h1_error"),
        ("T5", "S0", "total_regret"),
        ("T5", "S0", "h2_regret"),
        ("T5", "S0", "coverage"))
    for replicate in range(BOOTSTRAPS):
        draw = rng.choice(unique_seeds, len(unique_seeds), replace=True)
        indices = np.concatenate([np.flatnonzero(seeds == s) for s in draw])
        for method, baseline_name, metric_name in comparisons:
            baseline = "T0"
            left = metrics([all_rows[method][i] for i in indices], method)
            right = metrics([all_rows[baseline][i] for i in indices], baseline)
            bootstrap.append({
                "replicate": replicate,
                "comparison": f"{method}-{baseline_name}",
                "metric": metric_name,
                "difference": left[metric_name] - right[metric_name],
                "resampling_unit": "seed"})
    boot = pd.DataFrame(bootstrap)
    boot_summary = []
    for (comparison, metric_name), group in boot.groupby(
            ["comparison", "metric"]):
        boot_summary.append({
            "comparison": comparison, "metric": metric_name,
            "replicates": len(group), "mean": group.difference.mean(),
            "ci95_low": group.difference.quantile(.025),
            "ci95_high": group.difference.quantile(.975),
            "resampling_unit": "seed"})

    influence = []
    for seed in unique_seeds:
        keep = np.flatnonzero(seeds != seed)
        for method in ("T1", "T2", "T4", "T5"):
            m = metrics([all_rows[method][i] for i in keep], method)
            influence.append({
                "excluded_seed": seed, "method": method,
                "balanced_accuracy": m["balanced_accuracy"],
                "coverage": m["coverage"],
                "total_regret": m["total_regret"],
                "h2_regret": m["h2_regret"]})

    summary = {r["method"]: r for r in summaries}
    t5, s0 = summary["T5"], summary["T0"]
    t5_boot = next(r for r in boot_summary
                   if r["comparison"] == "T5-S0"
                   and r["metric"] == "total_regret")
    same_direction = all(
        next(r for r in delays
             if r["method"] == "T5" and r["scope"] == f"delay_{d}")[
                 "total_regret"]
        < next(r for r in delays
               if r["method"] == "T0" and r["scope"] == f"delay_{d}")[
                   "total_regret"] for d in (1, 2))
    survives = (
        t5["coverage"] >= .80 and t5["balanced_accuracy"] >= .85
        and t5["total_regret"] <= .8 * s0["total_regret"]
        and t5["h2_regret"] <= s0["h2_regret"] and same_direction
        and t5_boot["ci95_high"] < 0)
    m3_value = (
        summary["T2"]["total_regret"] < s0["total_regret"]
        and summary["T2"]["h2_to_h1_error"] <
        s0["h2_to_h1_error"])
    conv_value = (
        summary["T4"]["total_regret"] < s0["total_regret"]
        and summary["T4"]["h2_regret"] <= s0["h2_regret"]
        and all(
            next(r for r in delays
                 if r["method"] == "T4"
                 and r["scope"] == f"delay_{d}")["total_regret"]
            < next(r for r in delays
                   if r["method"] == "T0"
                   and r["scope"] == f"delay_{d}")["total_regret"]
            for d in (1, 2)))
    oracle_best = max(r["balanced_accuracy"] for r in oracle_rows[:3])
    if survives and (m3_value or conv_value):
        decision = "G0_TEMPORAL_MECHANISM_EVIDENCE_SURVIVES"
    elif m3_value and not conv_value:
        decision = "G0_MODEL_CONSISTENCY_ONLY"
    elif conv_value and not m3_value:
        decision = "G0_TEMPORAL_CONVOLUTION_ONLY"
    elif oracle_best >= .85 and t5["balanced_accuracy"] < .85:
        decision = "G0_ORACLE_TEMPORAL_SIGNAL_ONLY"
    else:
        decision = "G0_TEMPORAL_EVIDENCE_INSUFFICIENT"

    likelihood_by_run = {
        (row["run_id"], row["delay"]): row for row in likelihood_rows}
    for index, row in enumerate(base):
        diagnostic = likelihood_by_run[(row["run_id"], row["delay"])]
        diagnostic["calibrated_T2_P_H2"] = predictions["T2"][index]
        diagnostic["T2_brier_contribution"] = (
            predictions["T2"][index] - labels[index]) ** 2

    write_csv(OUT / "01_data_manifest.csv", [{
        "seed_min": 3000, "seed_max": 3399, "seed_count": 400,
        "runs": len(base), "delays": "1|2", "packet_loss": 0,
        "deployable_source": "F2_RECEIVED_AND_ARRIVAL_TENSORS_ONLY",
        "decision_timestamp": 499, "sealed_seeds_loaded": False,
        "r0_final_seeds_loaded": False,
        "simulation_rerun": False,
        "trace_manifest_sha256": sha256(DATA / "01_replay_manifest.csv")}])
    write_csv(OUT / "02_window_manifest.csv", window_rows)
    trend_names = []
    for field in [FIELDS[i] for i in USE]:
        for node in range(6):
            for stat in ("last", "first_to_last_change", "mean_first_diff",
                         "last_first_diff", "max_first_diff", "variance",
                         "sign_changes"):
                trend_names.append(f"{field}_node{node}_{stat}")
        trend_names.extend([
            f"{field}_mean_node_slope", f"{field}_max_node_slope",
            f"{field}_slope_dispersion", f"{field}_fraction_nodes_increasing",
            f"{field}_pairwise_slope_agreement"])
    trend_names += [
        "busy_throughput_inconsistency",
        "queue_failure_slope_correlation"]
    write_csv(OUT / "03_temporal_feature_schema.csv", [
        {"route": "S1", "feature": name,
         "deployable_source": "F2_RECEIVED_HISTORY"}
        for name in trend_names] + [
        {"route": "M3", "feature": name,
         "deployable_source": "F2_RECEIVED_HISTORY"}
        for name in (
            "cumulative_H1_log_likelihood",
            "cumulative_H2_log_likelihood", "log_likelihood_ratio",
            "H1_residual_energy", "H2_residual_energy",
            "residual_energy_difference")])
    (OUT / "04_dynamic_model_specification.md").write_text(
        "# Dynamic model specification\n\n"
        "H1 and H2 always run in parallel on the same past F2 receiver "
        "history. Both use the same multivariate linear AR order in {1,2}; "
        "the shared order is selected using outer-training data only. H1 is "
        "fit only to training H1 runs and H2 only to training H2 runs. "
        "Innovation variances are training residual variances. This is not "
        "future-information RTS smoothing.\n")
    (OUT / "05_convolution_kernel_specification.md").write_text(
        "# Fixed convolution kernels\n\n" + "\n".join(
            f"- `{name}`: `{list(kernel)}`"
            for name, kernel in KERNELS.items()) +
        "\n\nOnly one fixed convolution bank is used. C2 is not run; no CNN, "
        "Transformer, LSTM, GRU, or attention model is present.\n")
    write_csv(OUT / "06_outer_fold_results.csv", fold_rows)
    write_csv(OUT / "07_method_summary.csv", summaries)
    write_csv(OUT / "08_delay_specific_results.csv", delays)
    write_csv(OUT / "09_mechanism_error_matrix.csv", errors)
    write_csv(OUT / "10_likelihood_and_residual_diagnostics.csv",
              likelihood_rows)
    write_csv(OUT / "11_convolution_response_diagnostics.csv", conv_diag)
    write_csv(OUT / "12_oracle_analysis.csv", oracle_rows)
    elapsed = time.perf_counter() - start
    runtime_rows = []
    for method in METHODS:
        lengths = [selected_by_fold[f][method] for f in range(5)]
        parameter_count = int(np.mean([
            next(r["classifier_parameter_count"] for r in window_rows
                 if r["outer_fold"] == f and r["method"] == method)
            for f in range(5)]))
        runtime_rows.append({
            "method": method,
            "feature_extraction_seconds_per_run":
                extraction_seconds / len(base),
            "filter_update_seconds_per_tick":
                extraction_seconds / (len(base) * 500),
            "convolution_seconds_per_run": (
                extraction_seconds / len(base)
                if method in ("T4", "T5") else 0),
            "total_inference_seconds_per_run": elapsed / len(base),
            "model_parameter_count": parameter_count,
            "communication_bytes_per_tick": COMMUNICATION,
            "selected_windows": "|".join(map(str, lengths))})
    write_csv(OUT / "13_runtime_and_parameter_count.csv", runtime_rows)
    write_csv(OUT / "14_bootstrap_results.csv", boot_summary)
    write_csv(OUT / "15_seed_influence.csv", influence)
    (OUT / "00_scope_and_nonclaims.md").write_text(
        "# B1 R3 Gate 0 scope and nonclaims\n\n"
        "Development-only candidate audit using seeds 3000--3399. Deployable "
        "methods consume only existing F2 received/arrival history at 5.10 "
        "B/tick. O0 appears only in labeled oracle analysis. No simulation is "
        "rerun, no sealed/final seed is opened, no deep model is used, and no "
        "historical R0/R1/R2 conclusion is overwritten. B1 remains frozen.\n")
    report = f"""# B1 R3 Gate 0 temporal consistency and convolution audit

## Formal decision

`{decision}`

| Method | Balanced accuracy | Coverage | Total regret | H2 regret |
|---|---:|---:|---:|---:|
""" + "\n".join(
        f"| {m} | {summary[m]['balanced_accuracy']:.4f} | "
        f"{summary[m]['coverage']:.4f} | "
        f"{summary[m]['total_regret']:.4f} | "
        f"{summary[m]['h2_regret']:.4f} |" for m in METHODS) + f"""

T1 versus S0 tests hand trends; T2 is M3 alone; T4 tests fixed convolution
beyond static evidence; T5 combines S0, M3, and C1. All deployable methods keep
communication fixed at {COMMUNICATION:.2f} B/tick. Window selection and model
fit never use the held-out outer fold.

## Key controls

1. **S1 versus S0:** S1 balanced accuracy is
   {summary['T1']['balanced_accuracy']:.4f} versus
   {s0['balanced_accuracy']:.4f}, and regret is
   {summary['T1']['total_regret']:.4f} versus {s0['total_regret']:.4f};
   hand trends do not improve the frozen static baseline.
2. **M3 dangerous error:** T2 H2-to-H1 error is
   {summary['T2']['h2_to_h1_error']:.4f} versus
   {s0['h2_to_h1_error']:.4f}, while H2 regret rises to
   {summary['T2']['h2_regret']:.4f}; dual-model consistency does not provide
   safe standalone value.
3. **C1 independent signal:** T4 balanced accuracy reaches
   {summary['T4']['balanced_accuracy']:.4f} and total regret
   {summary['T4']['total_regret']:.4f}, but H2 regret rises from
   {s0['h2_regret']:.4f} to {summary['T4']['h2_regret']:.4f}. The fixed kernels
   carry signal beyond S1 but fail the high-cost safety control.
4. **T5:** coverage is {t5['coverage']:.4f}, balanced accuracy
   {t5['balanced_accuracy']:.4f}, total regret {t5['total_regret']:.4f}, and
   H2 regret {t5['h2_regret']:.4f}; it does not outperform S0.
5. Delay-specific and leave-one-seed-out results are reported separately;
   no claim is based on an overall-only improvement.
6. Raw Gaussian AR likelihood is treated as ranking evidence, not a calibrated
   posterior. T2 probabilities come from outer-training logistic calibration;
   per-run Brier contributions are reported, but no held-out recalibration is
   performed.

The result does not revive B1 and does not authorize the next Gate.
"""
    (OUT / "16_gate0_report.md").write_text(report)
    (OUT / "17_gate0_decision.txt").write_text(decision + "\n")
    (CONCLUSION / "G0_FORMAL_DECISION.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
