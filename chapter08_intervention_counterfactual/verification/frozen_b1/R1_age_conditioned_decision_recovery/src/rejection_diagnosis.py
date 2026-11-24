"""Read-only diagnostics for the frozen Gate 1 rejection score.

This module contains no fitting or simulation code.  It reconstructs the
frozen M5 risk score from saved M4 posterior probabilities and the immutable
wrong-action regret estimates in the frozen receiver manifest.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


F2_BYTES = 5.10
FROZEN_CONFIG_ID = "PHASE3C_FIXED_MATCHED_CONFIG_PAIR"
SEALED_R1_SEEDS = set(range(2000, 2030))
FORBIDDEN_R0_SEEDS = set(range(120, 150))


def recover_score(probability_h2, h1_wrong_regret, h2_wrong_regret):
    """Return the exact score used by ``transfer_or_reject`` in Gate 1."""
    p = np.asarray(probability_h2, dtype=float)
    return np.where(
        p >= 0.5,
        (1.0 - p) * float(h1_wrong_regret),
        p * float(h2_wrong_regret),
    )


def prepare_f2_frame(predictions, frozen_manifest):
    """Select saved F2/M4 rows and add only deterministically derived fields."""
    frame = predictions[
        (predictions["method"] == "M4_AGE_MASK_SILENCE")
        & np.isclose(predictions["protocol_bytes_per_tick"], F2_BYTES)
    ].copy()
    if frame.empty:
        raise ValueError("no saved F2 M4 prediction rows")
    seeds = set(frame["seed"].astype(int))
    if seeds & SEALED_R1_SEEDS:
        raise ValueError("sealed R1 final seeds were loaded")
    if seeds & FORBIDDEN_R0_SEEDS:
        raise ValueError("forbidden R0 final seeds were loaded")
    reject = frozen_manifest["cost_sensitive_reject"]
    frame["rejection_score"] = recover_score(
        frame["prediction_probability"],
        reject["h1_wrong_regret_estimate"],
        reject["h2_wrong_regret_estimate"],
    )
    frame["reject_threshold"] = float(reject["reject_cost"])
    frame["accepted_current"] = (
        frame["rejection_score"] <= frame["reject_threshold"]
    )
    frame["config_id"] = FROZEN_CONFIG_ID
    frame["oracle_action"] = np.where(
        frame["true_label"].astype(int) == 1, "SYNC_ACTION", "CAPACITY_ACTION"
    )
    frame["unrejected_action"] = np.where(
        frame["predicted_label"].astype(int) == 1,
        "SYNC_ACTION",
        "CAPACITY_ACTION",
    )
    frame["wrong_action"] = (
        frame["unrejected_action"] != frame["oracle_action"]
    ).astype(int)
    frame["y1_classification_error"] = frame["classification_error"].astype(int)
    frame["y2_wrong_action"] = frame["wrong_action"].astype(int)
    return frame


def add_high_regret_labels(frame):
    """Add all preregistered fixed positive-regret quantile labels."""
    positive = frame.loc[
        frame["wrong_transfer_regret"] > 0, "wrong_transfer_regret"
    ].to_numpy(float)
    if not len(positive):
        raise ValueError("no positive wrong-transfer regret values")
    thresholds = {}
    for quantile, name in ((0.50, "p50"), (0.75, "p75"), (0.90, "p90")):
        threshold = float(np.quantile(positive, quantile))
        thresholds[name] = threshold
        frame[f"y3_high_regret_{name}"] = (
            frame["wrong_transfer_regret"] >= threshold
        ).astype(int)
    return thresholds


def binary_auroc(labels, scores):
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return math.nan
    ordered = np.sort(neg)
    lower = np.searchsorted(ordered, pos, side="left")
    upper = np.searchsorted(ordered, pos, side="right")
    return float(np.mean((lower + 0.5 * (upper - lower)) / len(neg)))


def average_precision(labels, scores):
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    positives = int(y.sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-s, kind="mergesort")
    ranked = y[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positives)


def overlap_coefficient(labels, scores, bins=20):
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    a, b = s[y == 1], s[y == 0]
    if not len(a) or not len(b):
        return math.nan
    lo, hi = float(s.min()), float(s.max())
    if hi == lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, edges)
    hb, _ = np.histogram(b, edges)
    return float(np.minimum(ha / len(a), hb / len(b)).sum())


def point_biserial(labels, scores):
    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    if y.std() == 0 or s.std() == 0:
        return math.nan
    return float(np.corrcoef(y, s)[0, 1])


def discrimination_row(frame, label, scope, group_value):
    y = frame[label].to_numpy(int)
    score = frame["rejection_score"].to_numpy(float)
    auc = binary_auroc(y, score)
    reject_scores, accept_scores = score[y == 1], score[y == 0]
    return {
        "scope": scope,
        "group_value": group_value,
        "target": label,
        "n": len(frame),
        "positive_n": int(y.sum()),
        "positive_rate": float(y.mean()),
        "auroc": auc,
        "auprc": average_precision(y, score),
        "mean_score_should_reject": (
            float(reject_scores.mean()) if len(reject_scores) else math.nan
        ),
        "mean_score_should_accept": (
            float(accept_scores.mean()) if len(accept_scores) else math.nan
        ),
        "median_score_should_reject": (
            float(np.median(reject_scores)) if len(reject_scores) else math.nan
        ),
        "median_score_should_accept": (
            float(np.median(accept_scores)) if len(accept_scores) else math.nan
        ),
        "distribution_overlap": overlap_coefficient(y, score),
        "mann_whitney_rank_biserial": (
            2.0 * auc - 1.0 if np.isfinite(auc) else math.nan
        ),
        "point_biserial_correlation": point_biserial(y, score),
    }


def balanced_accuracy(labels, predictions):
    y = np.asarray(labels, dtype=int)
    pred = np.asarray(predictions, dtype=int)
    rates = []
    for value in (0, 1):
        mask = y == value
        if mask.any():
            rates.append(float((pred[mask] == value).mean()))
    return float(np.mean(rates)) if rates else math.nan


def curve_row(frame, threshold, reject_cost, h1_wrong, h2_wrong):
    accepted = frame["rejection_score"].to_numpy(float) <= float(threshold)
    errors = frame["wrong_action"].to_numpy(int)
    regret = frame["wrong_transfer_regret"].to_numpy(float)
    mechanism = frame["mechanism"].astype(str).to_numpy()
    accepted_n = int(accepted.sum())
    rejected_n = len(frame) - accepted_n
    accepted_regret = regret * accepted
    h1 = np.char.startswith(mechanism.astype(str), "H1")
    h2 = np.char.startswith(mechanism.astype(str), "H2")
    wrong_reference = np.where(h1, h1_wrong, h2_wrong)
    reference_mean = float(wrong_reference.mean())
    total_cost = accepted_regret + rejected_n * 0.0
    total_cost = float(accepted_regret.sum() + reject_cost * rejected_n) / len(frame)
    return {
        "threshold": float(threshold),
        "reject_cost": float(reject_cost),
        "n": len(frame),
        "coverage": float(accepted.mean()),
        "reject_rate": float(1.0 - accepted.mean()),
        "accepted_case_balanced_accuracy": (
            balanced_accuracy(
                frame.loc[accepted, "true_label"],
                frame.loc[accepted, "predicted_label"],
            )
            if accepted_n else math.nan
        ),
        "overall_balanced_accuracy": balanced_accuracy(
            frame["true_label"], frame["predicted_label"]
        ),
        "wrong_action_rate_counting_reject_as_no_transfer": float(
            (errors * accepted).mean()
        ),
        "overall_potential_regret_without_rejection": float(regret.mean()),
        "accepted_case_mean_regret": (
            float(regret[accepted].mean()) if accepted_n else math.nan
        ),
        "regret_counting_rejected_as_zero_action": float(accepted_regret.mean()),
        "mean_total_cost_with_reject_cost": total_cost,
        "h1_regret_counting_rejected_as_zero": (
            float(accepted_regret[h1].mean()) if h1.any() else math.nan
        ),
        "h2_regret_counting_rejected_as_zero": (
            float(accepted_regret[h2].mean()) if h2.any() else math.nan
        ),
        "utility_gain_retention": (
            float((reference_mean - total_cost) / reference_mean)
            if reference_mean else math.nan
        ),
        "accepted_h1": int((accepted & h1).sum()),
        "accepted_h2": int((accepted & h2).sum()),
    }


def threshold_grid(scores, current_threshold):
    score = np.asarray(scores, dtype=float)
    quantiles = np.quantile(score, np.linspace(0.0, 1.0, 101))
    return np.unique(np.r_[score.min() - 1e-12, quantiles, current_threshold,
                           score.max() + 1e-12])


def curve(frame, thresholds, reject_costs, h1_wrong, h2_wrong):
    return pd.DataFrame([
        curve_row(frame, threshold, reject_cost, h1_wrong, h2_wrong)
        for threshold in thresholds for reject_cost in reject_costs
    ])


def best_at_coverage(frame, minimum_coverage=0.80):
    """Diagnostic oracle using free-rejection transfer regret as its objective."""
    score = frame["rejection_score"].to_numpy(float)
    regret = frame["wrong_transfer_regret"].to_numpy(float)
    order = np.argsort(score, kind="mergesort")
    ordered_score = score[order]
    cumulative_regret = np.cumsum(regret[order])
    # A threshold must accept every tied score.  Keep only tie-block endpoints.
    endpoints = np.r_[ordered_score[1:] != ordered_score[:-1], True]
    indices = np.flatnonzero(endpoints)
    coverages = (indices + 1) / len(frame)
    eligible = indices[coverages >= minimum_coverage]
    if not len(eligible):
        return None
    chosen = min(
        eligible,
        key=lambda index: (
            cumulative_regret[index] / len(frame),
            (index + 1) / len(frame),
            ordered_score[index],
        ),
    )
    return curve_row(frame, ordered_score[chosen], 0.0, 1.0, 1.0)
