#!/usr/bin/env python3
"""F0 unified analysis: Pair 1 (T0/T4/R4) + Pair 2 (M0/M1/M2) metrics,
pairwise CIs, harm-mass reconstruction, risk-direction prediction, joint
seed-level bootstrap/LOSO. Reuses CF1's/P3's own metric/decomposition
definitions (only the seed set is new).

Bootstrap/LOSO are vectorized over seed *blocks* (each seed contributes a
fixed number of rows per method, so a seed-level resample is just a
fancy-index over the block axis) rather than repeated per-replicate
DataFrame filtering, which does not scale to 2000 replicates x 400 seeds.

Usage: analyze_f0.py <pair1_dir> <pair2_dir> <out_dir>
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BOOTS = 2000
REJECT_COST = 5.0
H2_NAME = "H2_DISTRIBUTED_CAPACITY_SUPPRESSION"


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------
# Pair 1 (H1/H2, T0/T4/R4) -- seed-block arrays
# --------------------------------------------------------------------
def p1_blocks(df, method, seeds_sorted):
    sub = df[df.method == method].sort_values("seed", kind="stable")
    n_seeds = len(seeds_sorted)
    rows_per_seed = len(sub) // n_seeds
    assert len(sub) == n_seeds * rows_per_seed, "ragged per-seed row count"
    shape = (n_seeds, rows_per_seed)
    return {
        "y_h2": (sub.mechanism == H2_NAME).to_numpy().reshape(shape),
        "pred_h2": (sub.predicted_mechanism == H2_NAME).to_numpy().reshape(shape),
        "accepted": sub.accepted.to_numpy().astype(bool).reshape(shape),
        "regret": sub.system_regret.to_numpy(dtype=float).reshape(shape),
    }


def p1_metrics_from_flat(y_h2, pred_h2, accepted, regret):
    ba = .5 * (np.mean(pred_h2[y_h2]) + np.mean(~pred_h2[~y_h2]))
    return {
        "n": int(len(y_h2)), "balanced_accuracy": float(ba),
        "coverage": float(np.mean(accepted)),
        "total_regret": float(np.mean(regret)),
        "h1_regret": float(np.mean(regret[~y_h2])),
        "h2_regret": float(np.mean(regret[y_h2])),
        "h1_to_h2_error": float(np.mean(pred_h2[~y_h2])),
        "h2_to_h1_error": float(np.mean(~pred_h2[y_h2])),
    }


def p1_metrics(df):
    return p1_metrics_from_flat(
        (df.mechanism == H2_NAME).to_numpy(),
        (df.predicted_mechanism == H2_NAME).to_numpy(),
        df.accepted.to_numpy().astype(bool),
        df.system_regret.to_numpy(dtype=float))


def p1_decomposition(df, a, b):
    result = []
    for mech, col in (("H1_CAPACITY_SHORTAGE", "H1"), (H2_NAME, "H2")):
        sub = df[df.mechanism == mech].copy()
        sub["key"] = sub.run_id.astype(str) + "|" + sub.delay.astype(str)
        by_a = sub[sub.method == a].set_index("key").sort_index()
        by_b = sub[sub.method == b].set_index("key").sort_index()
        assert (by_a.index == by_b.index).all()
        aa, bb = by_a.accepted.to_numpy(bool), by_b.accepted.to_numpy(bool)
        ra, rb = by_a.system_regret.to_numpy(float), by_b.system_regret.to_numpy(float)
        both = aa & bb
        diff = rb - ra
        corrected = float(-diff[both & (diff < 0)].sum())
        new = float(diff[both & (diff > 0)].sum())
        loss = float(diff[aa & ~bb].sum())
        gain = float(diff[~aa & bb].sum())
        lhs = float(diff.sum())
        rhs = new - corrected + loss + gain
        result.append({
            "comparison": f"{a}_to_{b}", "mechanism": col, "n": len(by_a),
            "corrected_harm_mass": corrected, "new_harm_mass": new,
            "coverage_loss_mass": loss, "coverage_gain_mass": gain,
            "lhs_total_regret_change": lhs, "rhs_reconstruction": rhs,
            "absolute_error": abs(lhs - rhs)})
    return result


def bootstrap_diff_p1(df, a, b, seeds_sorted, rng):
    ba_ = p1_blocks(df, a, seeds_sorted)
    bb_ = p1_blocks(df, b, seeds_sorted)
    n_seeds = len(seeds_sorted)
    out = {k: np.empty(BOOTS) for k in
           ("balanced_accuracy", "total_regret", "h1_regret", "h2_regret")}
    for i in range(BOOTS):
        idx = rng.integers(0, n_seeds, n_seeds)
        ma = p1_metrics_from_flat(
            ba_["y_h2"][idx].ravel(), ba_["pred_h2"][idx].ravel(),
            ba_["accepted"][idx].ravel(), ba_["regret"][idx].ravel())
        mb = p1_metrics_from_flat(
            bb_["y_h2"][idx].ravel(), bb_["pred_h2"][idx].ravel(),
            bb_["accepted"][idx].ravel(), bb_["regret"][idx].ravel())
        for k in out:
            out[k][i] = mb[k] - ma[k]
    return [{
        "comparison": f"{b}-{a}", "metric": k,
        "mean": float(out[k].mean()),
        "ci95_low": float(np.quantile(out[k], .025)),
        "ci95_high": float(np.quantile(out[k], .975))}
        for k in out]


def loso_p1(df, a, b, seeds_sorted):
    ba_ = p1_blocks(df, a, seeds_sorted)
    bb_ = p1_blocks(df, b, seeds_sorted)
    n_seeds = len(seeds_sorted)
    rows = []
    for i in range(n_seeds):
        mask = np.arange(n_seeds) != i
        ma = p1_metrics_from_flat(
            ba_["y_h2"][mask].ravel(), ba_["pred_h2"][mask].ravel(),
            ba_["accepted"][mask].ravel(), ba_["regret"][mask].ravel())
        mb = p1_metrics_from_flat(
            bb_["y_h2"][mask].ravel(), bb_["pred_h2"][mask].ravel(),
            bb_["accepted"][mask].ravel(), bb_["regret"][mask].ravel())
        rows.append({"excluded_seed": int(seeds_sorted[i]),
                     f"p1_{b}-{a}_total_regret": mb["total_regret"] - ma["total_regret"]})
    return rows


# --------------------------------------------------------------------
# Pair 2 (H3/H4, M0/M1/M2)
# --------------------------------------------------------------------
def p2_blocks(df, method, seeds_sorted):
    sub = df.sort_values("seed", kind="stable")
    n_seeds = len(seeds_sorted)
    rows_per_seed = len(sub) // n_seeds
    assert len(sub) == n_seeds * rows_per_seed
    shape = (n_seeds, rows_per_seed)
    return {
        "y": sub.y.to_numpy().reshape(shape),
        "pred": sub[f"{method}_pred"].to_numpy().reshape(shape),
        "accept": sub[f"{method}_accept"].to_numpy(bool).reshape(shape),
        "regret": sub[f"{method}_regret"].to_numpy(float).reshape(shape),
    }


def _ba(y, p):
    vals = [np.mean(p[y == c] == c) if np.any(y == c) else np.nan for c in (0, 1)]
    return float(np.nanmean(vals))


def p2_metrics_from_flat(y, pred, accept, regret):
    return {
        "balanced_accuracy": _ba(y, pred), "coverage": float(accept.mean()),
        "total_regret": float(regret.mean()),
        "H3_regret": float(regret[y == 0].mean()),
        "H4_regret": float(regret[y == 1].mean())}


def p2_metric(rows, method):
    y = np.array([r["y"] for r in rows])
    pred = np.array([r[f"{method}_pred"] for r in rows])
    acc = np.array([r[f"{method}_accept"] for r in rows])
    reg = np.array([r[f"{method}_regret"] for r in rows], dtype=float)
    return {"method": method, **p2_metrics_from_flat(y, pred, acc, reg)}


def p2_decomposition(rows, a, b):
    result = []
    for mech, yval in (("H3", 0), ("H4", 1)):
        rr = [r for r in rows if r["y"] == yval]
        aa = np.array([r[f"{a}_accept"] for r in rr], bool)
        bb = np.array([r[f"{b}_accept"] for r in rr], bool)
        ra = np.array([r[f"{a}_regret"] for r in rr], float)
        rb = np.array([r[f"{b}_regret"] for r in rr], float)
        both = aa & bb
        diff = rb - ra
        corrected = float(-diff[both & (diff < 0)].sum())
        new = float(diff[both & (diff > 0)].sum())
        loss = float(diff[aa & ~bb].sum())
        gain = float(diff[~aa & bb].sum())
        lhs = float(diff.sum())
        rhs = new - corrected + loss + gain
        result.append(dict(
            comparison=f"{a}_to_{b}", mechanism=mech, n=len(rr),
            corrected_harm_mass=corrected, new_harm_mass=new,
            coverage_loss_mass=loss, coverage_gain_mass=gain,
            lhs_total_regret_change=lhs, rhs_reconstruction=rhs,
            absolute_error=abs(lhs - rhs)))
    return result


def bootstrap_diff_p2(df, a, b, seeds_sorted, rng):
    ba_ = p2_blocks(df, a, seeds_sorted)
    bb_ = p2_blocks(df, b, seeds_sorted)
    n_seeds = len(seeds_sorted)
    keys = ("balanced_accuracy", "total_regret", "H3_regret", "H4_regret")
    out = {k: np.empty(BOOTS) for k in keys}
    for i in range(BOOTS):
        idx = rng.integers(0, n_seeds, n_seeds)
        ma = p2_metrics_from_flat(
            ba_["y"][idx].ravel(), ba_["pred"][idx].ravel(),
            ba_["accept"][idx].ravel(), ba_["regret"][idx].ravel())
        mb = p2_metrics_from_flat(
            bb_["y"][idx].ravel(), bb_["pred"][idx].ravel(),
            bb_["accept"][idx].ravel(), bb_["regret"][idx].ravel())
        for k in keys:
            out[k][i] = mb[k] - ma[k]
    return [{
        "comparison": f"{b}-{a}", "metric": k,
        "mean": float(out[k].mean()),
        "ci95_low": float(np.quantile(out[k], .025)),
        "ci95_high": float(np.quantile(out[k], .975))}
        for k in keys]


def loso_p2(df, a, b, seeds_sorted):
    ba_ = p2_blocks(df, a, seeds_sorted)
    bb_ = p2_blocks(df, b, seeds_sorted)
    n_seeds = len(seeds_sorted)
    rows = []
    for i in range(n_seeds):
        mask = np.arange(n_seeds) != i
        ma = p2_metrics_from_flat(
            ba_["y"][mask].ravel(), ba_["pred"][mask].ravel(),
            ba_["accept"][mask].ravel(), ba_["regret"][mask].ravel())
        mb = p2_metrics_from_flat(
            bb_["y"][mask].ravel(), bb_["pred"][mask].ravel(),
            bb_["accept"][mask].ravel(), bb_["regret"][mask].ravel())
        rows.append({"excluded_seed": int(seeds_sorted[i]),
                     f"p2_{b}-{a}_total_regret": mb["total_regret"] - ma["total_regret"]})
    return rows


def main():
    pair1_dir, pair2_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    p1_df = pd.read_csv(pair1_dir / "per_sample_strict_action_outcomes.csv")
    p2_df = pd.read_csv(pair2_dir / "out_of_fold_predictions.csv")
    p2_rows = p2_df.to_dict("records")

    p1_seeds = np.array(sorted(p1_df.seed.unique()))
    p2_seeds = np.array(sorted(p2_df.seed.unique()))
    assert set(p1_seeds.tolist()) == set(p2_seeds.tolist()), \
        "Pair 1 / Pair 2 seed sets diverge -- same-seed joint generation violated"

    # ---- Pair 1 method results ----
    p1_summary = [{"method": m, **p1_metrics(p1_df[p1_df.method == m])}
                  for m in ("T0", "T4", "R4")]
    write_csv(out / "12_pair1_method_results.csv", p1_summary)

    rng1 = np.random.default_rng(710307)
    p1_pairwise = []
    for a, b in (("T0", "T4"), ("T4", "R4"), ("T0", "R4")):
        p1_pairwise += bootstrap_diff_p1(p1_df, a, b, p1_seeds, rng1)
    write_csv(out / "13_pair1_pairwise_results.csv", p1_pairwise)

    # matched coverage: T4 rows re-ranked by certainty to R4's coverage
    t4_df = p1_df[p1_df.method == "T4"].reset_index(drop=True)
    r4_summary = next(r for r in p1_summary if r["method"] == "R4")
    n = len(t4_df)
    n_accept = int(round(r4_summary["coverage"] * n))
    certainty = np.abs(t4_df.p_h2.to_numpy() - .5)
    order = np.argsort(-certainty)
    keep = np.zeros(n, bool)
    keep[order[:n_accept]] = True
    matched_regret = np.where(keep, t4_df.system_regret.to_numpy(), REJECT_COST)
    y_h2 = (t4_df.mechanism == H2_NAME).to_numpy()
    write_csv(out / "14_pair1_matched_coverage.csv", [
        {"candidate": "T4_matched_to_R4_coverage",
         "coverage": n_accept / n, "total_regret": float(matched_regret.mean()),
         "h1_regret": float(matched_regret[~y_h2].mean()),
         "h2_regret": float(matched_regret[y_h2].mean())},
        {"candidate": "R4_actual", **{k: r4_summary[k] for k in
         ("coverage", "total_regret", "h1_regret", "h2_regret")}}])

    p1_harm = (p1_decomposition(p1_df, "T0", "T4") + p1_decomposition(p1_df, "T4", "R4")
               + p1_decomposition(p1_df, "T0", "R4"))

    # ---- Pair 2 method results ----
    p2_summary = [p2_metric(p2_rows, m) for m in ("M0", "M1", "M2")]
    write_csv(out / "16_pair2_method_results.csv", p2_summary)

    rng2 = np.random.default_rng(94117)
    p2_pairwise = (bootstrap_diff_p2(p2_df, "M0", "M1", p2_seeds, rng2)
                   + bootstrap_diff_p2(p2_df, "M1", "M2", p2_seeds, rng2))
    write_csv(out / "17_pair2_pairwise_results.csv", p2_pairwise)

    p2_harm = (p2_decomposition(p2_rows, "M0", "M1") + p2_decomposition(p2_rows, "M1", "M2")
               + p2_decomposition(p2_rows, "M0", "M2"))

    all_harm = p1_harm + p2_harm
    write_csv(out / "18_harm_mass_results.csv", all_harm)
    max_err = max(r["absolute_error"] for r in all_harm)
    if max_err >= 1e-9:
        print(f"F0_HARM_MASS_RECONSTRUCTION_FAILED: max_abs_error={max_err}")
        raise SystemExit(1)

    # ---- Joint seed-level bootstrap (Section 15): SAME resampled seed
    # index used for both pairs in each replicate (same-seed joint pull) ----
    n_seeds = len(p1_seeds)
    p1_t0, p1_t4, p1_r4 = (p1_blocks(p1_df, m, p1_seeds) for m in ("T0", "T4", "R4"))
    p2_m0, p2_m1, p2_m2 = (p2_blocks(p2_df, m, p1_seeds) for m in ("M0", "M1", "M2"))
    rng_joint = np.random.default_rng(571307)
    keys = ("p1_T4-T0_total_regret", "p1_T4-T0_h2_regret",
            "p1_R4-T4_total_regret", "p1_R4-T4_h2_regret",
            "p2_M1-M0_total_regret", "p2_M2-M1_total_regret")
    joint = {k: np.empty(BOOTS) for k in keys}
    for i in range(BOOTS):
        idx = rng_joint.integers(0, n_seeds, n_seeds)
        t0m = p1_metrics_from_flat(p1_t0["y_h2"][idx].ravel(), p1_t0["pred_h2"][idx].ravel(),
                                    p1_t0["accepted"][idx].ravel(), p1_t0["regret"][idx].ravel())
        t4m = p1_metrics_from_flat(p1_t4["y_h2"][idx].ravel(), p1_t4["pred_h2"][idx].ravel(),
                                    p1_t4["accepted"][idx].ravel(), p1_t4["regret"][idx].ravel())
        r4m = p1_metrics_from_flat(p1_r4["y_h2"][idx].ravel(), p1_r4["pred_h2"][idx].ravel(),
                                    p1_r4["accepted"][idx].ravel(), p1_r4["regret"][idx].ravel())
        m0m = p2_metrics_from_flat(p2_m0["y"][idx].ravel(), p2_m0["pred"][idx].ravel(),
                                    p2_m0["accept"][idx].ravel(), p2_m0["regret"][idx].ravel())
        m1m = p2_metrics_from_flat(p2_m1["y"][idx].ravel(), p2_m1["pred"][idx].ravel(),
                                    p2_m1["accept"][idx].ravel(), p2_m1["regret"][idx].ravel())
        m2m = p2_metrics_from_flat(p2_m2["y"][idx].ravel(), p2_m2["pred"][idx].ravel(),
                                    p2_m2["accept"][idx].ravel(), p2_m2["regret"][idx].ravel())
        joint["p1_T4-T0_total_regret"][i] = t4m["total_regret"] - t0m["total_regret"]
        joint["p1_T4-T0_h2_regret"][i] = t4m["h2_regret"] - t0m["h2_regret"]
        joint["p1_R4-T4_total_regret"][i] = r4m["total_regret"] - t4m["total_regret"]
        joint["p1_R4-T4_h2_regret"][i] = r4m["h2_regret"] - t4m["h2_regret"]
        joint["p2_M1-M0_total_regret"][i] = m1m["total_regret"] - m0m["total_regret"]
        joint["p2_M2-M1_total_regret"][i] = m2m["total_regret"] - m1m["total_regret"]
    write_csv(out / "20_main_bootstrap.csv", [
        {"comparison": k, "mean": float(joint[k].mean()),
         "ci95_low": float(np.quantile(joint[k], .025)),
         "ci95_high": float(np.quantile(joint[k], .975))}
        for k in keys])

    # ---- LOSO ----
    loso_p1_rows = loso_p1(p1_df, "T0", "T4", p1_seeds)
    loso_p1_rows2 = loso_p1(p1_df, "T4", "R4", p1_seeds)
    loso_p2_rows = loso_p2(p2_df, "M0", "M1", p1_seeds)
    loso_p2_rows2 = loso_p2(p2_df, "M1", "M2", p1_seeds)
    loso_merged = []
    for i, s in enumerate(p1_seeds):
        loso_merged.append({
            "excluded_seed": int(s),
            **{k: v for k, v in loso_p1_rows[i].items() if k != "excluded_seed"},
            **{k: v for k, v in loso_p1_rows2[i].items() if k != "excluded_seed"},
            **{k: v for k, v in loso_p2_rows[i].items() if k != "excluded_seed"},
            **{k: v for k, v in loso_p2_rows2[i].items() if k != "excluded_seed"},
        })
    write_csv(out / "22_leave_one_seed_out.csv", loso_merged)

    print(json.dumps({
        "p1_summary": p1_summary, "p2_summary": p2_summary,
        "max_harm_mass_error": max_err}, indent=2, default=str))


if __name__ == "__main__":
    main()
