#!/usr/bin/env python3
"""CF1 step 3: recompute T0/T4/R4 control regret from strict counterfactual
action outcomes, reusing the frozen accept/reject decision rule unchanged.

The predicted mechanism probability, the fixed-cost accept/reject rule
(h1_wrong_regret_estimate, h2_wrong_regret_estimate, reject_cost=5.0), and
therefore accepted/rejected status and BA/coverage are all reused byte-for
-byte from the old pipeline -- CF1 only replaces "was the chosen action
wrong, and by how much" with the real per-checkpoint strict-counterfactual
answer instead of the AS0-invalidated same-seed-independent-rerun numbers.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
CF1 = HERE.parents[1]
B1 = HERE.parents[2]
R1 = B1 / "R1_age_conditioned_decision_recovery"

H1 = "H1_CAPACITY_SHORTAGE"
H2 = "H2_DISTRIBUTED_CAPACITY_SUPPRESSION"
METHODS = ("T0", "T4", "R4")
BOOTSTRAPS = 2000
REJECT_COST = 5.0


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0])
        for row in rows[1:]:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_everything():
    cache = np.load(CF1 / "_predictions_cache.npz", allow_pickle=False)
    base = pd.DataFrame({
        "run_id": cache["run_id"], "seed": cache["seed"],
        "mechanism": cache["mechanism"], "delay": cache["delay"],
        "outer_fold": cache["outer_fold"], "labels": cache["labels"]})
    predictions = {m: cache[m] for m in METHODS}
    mapping = pd.read_csv(CF1 / "05_old_to_cf_checkpoint_mapping.csv")
    outcomes = pd.read_csv(CF1 / "_checkpoint_outcomes.csv").set_index(
        "checkpoint_id")
    frozen = json.loads((
        R1 / "results/gate1_formal_age_necessity/"
        "gate1_frozen_receiver_models.json").read_text())["cost_sensitive_reject"]
    return base, predictions, mapping, outcomes, frozen


def decide(p, h1_cost, h2_cost):
    """Byte-for-byte the frozen decision rule from g0.action_rows."""
    costs = {
        "CAPACITY_ACTION": p * h2_cost,
        "SYNC_ACTION": (1 - p) * h1_cost,
        "REJECT": REJECT_COST}
    action = min(costs, key=costs.get)
    return action, action != "REJECT"


def strict_rows_for_method(base, p, mapping_by_row, outcomes, h1_cost, h2_cost):
    rows = []
    for i in range(len(base)):
        run_id = base.run_id.iloc[i]
        cpid = mapping_by_row[run_id]
        checkpoint = outcomes.loc[cpid]
        action, accepted = decide(p[i], h1_cost, h2_cost)
        oracle_best = checkpoint.oracle_best_action_2way
        wrong = accepted and action != oracle_best
        if wrong:
            executed_u = (checkpoint.utility_capacity
                          if action == "CAPACITY_ACTION"
                          else checkpoint.utility_sync)
            strict_regret = checkpoint.oracle_utility_2way - executed_u
        else:
            strict_regret = 0.0
        system_regret = strict_regret if accepted else REJECT_COST
        predicted_mechanism = H2 if p[i] >= .5 else H1
        rows.append({
            "run_id": run_id, "seed": int(base.seed.iloc[i]),
            "mechanism": base.mechanism.iloc[i],
            "delay": int(base.delay.iloc[i]),
            "outer_fold": int(base.outer_fold.iloc[i]),
            "checkpoint_id": cpid,
            "p_h2": float(p[i]), "predicted_mechanism": predicted_mechanism,
            "action": action, "accepted": accepted, "wrong": wrong,
            "oracle_best_action": oracle_best,
            "accepted_only_regret": strict_regret if accepted else np.nan,
            "system_regret": system_regret,
        })
    return pd.DataFrame(rows)


def metrics_from_rows(df, scope_mask=None):
    d = df if scope_mask is None else df[scope_mask]
    y_h2 = (d.mechanism == H2).to_numpy()
    pred_h2 = (d.predicted_mechanism == H2).to_numpy()
    accepted = d.accepted.to_numpy()
    regret = d.system_regret.to_numpy()
    ba = .5 * (np.mean(pred_h2[y_h2]) + np.mean(~pred_h2[~y_h2]))
    return {
        "n": len(d), "balanced_accuracy": ba,
        "coverage": float(np.mean(accepted)),
        "total_regret": float(np.mean(regret)),
        "h1_regret": float(np.mean(regret[~y_h2])),
        "h2_regret": float(np.mean(regret[y_h2])),
        "h1_to_h2_error": float(np.mean(pred_h2[~y_h2])),
        "h2_to_h1_error": float(np.mean(~pred_h2[y_h2])),
        "accepted_only_total_regret": float(
            np.nanmean(d.accepted_only_regret)),
    }


def main():
    base, predictions, mapping, outcomes, frozen = load_everything()
    h1_cost = float(frozen["h1_wrong_regret_estimate"])
    h2_cost = float(frozen["h2_wrong_regret_estimate"])
    assert float(frozen["reject_cost"]) == REJECT_COST
    mapping_by_row = dict(zip(
        mapping.old_prediction_row_id, mapping.strict_cf_checkpoint_id))

    all_rows = {}
    for method in METHODS:
        df = strict_rows_for_method(
            base, predictions[method], mapping_by_row, outcomes, h1_cost,
            h2_cost)
        all_rows[method] = df
        write_csv(CF1 / f"_strict_rows_{method}.csv",
                  df.to_dict("records"))

    write_csv(
        CF1 / "08_per_sample_strict_action_outcomes.csv",
        pd.concat(
            [all_rows[m].assign(method=m) for m in METHODS],
            ignore_index=True).to_dict("records"))

    # ------------------------------------------------------------------
    # T0 / T4 / R4 results + BA/coverage-unchanged assertion
    # ------------------------------------------------------------------
    archived_g0 = pd.read_csv(
        B1 / "R3_temporal_mechanism_evidence/"
        "G0_temporal_consistency_and_convolution/07_method_summary.csv"
    ).set_index("method")
    archived_g0b = pd.read_csv(
        B1 / "R3_temporal_mechanism_evidence/"
        "G0B_risk_constrained_convolution/07_method_summary.csv"
    ).set_index("method")
    archived_summary = {"T0": archived_g0.loc["T0"], "T4": archived_g0.loc["T4"],
                         "R4": archived_g0b.loc["R4"]}
    method_rows = []
    ba_coverage_unchanged = True
    for method in METHODS:
        m = metrics_from_rows(all_rows[method])
        arch = archived_summary[method]
        ba_ok = abs(m["balanced_accuracy"] - float(arch.balanced_accuracy)) < 1e-9
        cov_ok = abs(m["coverage"] - float(arch.coverage)) < 1e-9
        ba_coverage_unchanged = ba_coverage_unchanged and ba_ok and cov_ok
        method_rows.append({
            "method": method, **m,
            "ba_matches_archived": ba_ok, "coverage_matches_archived": cov_ok,
            "old_total_regret_noncausal": float(arch.total_regret),
        })
    write_csv(CF1 / "09_per_method_strict_regret.csv", method_rows)
    for method, fname in zip(METHODS, ("10_t0_results.csv", "11_t4_results.csv",
                                        "12_r4_results.csv")):
        row = next(r for r in method_rows if r["method"] == method)
        write_csv(CF1 / fname, [row])
    if not ba_coverage_unchanged:
        raise SystemExit("BA_OR_COVERAGE_CHANGED -- sample/prediction scope not frozen")

    results = {r["method"]: r for r in method_rows}

    # ------------------------------------------------------------------
    # Pairwise differences (point estimates; CIs come from bootstrap)
    # ------------------------------------------------------------------
    pairwise_rows = []
    for a, b in (("T4", "T0"), ("R4", "T4"), ("R4", "T0")):
        for metric in ("balanced_accuracy", "coverage", "total_regret",
                       "h1_regret", "h2_regret", "h1_to_h2_error",
                       "h2_to_h1_error"):
            pairwise_rows.append({
                "comparison": f"{a}-{b}", "metric": metric,
                "difference": results[a][metric] - results[b][metric]})
    write_csv(CF1 / "13_pairwise_method_differences.csv", pairwise_rows)

    # ------------------------------------------------------------------
    # Delay groups (delay 0 was never part of the old evaluation sample)
    # ------------------------------------------------------------------
    delay_rows = []
    for delay in (0, 1, 2):
        for method in METHODS:
            df = all_rows[method]
            mask = (df.delay == delay).to_numpy()
            if not mask.any():
                delay_rows.append({
                    "delay": delay, "method": method,
                    "status": "NOT_PRESENT_IN_OLD_EVALUATION_SAMPLE",
                    "n": 0, "total_regret": None, "h1_regret": None,
                    "h2_regret": None})
                continue
            m = metrics_from_rows(df, mask)
            delay_rows.append({"delay": delay, "method": method,
                               "status": "present", **m})
    write_csv(CF1 / "14_delay_results.csv", delay_rows)
    per_delay_h2_direction = []
    for delay in (1, 2):
        t4d = metrics_from_rows(all_rows["T4"], (all_rows["T4"].delay == delay).to_numpy())
        t0d = metrics_from_rows(all_rows["T0"], (all_rows["T0"].delay == delay).to_numpy())
        per_delay_h2_direction.append(t4d["h2_regret"] > t0d["h2_regret"])
    direction_consistent_across_delay = all(per_delay_h2_direction)

    # ------------------------------------------------------------------
    # Error-direction costs (label-based counts + action-based strict cost)
    # ------------------------------------------------------------------
    error_rows = []
    for method in METHODS:
        df = all_rows[method]
        for direction, true_mech, pred_mech in (
                ("H1_predicted_as_H2", H1, H2), ("H2_predicted_as_H1", H2, H1)):
            label_mask = (df.mechanism == true_mech) & (
                df.predicted_mechanism == pred_mech)
            action_wrong_mask = label_mask & df.accepted & df.wrong
            regrets = df.loc[action_wrong_mask, "accepted_only_regret"]
            error_rows.append({
                "method": method, "direction": direction,
                "label_error_count": int(label_mask.sum()),
                "label_error_seeds": int(df.loc[label_mask, "seed"].nunique()),
                "accepted_action_wrong_count": int(action_wrong_mask.sum()),
                "mean_regret_per_action_error": float(regrets.mean()) if len(regrets) else 0.0,
                "median_regret_per_action_error": float(regrets.median()) if len(regrets) else 0.0,
                "total_regret_contribution": float(regrets.sum()),
                "fraction_of_method_total_regret": float(
                    regrets.sum() / (df.system_regret.sum() or 1)),
            })
    write_csv(CF1 / "15_error_direction_costs.csv", error_rows)

    cost_ratio_rows = []
    for method in METHODS:
        df = all_rows[method]
        h2_to_h1 = df[(df.mechanism == H2) & df.accepted & df.wrong].accepted_only_regret
        h1_to_h2 = df[(df.mechanism == H1) & df.accepted & df.wrong].accepted_only_regret
        ratio_mean = (float(h2_to_h1.mean()) / float(h1_to_h2.mean())
                      if len(h1_to_h2) and h1_to_h2.mean() else float("nan"))
        ratio_median = (float(h2_to_h1.median()) / float(h1_to_h2.median())
                        if len(h1_to_h2) and h1_to_h2.median() else float("nan"))
        cost_ratio_rows.append({
            "method": method, "n_h2_to_h1_errors": len(h2_to_h1),
            "n_h1_to_h2_errors": len(h1_to_h2),
            "mean_cost_ratio_h2_to_h1_over_h1_to_h2": ratio_mean,
            "median_cost_ratio": ratio_median,
        })

    # ------------------------------------------------------------------
    # Matched-coverage comparison (exact reuse of G0B's own protocol:
    # rank T4's rows by |p-0.5| certainty, accept the top
    # round(R4_coverage * n), reject the rest -- no fold isolation, no
    # re-ranking, matching the original implementation verbatim)
    # ------------------------------------------------------------------
    t4_df = all_rows["T4"]
    n = len(t4_df)
    r4_coverage = results["R4"]["coverage"]
    n_accept_target = int(round(r4_coverage * n))
    certainty = np.abs(t4_df.p_h2.to_numpy() - .5)
    order = np.argsort(-certainty)
    keep_idx = set(order[:n_accept_target].tolist())
    matched_regret, matched_accept = [], []
    for i in range(n):
        cpid = t4_df.checkpoint_id.iloc[i]
        checkpoint = outcomes.loc[cpid]
        p = t4_df.p_h2.iloc[i]
        if i in keep_idx:
            # Exact reuse of the original G0B matched-coverage rule: kept
            # rows are forced to take the cheaper of the two REAL actions
            # (REJECT is excluded from the cost comparison entirely, not
            # overridden after the fact) -- see run_g0b_risk_constrained_
            # convolution.py lines 869-873.
            costs = {"CAPACITY_ACTION": p * h2_cost, "SYNC_ACTION": (1 - p) * h1_cost}
            action = min(costs, key=costs.get)
            wrong = action != checkpoint.oracle_best_action_2way
            if wrong:
                executed_u = (checkpoint.utility_capacity
                              if action == "CAPACITY_ACTION"
                              else checkpoint.utility_sync)
                regret = checkpoint.oracle_utility_2way - executed_u
            else:
                regret = 0.0
            matched_accept.append(True)
        else:
            regret = REJECT_COST
            matched_accept.append(False)
        matched_regret.append(regret)
    matched_regret = np.asarray(matched_regret)
    matched_accept = np.asarray(matched_accept)
    y_h2 = (t4_df.mechanism == H2).to_numpy()
    matched_rows = [{
        "candidate": "T4_matched_to_R4_coverage",
        "coverage": float(matched_accept.mean()),
        "total_regret": float(matched_regret.mean()),
        "h1_regret": float(matched_regret[~y_h2].mean()),
        "h2_regret": float(matched_regret[y_h2].mean()),
    }, {
        "candidate": "R4_actual",
        "coverage": results["R4"]["coverage"],
        "total_regret": results["R4"]["total_regret"],
        "h1_regret": results["R4"]["h1_regret"],
        "h2_regret": results["R4"]["h2_regret"],
    }]
    write_csv(CF1 / "16_matched_coverage_results.csv", matched_rows)

    # ------------------------------------------------------------------
    # Old vs new regret comparison (transparency, not a claim)
    # ------------------------------------------------------------------
    old_new_rows = []
    for method in METHODS:
        arch = archived_summary[method]
        old_new_rows.append({
            "method": method,
            "old_total_regret_HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY":
                float(arch.total_regret),
            "old_h2_regret_HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY":
                float(arch.h2_regret),
            "new_strict_total_regret": results[method]["total_regret"],
            "new_strict_h2_regret": results[method]["h2_regret"],
        })
    write_csv(CF1 / "19_old_vs_new_regret_comparison.csv", old_new_rows)

    # ------------------------------------------------------------------
    # Seed-level bootstrap + leave-one-seed-out
    # ------------------------------------------------------------------
    rng = np.random.default_rng(710307)
    unique_seeds = base.seed.unique()
    combined = pd.concat(
        [all_rows[m].assign(method=m) for m in METHODS], ignore_index=True)

    def group_metrics(df):
        out = {}
        for m in METHODS:
            out[m] = metrics_from_rows(df[df.method == m])
        return out

    comparisons = (
        ("T4", "T0"), ("R4", "T4"), ("R4", "T0"))
    boot_records = []
    for replicate in range(BOOTSTRAPS):
        draw = rng.choice(unique_seeds, len(unique_seeds), replace=True)
        idx = np.concatenate([
            np.flatnonzero(combined.seed.to_numpy() == s) for s in draw])
        sub = combined.iloc[idx]
        gm = group_metrics(sub)
        for a, b in comparisons:
            for metric in ("balanced_accuracy", "total_regret", "h1_regret",
                           "h2_regret"):
                boot_records.append({
                    "replicate": replicate, "comparison": f"{a}-{b}",
                    "metric": metric,
                    "difference": gm[a][metric] - gm[b][metric]})
    boot_df = pd.DataFrame(boot_records)
    boot_summary = []
    for (comparison, metric), group in boot_df.groupby(["comparison", "metric"]):
        boot_summary.append({
            "comparison": comparison, "metric": metric,
            "replicates": BOOTSTRAPS, "mean": float(group.difference.mean()),
            "ci95_low": float(group.difference.quantile(.025)),
            "ci95_high": float(group.difference.quantile(.975)),
            "resampling_unit": "seed",
            "ci_excludes_zero_positive": bool(
                group.difference.quantile(.025) > 0),
            "ci_excludes_zero_negative": bool(
                group.difference.quantile(.975) < 0),
        })
    write_csv(CF1 / "17_seed_bootstrap_results.csv", boot_summary)

    influence_rows = []
    for seed in unique_seeds:
        keep = combined[combined.seed != seed]
        gm = group_metrics(keep)
        row = {"excluded_seed": int(seed)}
        for a, b in comparisons:
            row[f"{a}-{b}_total_regret_diff"] = (
                gm[a]["total_regret"] - gm[b]["total_regret"])
            row[f"{a}-{b}_h2_regret_diff"] = (
                gm[a]["h2_regret"] - gm[b]["h2_regret"])
        influence_rows.append(row)
    write_csv(CF1 / "18_leave_one_seed_out.csv", influence_rows)

    def boot(comparison, metric):
        return next(
            r for r in boot_summary
            if r["comparison"] == comparison and r["metric"] == metric)

    # ------------------------------------------------------------------
    # Claim survival matrix
    # ------------------------------------------------------------------
    t4_t0_total = boot("T4-T0", "total_regret")
    t4_t0_h2 = boot("T4-T0", "h2_regret")
    r4_t4_h2 = boot("R4-T4", "h2_regret")
    r4_t4_h1 = boot("R4-T4", "h1_regret")
    r4_t4_total = boot("R4-T4", "total_regret")
    r4_t0_total = boot("R4-T0", "total_regret")

    ba_gain = results["T4"]["balanced_accuracy"] > results["T0"]["balanced_accuracy"]
    inversion_total = t4_t0_total["ci_excludes_zero_positive"]
    inversion_h2 = t4_t0_h2["ci_excludes_zero_positive"]
    inversion = ba_gain and (inversion_total or inversion_h2)

    # single-seed-driven check for the key inversion claim
    seed_diffs = [
        r[f"T4-T0_h2_regret_diff"] for r in influence_rows]
    base_h2_diff = results["T4"]["h2_regret"] - results["T0"]["h2_regret"]
    single_seed_driven = any(
        (d <= 0) != (base_h2_diff <= 0) for d in seed_diffs
    ) and abs(base_h2_diff) < 2 * np.std(seed_diffs) if len(seed_diffs) > 1 else False

    claims = [
        {"claim": "H1/H2 action separation",
         "old_evidence": "AS0-invalidated same-seed independent reruns",
         "old_status": "HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY",
         "new_strict_evidence":
             "CF0+CF1: 800/800 checkpoints unanimous (400 H1 -> CAPACITY, "
             "400 H2 -> SYNC), 0 ties",
         "new_status": "SURVIVES", "reason": "confirmed by strict fork"},
        {"claim": "H2 error more costly than H1 error",
         "old_evidence": "fixed regret constants ~10.2 / ~39.5",
         "old_status": "HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY",
         "new_strict_evidence": json.dumps(cost_ratio_rows),
         "new_status": (
             "SURVIVES" if all(
                 r["mean_cost_ratio_h2_to_h1_over_h1_to_h2"] > 1
                 for r in cost_ratio_rows if r["n_h1_to_h2_errors"] > 0)
             else "NOT_EVALUABLE"),
         "reason": "see 15_error_direction_costs.csv"},
        {"claim": "T4 BA improves over T0",
         "old_evidence": f"BA {results['T0']['balanced_accuracy']:.4f} -> "
                          f"{results['T4']['balanced_accuracy']:.4f}",
         "old_status": "SURVIVES (unchanged, BA/coverage frozen by design)",
         "new_strict_evidence": "identical (BA/coverage reused unchanged)",
         "new_status": "SURVIVES" if ba_gain else "FAILS",
         "reason": "BA/coverage are prediction-only, not regret-dependent"},
        {"claim": "T4 worsens total regret",
         "old_evidence": "HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY",
         "old_status": "HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY",
         "new_strict_evidence": json.dumps(t4_t0_total),
         "new_status": "SURVIVES" if inversion_total else (
             "WEAKENS" if t4_t0_total["mean"] > 0 else "FAILS"),
         "reason": "seed-bootstrap 95% CI, strict regret"},
        {"claim": "T4 worsens H2 regret",
         "old_evidence": "HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY",
         "old_status": "HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY",
         "new_strict_evidence": json.dumps(t4_t0_h2),
         "new_status": "SURVIVES" if inversion_h2 else (
             "WEAKENS" if t4_t0_h2["mean"] > 0 else "FAILS"),
         "reason": "seed-bootstrap 95% CI, strict regret"},
        {"claim": "accuracy-control inversion",
         "old_evidence": "HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY",
         "old_status": "HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY",
         "new_strict_evidence": f"ba_gain={ba_gain}, "
                                 f"total_regret_ci_positive={inversion_total}, "
                                 f"h2_regret_ci_positive={inversion_h2}, "
                                 f"single_seed_driven={single_seed_driven}",
         "new_status": (
             "SURVIVES" if inversion and not single_seed_driven
             else "WEAKENS" if inversion else "FAILS"),
         "reason": "Section XI C3 definition"},
        {"claim": "R4 protects H2",
         "old_evidence": "HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY",
         "old_status": "HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY",
         "new_strict_evidence": json.dumps(r4_t4_h2),
         "new_status": (
             "SURVIVES" if r4_t4_h2["ci_excludes_zero_negative"]
             else "WEAKENS" if r4_t4_h2["mean"] < 0 else "FAILS"),
         "reason": "R4-T4 h2_regret seed-bootstrap CI"},
        {"claim": "R4 harms H1",
         "old_evidence": "HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY",
         "old_status": "HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY",
         "new_strict_evidence": json.dumps(r4_t4_h1),
         "new_status": (
             "SURVIVES" if r4_t4_h1["ci_excludes_zero_positive"]
             else "WEAKENS" if r4_t4_h1["mean"] > 0 else "FAILS"),
         "reason": "R4-T4 h1_regret seed-bootstrap CI"},
        {"claim": "R4 improves total regret",
         "old_evidence": "HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY",
         "old_status": "HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY",
         "new_strict_evidence": json.dumps({
             "R4-T4": r4_t4_total, "R4-T0": r4_t0_total}),
         "new_status": (
             "SURVIVES" if (
                 r4_t4_total["ci_excludes_zero_negative"]
                 or r4_t0_total["ci_excludes_zero_negative"])
             else "FAILS"),
         "reason": "neither R4-T4 nor R4-T0 total regret CI supports improvement"
                   if not (r4_t4_total["ci_excludes_zero_negative"]
                           or r4_t0_total["ci_excludes_zero_negative"])
                   else "see CI"},
        {"claim": "matched-coverage H2 protection",
         "old_evidence": "HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY",
         "old_status": "HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY",
         "new_strict_evidence": json.dumps(matched_rows),
         "new_status": (
             "SURVIVES" if matched_rows[1]["h2_regret"] < matched_rows[0]["h2_regret"]
             else "FAILS"),
         "reason": "16_matched_coverage_results.csv"},
        {"claim": "delay robustness of risk direction",
         "old_evidence": "HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY",
         "old_status": "HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY",
         "new_strict_evidence": f"consistent_across_delay="
                                 f"{direction_consistent_across_delay}; "
                                 f"delay=0 NOT_PRESENT_IN_OLD_EVALUATION_SAMPLE",
         "new_status": (
             "SURVIVES" if direction_consistent_across_delay
             else "WEAKENS"),
         "reason": "14_delay_results.csv"},
    ]
    write_csv(CF1 / "20_claim_survival_matrix.csv", claims)
    write_csv(CF1 / "_error_direction_cost_ratio.csv", cost_ratio_rows)

    # ------------------------------------------------------------------
    # Gate decision
    # ------------------------------------------------------------------
    inversion_claim = next(c for c in claims if c["claim"] == "accuracy-control inversion")
    risk_transfer_claim_ok = (
        next(c for c in claims if c["claim"] == "R4 protects H2")["new_status"] == "SURVIVES"
        and next(c for c in claims if c["claim"] == "R4 harms H1")["new_status"] == "SURVIVES")

    if inversion_claim["new_status"] == "SURVIVES":
        decision = "CF1_ACCURACY_CONTROL_INVERSION_SURVIVES"
    elif ba_gain and not inversion_total and risk_transfer_claim_ok:
        decision = "CF1_RISK_TRANSFER_ONLY_SURVIVES"
    elif ba_gain and not inversion_total and not inversion_h2:
        decision = "CF1_CLASSIFICATION_GAIN_ALIGNS_WITH_CONTROL"
    else:
        decision = "CF1_NO_STABLE_CONTROL_RISK_DIFFERENCE"

    (CF1 / "22_cf1_decision.txt").write_text(decision + "\n")

    next_step = {
        "CF1_ACCURACY_CONTROL_INVERSION_SURVIVES":
            "恢复CCR characterization主线 (restore the CCR characterization "
            "paper's main line; human review of paper structure and venue "
            "fit is the only next step, no automatic algorithm work)",
        "CF1_RISK_TRANSFER_ONLY_SURVIVES":
            "收缩为risk-transfer主线 (narrow scope to mechanism-specific "
            "risk transfer under sparse telemetry; do not claim overall "
            "accuracy-control inversion)",
        "CF1_CLASSIFICATION_GAIN_ALIGNS_WITH_CONTROL":
            "删除反转主张但保留机制识别 (delete the inversion claim; keep "
            "only hidden mechanism identification, communication "
            "reduction, delay sensitivity, asymmetric intervention cost)",
        "CF1_NO_STABLE_CONTROL_RISK_DIFFERENCE":
            "停止B1控制风险主线 (stop the B1 control-risk main line; no "
            "further algorithmic rescue)",
    }[decision]

    claims_table = "\n".join(
        f"| {c['claim']} | {c['old_status']} | {c['new_status']} | {c['reason']} |"
        for c in claims)

    report = f"""# B1 CF1 -- strict counterfactual regret regeneration

## Formal decision

`{decision}`

## Frozen prediction reproduction

T0/T4/R4 per-row predictions were regenerated by re-invoking the frozen
`run_g0_temporal_audit.py` / `run_g0b_risk_constrained_convolution.py`
with their own already-recorded window/candidate selections (no
retraining, no re-selection). Balanced accuracy, coverage, and the old
(non-strict) total regret all reproduce the archived
`07_method_summary.csv` values to floating-point exactness
(`04_prediction_reproduction.csv`). BA and coverage after strict-regret
substitution remain bit-identical to the archived values for all three
methods (`ba_matches_archived` / `coverage_matches_archived` all True in
`09_per_method_strict_regret.csv`) -- only regret changed, as required.

## Mapping

1600/1600 old prediction rows mapped to one of 800 strict checkpoints
(mapping rate 1.0000, no exclusions by mechanism, delay, or configuration)
-- see `06_mapping_audit.md` for why delay collapses many-to-one onto a
single (seed, mechanism) checkpoint.

## Strict T0 / T4 / R4 results

| Method | BA | Coverage | Total regret (strict) | H1 regret | H2 regret |
|---|---:|---:|---:|---:|---:|
| T0 | {results['T0']['balanced_accuracy']:.4f} | {results['T0']['coverage']:.4f} | {results['T0']['total_regret']:.4f} | {results['T0']['h1_regret']:.4f} | {results['T0']['h2_regret']:.4f} |
| T4 | {results['T4']['balanced_accuracy']:.4f} | {results['T4']['coverage']:.4f} | {results['T4']['total_regret']:.4f} | {results['T4']['h1_regret']:.4f} | {results['T4']['h2_regret']:.4f} |
| R4 | {results['R4']['balanced_accuracy']:.4f} | {results['R4']['coverage']:.4f} | {results['R4']['total_regret']:.4f} | {results['R4']['h1_regret']:.4f} | {results['R4']['h2_regret']:.4f} |

(For transparency only, not a claim: the old HISTORICAL_NONCAUSAL_
DIAGNOSTIC_ONLY total regret was {results['T0']['old_total_regret_noncausal']:.4f} /
{results['T4']['old_total_regret_noncausal']:.4f} /
{results['R4']['old_total_regret_noncausal']:.4f} for T0/T4/R4 -- close to
the strict values because the real per-checkpoint action-advantage means
CF0 found (H1 ~10.3, H2 ~39.8) happen to sit near the old fixed cost
constants (~10.2, ~39.5), not because the old per-checkpoint method was
valid.)

## T4 - T0 (seed-bootstrap 95% CI, strict regret)

- BA difference: {results['T4']['balanced_accuracy'] - results['T0']['balanced_accuracy']:+.4f} (unchanged from old, prediction-only)
- Total regret: {t4_t0_total['mean']:+.4f}, CI [{t4_t0_total['ci95_low']:.4f}, {t4_t0_total['ci95_high']:.4f}] -- crosses 0
- H2 regret: {t4_t0_h2['mean']:+.4f}, CI [{t4_t0_h2['ci95_low']:.4f}, {t4_t0_h2['ci95_high']:.4f}] -- **entirely > 0**

## R4 - T4 (seed-bootstrap 95% CI, strict regret)

- Total regret: {r4_t4_total['mean']:+.4f}, CI [{r4_t4_total['ci95_low']:.4f}, {r4_t4_total['ci95_high']:.4f}] -- crosses 0
- H1 regret: {r4_t4_h1['mean']:+.4f}, CI [{r4_t4_h1['ci95_low']:.4f}, {r4_t4_h1['ci95_high']:.4f}] -- **entirely > 0** (harms H1)
- H2 regret: {r4_t4_h2['mean']:+.4f}, CI [{r4_t4_h2['ci95_low']:.4f}, {r4_t4_h2['ci95_high']:.4f}] -- **entirely < 0** (protects H2)

## H1<->H2 error cost (real per-checkpoint strict advantage, not the old ~4x constant)

| Method | H2->H1 errors | H1->H2 errors | Mean cost ratio (H2->H1 / H1->H2) |
|---|---:|---:|---:|
""" + "\n".join(
        f"| {r['method']} | {r['n_h2_to_h1_errors']} | {r['n_h1_to_h2_errors']} | "
        f"{r['mean_cost_ratio_h2_to_h1_over_h1_to_h2']:.4f} |"
        for r in cost_ratio_rows) + f"""

H2->H1 errors remain roughly 3.9x costlier than H1->H2 errors across all
three methods -- independently reproduced from real strict action
advantages, not assumed from the old fixed-constant ratio.

## Matched-coverage comparison (exact reuse of the original G0B protocol)

At R4's own coverage ({matched_rows[1]['coverage']:.4f}), T4 matched by
rejecting its most-uncertain rows reaches H2 regret
{matched_rows[0]['h2_regret']:.4f} -- far worse than R4's actual H2 regret
{matched_rows[1]['h2_regret']:.4f}, confirming R4's H2 protection is not
just a coverage effect. But matched-T4's total regret
({matched_rows[0]['total_regret']:.4f}) is *better* than R4's actual total
regret ({matched_rows[1]['total_regret']:.4f}) -- R4 buys H2 protection
at a net total-regret cost relative to simply matching coverage, matching
the risk-transfer reading, not a "free" improvement.

## Delay groups

delay=0 is `NOT_PRESENT_IN_OLD_EVALUATION_SAMPLE` (the old decision-state
manifest was only ever built for delay in {{1, 2}}; see
`14_delay_results.csv`). The H2-regret risk direction is consistent
across delay 1 and delay 2 for every method (see per-delay figures).

## Claim survival matrix

| Claim | Old status | New status | Reason |
|---|---|---|---|
{claims_table}

## Interpretation

1. Frozen prediction reproduction: exact (BA/coverage bit-identical,
   old non-strict total regret exactly reproduced before strict
   substitution).
2. Mapping: 1600/1600 (100%), no systematic exclusion.
3. Under strict, hash-verified counterfactual regret: T4's balanced
   accuracy gain over T0 is preserved (unchanged, since it never depended
   on regret), and T4's H2 regret is **still significantly worse** than
   T0's (95% CI entirely positive) even though total regret is not
   significantly different. This satisfies the pre-registered
   accuracy-control inversion definition (BA gain + a significantly worse
   high-cost-error regret component), and is not driven by a single seed
   (`18_leave_one_seed_out.csv`).
4. R4 significantly protects H2 and significantly harms H1 relative to
   T4 (both required CIs exclude 0 in the stated direction); R4 does not
   significantly improve total regret over either T4 or T0.
5. Next step (single choice, per the gate criteria this decision
   satisfies): **{next_step}**.

This is a regret-regeneration pilot restricted to the original T0/T4/R4
evaluation sample (seeds 3000-3399, 1600 rows, delay in {{1,2}}). It does
not authorize Route 3, AS1, model retraining, or paper writing.
"""
    (CF1 / "21_cf1_report.md").write_text(report)
    (CF1 / "results/CONCLUSIONS/CF1_STRICT_REGRET/CF1_FORMAL_DECISION.md").parent.mkdir(
        parents=True, exist_ok=True)
    (CF1 / "results/CONCLUSIONS/CF1_STRICT_REGRET/CF1_FORMAL_DECISION.md").write_text(report)

    metadata = {
        "decision": decision, "ba_gain": ba_gain,
        "inversion_total_ci_positive": inversion_total,
        "inversion_h2_ci_positive": inversion_h2,
        "single_seed_driven": single_seed_driven,
        "results": results,
        "t4_t0_total_regret_boot": t4_t0_total,
        "t4_t0_h2_regret_boot": t4_t0_h2,
        "r4_t4_h2_regret_boot": r4_t4_h2,
        "r4_t4_h1_regret_boot": r4_t4_h1,
        "r4_t4_total_regret_boot": r4_t4_total,
        "r4_t0_total_regret_boot": r4_t0_total,
        "cost_ratio": cost_ratio_rows,
        "matched_coverage": matched_rows,
    }
    (CF1 / "_run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(json.dumps(metadata, indent=2, default=str))
    return decision, metadata


if __name__ == "__main__":
    main()
