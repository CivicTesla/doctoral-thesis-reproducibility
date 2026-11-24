"""Shared analysis for corrected valid-future E22/E23 experiments."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.helpers.rebuttal_experiment_helpers import (
    CALM_COLORS,
    FIXED_QUOTA_POLICIES,
    add_paper_labels,
    bootstrap_ci,
    oracle_audit,
    save_fig,
    setup_plot,
    write_oracle_report,
)


N_AGENTS = 12
HORIZON = 10
N_BOOT = 1000
RNG_SEED = 20260701
MAIN = "rgsa_prefix_risk_diverse_soft"


def _ordinary_bootstrap(arr: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if arr.size <= 1:
        v = float(arr.mean()) if arr.size else float("nan")
        return v, v
    idx = np.arange(arr.size)
    vals = np.array([arr[rng.choice(idx, size=arr.size, replace=True)].mean() for _ in range(N_BOOT)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _stratified_bootstrap(df: pd.DataFrame, delta_col: str, rng: np.random.Generator) -> tuple[float, float]:
    groups = [g[delta_col].to_numpy(dtype=float) for _, g in df.groupby(["scenario", "regime"], sort=True)]
    if not groups:
        return float("nan"), float("nan")
    vals = []
    for _ in range(N_BOOT):
        vals.append(float(np.mean([rng.choice(arr, size=len(arr), replace=True).mean() for arr in groups])))
    vals = np.asarray(vals, dtype=float)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _wilcoxon(deltas: np.ndarray) -> dict:
    vals = np.asarray(deltas, dtype=float)
    vals = vals[np.abs(vals) > 1e-12]
    n = int(len(vals))
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n_nonzero": 0}
    abs_vals = np.abs(vals)
    order = np.argsort(abs_vals)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs_vals[order[j]] == abs_vals[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    stat = min(float(ranks[vals > 0].sum()), float(ranks[vals < 0].sum()))
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    z = (stat - mean_w + 0.5) / math.sqrt(var_w)
    return {"statistic": stat, "p_value": float(math.erfc(abs(z) / math.sqrt(2.0))), "n_nonzero": n}


def _policy_summary(df: pd.DataFrame, out: Path, prefix: str) -> pd.DataFrame:
    rows = []
    optional = [
        "global_field_loss", "hidden_congestion_score", "ghost_pressure_score",
        "mean_route_decision_regret", "amortized_runtime_per_tick_ms",
        "admitted_same_agent_concentration", "admitted_same_agent_concentration_p90",
        "admitted_unique_agents_per_tick_mean", "live_queue_size_raw",
        "live_queue_size_valid_for_admission", "candidate_count_before_valid_filter",
        "candidate_count_after_valid_filter", "filtered_nonpositive_remaining_horizon_fraction",
    ]
    for keys, sub in df.groupby(["budget_per_tick", "admission_ratio", "policy", "paper_label"], sort=True):
        row = dict(zip(["budget_per_tick", "admission_ratio", "policy", "paper_label"], keys))
        vals = sub["average_route_cost"].astype(float)
        row.update({
            "paired_units": int(len(sub)),
            "mean_route_cost": float(vals.mean()),
            "median_route_cost": float(vals.median()),
            "route_cost_ci95_low": bootstrap_ci(vals, seed=RNG_SEED)[0],
            "route_cost_ci95_high": bootstrap_ci(vals, seed=RNG_SEED)[1],
        })
        for col in optional:
            if col in sub.columns:
                row[f"mean_{col}"] = float(sub[col].mean())
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["budget_per_tick", "mean_route_cost"])
    summary.to_csv(out / f"{prefix}_policy_summary.csv", index=False)
    return summary


def _best_fq(summary: pd.DataFrame, out: Path, prefix: str, budget_sweep: bool) -> pd.DataFrame:
    fq = summary[summary["policy"].isin(FIXED_QUOTA_POLICIES)].copy()
    rows = []
    for b, sub in fq.groupby("budget_per_tick", sort=True):
        best = sub.sort_values("mean_route_cost").iloc[0]
        rows.append({
            "budget_per_tick": int(b),
            "admission_ratio": float(best["admission_ratio"]),
            "best_fixed_quota_policy": best["policy"],
            "best_fixed_quota_label": best["paper_label"],
            "best_fixed_quota_mean_route_cost": float(best["mean_route_cost"]),
        })
    best_df = pd.DataFrame(rows)
    name = f"{prefix}_best_fixed_quota_by_B.csv" if budget_sweep else f"{prefix}_best_fixed_quota.csv"
    best_df.to_csv(out / name, index=False)
    return best_df


def _comparison_list(budget_sweep: bool) -> list[tuple[str, str, str]]:
    if budget_sweep:
        return [
            (MAIN, "aoi_priority_scheduler", "RGSA-prefix-diverse - AoIOnly"),
            (MAIN, "__BEST_FQ__", "RGSA-prefix-diverse - global best FQ"),
            (MAIN, "rgsa_current", "RGSA-prefix-diverse - RGSA-current"),
            ("rgsa_current", "aoi_priority_scheduler", "RGSA-current - AoIOnly"),
        ]
    return [
        (MAIN, "aoi_priority_scheduler", "RGSA-prefix-diverse - AoIOnly"),
        (MAIN, "__BEST_FQ__", "RGSA-prefix-diverse - global best FQ"),
        (MAIN, "rgsa_current", "RGSA-prefix-diverse - RGSA-current"),
        (MAIN, "decision_critical_scheduler", "RGSA-prefix-diverse - DecisionCritical"),
        (MAIN, "random_budget", "RGSA-prefix-diverse - Random"),
        ("rgsa_prefix_risk", "rgsa_current", "RGSA-prefix-risk - RGSA-current"),
        (MAIN, "rgsa_prefix_risk", "RGSA-prefix-diverse - RGSA-prefix-risk"),
    ]


def _paired(df: pd.DataFrame, best_fq: pd.DataFrame, out: Path, prefix: str, budget_sweep: bool) -> pd.DataFrame:
    piv = df.pivot_table(
        index=["scenario", "regime", "budget_per_tick", "seed"],
        columns="policy",
        values="average_route_cost",
        aggfunc="mean",
    ).reset_index()
    piv = piv.merge(best_fq[["budget_per_tick", "best_fixed_quota_policy", "best_fixed_quota_label"]],
                    on="budget_per_tick", how="left")
    rows = []
    rng = np.random.default_rng(RNG_SEED)
    for b, sub in piv.groupby("budget_per_tick", sort=True):
        best_policy = str(sub["best_fixed_quota_policy"].iloc[0])
        for first, second, label in _comparison_list(budget_sweep):
            actual_second = best_policy if second == "__BEST_FQ__" else second
            if first not in sub.columns or actual_second not in sub.columns:
                continue
            pair = sub.dropna(subset=[first, actual_second]).copy()
            if pair.empty:
                continue
            pair["delta"] = pair[first] - pair[actual_second]
            d = pair["delta"].to_numpy(dtype=float)
            olo, ohi = _ordinary_bootstrap(d, rng)
            slo, shi = _stratified_bootstrap(pair, "delta", rng)
            wil = _wilcoxon(d)
            rows.append({
                "budget_per_tick": int(b),
                "admission_ratio": float(b) / (N_AGENTS * HORIZON),
                "first_policy": first,
                "second_policy": actual_second,
                "comparison_label": label,
                "paired_units": int(len(pair)),
                "mean_paired_delta": float(d.mean()),
                "median_paired_delta": float(np.median(d)),
                "paired_win_rate": float((d < 0).mean()),
                "ordinary_ci_low": olo,
                "ordinary_ci_high": ohi,
                "stratified_ci_low": slo,
                "stratified_ci_high": shi,
                "wilcoxon_statistic": float(wil["statistic"]),
                "wilcoxon_p_value": float(wil["p_value"]),
                "support_flag": bool(ohi < 0 and shi < 0 and wil["p_value"] < 0.05),
                "best_fixed_quota_policy": best_policy if second == "__BEST_FQ__" else "",
                "best_fixed_quota_label": str(sub["best_fixed_quota_label"].iloc[0]) if second == "__BEST_FQ__" else "",
            })
    pairwise = pd.DataFrame(rows)
    pairwise.to_csv(out / f"{prefix}_pairwise_deltas.csv", index=False)
    return pairwise


def _diagnostic_tables(df: pd.DataFrame, out: Path, prefix: str, budget_sweep: bool) -> None:
    cell_cols = ["scenario", "regime", "budget_per_tick", "admission_ratio", "policy", "paper_label"]
    by_cell = df.groupby(cell_cols, sort=True).agg(
        paired_units=("seed", "nunique"),
        mean_route_cost=("average_route_cost", "mean"),
        mean_runtime_ms_tick=("amortized_runtime_per_tick_ms", "mean"),
        mean_hidden_congestion=("hidden_congestion_score", "mean"),
        mean_ghost_pressure=("ghost_pressure_score", "mean"),
        mean_field_loss=("global_field_loss", "mean"),
    ).reset_index()
    by_cell_name = f"{prefix}_by_scenario_regime_budget.csv" if budget_sweep else f"{prefix}_by_scenario_regime.csv"
    by_cell.to_csv(out / by_cell_name, index=False)

    grp = ["budget_per_tick", "admission_ratio", "policy", "paper_label"]
    qcols = [
        "live_queue_size_raw", "live_queue_size_valid_for_admission", "live_queue_size_mean",
        "candidate_count_before_valid_filter", "candidate_count_after_valid_filter",
        "filtered_nonpositive_remaining_horizon_count", "filtered_nonpositive_remaining_horizon_fraction",
        "admitted_points_per_tick_mean",
    ]
    df.groupby(grp, sort=True)[[c for c in qcols if c in df.columns]].mean().reset_index().to_csv(
        out / f"{prefix}_queue_diagnostics.csv", index=False)
    acols = [
        "admitted_unique_agents_per_tick_mean", "admitted_same_agent_concentration",
        "admitted_same_agent_concentration_p90", "fraction_ticks_single_agent_monopoly",
    ]
    df.groupby(grp, sort=True)[[c for c in acols if c in df.columns]].mean().reset_index().to_csv(
        out / f"{prefix}_agent_concentration.csv", index=False)
    rcols = [c for c in df.columns if c.startswith("admitted_remaining_horizon_fraction_")]
    rem_rows = []
    for col in rcols:
        label = col.replace("admitted_remaining_horizon_fraction_", "")
        rem = -int(label[1:]) if label.startswith("m") else int(label)
        rem_rows.append(df[["budget_per_tick", "admission_ratio", "policy", "paper_label", col]].assign(
            remaining_horizon=rem,
            admitted_fraction=df[col].astype(float),
        )[["budget_per_tick", "admission_ratio", "policy", "paper_label", "remaining_horizon", "admitted_fraction"]])
    rem = pd.concat(rem_rows, ignore_index=True) if rem_rows else pd.DataFrame()
    if not rem.empty:
        rem = rem.groupby(["budget_per_tick", "admission_ratio", "policy", "paper_label", "remaining_horizon"], as_index=False).agg(
            mean_admitted_fraction=("admitted_fraction", "mean"))
    rem.to_csv(out / f"{prefix}_remaining_horizon_distribution.csv", index=False)
    runtime_cols = ["budget_per_tick", "admission_ratio", "policy", "paper_label", "amortized_runtime_per_tick_ms", "runtime_per_decision_ms_mean"]
    df[[c for c in runtime_cols if c in df.columns]].groupby(
        ["budget_per_tick", "admission_ratio", "policy", "paper_label"], as_index=False
    ).mean().to_csv(out / f"{prefix}_runtime.csv", index=False)


def _invariant_report(df: pd.DataFrame, out: Path, prefix: str) -> None:
    admitted_ok = bool((df["fraction_admitted_remaining_horizon_le_0"].astype(float) == 0.0).all())
    field_ok = bool((df["fraction_field_updates_remaining_horizon_le_0"].astype(float) == 0.0).all())
    min_ok = bool((df["admitted_remaining_horizon_min"].astype(float) >= 1.0).all())
    sim_ok = bool(df["valid_future_invariant_pass"].astype(bool).all())
    status = "PASS" if admitted_ok and field_ok and min_ok and sim_ok else "FAIL"
    lines = [
        f"# {prefix.upper()} invariant check",
        "",
        f"1. Verdict: **{status}**",
        f"2. Can corrected results be trusted? {'yes' if status == 'PASS' else 'no'}",
        f"3. Did all future-validity invariants pass? {'yes' if status == 'PASS' else 'no'}",
        f"4. Is a larger rerun recommended? {'yes' if status == 'PASS' else 'no'}",
        "5. Main finding: All admitted and field-updating points have positive remaining horizon." if status == "PASS" else "5. Main finding: At least one future-validity invariant failed.",
        "",
        "| policy | B | admitted <=0 | field updates <=0 | min remaining | invariant pass |",
        "|---|---:|---:|---:|---:|---|",
    ]
    grp = df.groupby(["policy", "budget_per_tick"], sort=True).agg(
        admitted_le0=("fraction_admitted_remaining_horizon_le_0", "mean"),
        field_le0=("fraction_field_updates_remaining_horizon_le_0", "mean"),
        min_remaining=("admitted_remaining_horizon_min", "min"),
        pass_flag=("valid_future_invariant_pass", "min"),
    ).reset_index()
    for _, r in grp.iterrows():
        lines.append(f"| {r['policy']} | {int(r['budget_per_tick'])} | {r['admitted_le0']:.6f} | {r['field_le0']:.6f} | {r['min_remaining']:.1f} | {bool(r['pass_flag'])} |")
    (out / f"{prefix}_invariant_check_report.md").write_text("\n".join(lines), encoding="utf-8")


def _plots(summary: pd.DataFrame, pairwise: pd.DataFrame, out: Path, prefix: str, budget_sweep: bool) -> None:
    setup_plot()
    labels = ["AoI", "Random", "DecisionCritical", "FQ-AoI", "FQ-Crit", "FQ-Cov", "FQ-Ghost", "FQ-Bal", "RGSA current", "RGSA prefix", "RGSA prefix diverse"]
    if budget_sweep:
        fig, ax = plt.subplots(figsize=(5.6, 3.1))
        for label in labels:
            sub = summary[summary["paper_label"] == label].sort_values("budget_per_tick")
            if len(sub):
                ax.plot(sub["budget_per_tick"], sub["mean_route_cost"], marker="o", label=label, color=CALM_COLORS.get(label))
        ax.set_xlabel("B")
        ax.set_ylabel("Route cost")
        ax.legend(frameon=False, ncol=3)
        save_fig(fig, out, f"{prefix}_route_cost_vs_budget")
        for comp, stem in [
            ("RGSA-prefix-diverse - AoIOnly", "delta_vs_aoi"),
            ("RGSA-prefix-diverse - global best FQ", "delta_vs_best_fq"),
            ("RGSA-prefix-diverse - RGSA-current", "delta_vs_current_rgsa"),
        ]:
            fig, ax = plt.subplots(figsize=(4.6, 2.8))
            sub = pairwise[pairwise["comparison_label"] == comp].sort_values("budget_per_tick")
            ax.plot(sub["budget_per_tick"], sub["mean_paired_delta"], marker="o")
            ax.axhline(0, color="#444444", lw=0.8)
            ax.set_xlabel("B")
            ax.set_ylabel("Route cost delta")
            save_fig(fig, out, f"{prefix}_{stem}")
    else:
        fig, ax = plt.subplots(figsize=(5.8, 3.2))
        sub = summary[summary["budget_per_tick"] == summary["budget_per_tick"].min()].copy()
        sub = sub.sort_values("mean_route_cost")
        ax.bar(sub["paper_label"], sub["mean_route_cost"], color=[CALM_COLORS.get(x, "#999999") for x in sub["paper_label"]])
        ax.set_ylabel("Route cost")
        ax.tick_params(axis="x", rotation=35)
        save_fig(fig, out, f"{prefix}_route_cost_bar")
        fig, ax = plt.subplots(figsize=(5.8, 3.2))
        p = pairwise.sort_values("mean_paired_delta")
        ax.barh(p["comparison_label"], p["mean_paired_delta"], color="#4e79a7")
        ax.axvline(0, color="#444444", lw=0.8)
        ax.set_xlabel("Route cost delta")
        save_fig(fig, out, f"{prefix}_pairwise_delta_bar")

    agent = pd.read_csv(out / f"{prefix}_agent_concentration.csv")
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for label in ["AoI", "RGSA current", "RGSA prefix diverse"]:
        sub = agent[agent["paper_label"] == label].sort_values("budget_per_tick")
        if len(sub):
            x = sub["budget_per_tick"] if budget_sweep else np.arange(len(sub))
            ax.plot(x, sub["admitted_same_agent_concentration"], marker="o", label=label)
    ax.set_xlabel("B" if budget_sweep else "Policy group")
    ax.set_ylabel("Same-agent concentration")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    save_fig(fig, out, f"{prefix}_same_agent_concentration" + ("_vs_budget" if budget_sweep else ""))

    runtime = pd.read_csv(out / f"{prefix}_runtime.csv")
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for label in ["RGSA current", "RGSA prefix diverse"]:
        sub = runtime[runtime["paper_label"] == label].sort_values("budget_per_tick")
        if len(sub):
            ax.plot(sub["budget_per_tick"], sub["amortized_runtime_per_tick_ms"], marker="o", label=label)
    ax.set_xlabel("B")
    ax.set_ylabel("ms/tick")
    ax.legend(frameon=False)
    save_fig(fig, out, f"{prefix}_runtime" + ("_vs_budget" if budget_sweep else ""))

    rem = pd.read_csv(out / f"{prefix}_remaining_horizon_distribution.csv")
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for label in ["AoI", "RGSA current", "RGSA prefix diverse"]:
        sub = rem[rem["paper_label"] == label].groupby("remaining_horizon")["mean_admitted_fraction"].mean().reset_index()
        if len(sub):
            ax.plot(sub["remaining_horizon"], sub["mean_admitted_fraction"], marker="o", label=label)
    ax.axvline(0, color="#444444", lw=0.8)
    ax.set_xlabel("Remaining horizon")
    ax.set_ylabel("Admitted fraction")
    ax.legend(frameon=False)
    save_fig(fig, out, f"{prefix}_remaining_horizon_distribution")


def _reports(summary: pd.DataFrame, pairwise: pd.DataFrame, best_fq: pd.DataFrame,
             out: Path, prefix: str, budget_sweep: bool) -> None:
    if prefix == "e23b":
        report_name = "e23b_boundary_budget_report.md"
    else:
        report_name = f"{prefix}_budget_sweep_report.md" if budget_sweep else f"{prefix}_final_validation_report.md"
    title = "budget sweep" if budget_sweep else "final validation"
    lines = [f"# {prefix.upper()} valid-future {title} report", ""]
    lines.append("All reported comparisons use `--valid-future-only`; negative deltas favor the first policy.")
    lines.append("")
    lines.append("## Best fixed quota")
    for _, r in best_fq.iterrows():
        lines.append(f"- B={int(r['budget_per_tick'])}: {r['best_fixed_quota_label']} mean route cost {r['best_fixed_quota_mean_route_cost']:.6f}.")
    lines.append("")
    lines.append("## Pairwise comparisons")
    for _, r in pairwise.iterrows():
        lines.append(
            f"- B={int(r['budget_per_tick'])} `{r['comparison_label']}`: "
            f"mean delta {r['mean_paired_delta']:.6f}, median {r['median_paired_delta']:.6f}, "
            f"win rate {r['paired_win_rate']:.3f}, ordinary CI [{r['ordinary_ci_low']:.6f}, {r['ordinary_ci_high']:.6f}], "
            f"stratified CI [{r['stratified_ci_low']:.6f}, {r['stratified_ci_high']:.6f}], "
            f"Wilcoxon p={r['wilcoxon_p_value']:.4g}, support={r['support_flag']}."
        )
    (out / report_name).write_text("\n".join(lines), encoding="utf-8")

    paper = [
        f"# {prefix.upper()} paper text suggestions",
        "",
        "## Methods",
        "We reran the final validation with valid-future admission: a candidate future-intention point is admissible only when its target tick is strictly greater than the current tick. This keeps B as a point-level semantic future-intention budget while preventing stale points from entering the admitted set or updating the communicated field.",
        "",
        "## Results",
    ]
    for _, r in pairwise.iterrows():
        paper.append(f"- B={int(r['budget_per_tick'])} `{r['comparison_label']}` mean delta {r['mean_paired_delta']:.6f}, win rate {r['paired_win_rate']:.3f}, support={r['support_flag']}.")
    paper.extend(["", "## Discussion"])
    if not budget_sweep:
        key = pairwise[pairwise["comparison_label"].isin([
            "RGSA-prefix-diverse - AoIOnly",
            "RGSA-prefix-diverse - global best FQ",
            "RGSA-prefix-diverse - RGSA-current",
        ])]
        if len(key) == 3 and key["support_flag"].astype(bool).all():
            paper.append("The corrected validation supports replacing current RGSA with the prefix-risk diverse variant as the main method.")
        elif len(key) and (key["comparison_label"].eq("RGSA-prefix-diverse - AoIOnly") & key["support_flag"].astype(bool)).any():
            paper.append("The corrected validation supports prefix-risk diverse as a stronger adaptive variant against AoIOnly, but not as a universally dominant method over all baselines.")
        else:
            paper.append("The corrected validation does not support using prefix-risk diverse as the main method under these criteria.")
    else:
        paper.append("The budget sweep should be interpreted as scarce valid future-intention admission if gains are concentrated at lower B values.")
    (out / f"{prefix}_paper_text.md").write_text("\n".join(paper), encoding="utf-8")


def analyze_valid_future(input_dir: Path, prefix: str, budget_sweep: bool) -> None:
    out = input_dir / "final_analysis"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_dir / f"{prefix}_by_seed.csv")
    df = add_paper_labels(df)
    if "admission_ratio" not in df.columns:
        df["admission_ratio"] = df["budget_per_tick"].astype(float) / (N_AGENTS * HORIZON)
    summary = _policy_summary(df, out, prefix)
    best = _best_fq(summary, out, prefix, budget_sweep)
    pairwise = _paired(df, best, out, prefix, budget_sweep)
    _diagnostic_tables(df, out, prefix, budget_sweep)
    audit = oracle_audit(df, {
        "valid_future_only_rows": bool(df.get("valid_future_only", pd.Series(False, index=df.index)).astype(bool).all()),
        "admitted_remaining_horizon_positive": bool((df["fraction_admitted_remaining_horizon_le_0"].astype(float) == 0.0).all()),
        "field_updates_remaining_horizon_positive": bool((df["fraction_field_updates_remaining_horizon_le_0"].astype(float) == 0.0).all()),
        "sim_invariant_pass": bool(df["valid_future_invariant_pass"].astype(bool).all()),
    })
    write_oracle_report(audit, out, f"{prefix}_oracle_check_report.md", f"{prefix.upper()} oracle check")
    _invariant_report(df, out, prefix)
    _plots(summary, pairwise, out, prefix, budget_sweep)
    _reports(summary, pairwise, best, out, prefix, budget_sweep)
    print(f"{prefix.upper()} analysis -> {out.resolve()}")
