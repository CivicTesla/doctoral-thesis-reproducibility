"""Shared analysis helpers for E12 staged analysis scripts."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

from a2field.e12_staged_sim import DISPLAY, ACTION_NAMES

PALETTE = {
    "Random":                   "#aaaaaa",
    "AoIOnly":                  "#4c72b0",
    "DecisionCritical":         "#dd8452",
    "FixedQuota_AoIHeavy":      "#8c564b",
    "FixedQuota_CriticalHeavy": "#e377c2",
    "FixedQuota_CoverageHeavy": "#7f7f7f",
    "FixedQuota_Balanced":      "#bcbd22",
    "FixedQuota_GhostSafe":     "#17becf",
    "ContextGatedRule":         "#55a868",
    "MCRolloutGated":           "#c44e52",
    "MC_H3_K4_E5":              "#c44e52",
    "MC_H3_K4_E10":             "#e86b6b",
    "MC_H5_K4_E5":              "#a83232",
    "MC_H5_K8_E5":              "#7a1f1f",
    "MC_H2_K4_E5":              "#f08080",
    "MC_H3_K2_E5":              "#cd5c5c",
    "MC_RouteOnly":             "#2ca02c",
    "MC_RegretBalanced":        "#1f77b4",
    "MC_RegretHeavy":           "#9467bd",
    "MC_GhostRobust":           "#ff7f0e",
    "Adaptive":                 "#8172b2",
    "PeriodicFull":             "#937860",
}


def _color(policy: str) -> str:
    d = DISPLAY.get(policy, policy)
    return PALETTE.get(d, "#999999")


def fig_route_cost_summary(
    df: pd.DataFrame, out: Path, prefix: str, title: str = "",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, rg in zip(axes, ("normal", "ghost_stress")):
        sub = df[df["regime"] == rg]
        m   = sub.groupby("policy")["average_route_cost"].mean().reset_index()
        m["display"] = m["policy"].map(lambda p: DISPLAY.get(p, p))
        m   = m.sort_values("average_route_cost")
        colors = [_color(p) for p in m["policy"]]
        ax.barh(m["display"], m["average_route_cost"], color=colors)
        ax.set_xlabel("Mean Route Cost")
        ax.set_title(f"Regime: {rg}")
        for _, row in m.iterrows():
            ax.text(row["average_route_cost"], row["display"],
                    f"  {row['average_route_cost']:.4f}", va="center", fontsize=7)
    fig.suptitle(f"{prefix} Route Cost by Policy{title}")
    fig.tight_layout()
    _save(fig, out, f"{prefix}_fig_policy_route_cost_summary")


def fig_decision_regret(df: pd.DataFrame, out: Path, prefix: str) -> None:
    policies = list(df["policy"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, rg in zip(axes, ("normal", "ghost_stress")):
        sub    = df[df["regime"] == rg]
        data   = [sub[sub["policy"] == p]["mean_route_decision_regret"].dropna().values
                   for p in policies]
        labels = [DISPLAY.get(p, p) for p in policies]
        colors = [_color(p) for p in policies]
        bp     = ax.boxplot(data, labels=labels, patch_artist=True, vert=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("Decision Regret")
        ax.set_title(f"Regime: {rg}")
    fig.suptitle(f"{prefix} Decision Regret by Policy")
    fig.tight_layout()
    _save(fig, out, f"{prefix}_fig_decision_regret_by_policy")


def fig_best_by_case_matrix(
    df: pd.DataFrame, scenarios: list[str], out: Path, prefix: str,
) -> None:
    policies = sorted(df["policy"].unique())
    rows_lb  = [f"{sc[:10]}|{rg[:6]}" for sc in scenarios
                for rg in ("normal", "ghost_stress")]
    wins = np.zeros((len(policies), len(rows_lb)), dtype=int)
    for j, (sc, rg) in enumerate([
        (sc, rg) for sc in scenarios for rg in ("normal", "ghost_stress")
    ]):
        sub = df[(df["scenario"] == sc) & (df["regime"] == rg)]
        if sub.empty:
            continue
        m   = sub.groupby("policy")["average_route_cost"].mean()
        bi  = m.idxmin()
        for i, po in enumerate(policies):
            if po == bi:
                wins[i, j] = 1
    fig, ax = plt.subplots(figsize=(max(8, len(rows_lb) * 0.8 + 2), max(4, len(policies) * 0.5 + 1)))
    ax.imshow(wins, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(rows_lb)))
    ax.set_yticks(range(len(policies)))
    ax.set_xticklabels(rows_lb, rotation=40, ha="right", fontsize=7)
    ax.set_yticklabels([DISPLAY.get(p, p) for p in policies], fontsize=7)
    ax.set_title(f"{prefix} Best-by-Case Winner Matrix")
    fig.tight_layout()
    _save(fig, out, f"{prefix}_fig_best_by_case_matrix")


def fig_ghost_stress_robustness(df: pd.DataFrame, out: Path, prefix: str) -> None:
    m = df.groupby(["policy", "regime"])["average_route_cost"].mean().unstack("regime")
    if "normal" not in m.columns or "ghost_stress" not in m.columns:
        return
    m["delta"] = m["ghost_stress"] - m["normal"]
    m["display"] = [DISPLAY.get(p, p) for p in m.index]
    m = m.sort_values("delta")
    fig, ax = plt.subplots(figsize=(8, max(3, len(m) * 0.45)))
    colors  = [_color(p) for p in m.index]
    ax.barh(m["display"], m["delta"], color=colors)
    ax.axvline(0, c="k", lw=0.8)
    ax.set_xlabel("Route cost delta (ghost_stress − normal)")
    ax.set_title(f"{prefix} Ghost-Stress Robustness")
    fig.tight_layout()
    _save(fig, out, f"{prefix}_fig_ghost_stress_robustness")


def fig_action_distribution(df: pd.DataFrame, out: Path, prefix: str) -> None:
    count_cols = [f"mc_action_count_{a}" for a in ACTION_NAMES
                  if f"mc_action_count_{a}" in df.columns]
    if not count_cols:
        return
    mc_pols = [p for p in df["policy"].unique()
               if any(df[df["policy"] == p][count_cols].sum(axis=1).sum() > 0)]
    if not mc_pols:
        return
    fig, axes = plt.subplots(len(mc_pols), 2,
                              figsize=(12, max(4, len(mc_pols) * 3)),
                              sharey=False, squeeze=False)
    cmap = plt.get_cmap("tab10")
    act_colors = [cmap(i) for i in range(len(ACTION_NAMES))]
    for row_i, po in enumerate(mc_pols):
        sub_po = df[df["policy"] == po]
        for col_i, rg in enumerate(("normal", "ghost_stress")):
            ax  = axes[row_i][col_i]
            sub = sub_po[sub_po["regime"] == rg]
            if sub.empty:
                ax.set_visible(False)
                continue
            grp    = sub.groupby("scenario")[count_cols].mean()
            totals = grp.sum(axis=1).replace(0, 1)
            grp    = grp.div(totals, axis=0)
            bottom = np.zeros(len(grp))
            for k, col in enumerate(count_cols):
                aname = col.replace("mc_action_count_", "")
                ax.bar(range(len(grp)), grp[col].values, bottom=bottom,
                       color=act_colors[k], label=aname, width=0.6)
                bottom += grp[col].values
            ax.set_xticks(range(len(grp)))
            ax.set_xticklabels(grp.index, rotation=30, ha="right", fontsize=6)
            ax.set_ylim(0, 1)
            ax.set_title(f"{DISPLAY.get(po, po)} — {rg}", fontsize=8)
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=6, loc="upper right")
    fig.suptitle(f"{prefix} MC Action Distribution by Scenario")
    fig.tight_layout()
    _save(fig, out, f"{prefix}_fig_action_distribution")


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    for ext in (".pdf", ".png"):
        fig.savefig(out / (name + ext), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {name}.pdf/.png")


def go_no_go_answers(
    df: pd.DataFrame,
    questions: list[tuple[str, str, bool, str]],  # (tag, question, passes, detail)
) -> tuple[int, int, list[tuple]]:
    go  = sum(1 for _, _, p, _ in questions if p)
    ngo = sum(1 for _, _, p, _ in questions if not p)
    return go, ngo, questions


def verdict_text(go: int, total: int, threshold_strong: int, threshold_go: int) -> str:
    if go >= threshold_strong:
        return "**VERDICT: STRONG GO**"
    elif go >= threshold_go:
        return "**VERDICT: CONDITIONAL GO**"
    else:
        return "**VERDICT: NO GO**"


def bootstrap_ci(
    df: pd.DataFrame,
    policy_a: str,
    policy_b: str,
    metric: str = "average_route_cost",
    n_boot: int = 1000,
    ci_level: float = 0.95,
) -> dict[str, float]:
    """Bootstrap CI for mean(A) - mean(B)."""
    a = df[df["policy"] == policy_a][metric].dropna().values
    b = df[df["policy"] == policy_b][metric].dropna().values
    if len(a) < 3 or len(b) < 3:
        return {"mean_diff": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    rng    = np.random.default_rng(0)
    diffs  = []
    for _ in range(n_boot):
        da = rng.choice(a, size=len(a), replace=True).mean()
        db = rng.choice(b, size=len(b), replace=True).mean()
        diffs.append(da - db)
    diffs = np.array(diffs)
    alpha  = (1 - ci_level) / 2
    return {
        "mean_diff": float(a.mean() - b.mean()),
        "ci_lo":     float(np.percentile(diffs, alpha * 100)),
        "ci_hi":     float(np.percentile(diffs, (1 - alpha) * 100)),
    }


def win_rate(
    df: pd.DataFrame,
    policy_a: str,
    policy_b: str,
    metric: str = "average_route_cost",
    group_cols: list[str] | None = None,
) -> float:
    """Fraction of (scenario, regime) cases where mean(A) < mean(B)."""
    if group_cols is None:
        group_cols = ["scenario", "regime"]
    a_grp = df[df["policy"] == policy_a].groupby(group_cols)[metric].mean()
    b_grp = df[df["policy"] == policy_b].groupby(group_cols)[metric].mean()
    common = a_grp.index.intersection(b_grp.index)
    if len(common) == 0:
        return float("nan")
    return float((a_grp[common] < b_grp[common]).mean())


def interpretation_notes(
    df: pd.DataFrame,
    scenarios: list[str],
    mc_policy:  str,
    out: Path,
    prefix: str,
    extra_context: str = "",
) -> None:
    """Write paper-facing interpretation_notes.md answering 10 standard questions."""
    aoi = "aoi_priority_scheduler"
    dc  = "decision_critical_scheduler"
    fq  = "fixed_quota_aoi_heavy"

    def _rc(p: str) -> float:
        sub = df[df["policy"] == p]
        return float(sub["average_route_cost"].mean()) if not sub.empty else float("nan")

    def _rg(p: str) -> float:
        sub = df[df["policy"] == p]
        return float(sub["mean_route_decision_regret"].mean()) if not sub.empty else float("nan")

    def _delta_gs(p: str) -> float:
        n_  = df[(df["policy"] == p) & (df["regime"] == "normal")]["average_route_cost"].mean()
        gs_ = df[(df["policy"] == p) & (df["regime"] == "ghost_stress")]["average_route_cost"].mean()
        return float(gs_ - n_) if (not np.isnan(n_) and not np.isnan(gs_)) else float("nan")

    mc_rc  = _rc(mc_policy)
    aoi_rc = _rc(aoi)
    dc_rc  = _rc(dc)
    fq_rc  = _rc(fq)
    mc_rg  = _rg(mc_policy)
    aoi_rg = _rg(aoi)

    count_cols = [f"mc_action_count_{a}" for a in ACTION_NAMES
                  if f"mc_action_count_{a}" in df.columns]
    mc_sub = df[df["policy"] == mc_policy]
    if count_cols and not mc_sub.empty:
        ac = mc_sub[count_cols].mean()
        top_action = ac.idxmax().replace("mc_action_count_", "")
        top_frac   = ac.max() / ac.sum() if ac.sum() > 0 else 0.0
    else:
        top_action = "unknown"
        top_frac   = float("nan")

    rt = float(mc_sub["amortized_runtime_per_tick_ms"].mean()) if (
        not mc_sub.empty and "amortized_runtime_per_tick_ms" in mc_sub.columns) else float("nan")

    lines = [
        f"# {prefix} Interpretation Notes\n\n",
        f"MC policy evaluated: `{mc_policy}` ({DISPLAY.get(mc_policy, mc_policy)})\n\n",
    ]
    if extra_context:
        lines.append(f"{extra_context}\n\n")

    def _ans(q: str, a: str) -> str:
        return f"**{q}**\n{a}\n\n"

    lines.append(_ans(
        "Q1 Is MC just AoIHeavy?",
        f"MC selects `{top_action}` most often ({top_frac:.1%} of ticks). "
        + (f"MC route cost {mc_rc:.5f} vs FixedQuota_AoIHeavy {fq_rc:.5f} — "
           + ("MC adds value beyond fixed AoI." if mc_rc < fq_rc else "MC does NOT beat fixed AoI.")
           if not math.isnan(fq_rc) else "FixedQuota_AoIHeavy not available for comparison.")
    ))
    lines.append(_ans(
        "Q2 Does MC beat fixed-quota controls?",
        f"MC route cost {mc_rc:.5f}. "
        + f"FixedQuota_AoIHeavy {fq_rc:.5f}. "
        + ("MC beats best fixed quota." if mc_rc < fq_rc else "MC does not beat best fixed quota.")
        if not math.isnan(fq_rc) else "Fixed quota not in this stage."
    ))
    lines.append(_ans(
        "Q3 Does MC beat AoIOnly?",
        f"MC {mc_rc:.5f} vs AoI {aoi_rc:.5f} — "
        + (f"delta={aoi_rc - mc_rc:.5f} " + ("YES" if mc_rc < aoi_rc else "NO"))
        if not math.isnan(aoi_rc) else "AoI not available."
    ))
    lines.append(_ans(
        "Q4 Does MC beat DecisionCritical?",
        f"MC {mc_rc:.5f} vs DC {dc_rc:.5f} — "
        + (f"delta={dc_rc - mc_rc:.5f} " + ("YES" if mc_rc < dc_rc else "NO"))
        if not math.isnan(dc_rc) else "DC not available."
    ))
    lines.append(_ans(
        "Q5 Does MC improve route cost but hurt regret?",
        f"MC regret {mc_rg:.5f} vs AoI regret {aoi_rg:.5f}. "
        + ("MC has lower regret." if mc_rg < aoi_rg else "MC has higher regret — tradeoff.")
        if not math.isnan(aoi_rg) else "AoI not available."
    ))
    lines.append(_ans(
        "Q6 Does MC adapt differently under ghost stress?",
        f"MC ghost_stress delta = {_delta_gs(mc_policy):.5f}. "
        f"AoI ghost_stress delta = {_delta_gs(aoi):.5f}."
        if not math.isnan(_delta_gs(aoi)) else "N/A"
    ))
    lines.append(_ans(
        "Q7 Which action does MC choose most and why?",
        f"Most selected action: {top_action} ({top_frac:.1%}). "
        "Review ctx_mean_stale_score and ctx_mean_ghost_score for explanation."
    ))
    lines.append(_ans(
        "Q8 Are context features predictive of action selection?",
        "See action_distribution figure and ctx_* columns in seed CSV."
    ))
    lines.append(_ans(
        "Q9 Is runtime acceptable?",
        f"Amortized runtime: {rt:.2f} ms/tick. "
        + ("STRONG (< 10ms)" if rt < 10 else "ACCEPTABLE (< 50ms)" if rt < 50
           else "HIGH (> 50ms) — consider increasing decision_epoch or reducing K/H")
        if not math.isnan(rt) else "Runtime not available."
    ))
    lines.append(_ans(
        "Q10 Is the method paper-worthy?",
        f"MC route cost {mc_rc:.5f} vs AoI {aoi_rc:.5f}. "
        + ("Promising for paper." if mc_rc < aoi_rc else "Needs further tuning.")
        if not math.isnan(aoi_rc) else "Insufficient data."
    ))

    rpt = out / f"{prefix}_interpretation_notes.md"
    rpt.write_text("".join(lines))
    print(f"  Saved: {rpt.name}")
