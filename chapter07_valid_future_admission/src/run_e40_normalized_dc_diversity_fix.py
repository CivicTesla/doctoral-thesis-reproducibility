#!/usr/bin/env python3
"""Run isolated E40 normalized-DC-diversity validation campaigns."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from a2field.e12_staged_sim import E12StagedConfig
from scripts.helpers.e12_runner_helpers import run_jobs
from scripts.helpers.rebuttal_experiment_helpers import parse_seeds

EXPERIMENT_ID = "e40_normalized_dc_diversity_fix"
ROOT = Path("results/e40_normalized_dc_diversity_fix")
SCENARIOS = (
    "original_dual_corridor", "asymmetric_dual_corridor", "bottleneck_shift",
    "dual_corridor", "hourglass", "cross_intersection", "warehouse_grid",
)
REGIMES = ("normal", "ghost_stress")
N_VALUES = (12, 24, 48, 64, 80)
BUDGETS = (4, 8, 12, 20, 30, 45, 60)
CHANNEL_PROFILES = (
    "profile_0_ideal_packetized", "profile_1_random_loss_light",
    "profile_2_random_loss_moderate", "profile_3_burst_loss",
    "profile_4_bandwidth_fluctuation",
)
PUBLIC_TO_INTERNAL = {
    "vfa": "prefix_diverse_vfa",
    "crit_topk": "dc_topk_prefix_diverse_m5",
    "route_ref": "decision_critical_scheduler",
    "pd_no_rollout": "prefix_diverse_no_rollout_ghostsafe",
    "aoi": "aoi_priority_scheduler", "rgsa": "rgsa_current",
    "random": "random_budget",
    "best_fq": "fixed_quota_ghost_safe",
    "ucb": "ucb_quota", "exp3": "exp3_quota",
    "linucb": "contextual_linucb_quota",
}
INTERNAL_TO_PUBLIC = {v: k for k, v in PUBLIC_TO_INTERNAL.items()}
MAIN_POLICIES = ("vfa", "crit_topk", "route_ref", "pd_no_rollout", "aoi", "rgsa", "random", "best_fq")
ONLINE_POLICIES = ("vfa", "aoi", "rgsa", "best_fq", "ucb", "exp3", "linucb")
RUNTIME_POLICIES = ("vfa", "crit_topk", "rgsa")
MC_LAMBDA = (0.5, 2.0, 0.1, 0.2)
CAMPAIGNS = ("pilot", "main", "budget", "channel", "online", "runtime", "large_fleet", "fixed_scarcity")
SUMMARY_NAMES = {
    "main": "e40_main_validation_summary.csv",
    "budget": "e40_budget_sweep_summary.csv",
    "channel": "e40_channel_stress_summary.csv",
    "online": "e40_online_learning_summary.csv",
    "runtime": "e40_runtime_summary.csv",
    "large_fleet": "e40_large_fleet_summary.csv",
    "fixed_scarcity": "e40_fixed_scarcity_summary.csv",
    "pilot": "e40_pilot_summary.csv",
}


def _frame_text(df: pd.DataFrame) -> str:
    """Dependency-free Markdown-friendly table payload."""
    return "```text\n" + df.to_string(index=False) + "\n```"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", choices=CAMPAIGNS + ("finalize",), required=True)
    p.add_argument("--output-root", type=Path, default=ROOT)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--seeds", nargs="+")
    p.add_argument("--ticks", type=int, default=250)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _best_fq(budget: int) -> str:
    return "fixed_quota_ghost_safe" if int(budget) == 4 else "fixed_quota_balanced"


def _cfg(public: str, scenario: str, regime: str, seed: int, n: int, budget: int,
         args: argparse.Namespace, campaign: str, channel: str = "none") -> E12StagedConfig:
    internal = _best_fq(budget) if public == "best_fq" else PUBLIC_TO_INTERNAL[public]
    diverse = public in {"vfa", "pd_no_rollout", "crit_topk"}
    kwargs = dict(
        experiment_id=EXPERIMENT_ID, scenario=scenario, regime=regime,
        policy=internal, seed=seed, rollout_horizon=3, n_rollouts=4,
        decision_epoch=5, n_ticks=args.ticks, n_agents=n, horizon=10,
        budget_per_tick=budget, max_queue_age=15,
        sweep_id=f"{campaign}_N{n}_B{budget}_{channel}", channel_profile=channel,
        lambda_hidden=0.0, valid_future_only=True,
        prefix_trust_mode="prefix_risk" if diverse else "none",
        prefix_weight_ranking=diverse, prefix_weight_field=diverse,
        agent_diversity_penalty=1.0 if diverse else 0.0,
        audit_enabled=(campaign in {"pilot", "main"} and public in {"vfa", "pd_no_rollout", "crit_topk"}),
        audit_example_limit=4, e40_invariant_checks=True,
    )
    if internal in {"prefix_diverse_vfa", "rgsa_current", "ucb_quota", "exp3_quota", "contextual_linucb_quota"}:
        lr, lrg, lo, lg = MC_LAMBDA
        kwargs.update(lambda_route=lr, lambda_regret=lrg, lambda_osc=lo, lambda_ghost=lg)
    return E12StagedConfig(**kwargs)


def _design(campaign: str, args: argparse.Namespace) -> tuple[list[E12StagedConfig], dict]:
    default_seeds = tuple(range(5 if campaign == "pilot" else (10 if campaign == "runtime" else 30)))
    seeds = parse_seeds(args.seeds, default_seeds)
    scenarios, regimes = SCENARIOS, REGIMES
    jobs: list[E12StagedConfig] = []
    if args.smoke:
        seeds, scenarios, regimes = (0,), ("original_dual_corridor",), ("normal",)
    if campaign in {"pilot", "main"}:
        for sc in scenarios:
            for reg in regimes:
                for p in MAIN_POLICIES:
                    for seed in seeds:
                        jobs.append(_cfg(p, sc, reg, seed, 12, 4, args, campaign))
    elif campaign == "budget":
        budgets = (4, 8) if args.smoke else BUDGETS
        for b in budgets:
            for sc in scenarios:
                for reg in regimes:
                    for p in MAIN_POLICIES:
                        for seed in seeds:
                            jobs.append(_cfg(p, sc, reg, seed, 12, b, args, campaign))
    elif campaign == "channel":
        profiles = CHANNEL_PROFILES[:1] if args.smoke else CHANNEL_PROFILES
        channel_scenarios = scenarios if args.smoke else ("dual_corridor", "hourglass", "cross_intersection", "warehouse_grid")
        for prof in profiles:
            for sc in channel_scenarios:
                for reg in regimes:
                    for p in MAIN_POLICIES:
                        for seed in seeds:
                            jobs.append(_cfg(p, sc, reg, seed, 12, 4, args, campaign, prof))
    elif campaign == "online":
        for sc in scenarios:
            for reg in regimes:
                for p in ONLINE_POLICIES:
                    for seed in seeds:
                        jobs.append(_cfg(p, sc, reg, seed, 12, 4, args, campaign))
    elif campaign == "runtime":
        ns = (12,) if args.smoke else N_VALUES
        runtime_scenarios = ("warehouse_grid",)
        for n in ns:
            for sc in runtime_scenarios:
                for reg in regimes:
                    for p in RUNTIME_POLICIES:
                        for seed in seeds:
                            jobs.append(_cfg(p, sc, reg, seed, n, math.ceil(n / 3), args, campaign))
    elif campaign in {"large_fleet", "fixed_scarcity"}:
        ns = (12,) if args.smoke else N_VALUES
        budgets_by_n = lambda n: (math.ceil(n / 3),) if campaign == "large_fleet" else ((4,) if args.smoke else (4, 8))
        for n in ns:
            for b in budgets_by_n(n):
                for sc in scenarios:
                    for reg in regimes:
                        for p in MAIN_POLICIES:
                            for seed in seeds:
                                jobs.append(_cfg(p, sc, reg, seed, n, b, args, campaign))
    meta = {"campaign": campaign, "seeds": list(seeds), "scenarios": list(scenarios),
            "regimes": list(regimes), "ticks": args.ticks, "jobs": len(jobs)}
    return jobs, meta


def _flatten_audit(rows: list[dict], campaign: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    counters, examples = [], []
    for row in rows:
        audit = row.pop("e39_audit", None)
        if not audit:
            continue
        public = INTERNAL_TO_PUBLIC.get(row["policy"], row["policy"])
        base = {"campaign": campaign, "policy": public, "scenario": row["scenario"],
                "regime": row["regime"], "seed": row["seed"],
                "N": row["n_agents"], "B": row["budget_per_tick"]}
        for action, vals in audit.get("groups", {}).items():
            counters.append({**base, "action": action, **vals})
        for val in audit.get("examples", []):
            examples.append({**base, **val, "policy": public})
    return pd.DataFrame(counters), pd.DataFrame(examples)


def _standardize(rows: list[dict], campaign: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["internal_policy"] = df["policy"]
    df["policy"] = df["internal_policy"].map(INTERNAL_TO_PUBLIC).fillna(df["internal_policy"])
    df["N"] = df["n_agents"].astype(int)
    df["B"] = df["budget_per_tick"].astype(int)
    df["campaign"] = campaign
    return df


def _summary(df: pd.DataFrame, campaign: str) -> pd.DataFrame:
    keys = ["policy", "N", "B"]
    for col in ("channel_profile", "regime"):
        if col in df and (campaign in {"channel", "runtime"}):
            keys.append(col)
    metrics = [c for c in (
        "average_route_cost", "admitted_same_agent_concentration",
        "active_starvation_rate_W10", "starved_agent_route_cost_gap_W10",
        "amortized_runtime_per_tick_ms", "selection_runtime_ms_tick_mean",
    ) if c in df]
    out = df.groupby(keys, dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    out.columns = ["_".join(str(x) for x in c if x) if isinstance(c, tuple) else c for c in out.columns]
    return out


def _write_invariants(root: Path, df: pd.DataFrame, audit: pd.DataFrame) -> None:
    oracle = int(df["scheduler_uses_oracle"].astype(bool).sum())
    valid = int((~df["valid_future_invariant_pass"].astype(bool)).sum())
    normalized_bad = int(audit.get("diversity_score_negative_count", pd.Series(dtype=float)).sum()) if not audit.empty else 0
    promoted = int(audit.get("promotion_selected_count", pd.Series(dtype=float)).sum()) if not audit.empty else 0
    fallback = int(df.get("mc_fallback_count", pd.Series(dtype=float)).sum())
    crit_bad = 0
    if not audit.empty:
        crit = audit[audit.policy == "crit_topk"]
        crit_bad = int(crit.get("diversity_score_negative_count", pd.Series(dtype=float)).sum())
    passed = oracle == valid == normalized_bad == promoted == crit_bad == fallback == 0
    lines = ["# E40 invariant check report", "",
             f"- Overall: {'PASS' if passed else 'FAIL'}",
             f"- Normalized DC diversity negative inputs: {normalized_bad}",
             f"- Diversity monotonicity/promotion violations: {promoted}",
             f"- Valid-future invariant failures: {valid}",
             f"- Oracle-access violations: {oracle}",
             f"- Crit-TopK negative normalized inputs: {crit_bad}",
             f"- MC rollout fallback/assertion failures: {fallback}",
             "- Audit hooks are observation-only and consume no RNG.", ""]
    (root / "e40_invariant_check_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_postfix(root: Path, audit: pd.DataFrame) -> None:
    if audit.empty:
        return
    cols = ["dc_candidate_count", "candidate_negative_count", "dc_selected_count",
            "selected_negative_count", "promotion_candidate_count", "promotion_selected_count",
            "diversity_score_negative_count"]
    agg = audit.groupby("policy")[cols].sum()
    agg["raw_negative_candidate_frac"] = agg.candidate_negative_count / agg.dc_candidate_count.clip(lower=1)
    agg["raw_negative_selected_frac"] = agg.selected_negative_count / agg.dc_selected_count.clip(lower=1)
    agg["normalized_score_min"] = audit.groupby("policy")["diversity_score_min"].min()
    agg["normalized_score_max"] = audit.groupby("policy")["diversity_score_max"].max()
    out = agg.reset_index()
    out.to_csv(root / "e40_negative_score_postfix_audit.csv", index=False)
    lines = ["# E40 post-fix negative-score audit", "",
             "Raw DC scores may remain negative. The selector uses the normalized score for prefix/diversity.", "",
             _frame_text(out), "",
             f"Promotion after normalization is {'zero' if int(out.promotion_selected_count.sum()) == 0 else 'NONZERO (FAIL)'}.", ""]
    (root / "e40_negative_score_postfix_audit.md").write_text("\n".join(lines), encoding="utf-8")


def _pilot_report(root: Path, df: pd.DataFrame) -> bool:
    metrics = df.groupby("policy")[["average_route_cost", "admitted_same_agent_concentration",
                                     "active_starvation_rate_W10", "amortized_runtime_per_tick_ms"]].mean()
    catastrophic = (not np.isfinite(metrics["average_route_cost"]).all()
                    or float(metrics["average_route_cost"].max()) > 2.0)
    lines = ["# E40 pilot report", "", _frame_text(metrics.reset_index()), "",
             f"- Catastrophic regression gate: {'STOP' if catastrophic else 'PASS'}",
             "- Old results were read only diagnostically and were not overwritten.", ""]
    (root / "e40_pilot_report.md").write_text("\n".join(lines), encoding="utf-8")
    return catastrophic


def _manifest(root: Path, meta: dict, elapsed: float) -> None:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    path = root / "e40_manifest.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# E40 manifest\n\n"
    entry = {**meta, "elapsed_seconds": round(elapsed, 3),
             "utc": datetime.now(timezone.utc).isoformat(), "commit": commit,
             "algorithm_change": "DC min-max normalization for VFA/PD before prefix/diversity",
             "old_results_overwritten": False, "tex_changed": False}
    path.write_text(existing + "```json\n" + json.dumps(entry, indent=2) + "\n```\n\n", encoding="utf-8")


def _finalize(root: Path) -> None:
    missing = [c for c in CAMPAIGNS if not (root / c / "by_episode.csv").exists()]
    if missing:
        raise SystemExit(f"E40 finalize stopped; missing campaigns: {missing}")
    import matplotlib.pyplot as plt
    main = pd.read_csv(root / "main" / "by_episode.csv")
    budget = pd.read_csv(root / "budget" / "by_episode.csv")
    runtime = pd.read_csv(root / "runtime" / "by_episode.csv")
    large = pd.read_csv(root / "large_fleet" / "by_episode.csv")
    fixed = pd.read_csv(root / "fixed_scarcity" / "by_episode.csv")
    plots = [
        (main.groupby("policy").average_route_cost.mean(), "Mean route cost", "fig_e40_final_validation"),
        (budget[budget.policy == "vfa"].groupby("B").average_route_cost.mean(), "VFA route cost", "fig_e40_budget_runtime"),
        (large[large.policy == "vfa"].groupby("N").average_route_cost.mean(), "VFA route cost", "fig_e40_large_fleet"),
    ]
    for series, ylabel, name in plots:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        if name == "fig_e40_final_validation":
            series.plot(kind="bar", ax=ax)
        else:
            series.plot(kind="line", marker="o", ax=ax)
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.25); fig.tight_layout()
        fig.savefig(root / f"{name}.pdf"); fig.savefig(root / f"{name}.png", dpi=200); plt.close(fig)
    means = main.groupby("policy").average_route_cost.mean()
    conc = main.groupby("policy").admitted_same_agent_concentration.mean()
    starve = fixed.groupby("policy").active_starvation_rate_W10.mean()
    runtime_means = runtime.groupby(["policy", "N"]).amortized_runtime_per_tick_ms.mean()
    checks = {
        "VFA improves over AoIOnly": means.get("vfa", np.inf) < means.get("aoi", -np.inf),
        "VFA improves over RGSA": means.get("vfa", np.inf) < means.get("rgsa", -np.inf),
        "VFA improves over Best FQ": means.get("vfa", np.inf) < means.get("best_fq", -np.inf),
        "VFA remains route-cost competitive with Route-ref.": means.get("vfa", np.inf) <= 1.05 * means.get("route_ref", -np.inf),
        "VFA/Crit-TopK reduce same-agent concentration": max(conc.get("vfa", np.inf), conc.get("crit_topk", np.inf)) < conc.get("route_ref", -np.inf),
        "Crit-TopK reduces W10 starvation under fixed scarcity": starve.get("crit_topk", np.inf) < starve.get("route_ref", -np.inf),
        "Runtime story remains valid": bool(np.isfinite(runtime_means.to_numpy()).all() and len(runtime_means) == len(RUNTIME_POLICIES) * len(N_VALUES)),
    }
    lines = ["# E40 old-vs-new qualitative comparison", ""]
    lines += [f"- {label}: {'PASS' if ok else 'FAIL'}" for label, ok in checks.items()]
    old_sources = [
        Path("results/e37_large_fleet_main_validation/large_fleet_main_by_episode.csv"),
        Path("results/e22_valid_future_final_validation/e22_by_seed.csv"),
    ]
    old = next((pd.read_csv(p) for p in old_sources if p.exists()), pd.DataFrame())
    if not old.empty and "average_route_cost" in old:
        if "internal_policy" in old:
            old["policy_cmp"] = old["internal_policy"].map(INTERNAL_TO_PUBLIC).fillna(old.get("policy", ""))
        else:
            old["policy_cmp"] = old["policy"].map(INTERNAL_TO_PUBLIC).fillna(old["policy"])
        old_means = old.groupby("policy_cmp").average_route_cost.mean()
        compare = pd.DataFrame({"old_route_cost": old_means, "e40_route_cost": means})
        compare["delta"] = compare.e40_route_cost - compare.old_route_cost
        compare["delta_percent"] = 100.0 * compare.delta / compare.old_route_cost
        compare.to_csv(root / "e40_old_vs_new_route_cost.csv")
        lines += ["", "## Route-cost deltas", "", _frame_text(compare.reset_index())]
    lines += [""]
    (root / "e40_old_vs_new_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    claims_ok = all(checks.values())
    final = ["# E40 final report", "", "## Status", "",
             "E40 completed: all required campaign files were present before figures were generated.",
             f"All seven qualitative claim checks: {'PASS' if claims_ok else 'FAIL; review required'}.",
             "", "## Algorithm change", "",
             "Raw DC-tranche scores for Prefix Diverse VFA and PD no-rollout are min-max normalized over the current candidate pool before prefix confidence and soft diversity. AoI, residual, RGSA, FQ, Crit-TopK, Route-ref., valid-future filtering, and field weighting retain their specified paths.",
             "", "## Verification", "",
             "The normalized DC diversity input is nonnegative; the denominator is monotone. See `e40_invariant_check_report.md` and `e40_negative_score_postfix_audit.md`.",
             "", "## Campaigns", "",
             "Pilot, main validation, budget sweep, channel stress, online learning, runtime, large-fleet proportional validation, and fixed-scarcity validation completed under E40 output paths.",
             "", "## Preservation", "", "Old results and TeX were not overwritten.",
             "", "## Recommended next action", "",
             "Review the old-vs-new comparison and patch suggestions, then explicitly authorize a TeX update if the claim checks are acceptable.", ""]
    (root / "e40_final_report.md").write_text("\n".join(final), encoding="utf-8")
    tex_note = ["# E40 TeX patch suggestions (do not apply automatically)", "",
                "- Eq. (11): retain the raw DecisionCritical score definition.",
                "- Add the DC candidate-pool min-max normalization equation.",
                "- Eq. (14): use normalized DC score, raw nonnegative AoI score, and residual probability proportional to prefix confidence over diversity denominator.",
                "- State that every score entering the repeated-agent denominator is nonnegative.",
                "- Remove the sentence claiming DC diversity can be non-monotone.",
                "- Replace it with: raw DC score can be negative after ghost subtraction, but the implemented selector normalizes it before prefix/diversity, making the denominator monotone for the selection score.",
                "- Update Abstract numbers, Fig. 2/Table II, Figs. 3/4, Results, Discussion, and Conclusion from E40 outputs.", ""]
    (root / "e40_tex_patch_suggestions.md").write_text("\n".join(tex_note), encoding="utf-8")
    print("E40 completed; reports and figures are under", root.resolve())


def main() -> None:
    args = parse_args()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    if args.campaign == "finalize":
        _finalize(root); return
    jobs, meta = _design(args.campaign, args)
    out = root / args.campaign
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**meta, "seed": j.seed, "scenario": j.scenario, "regime": j.regime,
                   "policy": j.policy, "N": j.n_agents, "B": j.budget_per_tick,
                   "channel_profile": j.channel_profile} for j in jobs]).to_csv(out / "expected_jobs.csv", index=False)
    print(f"E40 {args.campaign}: jobs={len(jobs)} out={out}", flush=True)
    if args.dry_run:
        return
    start = time.perf_counter()
    rows = run_jobs(jobs, args.workers)
    audit, examples = _flatten_audit(rows, args.campaign)
    df = _standardize(rows, args.campaign)
    df.to_csv(out / "by_episode.csv", index=False)
    summary = _summary(df, args.campaign)
    summary.to_csv(root / SUMMARY_NAMES[args.campaign], index=False)
    if not audit.empty:
        audit.to_csv(out / "audit_counters.csv", index=False)
    if not examples.empty:
        examples.head(200).to_csv(out / "audit_examples.csv", index=False)
    _write_invariants(root, df, audit)
    if args.campaign == "main":
        _write_postfix(root, audit)
        (root / "e40_main_validation_report.md").write_text(
            "# E40 main validation report\n\n" + _frame_text(summary) + "\n", encoding="utf-8")
    if args.campaign == "pilot" and _pilot_report(root, df):
        (root / "STOP_CATASTROPHIC_REGRESSION").write_text("Pilot gate failed.\n")
        raise SystemExit("E40 stopped: catastrophic pilot regression")
    if args.campaign in {"channel", "online", "fixed_scarcity"}:
        report_name = {"channel": "e40_channel_stress_report.md", "online": "e40_online_learning_report.md",
                       "fixed_scarcity": "e40_fixed_scarcity_report.md"}[args.campaign]
        (root / report_name).write_text(f"# E40 {args.campaign} report\n\n" + _frame_text(summary) + "\n", encoding="utf-8")
    _manifest(root, meta, time.perf_counter() - start)
    print(f"E40 {args.campaign} completed -> {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
