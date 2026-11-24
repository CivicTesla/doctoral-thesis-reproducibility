"""Shared helpers for E12 staged runner scripts."""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a2field.e12_staged_sim import (
    E12StagedConfig, run_e12_staged_single,
    DISPLAY, ACTION_NAMES,
)

CORE_METRICS = (
    "average_route_cost",
    "route_balanced_loss_g2",
    "global_field_loss",
    "mean_abs_risk_gap_current",
    "mean_route_decision_agreement",
    "fraction_route_decision_disagreement",
    "mean_route_decision_regret",
    "p90_route_decision_regret",
    "fraction_positive_decision_regret",
    "oscillation_count",
    "false_positive_replans",
    "unnecessary_replans",
    "hidden_congestion_score",
    "ghost_pressure_score",
)

AUDIT_COLS = (
    "experiment_id", "scenario", "regime", "policy", "seed",
    "rollout_horizon", "n_rollouts", "decision_epoch", "quota_action_set",
    "lambda_route", "lambda_regret", "lambda_osc", "lambda_ghost", "lambda_hidden",
    "scheduler_uses_oracle", "commit_hash",
)


def run_jobs(
    jobs: list[E12StagedConfig],
    n_workers: int,
    print_interval_pct: float = 0.05,
) -> list[dict]:
    """Run jobs in parallel. Returns list of result dicts."""
    n_jobs   = len(jobs)
    workers  = min(n_workers, n_jobs)
    interval = max(1, int(n_jobs * print_interval_pct))
    t0       = time.time()
    rows:  list[dict] = []
    done = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_e12_staged_single, job): job for job in jobs}
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception as exc:
                job = futs[fut]
                print(f"  ERROR {job.scenario}/{job.regime}/{job.policy}/s{job.seed}: {exc}",
                      flush=True)
            done += 1
            if done % interval == 0 or done == n_jobs:
                elapsed = time.time() - t0
                eta     = elapsed / done * (n_jobs - done)
                print(f"  [{done}/{n_jobs}] {elapsed:.1f}s  ETA {eta:.0f}s", flush=True)

    return rows


def audit_hard_fails(
    df: pd.DataFrame,
    expected_configs: list[tuple],  # (scenario, regime, policy, seed)
) -> list[str]:
    fails: list[str] = []

    # 1. Oracle check
    if "scheduler_uses_oracle" in df.columns:
        bad = df[df["scheduler_uses_oracle"].astype(bool)]
        if not bad.empty:
            fails.append(f"FAIL:oracle_access: {len(bad)} rows have scheduler_uses_oracle=True")

    # 2. NaN / Inf in core metrics
    for col in CORE_METRICS:
        if col not in df.columns:
            continue
        if df[col].isna().any():
            fails.append(f"FAIL:nan_in_metric: {col} has NaN values")
        if np.isinf(df[col].values).any():
            fails.append(f"FAIL:inf_in_metric: {col} has Inf values")

    # 3. Duplicate (scenario, regime, policy, seed)
    group_cols = ["scenario", "regime", "policy", "seed"]
    counts = df.groupby(group_cols).size()
    dups   = counts[counts > 1]
    if not dups.empty:
        fails.append(f"FAIL:duplicate_rows: {len(dups)} (scenario,regime,policy,seed) groups have >1 row")

    # 4. Missing seeds
    actual_set   = set(map(tuple, df[group_cols].values.tolist()))
    expected_set = set(expected_configs)
    missing      = expected_set - actual_set
    if missing:
        fails.append(f"FAIL:missing_jobs: {len(missing)} expected jobs not found in results")

    return fails


def write_run_config(
    jobs: list[E12StagedConfig],
    out: Path,
    prefix: str,
) -> None:
    rows = []
    for cfg in jobs:
        rows.append({
            "experiment_id":    cfg.experiment_id,
            "scenario":         cfg.scenario,
            "regime":           cfg.regime,
            "policy":           cfg.policy,
            "seed":             cfg.seed,
            "rollout_horizon":  cfg.rollout_horizon,
            "n_rollouts":       cfg.n_rollouts,
            "decision_epoch":   cfg.decision_epoch,
            "quota_action_set": cfg.quota_action_set,
            "sweep_id":         cfg.sweep_id,
            "channel_profile":  cfg.channel_profile,
            "lambda_route":     cfg.lambda_route,
            "lambda_regret":    cfg.lambda_regret,
            "lambda_osc":       cfg.lambda_osc,
            "lambda_ghost":     cfg.lambda_ghost,
            "lambda_hidden":    cfg.lambda_hidden,
            "n_agents":         cfg.n_agents,
            "horizon":          cfg.horizon,
            "max_queue_age":    cfg.max_queue_age,
            "n_ticks":          cfg.n_ticks,
            "budget_per_tick":  cfg.budget_per_tick,
            "prefix_trust_mode": cfg.prefix_trust_mode,
            "prefix_beta":      cfg.prefix_beta,
            "prefix_confidence_floor": cfg.prefix_confidence_floor,
            "prefix_weight_ranking": cfg.prefix_weight_ranking,
            "prefix_weight_field": cfg.prefix_weight_field,
            "agent_diversity_penalty": cfg.agent_diversity_penalty,
            "valid_future_only": cfg.valid_future_only,
            "drop_expired_intention_points": cfg.drop_expired_intention_points,
            "drop_nonpositive_remaining_horizon": cfg.drop_nonpositive_remaining_horizon,
        })
    pd.DataFrame(rows).to_csv(out / f"{prefix}_run_config.csv", index=False)


def write_expected_vs_effective(
    jobs: list[E12StagedConfig],
    df:   pd.DataFrame,
    out:  Path,
    prefix: str,
) -> None:
    group_cols = ["scenario", "regime", "policy", "seed"]
    actual_set = set(map(tuple, df[group_cols].values.tolist()))
    rows = []
    for cfg in jobs:
        key = (cfg.scenario, cfg.regime, cfg.policy, cfg.seed)
        status = "present" if key in actual_set else "missing"
        rows.append({
            "scenario": cfg.scenario, "regime": cfg.regime,
            "policy": cfg.policy, "seed": cfg.seed,
            "expected_rollout_horizon": cfg.rollout_horizon,
            "expected_n_rollouts":      cfg.n_rollouts,
            "expected_decision_epoch":  cfg.decision_epoch,
            "status": status,
        })
    pd.DataFrame(rows).to_csv(
        out / f"{prefix}_expected_vs_effective_config.csv", index=False)


def policy_summary(seed_df: pd.DataFrame, prefix: str, out: Path) -> pd.DataFrame:
    grp  = ["scenario", "regime", "policy", "display_name"]
    grp  = [c for c in grp if c in seed_df.columns]
    cols = [c for c in CORE_METRICS if c in seed_df.columns]
    agg  = seed_df.groupby(grp)[cols].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join(filter(None, c)) if c[1] else c[0] for c in agg.columns
    ]
    agg.to_csv(out / f"{prefix}_policy_summary.csv", index=False)
    return agg


def pairwise_vs(
    seed_df: pd.DataFrame,
    ref_policy: str,
    scenarios: tuple[str, ...],
    prefix: str,
    out: Path,
) -> pd.DataFrame:
    rows: list[dict] = []
    others = [p for p in seed_df["policy"].unique() if p != ref_policy]
    for other in others:
        for sc in scenarios:
            for rg in ("normal", "ghost_stress"):
                sub = seed_df[(seed_df["scenario"] == sc) & (seed_df["regime"] == rg)]
                r_m = sub[sub["policy"] == ref_policy]["average_route_cost"].mean()
                o_m = sub[sub["policy"] == other]["average_route_cost"].mean()
                if np.isnan(r_m) or np.isnan(o_m):
                    continue
                rows.append({
                    "ref_policy":    ref_policy,
                    "other_policy":  other,
                    "other_display": DISPLAY.get(other, other),
                    "scenario": sc, "regime": rg,
                    "ref_cost":      r_m,
                    "other_cost":    o_m,
                    "mc_improvement": o_m - r_m,
                    "mc_wins":        int(r_m < o_m),
                })
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{prefix}_pairwise_policy_comparisons.csv", index=False)
    return df


def best_by_case(
    seed_df: pd.DataFrame,
    scenarios: tuple[str, ...],
    prefix: str,
    out: Path,
) -> pd.DataFrame:
    rows: list[dict] = []
    for sc in scenarios:
        for rg in ("normal", "ghost_stress"):
            sub = seed_df[(seed_df["scenario"] == sc) & (seed_df["regime"] == rg)]
            if sub.empty:
                continue
            m  = sub.groupby("policy")["average_route_cost"].mean()
            rr = sub.groupby("policy")["mean_route_decision_regret"].mean()
            bw = str(m.idxmin())
            rows.append({
                "scenario": sc, "regime": rg,
                "best_route_cost_policy": bw,
                "best_route_cost":        float(m[bw]),
                "best_regret_policy":     str(rr.idxmin()),
                "best_regret":            float(rr.min()),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{prefix}_best_by_case.csv", index=False)
    return df
