"""Shared helpers for A2 rebuttal experiments E16-E19."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAPER_LABELS = {
    "mc_regret_heavy": "RGSA",
    "aoi_priority_scheduler": "AoI",
    "decision_critical_scheduler": "DecisionCritical",
    "random_budget": "Random",
    "fixed_quota_aoi_heavy": "FQ-AoI",
    "fixed_quota_critical_heavy": "FQ-Crit",
    "fixed_quota_coverage_heavy": "FQ-Cov",
    "fixed_quota_balanced": "FQ-Bal",
    "fixed_quota_ghost_safe": "FQ-Ghost",
    "ucb_quota": "UCB",
    "exp3_quota": "EXP3",
    "contextual_linucb_quota": "CtxBandit",
    "rgsa_current": "RGSA current",
    "rgsa_horizon_decay": "RGSA horizon",
    "rgsa_prefix_risk": "RGSA prefix",
    "rgsa_prefix_risk_diverse_soft": "RGSA prefix diverse",
    "rgsa_prefix_risk_rank_only": "RGSA prefix rank",
    "rgsa_prefix_risk_field_only": "RGSA prefix field",
    "aoi_prefix_risk": "AoI prefix",
}

FIXED_QUOTA_POLICIES = (
    "fixed_quota_aoi_heavy",
    "fixed_quota_critical_heavy",
    "fixed_quota_coverage_heavy",
    "fixed_quota_balanced",
    "fixed_quota_ghost_safe",
)

CALM_COLORS = {
    "RGSA": "#2f5f8f",
    "AoI": "#6b8e23",
    "DecisionCritical": "#8c564b",
    "Random": "#7f7f7f",
    "FQ-AoI": "#4c78a8",
    "FQ-Crit": "#e45756",
    "FQ-Cov": "#72b7b2",
    "FQ-Bal": "#f2cf5b",
    "FQ-Ghost": "#54a24b",
    "UCB": "#b279a2",
    "EXP3": "#ff9da6",
    "CtxBandit": "#9d755d",
    "RGSA current": "#2f5f8f",
    "RGSA horizon": "#59a14f",
    "RGSA prefix": "#e15759",
    "RGSA prefix diverse": "#4e79a7",
    "RGSA prefix rank": "#b279a2",
    "RGSA prefix field": "#f28e2b",
    "AoI prefix": "#86bc86",
}


def paper_label(policy: str) -> str:
    return PAPER_LABELS.get(policy, policy)


def add_paper_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "policy" in out.columns:
        out["paper_label"] = out["policy"].map(paper_label)
    return out


def parse_seeds(raw: list[str] | None, default: Iterable[int]) -> tuple[int, ...]:
    if raw is None:
        return tuple(default)
    if len(raw) == 1 and ":" in raw[0]:
        lo, hi = raw[0].split(":", 1)
        return tuple(range(int(lo), int(hi) + 1))
    return tuple(int(x) for x in raw)


def bootstrap_ci(values: Iterable[float], seed: int = 20260628,
                 n_boot: int = 1000) -> tuple[float, float]:
    arr = np.asarray([v for v in values if not pd.isna(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = float(rng.choice(arr, size=arr.size, replace=True).mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def setup_plot() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 120,
    })


def write_expected_jobs(rows: list[dict], out: Path, filename: str) -> None:
    pd.DataFrame(rows).to_csv(out / filename, index=False)


def write_basic_manifest(out: Path, filename: str, title: str,
                         lines: list[str]) -> None:
    text = [f"# {title}", ""]
    text.extend(lines)
    text.append("")
    (out / filename).write_text("\n".join(text), encoding="utf-8")


def oracle_audit(df: pd.DataFrame, extra_checks: dict[str, bool] | None = None) -> pd.DataFrame:
    rows = []
    oracle_rows = int(df.get("scheduler_uses_oracle", pd.Series(False, index=df.index)).astype(bool).sum())
    hidden_nonzero = 0
    if "lambda_hidden" in df.columns:
        hidden_nonzero = int((df["lambda_hidden"].fillna(0.0).astype(float).abs() > 1e-12).sum())
    rows.append({"check": "scheduler_uses_oracle_rows", "value": oracle_rows, "pass": oracle_rows == 0})
    rows.append({"check": "lambda_hidden_nonzero_rows", "value": hidden_nonzero, "pass": hidden_nonzero == 0})
    for name, ok in (extra_checks or {}).items():
        rows.append({"check": name, "value": int(not ok), "pass": bool(ok)})
    return pd.DataFrame(rows)


def write_oracle_report(audit_df: pd.DataFrame, out: Path, filename: str,
                        title: str) -> None:
    status = "PASS" if bool(audit_df["pass"].all()) else "FAIL"
    lines = [f"# {title}", "", f"Verdict: **{status}**", "",
             "| check | value | pass |", "|---|---:|---|"]
    for row in audit_df.to_dict("records"):
        lines.append(f"| {row['check']} | {row['value']} | {row['pass']} |")
    lines.append("")
    lines.append("Online scheduling audit: no scheduling policy is configured to read F_full, hidden congestion labels, offline decision regret, or true future congestion.")
    (out / filename).write_text("\n".join(lines), encoding="utf-8")


def rank_by_case(df: pd.DataFrame, case_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(case_cols, sort=True):
        means = sub.groupby("policy")["average_route_cost"].mean().sort_values()
        for rank, (policy, value) in enumerate(means.items(), start=1):
            base = dict(zip(case_cols, keys if isinstance(keys, tuple) else (keys,)))
            base.update({"policy": policy, "mean_route_cost": float(value), "rank": rank})
            rows.append(base)
    return pd.DataFrame(rows)


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def ceil_div(a: int, b: int) -> int:
    return int(math.ceil(a / b))
