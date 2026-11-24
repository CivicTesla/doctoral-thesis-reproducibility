from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .agents import RollingAgent, generate_rolling_agents
from .buffer import IntentionBuffer
from .field import build_rolling_field, project_sum
from .grid import dilate_mask
from .messages import IntentionMessage, MotionTubePoint
from .metrics import compute_metrics
from .pathing import astar
from .policies import select_tube_points
from .scenarios import get_scenario


@dataclass(frozen=True)
class RollingConfig:
    scenario: str = "dual_corridor"
    agents: int = 60
    seed: int = 0
    sim_steps: int = 50
    horizon: int = 25
    upload_period: int = 5
    delay: int = 0
    budget_points: int = 800
    partial_access_ratio: float = 1.0
    message_loss_prob: float = 0.0
    policy_name: str = "periodic_full"
    buffer_mode: str = "overwrite_with_decay_TTL"
    base_radius: float = 1.5
    width_mode: str = "thin"
    width_growth: float = 0.0
    max_radius: float = 2.0
    beta: float = 0.03
    freshness_tau: float = 10.0
    ttl: int = 30
    top_k: int = 20
    peak_tolerance_radius: float = 3.0
    active_threshold_ratio: float = 0.05
    fixed_horizon: int = 10


def stable_seed(seed: int, *parts: object) -> int:
    text = ":".join(str(p) for p in parts)
    total = sum((i + 1) * ord(ch) for i, ch in enumerate(text))
    return seed * 100_003 + total


def _direction(path: list[tuple[int, int]], idx: int) -> tuple[int, int]:
    if not path:
        return (0, 0)
    cur = path[min(idx, len(path) - 1)]
    nxt = path[min(idx + 1, len(path) - 1)]
    return (nxt[0] - cur[0], nxt[1] - cur[1])


def _radius_for_step(base_radius: float, width_mode: str, width_growth: float, max_radius: float, k: int) -> float:
    if width_mode == "thin":
        return base_radius
    radius = base_radius + max(0.0, width_growth) * k
    if width_mode == "capped_growth":
        radius = min(radius, max_radius)
    return radius


def make_tube_points(
    agent: RollingAgent,
    current_time: int,
    horizon: int,
    base_radius: float,
    width_mode: str,
    width_growth: float,
    max_radius: float,
    beta: float,
    version_id: int,
) -> list[MotionTubePoint]:
    points: list[MotionTubePoint] = []
    if not agent.current_path:
        return points
    for k in range(1, horizon + 1):
        idx = min(agent.path_index + k, len(agent.current_path) - 1)
        y, x = agent.current_path[idx]
        dy, dx = _direction(agent.current_path, idx)
        points.append(
            MotionTubePoint(
                agent_id=agent.agent_id,
                version_id=version_id,
                upload_time=current_time,
                future_time_abs=current_time + k,
                k=k,
                y=y,
                x=x,
                dy=dy,
                dx=dx,
                radius=_radius_for_step(base_radius, width_mode, width_growth, max_radius, k),
                freshness=1.0,
                weight=float(np.exp(-beta * k)),
            )
        )
    return points


def make_messages_from_points(
    points: list[MotionTubePoint],
    current_time: int,
    delay: int,
    horizon: int,
    policy_name: str,
) -> list[IntentionMessage]:
    by_agent: dict[int, list[MotionTubePoint]] = defaultdict(list)
    version_by_agent: dict[int, int] = {}
    for p in points:
        by_agent[p.agent_id].append(p)
        version_by_agent[p.agent_id] = p.version_id
    messages: list[IntentionMessage] = []
    for agent_id, agent_points in by_agent.items():
        messages.append(
            IntentionMessage(
                agent_id=agent_id,
                version_id=version_by_agent[agent_id],
                upload_time=current_time,
                delivery_time=current_time + delay,
                horizon=horizon,
                tube_points=agent_points,
                policy_name=policy_name,
                uploaded_tube_points=len(agent_points),
            )
        )
    return messages


def run_rolling_simulation(cfg: RollingConfig) -> pd.DataFrame:
    scenario = get_scenario(cfg.scenario)
    rng = np.random.default_rng(cfg.seed)
    agents = generate_rolling_agents(scenario, cfg.agents, rng)
    buffer = IntentionBuffer(mode=cfg.buffer_mode, freshness_tau=cfg.freshness_tau, ttl=cfg.ttl)
    rows: list[dict] = []

    for current_time in range(cfg.sim_steps):
        for agent in agents:
            agent.step()

        oracle_points: list[MotionTubePoint] = []
        for agent in agents:
            oracle_points.extend(
                make_tube_points(
                    agent,
                    current_time,
                    cfg.horizon,
                    cfg.base_radius,
                    "thin",
                    0.0,
                    cfg.max_radius,
                    cfg.beta,
                    agent.version_id,
                )
            )
        R_full = build_rolling_field(oracle_points, scenario.grid, current_time, cfg.horizon)

        message_count = 0
        uploaded_tube_points = 0
        if current_time % max(1, cfg.upload_period) == 0:
            eligible = [a for a in agents if a.can_upload]
            if cfg.partial_access_ratio < 1.0 and eligible:
                n_keep = int(np.ceil(len(eligible) * np.clip(cfg.partial_access_ratio, 0.0, 1.0)))
                idx = rng.choice(len(eligible), size=n_keep, replace=False)
                eligible = [eligible[int(i)] for i in idx]

            candidates: list[MotionTubePoint] = []
            for agent in eligible:
                agent.version_id += 1
                candidates.extend(
                    make_tube_points(
                        agent,
                        current_time,
                        cfg.horizon,
                        cfg.base_radius,
                        cfg.width_mode,
                        cfg.width_growth,
                        cfg.max_radius,
                        cfg.beta,
                        agent.version_id,
                    )
                )
            policy_rng = np.random.default_rng(stable_seed(cfg.seed, cfg.policy_name, cfg.upload_period, cfg.delay, current_time))
            selected = select_tube_points(
                cfg.policy_name,
                candidates,
                policy_rng,
                cfg.budget_points,
                cfg.horizon,
                scenario.critical_mask,
                fixed_horizon=cfg.fixed_horizon,
            )
            selected_ids = {(p.agent_id, p.version_id, p.k) for p in selected}
            for msg in make_messages_from_points(selected, current_time, cfg.delay, cfg.horizon, cfg.policy_name):
                if cfg.message_loss_prob > 0.0 and rng.random() < cfg.message_loss_prob:
                    continue
                buffer.enqueue_message(msg)
                message_count += 1
                uploaded_tube_points += len(msg.tube_points)

        delivered_count = buffer.deliver_messages(current_time)
        buffer.update_buffer(current_time)
        comm_points = buffer.get_active_intentions(current_time)
        R_comm = build_rolling_field(comm_points, scenario.grid, current_time, cfg.horizon)
        metrics = compute_metrics(
            R_full,
            R_comm,
            cfg.policy_name,
            top_k=cfg.top_k,
            peak_tolerance_radius=cfg.peak_tolerance_radius,
            active_threshold_ratio=cfg.active_threshold_ratio,
        )
        stats = buffer.get_buffer_stats(current_time)
        metrics.update(
            {
                "seed": cfg.seed,
                "tick": current_time,
                "scenario": cfg.scenario,
                "agents": len(agents),
                "policy": cfg.policy_name,
                "buffer_mode": cfg.buffer_mode,
                "H": cfg.horizon,
                "U": cfg.upload_period,
                "delay": cfg.delay,
                "budget_points": cfg.budget_points,
                "width_mode": cfg.width_mode,
                "width_growth": cfg.width_growth,
                "message_count": message_count,
                "delivered_message_count": delivered_count,
                "uploaded_tube_points": uploaded_tube_points,
                "communication_budget_used": uploaded_tube_points,
                "active_intention_count": stats.active_intention_count,
                "mean_intention_age": stats.mean_intention_age,
                "max_intention_age": stats.max_intention_age,
                "stale_intention_ratio": stats.stale_intention_ratio,
                "expired_intention_count": stats.expired_intention_count,
                "overwrite_count": stats.overwrite_count,
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def quick_configs() -> list[RollingConfig]:
    configs: list[RollingConfig] = []
    for seed in [0, 1, 2]:
        for upload_period in [1, 5, 10]:
            for delay in [0, 5]:
                for buffer_mode in ["rolling_overwrite", "overwrite_with_decay_TTL"]:
                    for policy in ["periodic_full", "random_budget", "freshness_based", "wave_critical"]:
                        configs.append(
                            RollingConfig(
                                seed=seed,
                                sim_steps=10,
                                horizon=25,
                                upload_period=upload_period,
                                delay=delay,
                                budget_points=800,
                                policy_name=policy,
                                buffer_mode=buffer_mode,
                                freshness_tau=10.0,
                                ttl=30,
                            )
                        )
    return configs


def run_rolling_quick(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    configs = quick_configs()
    frames: list[pd.DataFrame] = []
    for i, cfg in enumerate(configs, start=1):
        print(
            f"[rolling {i}/{len(configs)}] seed={cfg.seed} policy={cfg.policy_name} "
            f"buffer={cfg.buffer_mode} U={cfg.upload_period} delay={cfg.delay}",
            flush=True,
        )
        frames.append(run_rolling_simulation(cfg))
    tick_df = pd.concat(frames, ignore_index=True)
    tick_df.to_csv(out_dir / "metrics_tick.csv", index=False)
    metric_cols = [
        "field_fidelity_score",
        "hidden_congestion",
        "ghost_pressure",
        "field_mass_ratio",
        "top_k_peak_overlap_tolerant",
        "entropy_diff",
        "mean_intention_age",
        "max_intention_age",
        "stale_intention_ratio",
        "active_intention_count",
        "expired_intention_count",
        "overwrite_count",
        "message_count",
        "uploaded_tube_points",
    ]
    summary = (
        tick_df.groupby(["policy", "buffer_mode", "H", "U", "delay"], dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(str(p) for p in col if p) if isinstance(col, tuple) else col for col in summary.columns]
    summary.to_csv(out_dir / "summary.csv", index=False)
    save_rolling_plots(tick_df, plots_dir)
    print_rolling_summary(tick_df, out_dir)
    return tick_df, summary


def save_rolling_plots(df: pd.DataFrame, plots_dir: Path) -> None:
    sanity = df[
        (df["seed"] == 0)
        & (df["policy"] == "periodic_full")
        & (df["U"] == 5)
        & (df["delay"] == 0)
        & (df["buffer_mode"] == "overwrite_with_decay_TTL")
    ].sort_values("tick")
    metrics = ["field_mass_ratio", "hidden_congestion", "ghost_pressure", "mean_intention_age"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, metric in zip(axes.ravel(), metrics):
        ax.plot(sanity["tick"], sanity[metric], marker="o", markersize=2)
        ax.set_title(metric)
        ax.set_xlabel("tick")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "rolling_field_sanity.png", dpi=150)
    plt.close(fig)

    sweep_metrics = ["field_fidelity_score", "hidden_congestion", "ghost_pressure", "mean_intention_age", "message_count"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    grouped = df.groupby(["U", "policy"])[sweep_metrics].mean().reset_index()
    for ax, metric in zip(axes.ravel(), sweep_metrics):
        for policy, sub in grouped.groupby("policy"):
            sub = sub.sort_values("U")
            ax.plot(sub["U"], sub[metric], marker="o", label=policy)
        ax.set_title(metric)
        ax.set_xlabel("U")
        ax.grid(alpha=0.25)
    axes.ravel()[-1].axis("off")
    axes[0, 1].legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "upload_period_sweep_quick.png", dpi=150)
    plt.close(fig)

    compare_metrics = ["hidden_congestion", "ghost_pressure", "top_k_peak_overlap_tolerant", "uploaded_tube_points"]
    comp = df.groupby("policy", sort=False)[compare_metrics].agg(["mean", "std"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, metric in zip(axes.ravel(), compare_metrics):
        means = comp[(metric, "mean")]
        stds = comp[(metric, "std")].fillna(0.0)
        ax.bar(means.index, means.values, yerr=stds.values, capsize=3)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "policy_comparison_quick.png", dpi=150)
    plt.close(fig)


def print_rolling_summary(df: pd.DataFrame, out_dir: Path) -> None:
    print("\nRolling quick run summary:")
    print(f"number of ticks simulated: {int(df['tick'].max()) + 1}")
    print(f"number of agents: {int(df['agents'].iloc[0])}")
    print(f"H: {sorted(df['H'].unique().tolist())}")
    print(f"U values: {sorted(df['U'].unique().tolist())}")
    print(f"delay values: {sorted(df['delay'].unique().tolist())}")
    print(f"policies: {', '.join(df['policy'].drop_duplicates().tolist())}")
    print(f"buffer modes: {', '.join(df['buffer_mode'].drop_duplicates().tolist())}")
    print(f"output directory: {out_dir}")

    close = df[
        (df["U"] == 1)
        & (df["delay"] == 0)
        & (df["policy"] == "periodic_full")
        & (df["buffer_mode"] == "rolling_overwrite")
    ]
    if not close.empty:
        print(
            "sanity periodic_full U=1 delay=0 rolling_overwrite: "
            f"hidden={close['hidden_congestion'].mean():.4f}, ghost={close['ghost_pressure'].mean():.4f}, "
            f"mass={close['field_mass_ratio'].mean():.4f}"
        )
    age_by_u = df.groupby("U")["mean_intention_age"].mean().sort_index()
    print("mean_intention_age by U:")
    for u, v in age_by_u.items():
        print(f"  U={int(u)}: {v:.3f}")
    delay_effect = df.groupby("delay")[["stale_intention_ratio", "hidden_congestion", "ghost_pressure"]].mean()
    print("delay effect:")
    print(delay_effect.to_string(float_format=lambda v: f"{v:.4f}"))
    wave = df[df["policy"] == "wave_critical"]
    if not wave.empty:
        max_upload = float(wave["uploaded_tube_points"].max())
        print(f"wave_critical max uploaded_tube_points per tick: {max_upload:.0f}")


@dataclass(frozen=True)
class E5BConfig:
    scenario: str = "dual_corridor"
    agents: int = 60
    seeds: tuple[int, ...] = (0, 1, 2)
    sim_steps: int = 80
    horizon: int = 25
    upload_period: int = 5
    route_switch_tick: int = 30
    switch_ratio: float = 0.5
    delay_list: tuple[int, ...] = (0, 2, 5, 10)
    policy: str = "periodic_full"
    buffer_modes: tuple[str, ...] = ("append_only", "rolling_overwrite", "overwrite_with_decay_TTL")
    budget_points: int = 800
    freshness_tau: float = 10.0
    ttl: int = 30
    base_radius: float = 1.5
    width_growth: float = 0.0
    force_upload_on_replan: bool = True
    plot_level: str = "minimal"
    beta: float = 0.03
    top_k: int = 20
    peak_tolerance_radius: float = 3.0
    active_threshold_ratio: float = 0.05
    selected_diagnostics: bool = False
    selected_diagnostics_selected_only: bool = False
    diagnostics_regime: str = "balanced"
    critical_budget_ratio: float = 0.7
    coverage_budget_ratio: float = 0.3
    new_version_weight: float = 1.0
    new_version_boost_window: int = 10
    old_version_weight: float = 2.0
    balance_time_bins: bool = False
    balance_agents: bool = False
    balance_spatial_bins: bool = False
    use_route_balance: bool = True
    post_switch_new_version_bonus: float = 0.0
    lambda_agent: float = 0.0
    lambda_time: float = 0.0
    lambda_space: float = 0.0
    distribution_mode: str = "random_uniform"


def generate_dual_corridor_path(scenario, start: tuple[int, int], goal: tuple[int, int], route_variant: str) -> list[tuple[int, int]]:
    h, w = scenario.grid.height, scenario.grid.width
    y = max(4, h // 2 - 17) if route_variant == "upper" else min(h - 5, h // 2 + 17)
    waypoints = [(y, w // 3), (y, 2 * w // 3)]
    full: list[tuple[int, int]] = []
    cur = start
    for wp in waypoints + [goal]:
        segment = astar(scenario.grid, cur, wp)
        if not segment:
            return astar(scenario.grid, start, goal)
        full.extend(segment if not full else segment[1:])
        cur = wp
    return full


def _free_cells_rect(obstacles: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for y in range(y0, y1):
        for x in range(x0, x1):
            if 0 <= y < obstacles.shape[0] and 0 <= x < obstacles.shape[1] and not obstacles[y, x]:
                cells.append((y, x))
    return cells


def generate_distribution_agents(scenario, n_agents: int, rng: np.random.Generator, mode: str) -> list[RollingAgent]:
    if mode == "random_uniform":
        return generate_rolling_agents(scenario, n_agents, rng)
    h, w = scenario.grid.height, scenario.grid.width
    obstacles = scenario.grid.obstacles
    left_all = _free_cells_rect(obstacles, 6, h - 6, 3, 9)
    goals_all = _free_cells_rect(obstacles, 6, h - 6, w - 9, w - 3)
    upper_left = _free_cells_rect(obstacles, 6, max(7, h // 2 - 8), 3, 9)
    lower_left = _free_cells_rect(obstacles, min(h - 7, h // 2 + 8), h - 6, 3, 9)
    mid_left = _free_cells_rect(obstacles, max(6, h // 2 - 5), min(h - 6, h // 2 + 6), 3, 9)
    starts: list[tuple[int, int]] = []
    if mode == "clustered_start":
        clusters = [upper_left or left_all, lower_left or left_all, mid_left or left_all]
        for i in range(n_agents):
            cluster = clusters[i % len(clusters)]
            starts.append(cluster[int(rng.integers(0, len(cluster)))])
    elif mode == "deterministic_grid":
        ordered = sorted(left_all)
        if not ordered:
            ordered = [scenario.sample_start(rng)]
        step = max(1, len(ordered) // max(1, n_agents))
        starts = [ordered[(i * step) % len(ordered)] for i in range(n_agents)]
    else:
        starts = [left_all[int(rng.integers(0, len(left_all)))] for _ in range(n_agents)]
    agents: list[RollingAgent] = []
    for agent_id, start in enumerate(starts):
        goal = goals_all[int(rng.integers(0, len(goals_all)))] if goals_all else scenario.sample_goal(rng)
        path = astar(scenario.grid, start, goal)
        if path:
            agents.append(RollingAgent(agent_id=agent_id, start=start, goal=goal, current_path=path))
    return agents


def initialize_route_variants(agents: list[RollingAgent], scenario, distribution_mode: str = "random_uniform") -> dict[int, str]:
    variants: dict[int, str] = {}
    for i, agent in enumerate(agents):
        if distribution_mode == "skewed_route":
            variant = "upper" if i < int(0.8 * len(agents)) else "lower"
        elif distribution_mode == "balanced_corridor":
            variant = "upper" if i % 2 == 0 else "lower"
        else:
            variant = "upper" if i % 2 == 0 else "lower"
        path = generate_dual_corridor_path(scenario, agent.start, agent.goal, variant)
        if path:
            agent.current_path = path
            agent.path_index = 0
        variants[agent.agent_id] = variant
    return variants


def switch_agent_route(agent: RollingAgent, scenario, current_time: int, horizon: int, cfg: E5BConfig, route_variants: dict[int, str]):
    old_variant = route_variants.get(agent.agent_id, "upper")
    new_variant = "lower" if old_variant == "upper" else "upper"
    old_points = make_tube_points(agent, current_time, horizon, cfg.base_radius, "thin", 0.0, 2.0, cfg.beta, agent.version_id)
    current_pos = agent.current_position
    new_path = generate_dual_corridor_path(scenario, current_pos, agent.goal, new_variant)
    if not new_path:
        return [], [], old_variant
    agent.current_path = new_path
    agent.path_index = 0
    route_variants[agent.agent_id] = new_variant
    agent.version_id += 1
    new_points = make_tube_points(agent, current_time, horizon, cfg.base_radius, "thin", 0.0, 2.0, cfg.beta, agent.version_id)
    return old_points, new_points, new_variant


def tube_points_mask(points: list[MotionTubePoint], shape: tuple[int, int], radius: int = 2) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    h, w = shape
    for p in points:
        y0, y1 = max(0, p.y - radius), min(h, p.y + radius + 1)
        x0, x1 = max(0, p.x - radius), min(w, p.x + radius + 1)
        mask[y0:y1, x0:x1] = True
    return mask


def route_specific_metrics(R_full: np.ndarray, R_comm: np.ndarray, old_mask: np.ndarray, new_mask: np.ndarray) -> tuple[float, float]:
    eps = 1e-12
    max_ref = max(float(np.max(R_full)) if R_full.size else 0.0, eps)
    high = 0.25 * max_ref
    denom = float(np.sum(R_full[R_full >= high])) + eps
    ghost = np.maximum(R_comm - R_full, 0.0)
    hidden = np.maximum(R_full - R_comm, 0.0)
    if old_mask.any():
        old_route_ghost = float(np.sum(ghost[:, old_mask]) / denom)
    else:
        old_route_ghost = 0.0
    if new_mask.any():
        new_route_hidden = float(np.sum(hidden[:, new_mask]) / denom)
    else:
        new_route_hidden = 0.0
    return old_route_ghost, new_route_hidden


def route_label_for_point(point: MotionTubePoint, height: int) -> str:
    if point.y < height * 0.42:
        return "upper"
    if point.y > height * 0.58:
        return "lower"
    return "middle"


def base_criticality_scores(points: list[MotionTubePoint], horizon: int, critical_mask: np.ndarray) -> dict[int, float]:
    if not points:
        return {}
    dilated = dilate_mask(critical_mask, 3)
    bins: dict[tuple[int, int, int], int] = defaultdict(int)
    for p in points:
        bins[(p.future_time_abs, p.y // 2, p.x // 2)] += 1
    max_overlap = max(bins.values()) if bins else 1
    scores: dict[int, float] = {}
    for i, p in enumerate(points):
        local_overlap = bins[(p.future_time_abs, p.y // 2, p.x // 2)] / max_overlap
        critical = 1.0 if 0 <= p.y < dilated.shape[0] and 0 <= p.x < dilated.shape[1] and dilated[p.y, p.x] else 0.0
        temporal = float(np.exp(-p.k / max(1, horizon)))
        width_penalty = max(0.0, p.radius - 2.0)
        scores[i] = float(local_overlap + critical + 0.5 * temporal + 0.5 * p.freshness - 0.2 * width_penalty)
    return scores


def dynamic_scores(
    points: list[MotionTubePoint],
    horizon: int,
    critical_mask: np.ndarray,
    current_time: int,
    freshness_tau: float,
    route_switch_tick: int,
    delivery_tick: int,
    new_version_by_agent: dict[int, int],
    new_version_weight: float = 1.0,
    new_version_boost_window: int = 10,
    old_version_weight: float = 2.0,
    post_switch_new_version_bonus: float = 0.0,
) -> dict[int, float]:
    base = base_criticality_scores(points, horizon, critical_mask)
    scores: dict[int, float] = {}
    for i, p in enumerate(points):
        age = max(0, current_time - p.upload_time)
        message_confidence = float(np.exp(-age / max(freshness_tau, 1e-12)))
        threshold = new_version_by_agent.get(p.agent_id)
        is_new = threshold is not None and p.version_id >= threshold
        is_old = threshold is not None and p.version_id < threshold
        new_bonus = 0.0
        if is_new and current_time >= route_switch_tick and current_time <= delivery_tick + new_version_boost_window:
            new_bonus = new_version_weight
        if is_new and delivery_tick <= current_time <= delivery_tick + new_version_boost_window:
            new_bonus += post_switch_new_version_bonus
        old_penalty = old_version_weight if is_old and current_time >= delivery_tick else 0.0
        scores[i] = base.get(i, 0.0) * message_confidence + new_bonus - old_penalty
    return scores


def select_wave_critical_dynamic(
    points: list[MotionTubePoint],
    budget_points: int,
    horizon: int,
    critical_mask: np.ndarray,
    current_time: int,
    freshness_tau: float,
    route_switch_tick: int,
    delivery_tick: int,
    new_version_by_agent: dict[int, int],
    grid_height: int,
    critical_budget_ratio: float = 0.7,
    coverage_budget_ratio: float = 0.3,
    new_version_weight: float = 1.0,
    new_version_boost_window: int = 10,
    old_version_weight: float = 2.0,
    balance_time_bins: bool = False,
    balance_agents: bool = False,
    balance_spatial_bins: bool = False,
    use_route_balance: bool = True,
    post_switch_new_version_bonus: float = 0.0,
    lambda_agent: float = 0.0,
    lambda_time: float = 0.0,
    lambda_space: float = 0.0,
) -> tuple[list[MotionTubePoint], dict[int, float]]:
    budget = min(max(0, int(budget_points)), len(points))
    scores = dynamic_scores(
        points,
        horizon,
        critical_mask,
        current_time,
        freshness_tau,
        route_switch_tick,
        delivery_tick,
        new_version_by_agent,
        new_version_weight=new_version_weight,
        new_version_boost_window=new_version_boost_window,
        old_version_weight=old_version_weight,
        post_switch_new_version_bonus=post_switch_new_version_bonus,
    )
    if budget <= 0 or not points:
        return [], scores
    if budget >= len(points):
        return list(points), scores

    critical_budget = int(round(budget * critical_budget_ratio))
    critical_budget = min(max(0, critical_budget), budget)
    coverage_budget = min(budget - critical_budget, int(round(budget * coverage_budget_ratio)))
    if critical_budget + coverage_budget < budget:
        coverage_budget = budget - critical_budget

    ordered = sorted(range(len(points)), key=lambda i: (scores.get(i, 0.0), -i), reverse=True)
    selected_idx: list[int] = []
    selected_set: set[int] = set()
    for i in ordered[:critical_budget]:
        selected_idx.append(i)
        selected_set.add(i)

    agent_counts: dict[int, int] = defaultdict(int)
    bin_counts: dict[int, int] = defaultdict(int)
    route_counts: dict[str, int] = defaultdict(int)
    new_counts: dict[int, int] = defaultdict(int)
    spatial_counts: dict[tuple[int, int], int] = defaultdict(int)
    active_agents = max(1, len({p.agent_id for p in points}))
    agent_cap = int(np.ceil(budget / active_agents * 1.5))
    expected_agent = max(1e-9, coverage_budget / active_agents)
    time_target = max(1, int(np.ceil(max(1, coverage_budget) / 5)))
    expected_time = max(1e-9, coverage_budget / 5)
    active_spatial = {(min(4, int(p.y * 5 / max(1, grid_height))), min(7, p.x // 10)) for p in points}
    spatial_target = max(1, int(np.ceil(max(1, coverage_budget) / max(1, len(active_spatial)))))
    expected_spatial = max(1e-9, coverage_budget / max(1, len(active_spatial)))

    def update_counts(i: int) -> None:
        p = points[i]
        agent_counts[p.agent_id] += 1
        bin_counts[min(4, int((p.k - 1) * 5 / max(1, horizon)))] += 1
        route_counts[route_label_for_point(p, grid_height)] += 1
        if p.agent_id in new_version_by_agent and p.version_id >= new_version_by_agent[p.agent_id]:
            new_counts[p.agent_id] += 1
        spatial_counts[(p.y // 4, p.x // 4)] += 1

    for i in selected_idx:
        update_counts(i)

    for _ in range(coverage_budget):
        best_i = None
        best_score = -np.inf
        for i, p in enumerate(points):
            if i in selected_set:
                continue
            time_bin = min(4, int((p.k - 1) * 5 / max(1, horizon)))
            route = route_label_for_point(p, grid_height)
            spatial = (min(4, int(p.y * 5 / max(1, grid_height))), min(7, p.x // 10))
            is_new = p.agent_id in new_version_by_agent and p.version_id >= new_version_by_agent[p.agent_id]
            coverage_score = (
                1.0 / (1.0 + agent_counts[p.agent_id])
                + 0.8 / (1.0 + bin_counts[time_bin])
                + 0.8 / (1.0 + spatial_counts[spatial])
                + (1.2 / (1.0 + new_counts[p.agent_id]) if is_new else 0.0)
            )
            if use_route_balance:
                coverage_score += 0.8 / (1.0 + route_counts[route])
            if balance_time_bins:
                coverage_score += 2.0 if bin_counts[time_bin] < time_target else -1.0 * (bin_counts[time_bin] - time_target + 1)
            if balance_agents and agent_counts[p.agent_id] >= agent_cap:
                coverage_score -= 4.0 * (agent_counts[p.agent_id] - agent_cap + 1)
            if balance_spatial_bins:
                coverage_score += 1.5 if spatial_counts[spatial] < spatial_target else -0.6 * (spatial_counts[spatial] - spatial_target + 1)
            if lambda_agent > 0.0:
                coverage_score -= lambda_agent * max(0.0, agent_counts[p.agent_id] / expected_agent - 1.0)
            if lambda_time > 0.0:
                coverage_score -= lambda_time * max(0.0, bin_counts[time_bin] / expected_time - 1.0)
            if lambda_space > 0.0:
                coverage_score -= lambda_space * max(0.0, spatial_counts[spatial] / expected_spatial - 1.0)
            adjusted = 0.35 * scores.get(i, 0.0) + coverage_score
            if adjusted > best_score:
                best_score = adjusted
                best_i = i
        if best_i is None:
            break
        selected_idx.append(best_i)
        selected_set.add(best_i)
        update_counts(best_i)

    if len(selected_idx) < budget:
        for i in ordered:
            if i not in selected_set:
                selected_idx.append(i)
                selected_set.add(i)
                if len(selected_idx) >= budget:
                    break
    return [points[i] for i in selected_idx[:budget]], scores


def _sample_indices(rng: np.random.Generator, indices: list[int], n: int) -> list[int]:
    if n <= 0 or not indices:
        return []
    if n >= len(indices):
        return list(indices)
    chosen = rng.choice(len(indices), size=n, replace=False)
    return [indices[int(i)] for i in chosen]


def _stratified_agent_indices(points: list[MotionTubePoint], rng: np.random.Generator, indices: list[int], quota: int) -> list[int]:
    by_agent: dict[int, list[int]] = defaultdict(list)
    for i in indices:
        by_agent[points[i].agent_id].append(i)
    agents = list(by_agent)
    rng.shuffle(agents)
    selected: list[int] = []
    if not agents or quota <= 0:
        return selected
    base = quota // len(agents)
    rem = quota % len(agents)
    for rank, agent_id in enumerate(agents):
        take = base + (1 if rank < rem else 0)
        selected.extend(_sample_indices(rng, by_agent[agent_id], take))
    if len(selected) < quota:
        remaining = [i for i in indices if i not in set(selected)]
        selected.extend(_sample_indices(rng, remaining, quota - len(selected)))
    return selected[:quota]


def _stratified_time_indices(points: list[MotionTubePoint], rng: np.random.Generator, indices: list[int], quota: int, horizon: int) -> list[int]:
    by_bin: dict[int, list[int]] = defaultdict(list)
    for i in indices:
        b = min(4, int((points[i].k - 1) * 5 / max(1, horizon)))
        by_bin[b].append(i)
    bins = list(by_bin)
    rng.shuffle(bins)
    selected: list[int] = []
    if not bins or quota <= 0:
        return selected
    base = quota // len(bins)
    rem = quota % len(bins)
    for rank, b in enumerate(bins):
        take = base + (1 if rank < rem else 0)
        selected.extend(_sample_indices(rng, by_bin[b], take))
    if len(selected) < quota:
        remaining = [i for i in indices if i not in set(selected)]
        selected.extend(_sample_indices(rng, remaining, quota - len(selected)))
    return selected[:quota]


def select_hybrid_random_wave(
    policy_name: str,
    points: list[MotionTubePoint],
    rng: np.random.Generator,
    budget_points: int,
    horizon: int,
    critical_mask: np.ndarray,
    current_time: int,
    freshness_tau: float,
    route_switch_tick: int,
    delivery_tick: int,
    new_version_by_agent: dict[int, int],
    grid_height: int,
) -> tuple[list[MotionTubePoint], dict[int, float], dict[tuple[int, int, int, int, int, int], str]]:
    budget = min(max(0, int(budget_points)), len(points))
    scores = dynamic_scores(
        points,
        horizon,
        critical_mask,
        current_time,
        freshness_tau,
        route_switch_tick,
        delivery_tick,
        new_version_by_agent,
        new_version_weight=2.0,
        new_version_boost_window=10,
        old_version_weight=2.0,
    )
    if budget <= 0 or not points:
        return [], scores, {}

    if policy_name == "hybrid_random10_wave90":
        random_ratio = 0.10
        mode = "uniform"
    elif policy_name == "hybrid_random25_wave75":
        random_ratio = 0.25
        mode = "uniform"
    elif policy_name in {"hybrid_random50_wave50", "hybrid_newversion_random50_wave50"}:
        random_ratio = 0.50
        mode = "latest" if policy_name == "hybrid_newversion_random50_wave50" else "uniform"
    elif policy_name == "hybrid_agent_stratified_random50_wave50":
        random_ratio = 0.50
        mode = "agent"
    elif policy_name == "hybrid_time_stratified_random50_wave50":
        random_ratio = 0.50
        mode = "time"
    elif policy_name == "hybrid_random75_wave25":
        random_ratio = 0.75
        mode = "uniform"
    elif policy_name == "hybrid_random90_wave10":
        random_ratio = 0.90
        mode = "uniform"
    else:
        raise ValueError(f"Unknown hybrid policy {policy_name!r}")

    random_quota = min(budget, int(round(random_ratio * budget)))
    all_indices = list(range(len(points)))
    random_pool = all_indices
    if mode == "latest":
        latest = [i for i, p in enumerate(points) if p.agent_id in new_version_by_agent and p.version_id >= new_version_by_agent[p.agent_id]]
        random_pool = latest if len(latest) >= random_quota else all_indices

    if mode == "agent":
        random_idx = _stratified_agent_indices(points, rng, random_pool, random_quota)
    elif mode == "time":
        random_idx = _stratified_time_indices(points, rng, random_pool, random_quota, horizon)
    else:
        random_idx = _sample_indices(rng, random_pool, random_quota)

    selected_idx: list[int] = []
    selected_set: set[int] = set()
    source_by_key: dict[tuple[int, int, int, int, int, int], str] = {}

    def key_for(p: MotionTubePoint) -> tuple[int, int, int, int, int, int]:
        return (p.agent_id, p.version_id, p.future_time_abs, p.k, p.y, p.x)

    for i in random_idx:
        if i not in selected_set:
            selected_idx.append(i)
            selected_set.add(i)
            source_by_key[key_for(points[i])] = "random"

    critical_quota = budget - len(selected_idx)
    ordered = sorted((i for i in all_indices if i not in selected_set), key=lambda i: (scores.get(i, 0.0), -i), reverse=True)
    for i in ordered[:critical_quota]:
        selected_idx.append(i)
        selected_set.add(i)
        source_by_key[key_for(points[i])] = "critical"
    return [points[i] for i in selected_idx[:budget]], scores, source_by_key


def log_selected_points(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    tick: int,
    policy: str,
    regime: str,
    points: list[MotionTubePoint],
    selected: list[MotionTubePoint],
    scores: dict[int, float],
    grid_height: int,
    old_route_mask: np.ndarray,
    new_route_mask: np.ndarray,
    new_version_by_agent: dict[int, int],
    selection_source_by_key: dict[tuple[int, int, int, int, int, int], str] | None = None,
    selected_only: bool = False,
) -> None:
    selected_keys = {(p.agent_id, p.version_id, p.future_time_abs, p.k, p.y, p.x) for p in selected}
    selection_source_by_key = selection_source_by_key or {}
    for i, p in enumerate(points):
        key = (p.agent_id, p.version_id, p.future_time_abs, p.k, p.y, p.x)
        if selected_only and key not in selected_keys:
            continue
        threshold = new_version_by_agent.get(p.agent_id)
        is_old_version = threshold is not None and p.version_id < threshold
        is_new_version = threshold is not None and p.version_id >= threshold
        in_old_route = bool(old_route_mask[p.y, p.x]) if old_route_mask.size and 0 <= p.y < old_route_mask.shape[0] and 0 <= p.x < old_route_mask.shape[1] else False
        in_new_route = bool(new_route_mask[p.y, p.x]) if new_route_mask.size and 0 <= p.y < new_route_mask.shape[0] and 0 <= p.x < new_route_mask.shape[1] else False
        rows.append(
            {
                "seed": seed,
                "tick": tick,
                "policy": policy,
                "regime": regime,
                "agent_id": p.agent_id,
                "plan_version": p.version_id,
                "route_label": route_label_for_point(p, grid_height),
                "future_step": p.k,
                "x": p.x,
                "y": p.y,
                "score": scores.get(i, np.nan),
                "selected": key in selected_keys,
                "selection_source": selection_source_by_key.get(key, "critical" if key in selected_keys else "candidate"),
                "is_old_version": is_old_version,
                "is_new_version": is_new_version,
                "is_old_route": in_old_route,
                "is_new_route": in_new_route,
                "time_bin": min(4, int((p.k - 1) * 5 / max(1, max(pp.k for pp in points) if points else 1))),
                "spatial_bin_y": min(4, int(p.y * 5 / max(1, grid_height))),
                "spatial_bin_x": min(7, p.x // 10),
            }
        )


def active_version_stats(buffer: IntentionBuffer, switched_ids: set[int], new_version_by_agent: dict[int, int]) -> tuple[int, int, int, int]:
    old_count = 0
    new_count = 0
    delivered_new = 0
    versions_by_agent: dict[int, set[int]] = defaultdict(set)
    for msg in buffer.get_active_messages():
        if msg.agent_id not in switched_ids:
            continue
        threshold = new_version_by_agent.get(msg.agent_id)
        if threshold is None:
            continue
        versions_by_agent[msg.agent_id].add(msg.version_id)
        if msg.version_id < threshold:
            old_count += 1
        else:
            new_count += 1
            delivered_new += 1
    mixed = 0
    for aid, versions in versions_by_agent.items():
        threshold = new_version_by_agent.get(aid, 10**9)
        if any(v < threshold for v in versions) and any(v >= threshold for v in versions):
            mixed += 1
    return old_count, new_count, mixed, delivered_new


def run_e5b_single(
    seed: int,
    delay: int,
    buffer_mode: str,
    cfg: E5BConfig,
    snapshots: list[tuple[str, int, np.ndarray, np.ndarray, object]],
    selected_point_rows: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    scenario = get_scenario(cfg.scenario)
    rng = np.random.default_rng(seed)
    agents = generate_distribution_agents(scenario, cfg.agents, rng, cfg.distribution_mode)
    route_variants = initialize_route_variants(agents, scenario, cfg.distribution_mode)
    buffer = IntentionBuffer(mode=buffer_mode, freshness_tau=cfg.freshness_tau, ttl=cfg.ttl)
    switched_ids: set[int] = set()
    new_version_by_agent: dict[int, int] = {}
    old_route_mask = np.zeros((scenario.grid.height, scenario.grid.width), dtype=bool)
    new_route_mask = np.zeros_like(old_route_mask)
    rows: list[dict] = []

    for current_time in range(cfg.sim_steps):
        for agent in agents:
            agent.step()

        forced_candidates: list[MotionTubePoint] = []
        switched_this_tick: list[RollingAgent] = []
        if current_time == cfg.route_switch_tick:
            n_switch = int(round(len(agents) * np.clip(cfg.switch_ratio, 0.0, 1.0)))
            switch_idx = rng.choice(len(agents), size=n_switch, replace=False)
            old_points_all: list[MotionTubePoint] = []
            new_points_all: list[MotionTubePoint] = []
            for idx in switch_idx:
                agent = agents[int(idx)]
                old_points, new_points, _ = switch_agent_route(agent, scenario, current_time, cfg.horizon, cfg, route_variants)
                if not new_points:
                    continue
                switched_ids.add(agent.agent_id)
                switched_this_tick.append(agent)
                new_version_by_agent[agent.agent_id] = agent.version_id
                old_points_all.extend(old_points)
                new_points_all.extend(new_points)
                if cfg.force_upload_on_replan:
                    forced_candidates.extend(new_points)
            old_route_mask = tube_points_mask(old_points_all, old_route_mask.shape, radius=2)
            new_route_mask = tube_points_mask(new_points_all, new_route_mask.shape, radius=2)

        oracle_points: list[MotionTubePoint] = []
        for agent in agents:
            oracle_points.extend(make_tube_points(agent, current_time, cfg.horizon, cfg.base_radius, "thin", 0.0, 2.0, cfg.beta, agent.version_id))
        R_full = build_rolling_field(oracle_points, scenario.grid, current_time, cfg.horizon)

        message_count = 0
        uploaded_tube_points = 0
        candidate_tube_points_per_upload = 0

        is_regular_upload = current_time % max(1, cfg.upload_period) == 0
        if is_regular_upload or forced_candidates:
            candidates: list[MotionTubePoint] = list(forced_candidates)
            skipped = {a.agent_id for a in switched_this_tick} if cfg.force_upload_on_replan else set()
            if is_regular_upload:
                for agent in agents:
                    if agent.agent_id in skipped:
                        continue
                    agent.version_id += 1
                    candidates.extend(make_tube_points(agent, current_time, cfg.horizon, cfg.base_radius, "thin", cfg.width_growth, 2.0, cfg.beta, agent.version_id))
            candidate_tube_points_per_upload = len(candidates)
            selector_policy = cfg.policy
            if cfg.policy == "fixed_hybrid_25_75":
                selector_policy = "hybrid_random25_wave75"
            elif cfg.policy == "adaptive_ratio_rule":
                if cfg.budget_points <= 500:
                    selector_policy = "hybrid_random50_wave50"
                elif cfg.budget_points < 900:
                    selector_policy = "hybrid_random25_wave75"
                else:
                    selector_policy = "hybrid_random10_wave90"
            is_hybrid = selector_policy.startswith("hybrid_")
            is_dynamic_wave = selector_policy.startswith("wave_critical_dynamic") or selector_policy.startswith("wave_dynamic_")
            source_by_key: dict[tuple[int, int, int, int, int, int], str] = {}
            if is_hybrid:
                selected, scores, source_by_key = select_hybrid_random_wave(
                    selector_policy,
                    candidates,
                    rng,
                    cfg.budget_points,
                    cfg.horizon,
                    scenario.critical_mask,
                    current_time,
                    cfg.freshness_tau,
                    cfg.route_switch_tick,
                    cfg.route_switch_tick + delay,
                    new_version_by_agent,
                    scenario.grid.height,
                )
            elif is_dynamic_wave:
                selected, scores = select_wave_critical_dynamic(
                    candidates,
                    cfg.budget_points,
                    cfg.horizon,
                    scenario.critical_mask,
                    current_time,
                    cfg.freshness_tau,
                    cfg.route_switch_tick,
                    cfg.route_switch_tick + delay,
                    new_version_by_agent,
                    scenario.grid.height,
                    critical_budget_ratio=cfg.critical_budget_ratio,
                    coverage_budget_ratio=cfg.coverage_budget_ratio,
                    new_version_weight=cfg.new_version_weight,
                    new_version_boost_window=cfg.new_version_boost_window,
                    old_version_weight=cfg.old_version_weight,
                    balance_time_bins=cfg.balance_time_bins,
                    balance_agents=cfg.balance_agents,
                    balance_spatial_bins=cfg.balance_spatial_bins,
                    use_route_balance=cfg.use_route_balance,
                    post_switch_new_version_bonus=cfg.post_switch_new_version_bonus,
                    lambda_agent=cfg.lambda_agent,
                    lambda_time=cfg.lambda_time,
                    lambda_space=cfg.lambda_space,
                )
            else:
                selected = select_tube_points(selector_policy, candidates, rng, cfg.budget_points, cfg.horizon, scenario.critical_mask)
                scores = base_criticality_scores(candidates, cfg.horizon, scenario.critical_mask)
            if cfg.selected_diagnostics and selected_point_rows is not None:
                log_selected_points(
                    selected_point_rows,
                    seed=seed,
                    tick=current_time,
                    policy=cfg.policy,
                    regime=cfg.diagnostics_regime,
                    points=candidates,
                    selected=selected,
                    scores=scores,
                    grid_height=scenario.grid.height,
                    old_route_mask=old_route_mask,
                    new_route_mask=new_route_mask,
                    new_version_by_agent=new_version_by_agent,
                    selection_source_by_key=source_by_key,
                    selected_only=cfg.selected_diagnostics_selected_only,
                )
            for msg in make_messages_from_points(selected, current_time, delay, cfg.horizon, cfg.policy):
                buffer.enqueue_message(msg)
                message_count += 1
                uploaded_tube_points += len(msg.tube_points)

        delivered_count = buffer.deliver_messages(current_time)
        buffer.update_buffer(current_time)
        comm_points = buffer.get_active_intentions(current_time)
        R_comm = build_rolling_field(comm_points, scenario.grid, current_time, cfg.horizon)
        metrics = compute_metrics(R_full, R_comm, cfg.policy, top_k=cfg.top_k, peak_tolerance_radius=cfg.peak_tolerance_radius, active_threshold_ratio=cfg.active_threshold_ratio)
        old_route_ghost, new_route_hidden = route_specific_metrics(R_full, R_comm, old_route_mask, new_route_mask)
        old_count, new_count, mixed_count, delivered_new = active_version_stats(buffer, switched_ids, new_version_by_agent)
        stats = buffer.get_buffer_stats(current_time)
        metrics.update(
            {
                "seed": seed,
                "tick": current_time,
                "scenario": cfg.scenario,
                "agents": len(agents),
                "horizon": cfg.horizon,
                "upload_period": cfg.upload_period,
                "delay": delay,
                "buffer_mode": buffer_mode,
                "policy": cfg.policy,
                "route_switch_tick": cfg.route_switch_tick,
                "switch_ratio": cfg.switch_ratio,
                "force_upload_on_replan": cfg.force_upload_on_replan,
                "old_route_ghost_pressure": old_route_ghost,
                "new_route_hidden_congestion": new_route_hidden,
                "mean_intention_age": stats.mean_intention_age,
                "max_intention_age": stats.max_intention_age,
                "stale_intention_ratio": stats.stale_intention_ratio,
                "active_intention_count": stats.active_intention_count,
                "old_version_active_count": old_count,
                "new_version_active_count": new_count,
                "mixed_version_agent_count": mixed_count,
                "expired_intention_count": stats.expired_intention_count,
                "overwrite_count": stats.overwrite_count,
                "switched_agent_count": len(switched_ids),
                "delivered_new_version_count": delivered_new,
                "message_count": message_count,
                "delivered_message_count": delivered_count,
                "candidate_tube_points_per_upload": candidate_tube_points_per_upload,
                "uploaded_tube_points": uploaded_tube_points,
            }
        )
        rows.append(metrics)

        snapshot_buffer_mode = getattr(cfg, "snapshot_buffer_mode", "rolling_overwrite")
        if seed == 0 and delay == 5 and buffer_mode == snapshot_buffer_mode:
            wanted = {cfg.route_switch_tick - 1, cfg.route_switch_tick + 2, cfg.route_switch_tick + delay + 2}
            if current_time in wanted:
                label_extra = f"tau={getattr(cfg, 'freshness_tau', np.nan):g} {buffer_mode}"
                snapshots.append((f"{label_extra} tick={current_time}", current_time, R_full.copy(), R_comm.copy(), scenario.grid))
    return pd.DataFrame(rows)


def run_e5b_stale_intention_ghost(cfg: E5BConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    snapshots: list[tuple[str, int, np.ndarray, np.ndarray, object]] = []
    total = len(cfg.seeds) * len(cfg.delay_list) * len(cfg.buffer_modes)
    run_idx = 0
    for seed in cfg.seeds:
        for delay in cfg.delay_list:
            for buffer_mode in cfg.buffer_modes:
                run_idx += 1
                print(f"[E5B {run_idx}/{total}] seed={seed} delay={delay} buffer={buffer_mode}", flush=True)
                frames.append(run_e5b_single(seed, delay, buffer_mode, cfg, snapshots))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_dir / "metrics_tick.csv", index=False)
    metric_cols = [
        "ghost_pressure",
        "hidden_congestion",
        "old_route_ghost_pressure",
        "new_route_hidden_congestion",
        "field_fidelity_score",
        "mean_intention_age",
        "old_version_active_count",
        "mixed_version_agent_count",
    ]
    summary = df.groupby(["delay", "buffer_mode", "policy"], dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(str(p) for p in col if p) if isinstance(col, tuple) else col for col in summary.columns]
    summary.to_csv(out_dir / "summary.csv", index=False)
    event = make_e5b_event_window_summary(df, cfg)
    event.to_csv(out_dir / "event_window_summary.csv", index=False)
    save_e5b_plots(df, event, snapshots, cfg, plots_dir)
    print_e5b_summary(df, event, cfg, out_dir)
    return df, summary


def make_e5b_event_window_summary(df: pd.DataFrame, cfg: E5BConfig) -> pd.DataFrame:
    rows = []
    for (delay, buffer_mode), sub in df.groupby(["delay", "buffer_mode"], dropna=False):
        start = cfg.route_switch_tick
        end = cfg.route_switch_tick + int(delay) + 10
        win = sub[(sub["tick"] >= start) & (sub["tick"] <= end)]
        if win.empty:
            continue
        ghost_peak_idx = win["old_route_ghost_pressure"].idxmax()
        ghost_peak_tick = int(win.loc[ghost_peak_idx, "tick"])
        peak_val = float(win.loc[ghost_peak_idx, "old_route_ghost_pressure"])
        after_peak = win[win["tick"] >= ghost_peak_tick]
        decay = after_peak[after_peak["old_route_ghost_pressure"] <= 0.5 * peak_val]
        rows.append(
            {
                "delay": delay,
                "buffer_mode": buffer_mode,
                "peak_ghost_pressure_after_switch": float(win["ghost_pressure"].max()),
                "peak_old_route_ghost_after_switch": peak_val,
                "peak_new_route_hidden_after_switch": float(win["new_route_hidden_congestion"].max()),
                "time_to_ghost_peak": ghost_peak_tick - cfg.route_switch_tick,
                "time_to_ghost_decay": np.nan if decay.empty else int(decay.iloc[0]["tick"]) - cfg.route_switch_tick,
                "old_version_duration": int((win["old_version_active_count"] > 0).sum()),
                "new_version_delivery_tick": np.nan if win[win["new_version_active_count"] > 0].empty else int(win[win["new_version_active_count"] > 0]["tick"].min()),
                "mean_ghost_pressure_window": float(win["ghost_pressure"].mean()),
                "mean_hidden_congestion_window": float(win["hidden_congestion"].mean()),
                "mean_field_fidelity_score_window": float(win["field_fidelity_score"].mean()),
            }
        )
    return pd.DataFrame(rows)


def save_e5b_plots(df: pd.DataFrame, event: pd.DataFrame, snapshots, cfg: E5BConfig, plots_dir: Path) -> None:
    ts = df[(df["seed"] == 0) & (df["delay"] == 5)]
    metrics = ["ghost_pressure", "hidden_congestion", "old_route_ghost_pressure", "new_route_hidden_congestion", "mean_intention_age", "old_version_active_count", "new_version_active_count"]
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    for ax, metric in zip(axes.ravel(), metrics):
        for mode, sub in ts.groupby("buffer_mode"):
            sub = sub.sort_values("tick")
            ax.plot(sub["tick"], sub[metric], label=mode)
        ax.axvline(cfg.route_switch_tick, color="k", linestyle="--", alpha=0.5)
        ax.axvline(cfg.route_switch_tick + 5, color="r", linestyle="--", alpha=0.35)
        ax.set_title(metric)
        ax.grid(alpha=0.25)
    axes.ravel()[-1].axis("off")
    axes[0, 1].legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "e5b_stale_ghost_timeseries.png", dpi=150)
    plt.close(fig)

    event5 = event[event["delay"] == 5]
    bar_metrics = ["peak_old_route_ghost_after_switch", "peak_new_route_hidden_after_switch", "mean_ghost_pressure_window", "mean_hidden_congestion_window"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, metric in zip(axes.ravel(), bar_metrics):
        ax.bar(event5["buffer_mode"], event5[metric])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "e5b_buffer_mode_comparison.png", dpi=150)
    plt.close(fig)

    sweep_metrics = ["peak_old_route_ghost_after_switch", "peak_new_route_hidden_after_switch", "old_version_duration", "mean_field_fidelity_score_window"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, metric in zip(axes.ravel(), sweep_metrics):
        for mode, sub in event.groupby("buffer_mode"):
            sub = sub.sort_values("delay")
            ax.plot(sub["delay"], sub[metric], marker="o", label=mode)
        ax.set_title(metric)
        ax.set_xlabel("delay")
        ax.grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "e5b_delay_sweep.png", dpi=150)
    plt.close(fig)

    save_e5b_representative_maps(snapshots, plots_dir / "e5b_representative_maps.png")


def save_e5b_representative_maps(snapshots, out_path: Path) -> None:
    if not snapshots:
        return
    fig, axes = plt.subplots(len(snapshots), 4, figsize=(17, 3.8 * len(snapshots)))
    if len(snapshots) == 1:
        axes = np.array([axes])
    for row, (label, tick, R_full, R_comm, grid) in enumerate(snapshots):
        R_full = R_full.copy()
        R_comm = R_comm.copy()
        R_full[np.abs(R_full) < 1e-9] = 0.0
        R_comm[np.abs(R_comm) < 1e-9] = 0.0
        full_sum = project_sum(R_full)
        comm_sum = project_sum(R_comm)
        ghost_sum = project_sum(np.maximum(R_comm - R_full, 0.0))
        hidden_sum = project_sum(np.maximum(R_full - R_comm, 0.0))
        common_vmax = max(float(np.max(full_sum)), 1e-12)
        panels = [("R_full", full_sum, "magma", common_vmax), ("R_comm", comm_sum, "magma", common_vmax), ("ghost_map", ghost_sum, "inferno", None), ("hidden_map", hidden_sum, "viridis", None)]
        for col, (title, data, cmap, vmax) in enumerate(panels):
            ax = axes[row, col]
            masked = np.ma.array(data, mask=grid.obstacles)
            ax.imshow(grid.obstacles, cmap="gray_r", alpha=0.35, interpolation="nearest")
            im = ax.imshow(masked, cmap=cmap, interpolation="nearest", vmin=0.0, vmax=vmax)
            ax.set_title(f"{title}\n{label}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, shrink=0.72)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_e5b_summary(df: pd.DataFrame, event: pd.DataFrame, cfg: E5BConfig, out_dir: Path) -> None:
    print("\nE5B Stale Intention Ghost Test Summary")
    before = df[df["tick"] < cfg.route_switch_tick]
    after = df[(df["tick"] >= cfg.route_switch_tick) & (df["tick"] <= cfg.route_switch_tick + 15)]
    print(f"before switch ghost={before['ghost_pressure'].mean():.4f}, old_route_ghost={before['old_route_ghost_pressure'].mean():.4f}")
    print(f"event window ghost={after['ghost_pressure'].mean():.4f}, old_route_ghost={after['old_route_ghost_pressure'].mean():.4f}, new_route_hidden={after['new_route_hidden_congestion'].mean():.4f}")
    print("\nEvent window summary:")
    print(event.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if (df["old_route_ghost_pressure"] <= 1e-12).all() and (df["ghost_pressure"] <= 1e-12).all():
        print("WARNING: No stale ghost was activated. Check route switch geometry and stale versions.")
    print(f"\nWrote E5B outputs to: {out_dir}")


@dataclass(frozen=True)
class E5CConfig:
    scenario: str = "dual_corridor"
    agents: int = 60
    seeds: tuple[int, ...] = (0, 1, 2)
    sim_steps: int = 80
    horizon: int = 25
    upload_period: int = 5
    route_switch_tick: int = 30
    switch_ratio: float = 0.5
    delay: int = 5
    policy: str = "periodic_full"
    buffer_mode: str = "overwrite_with_decay_TTL"
    freshness_tau_list: tuple[float, ...] = (2.0, 3.0, 5.0, 10.0, 20.0, 9999.0)
    ttl: int = 30
    base_radius: float = 1.5
    width_growth: float = 0.0
    force_upload_on_replan: bool = True
    plot_level: str = "minimal"
    beta: float = 0.03
    top_k: int = 20
    peak_tolerance_radius: float = 3.0
    active_threshold_ratio: float = 0.05
    snapshot_buffer_mode: str = "overwrite_with_decay_TTL"


def add_e5c_balanced_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ghost_norm = out["ghost_pressure"] / (1.0 + out["ghost_pressure"])
    hidden_norm = out["hidden_congestion"]
    old_ghost_norm = out["old_route_ghost_pressure"] / (1.0 + out["old_route_ghost_pressure"])
    new_hidden_norm = out["new_route_hidden_congestion"] / (1.0 + out["new_route_hidden_congestion"])
    out["ghost_norm_e5c"] = ghost_norm
    out["hidden_norm_e5c"] = hidden_norm
    out["old_route_ghost_norm"] = old_ghost_norm
    out["new_route_hidden_norm"] = new_hidden_norm
    out["balanced_loss_g1"] = hidden_norm + ghost_norm
    out["balanced_loss_g2"] = hidden_norm + 2.0 * ghost_norm
    out["balanced_fidelity_score_g1"] = -out["balanced_loss_g1"]
    out["balanced_fidelity_score_g2"] = -out["balanced_loss_g2"]
    out["route_balanced_loss_g1_tick"] = new_hidden_norm + old_ghost_norm
    out["route_balanced_loss_g2_tick"] = new_hidden_norm + 2.0 * old_ghost_norm
    return out


def make_e5c_event_window_summary(df: pd.DataFrame, cfg: E5CConfig) -> pd.DataFrame:
    rows = []
    start = cfg.route_switch_tick
    end = cfg.route_switch_tick + cfg.delay + 20
    for (seed, tau), sub in df.groupby(["seed", "freshness_tau"], dropna=False):
        win = sub[(sub["tick"] >= start) & (sub["tick"] <= end)].sort_values("tick")
        if win.empty:
            continue

        def auc(col: str) -> float:
            return float(np.trapz(win[col].to_numpy(dtype=float), win["tick"].to_numpy(dtype=float)))

        rows.append(
            {
                "seed": seed,
                "freshness_tau": tau,
                "peak_old_route_ghost_after_switch": float(win["old_route_ghost_pressure"].max()),
                "mean_old_route_ghost_after_switch": float(win["old_route_ghost_pressure"].mean()),
                "auc_old_route_ghost_after_switch": auc("old_route_ghost_pressure"),
                "peak_new_route_hidden_after_switch": float(win["new_route_hidden_congestion"].max()),
                "mean_new_route_hidden_after_switch": float(win["new_route_hidden_congestion"].mean()),
                "auc_new_route_hidden_after_switch": auc("new_route_hidden_congestion"),
                "peak_global_ghost_after_switch": float(win["ghost_pressure"].max()),
                "mean_global_ghost_after_switch": float(win["ghost_pressure"].mean()),
                "auc_global_ghost_after_switch": auc("ghost_pressure"),
                "peak_global_hidden_after_switch": float(win["hidden_congestion"].max()),
                "mean_global_hidden_after_switch": float(win["hidden_congestion"].mean()),
                "auc_global_hidden_after_switch": auc("hidden_congestion"),
                "ghost_duration_above_threshold": int((win["ghost_pressure"] > 0.05).sum()),
                "hidden_duration_above_threshold": int((win["hidden_congestion"] > 0.05).sum()),
                "mean_field_fidelity_score_window": float(win["field_fidelity_score"].mean()),
                "mean_balanced_loss_g1": float(win["balanced_loss_g1"].mean()),
                "mean_balanced_loss_g2": float(win["balanced_loss_g2"].mean()),
                "mean_balanced_fidelity_score_g1": float(win["balanced_fidelity_score_g1"].mean()),
                "mean_balanced_fidelity_score_g2": float(win["balanced_fidelity_score_g2"].mean()),
                "mean_route_balanced_loss_g1": float(win["route_balanced_loss_g1_tick"].mean()),
                "mean_route_balanced_loss_g2": float(win["route_balanced_loss_g2_tick"].mean()),
                "mean_old_route_ghost_norm": float(win["old_route_ghost_norm"].mean()),
                "mean_new_route_hidden_norm": float(win["new_route_hidden_norm"].mean()),
            }
        )
    return pd.DataFrame(rows)


def run_e5c_freshness_decay_sensitivity(cfg: E5CConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    snapshots: list[tuple[str, int, np.ndarray, np.ndarray, object]] = []
    total = len(cfg.seeds) * len(cfg.freshness_tau_list)
    run_idx = 0
    for seed in cfg.seeds:
        for tau in cfg.freshness_tau_list:
            run_idx += 1
            print(f"[E5C {run_idx}/{total}] seed={seed} freshness_tau={tau:g}", flush=True)
            e5b_cfg = E5BConfig(
                scenario=cfg.scenario,
                agents=cfg.agents,
                seeds=(seed,),
                sim_steps=cfg.sim_steps,
                horizon=cfg.horizon,
                upload_period=cfg.upload_period,
                route_switch_tick=cfg.route_switch_tick,
                switch_ratio=cfg.switch_ratio,
                delay_list=(cfg.delay,),
                policy=cfg.policy,
                buffer_modes=(cfg.buffer_mode,),
                freshness_tau=float(tau),
                ttl=cfg.ttl,
                base_radius=cfg.base_radius,
                width_growth=cfg.width_growth,
                force_upload_on_replan=cfg.force_upload_on_replan,
                plot_level=cfg.plot_level,
                beta=cfg.beta,
                top_k=cfg.top_k,
                peak_tolerance_radius=cfg.peak_tolerance_radius,
                active_threshold_ratio=cfg.active_threshold_ratio,
            )
            object.__setattr__(e5b_cfg, "snapshot_buffer_mode", cfg.snapshot_buffer_mode)
            frame = run_e5b_single(seed, cfg.delay, cfg.buffer_mode, e5b_cfg, snapshots)
            frame["freshness_tau"] = float(tau)
            frames.append(add_e5c_balanced_columns(frame))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_dir / "metrics_tick.csv", index=False)
    event = make_e5c_event_window_summary(df, cfg)
    event.to_csv(out_dir / "event_window_summary.csv", index=False)
    summary = event.groupby("freshness_tau", dropna=False).agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(str(p) for p in col if p) if isinstance(col, tuple) else col for col in summary.columns]
    summary.to_csv(out_dir / "summary_by_tau.csv", index=False)
    save_e5c_plots(df, event, snapshots, cfg, plots_dir)
    print_e5c_summary(event, cfg, out_dir)
    return df, event


def save_e5c_plots(df: pd.DataFrame, event: pd.DataFrame, snapshots, cfg: E5CConfig, plots_dir: Path) -> None:
    summary = event.groupby("freshness_tau", dropna=False).mean(numeric_only=True).reset_index().sort_values("freshness_tau")
    metrics = [
        "peak_old_route_ghost_after_switch",
        "auc_old_route_ghost_after_switch",
        "peak_new_route_hidden_after_switch",
        "auc_new_route_hidden_after_switch",
        "mean_balanced_loss_g1",
        "mean_balanced_loss_g2",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    x = np.arange(len(summary))
    labels = [f"{v:g}" for v in summary["freshness_tau"]]
    for ax, metric in zip(axes.ravel(), metrics):
        ax.plot(x, summary[metric], marker="o")
        ax.set_xticks(x, labels, rotation=25)
        ax.set_title(metric)
        ax.set_xlabel("freshness_tau")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "e5c_tau_tradeoff.png", dpi=150)
    plt.close(fig)

    ts = df[(df["seed"] == 0) & (df["freshness_tau"].isin([3.0, 10.0, 9999.0]))]
    ts_metrics = ["old_route_ghost_pressure", "new_route_hidden_congestion", "ghost_pressure", "hidden_congestion", "mean_intention_age"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for ax, metric in zip(axes.ravel(), ts_metrics):
        for tau, sub in ts.groupby("freshness_tau"):
            sub = sub.sort_values("tick")
            ax.plot(sub["tick"], sub[metric], label=f"tau={tau:g}")
        ax.axvline(cfg.route_switch_tick, color="k", linestyle="--", alpha=0.5)
        ax.axvline(cfg.route_switch_tick + cfg.delay, color="r", linestyle="--", alpha=0.35)
        ax.set_title(metric)
        ax.grid(alpha=0.25)
    axes.ravel()[-1].axis("off")
    axes[0, 1].legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "e5c_time_series_representative.png", dpi=150)
    plt.close(fig)

    save_e5c_representative_maps(snapshots, plots_dir / "e5c_representative_maps.png")
    save_e5c_best_tau_summary(event, plots_dir / "e5c_best_tau_summary.png")


def save_e5c_representative_maps(snapshots, out_path: Path) -> None:
    wanted = []
    for label, tick, R_full, R_comm, grid in snapshots:
        if any(f"tau={tau}" in label or f"tau={float(tau):g}" in label for tau in [3, 10, 9999]):
            wanted.append((label, tick, R_full, R_comm, grid))
    if not wanted:
        return
    fig, axes = plt.subplots(len(wanted), 4, figsize=(17, 3.3 * len(wanted)))
    if len(wanted) == 1:
        axes = np.array([axes])
    ghost_vmax = max(float(np.max(project_sum(np.maximum(R_comm - R_full, 0.0)))) for _, _, R_full, R_comm, _ in wanted)
    hidden_vmax = max(float(np.max(project_sum(np.maximum(R_full - R_comm, 0.0)))) for _, _, R_full, R_comm, _ in wanted)
    for row, (label, tick, R_full, R_comm, grid) in enumerate(wanted):
        R_full = R_full.copy()
        R_comm = R_comm.copy()
        R_full[np.abs(R_full) < 1e-9] = 0.0
        R_comm[np.abs(R_comm) < 1e-9] = 0.0
        full_sum = project_sum(R_full)
        comm_sum = project_sum(R_comm)
        ghost_sum = project_sum(np.maximum(R_comm - R_full, 0.0))
        hidden_sum = project_sum(np.maximum(R_full - R_comm, 0.0))
        common_vmax = max(float(np.max(full_sum)), 1e-12)
        panels = [
            ("R_full", full_sum, "magma", common_vmax),
            ("R_comm", comm_sum, "magma", common_vmax),
            ("ghost_map", ghost_sum, "inferno", max(ghost_vmax, 1e-12)),
            ("hidden_map", hidden_sum, "viridis", max(hidden_vmax, 1e-12)),
        ]
        for col, (title, data, cmap, vmax) in enumerate(panels):
            ax = axes[row, col]
            masked = np.ma.array(data, mask=grid.obstacles)
            ax.imshow(grid.obstacles, cmap="gray_r", alpha=0.35, interpolation="nearest")
            im = ax.imshow(masked, cmap=cmap, interpolation="nearest", vmin=0.0, vmax=vmax)
            ax.set_title(f"{title}\n{label}", fontsize=8)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, shrink=0.72)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_e5c_best_tau_summary(event: pd.DataFrame, out_path: Path) -> None:
    summary = event.groupby("freshness_tau", dropna=False).mean(numeric_only=True)
    g1_tau = float(summary["mean_balanced_loss_g1"].idxmin())
    g2_tau = float(summary["mean_balanced_loss_g2"].idxmin())
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    text = (
        "E5C best freshness_tau\n\n"
        f"best_tau_by_balanced_loss_g1: {g1_tau:g}\n"
        f"best_tau_by_balanced_loss_g2: {g2_tau:g}\n\n"
        f"g1 loss: {summary.loc[g1_tau, 'mean_balanced_loss_g1']:.4f}\n"
        f"g2 loss: {summary.loc[g2_tau, 'mean_balanced_loss_g2']:.4f}"
    )
    ax.text(0.05, 0.9, text, va="top", ha="left", fontsize=12, family="monospace")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_e5c_summary(event: pd.DataFrame, cfg: E5CConfig, out_dir: Path) -> None:
    print("\nE5C Freshness Decay Sensitivity Summary")
    summary = event.groupby("freshness_tau", dropna=False).mean(numeric_only=True).sort_index()
    cols = [
        "peak_old_route_ghost_after_switch",
        "auc_old_route_ghost_after_switch",
        "peak_new_route_hidden_after_switch",
        "auc_new_route_hidden_after_switch",
        "mean_balanced_loss_g1",
        "mean_balanced_loss_g2",
    ]
    print(summary[cols].to_string(float_format=lambda v: f"{v:.4f}"))
    g1_tau = float(summary["mean_balanced_loss_g1"].idxmin())
    g2_tau = float(summary["mean_balanced_loss_g2"].idxmin())
    print(f"\nbest_tau_by_balanced_loss_g1: {g1_tau:g}")
    print(f"best_tau_by_balanced_loss_g2: {g2_tau:g}")
    ghost_curve = summary["mean_old_route_ghost_after_switch"]
    hidden_curve = summary["mean_new_route_hidden_after_switch"]
    print(f"old_route_ghost trend tau_min->{summary.index[0]:g}: {ghost_curve.iloc[0]:.4f}, tau_max->{summary.index[-1]:g}: {ghost_curve.iloc[-1]:.4f}")
    print(f"new_route_hidden trend tau_min->{summary.index[0]:g}: {hidden_curve.iloc[0]:.4f}, tau_max->{summary.index[-1]:g}: {hidden_curve.iloc[-1]:.4f}")
    if ghost_curve.max() - ghost_curve.min() < 1e-3 and hidden_curve.max() - hidden_curve.min() < 1e-3:
        print("WARNING: Freshness tau has little effect. Check whether freshness weight is applied to stale old intentions before delivery.")
    print(f"\nWrote E5C outputs to: {out_dir}")


def run_e5c2_fine_tau_sweep(cfg: E5CConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    total = len(cfg.seeds) * len(cfg.freshness_tau_list)
    run_idx = 0
    for seed in cfg.seeds:
        for tau in cfg.freshness_tau_list:
            run_idx += 1
            print(f"[E5C2 {run_idx}/{total}] seed={seed} freshness_tau={tau:g}", flush=True)
            e5b_cfg = E5BConfig(
                scenario=cfg.scenario,
                agents=cfg.agents,
                seeds=(seed,),
                sim_steps=cfg.sim_steps,
                horizon=cfg.horizon,
                upload_period=cfg.upload_period,
                route_switch_tick=cfg.route_switch_tick,
                switch_ratio=cfg.switch_ratio,
                delay_list=(cfg.delay,),
                policy=cfg.policy,
                buffer_modes=(cfg.buffer_mode,),
                freshness_tau=float(tau),
                ttl=cfg.ttl,
                base_radius=cfg.base_radius,
                width_growth=cfg.width_growth,
                force_upload_on_replan=cfg.force_upload_on_replan,
                plot_level=cfg.plot_level,
                beta=cfg.beta,
                top_k=cfg.top_k,
                peak_tolerance_radius=cfg.peak_tolerance_radius,
                active_threshold_ratio=cfg.active_threshold_ratio,
            )
            frame = run_e5b_single(seed, cfg.delay, cfg.buffer_mode, e5b_cfg, [])
            frame["freshness_tau"] = float(tau)
            frames.append(add_e5c_balanced_columns(frame))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_dir / "metrics_tick.csv", index=False)
    event = make_e5c_event_window_summary(df, cfg)
    event.to_csv(out_dir / "event_window_summary.csv", index=False)
    summary = event.groupby("freshness_tau", dropna=False).agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(str(p) for p in col if p) if isinstance(col, tuple) else col for col in summary.columns]
    summary.to_csv(out_dir / "summary_by_tau.csv", index=False)
    best = make_e5c2_best_tau_summary(event)
    (out_dir / "best_tau_summary.txt").write_text(best["text"], encoding="utf-8")
    save_e5c2_plots(df, event, best, cfg, plots_dir)
    print_e5c2_summary(event, best, cfg, out_dir)
    return df, event


def make_e5c2_best_tau_summary(event: pd.DataFrame) -> dict:
    summary = event.groupby("freshness_tau", dropna=False).mean(numeric_only=True).sort_index()
    best_global_g1 = float(summary["mean_balanced_loss_g1"].idxmin())
    best_global_g2 = float(summary["mean_balanced_loss_g2"].idxmin())
    best_route_g1 = float(summary["mean_route_balanced_loss_g1"].idxmin())
    best_route_g2 = float(summary["mean_route_balanced_loss_g2"].idxmin())
    if best_route_g1 in {20.0, 25.0, 30.0} and best_route_g2 in {20.0, 25.0, 30.0}:
        candidates = [best_route_g1, best_route_g2]
        recommended = min(candidates, key=lambda t: float(summary.loc[t, "mean_route_balanced_loss_g1"] + summary.loc[t, "mean_route_balanced_loss_g2"]))
    elif best_route_g1 != best_route_g2:
        recommended = best_route_g2
    else:
        recommended = best_route_g1

    warnings: list[str] = []
    if recommended == 9999.0:
        warnings.append("WARNING: No decay remains best. Freshness decay may not be beneficial under current loss weights.")
    if recommended == 10.0:
        warnings.append("WARNING: Best tau is at lower boundary. Consider testing tau smaller than 10.")
    if recommended == 50.0:
        warnings.append("WARNING: Best tau is near upper boundary. Consider testing tau larger than 50.")
    text = "\n".join(
        [
            "E5C2 Fine Tau Sweep Best Tau Summary",
            f"best_tau_by_global_balanced_loss_g1: {best_global_g1:g}",
            f"best_tau_by_global_balanced_loss_g2: {best_global_g2:g}",
            f"best_tau_by_route_balanced_loss_g1: {best_route_g1:g}",
            f"best_tau_by_route_balanced_loss_g2: {best_route_g2:g}",
            f"recommended_tau: {recommended:g}",
            *warnings,
            "",
        ]
    )
    return {
        "summary": summary,
        "best_global_g1": best_global_g1,
        "best_global_g2": best_global_g2,
        "best_route_g1": best_route_g1,
        "best_route_g2": best_route_g2,
        "recommended": float(recommended),
        "warnings": warnings,
        "text": text,
    }


def save_e5c2_plots(df: pd.DataFrame, event: pd.DataFrame, best: dict, cfg: E5CConfig, plots_dir: Path) -> None:
    grouped = event.groupby("freshness_tau", dropna=False).agg(["mean", "std"]).reset_index()
    grouped.columns = ["_".join(str(p) for p in col if p) if isinstance(col, tuple) else col for col in grouped.columns]
    grouped = grouped.sort_values("freshness_tau")
    x = np.arange(len(grouped))
    labels = [f"{v:g}" for v in grouped["freshness_tau"]]
    metrics = [
        "peak_old_route_ghost_after_switch",
        "auc_old_route_ghost_after_switch",
        "peak_new_route_hidden_after_switch",
        "auc_new_route_hidden_after_switch",
        "mean_route_balanced_loss_g1",
        "mean_route_balanced_loss_g2",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for ax, metric in zip(axes.ravel(), metrics):
        ax.errorbar(x, grouped[f"{metric}_mean"], yerr=grouped[f"{metric}_std"].fillna(0.0), marker="o", capsize=3)
        ax.set_xticks(x, labels, rotation=25)
        ax.set_title(metric)
        ax.set_xlabel("freshness_tau")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "e5c2_tau_tradeoff_fine.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in ["mean_balanced_loss_g1", "mean_balanced_loss_g2", "mean_route_balanced_loss_g1", "mean_route_balanced_loss_g2"]:
        ax.errorbar(x, grouped[f"{metric}_mean"], yerr=grouped[f"{metric}_std"].fillna(0.0), marker="o", capsize=3, label=metric)
    ax.set_xticks(x, labels, rotation=25)
    ax.set_xlabel("freshness_tau")
    ax.set_ylabel("loss")
    ax.set_title("global vs route-specific balanced loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "e5c2_global_vs_route_loss.png", dpi=150)
    plt.close(fig)

    ts = df[(df["seed"] == 0) & (df["freshness_tau"].isin([10.0, 20.0, 9999.0]))]
    ts_metrics = ["old_route_ghost_pressure", "new_route_hidden_congestion", "ghost_pressure", "hidden_congestion", "mean_intention_age"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for ax, metric in zip(axes.ravel(), ts_metrics):
        for tau, sub in ts.groupby("freshness_tau"):
            sub = sub.sort_values("tick")
            ax.plot(sub["tick"], sub[metric], label=f"tau={tau:g}")
        ax.axvline(cfg.route_switch_tick, color="k", linestyle="--", alpha=0.5)
        ax.axvline(cfg.route_switch_tick + cfg.delay, color="r", linestyle="--", alpha=0.35)
        ax.set_title(metric)
        ax.grid(alpha=0.25)
    axes.ravel()[-1].axis("off")
    axes[0, 1].legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "e5c2_time_series_representative.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.05, 0.95, best["text"], va="top", ha="left", fontsize=11, family="monospace")
    fig.tight_layout()
    fig.savefig(plots_dir / "e5c2_best_tau_summary.png", dpi=150)
    plt.close(fig)


def print_e5c2_summary(event: pd.DataFrame, best: dict, cfg: E5CConfig, out_dir: Path) -> None:
    print("\nE5C2 Fine Tau Sweep Summary")
    summary = best["summary"]
    cols = [
        "mean_old_route_ghost_after_switch",
        "mean_new_route_hidden_after_switch",
        "mean_route_balanced_loss_g1",
        "mean_route_balanced_loss_g2",
        "mean_balanced_loss_g1",
        "mean_balanced_loss_g2",
    ]
    print(summary[cols].to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n" + best["text"])
    ghost = summary["mean_old_route_ghost_after_switch"]
    hidden = summary["mean_new_route_hidden_after_switch"]
    print(f"old_route_ghost tau_min={summary.index[0]:g}: {ghost.iloc[0]:.4f}, tau_max={summary.index[-1]:g}: {ghost.iloc[-1]:.4f}")
    print(f"new_route_hidden tau_min={summary.index[0]:g}: {hidden.iloc[0]:.4f}, tau_max={summary.index[-1]:g}: {hidden.iloc[-1]:.4f}")
    if ghost.max() - ghost.min() < 1e-3 and hidden.max() - hidden.min() < 1e-3:
        print("WARNING: Freshness decay may not be applied correctly to stale old intentions.")
    if best["recommended"] == 9999.0:
        print("No decay is still best under current setup.")
    print(f"\nWrote E5C2 outputs to: {out_dir}")


@dataclass(frozen=True)
class E5DConfig:
    scenario: str = "dual_corridor"
    agents: int = 60
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    sim_steps: int = 80
    horizon: int = 25
    upload_period: int = 5
    route_switch_tick: int = 30
    switch_ratio: float = 0.5
    delay: int = 5
    buffer_mode: str = "overwrite_with_decay_TTL"
    ttl: int = 30
    freshness_tau_regimes: tuple[tuple[str, float], ...] = (("no_decay", 9999.0), ("balanced", 20.0), ("ghost_sensitive", 3.0))
    policies: tuple[str, ...] = ("periodic_full", "random_budget", "freshness_based", "wave_critical")
    budget_points: int = 800
    base_radius: float = 1.5
    width_growth: float = 0.0
    force_upload_on_replan: bool = True
    plot_level: str = "minimal"
    beta: float = 0.03
    top_k: int = 20
    peak_tolerance_radius: float = 3.0
    active_threshold_ratio: float = 0.05


def make_e5d_event_window_summary(df: pd.DataFrame, cfg: E5DConfig) -> pd.DataFrame:
    rows: list[dict] = []
    start = cfg.route_switch_tick
    end = cfg.route_switch_tick + cfg.delay + 20
    for (seed, regime, tau, policy), sub in df.groupby(["seed", "freshness_regime", "freshness_tau", "policy"], dropna=False):
        win = sub[(sub["tick"] >= start) & (sub["tick"] <= end)].sort_values("tick")
        if win.empty:
            continue

        def auc(col: str) -> float:
            return float(np.trapz(win[col].to_numpy(dtype=float), win["tick"].to_numpy(dtype=float)))

        rows.append(
            {
                "seed": seed,
                "freshness_regime": regime,
                "freshness_tau": tau,
                "policy": policy,
                "auc_old_route_ghost_after_switch": auc("old_route_ghost_pressure"),
                "auc_new_route_hidden_after_switch": auc("new_route_hidden_congestion"),
                "peak_old_route_ghost_after_switch": float(win["old_route_ghost_pressure"].max()),
                "peak_new_route_hidden_after_switch": float(win["new_route_hidden_congestion"].max()),
                "mean_old_route_ghost_after_switch": float(win["old_route_ghost_pressure"].mean()),
                "mean_new_route_hidden_after_switch": float(win["new_route_hidden_congestion"].mean()),
                "mean_global_ghost_after_switch": float(win["ghost_pressure"].mean()),
                "mean_global_hidden_after_switch": float(win["hidden_congestion"].mean()),
                "mean_field_fidelity_score_window": float(win["field_fidelity_score"].mean()),
                "mean_active_shape_fidelity_score_window": float(win["active_shape_fidelity_score"].mean()),
                "mean_field_mass_ratio_window": float(win["field_mass_ratio"].mean()),
                "mean_top_k_peak_overlap_tolerant_window": float(win["top_k_peak_overlap_tolerant"].mean()),
                "mean_entropy_diff_window": float(win["entropy_diff"].mean()),
                "mean_balanced_loss_g1": float(win["balanced_loss_g1"].mean()),
                "mean_balanced_loss_g2": float(win["balanced_loss_g2"].mean()),
                "mean_route_balanced_loss_g1": float(win["route_balanced_loss_g1_tick"].mean()),
                "mean_route_balanced_loss_g2": float(win["route_balanced_loss_g2_tick"].mean()),
                "mean_old_route_ghost_norm": float(win["old_route_ghost_norm"].mean()),
                "mean_new_route_hidden_norm": float(win["new_route_hidden_norm"].mean()),
                "message_count": float(win["message_count"].mean()),
                "uploaded_tube_points": float(win["uploaded_tube_points"].mean()),
                "mean_intention_age": float(win["mean_intention_age"].mean()),
                "old_version_active_count": float(win["old_version_active_count"].mean()),
                "new_version_active_count": float(win["new_version_active_count"].mean()),
            }
        )
    return pd.DataFrame(rows)


def flatten_grouped_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = ["_".join(str(p) for p in col if p) if isinstance(col, tuple) else col for col in out.columns]
    return out


def make_e5d_policy_delta(event: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "auc_old_route_ghost_after_switch",
        "auc_new_route_hidden_after_switch",
        "mean_route_balanced_loss_g1",
        "mean_route_balanced_loss_g2",
        "mean_field_mass_ratio_window",
        "mean_top_k_peak_overlap_tolerant_window",
    ]
    rows: list[dict] = []
    means = event.groupby(["freshness_regime", "policy"], dropna=False)[metrics + ["uploaded_tube_points"]].mean()
    for regime in event["freshness_regime"].drop_duplicates():
        if (regime, "random_budget") not in means.index or (regime, "wave_critical") not in means.index:
            continue
        random_row = means.loc[(regime, "random_budget")]
        wave_row = means.loc[(regime, "wave_critical")]
        row = {
            "freshness_regime": regime,
            "random_uploaded_tube_points": float(random_row["uploaded_tube_points"]),
            "wave_uploaded_tube_points": float(wave_row["uploaded_tube_points"]),
        }
        for metric in metrics:
            row[f"delta_{metric}"] = float(wave_row[metric] - random_row[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def save_e5d_plots(df: pd.DataFrame, event: pd.DataFrame, delta: pd.DataFrame, cfg: E5DConfig, plots_dir: Path) -> None:
    regime_order = [name for name, _ in cfg.freshness_tau_regimes]
    policy_order = list(cfg.policies)
    grouped = event.groupby(["freshness_regime", "policy"], dropna=False).agg(["mean", "std"])
    metrics = [
        ("auc_old_route_ghost_after_switch", "old_route_ghost AUC"),
        ("auc_new_route_hidden_after_switch", "new_route_hidden AUC"),
        ("mean_route_balanced_loss_g1", "route_balanced_loss_g1"),
        ("mean_route_balanced_loss_g2", "route_balanced_loss_g2"),
    ]
    fig, axes = plt.subplots(len(regime_order), len(metrics), figsize=(18, 4.2 * len(regime_order)), squeeze=False)
    for r, regime in enumerate(regime_order):
        for c, (metric, title) in enumerate(metrics):
            ax = axes[r, c]
            means = []
            stds = []
            labels = []
            for policy in policy_order:
                key = (regime, policy)
                if key not in grouped.index:
                    continue
                labels.append(policy)
                means.append(float(grouped.loc[key, (metric, "mean")]))
                stds.append(float(grouped.loc[key, (metric, "std")]) if not pd.isna(grouped.loc[key, (metric, "std")]) else 0.0)
            ax.bar(labels, means, yerr=stds, capsize=3)
            ax.set_title(f"{regime}: {title}")
            ax.tick_params(axis="x", rotation=35)
            ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "e5d_policy_comparison_by_regime.png", dpi=150)
    plt.close(fig)

    if not delta.empty:
        delta_metrics = [
            ("delta_auc_old_route_ghost_after_switch", "delta_old_route_ghost_auc"),
            ("delta_auc_new_route_hidden_after_switch", "delta_new_route_hidden_auc"),
            ("delta_mean_route_balanced_loss_g1", "delta_route_balanced_loss_g1"),
            ("delta_mean_route_balanced_loss_g2", "delta_route_balanced_loss_g2"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for ax, (metric, title) in zip(axes.ravel(), delta_metrics):
            ax.bar(delta["freshness_regime"], delta[metric])
            ax.axhline(0.0, color="k", linewidth=1)
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle("wave_critical minus random_budget; negative is good for these metrics", y=1.02)
        fig.tight_layout()
        fig.savefig(plots_dir / "e5d_wave_vs_random_delta_by_regime.png", dpi=150)
        plt.close(fig)

    ts = df[(df["seed"] == 0) & (df["policy"].isin(["random_budget", "wave_critical"])) & (df["freshness_regime"].isin(regime_order))]
    ts_metrics = [
        "old_route_ghost_pressure",
        "new_route_hidden_congestion",
        "ghost_pressure",
        "hidden_congestion",
        "mean_intention_age",
    ]
    fig, axes = plt.subplots(len(ts_metrics), len(regime_order), figsize=(5.2 * len(regime_order), 13), squeeze=False)
    for c, regime in enumerate(regime_order):
        reg = ts[ts["freshness_regime"] == regime]
        for r, metric in enumerate(ts_metrics):
            ax = axes[r, c]
            for policy, sub in reg.groupby("policy", sort=False):
                sub = sub.sort_values("tick")
                ax.plot(sub["tick"], sub[metric], label=policy)
            ax.axvline(cfg.route_switch_tick, color="k", linestyle="--", alpha=0.5)
            ax.axvline(cfg.route_switch_tick + cfg.delay, color="r", linestyle="--", alpha=0.35)
            ax.set_title(f"{regime}: {metric}", fontsize=9)
            ax.grid(alpha=0.25)
            if r == 0 and c == len(regime_order) - 1:
                ax.legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "e5d_representative_timeseries.png", dpi=150)
    plt.close(fig)

    sanity = event.groupby(["freshness_regime", "policy"], dropna=False)[["uploaded_tube_points", "message_count"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(len(regime_order))
    width = 0.18
    for i, policy in enumerate(policy_order):
        sub = sanity[sanity["policy"] == policy].set_index("freshness_regime").reindex(regime_order)
        offset = (i - (len(policy_order) - 1) / 2) * width
        axes[0].bar(x + offset, sub["uploaded_tube_points"], width=width, label=policy)
        axes[1].bar(x + offset, sub["message_count"], width=width, label=policy)
    for ax, title in zip(axes, ["uploaded_tube_points", "message_count"]):
        ax.set_xticks(x, regime_order, rotation=20)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "e5d_uploaded_points_sanity.png", dpi=150)
    plt.close(fig)


def make_e5d_key_findings(event: pd.DataFrame, delta: pd.DataFrame, cfg: E5DConfig) -> str:
    lines = ["E5D Policy Comparison under Freshness Regimes", ""]
    summary = event.groupby(["freshness_regime", "policy"], dropna=False).mean(numeric_only=True)
    for regime, tau in cfg.freshness_tau_regimes:
        lines.append(f"Regime: {regime} (freshness_tau={tau:g})")
        for policy in cfg.policies:
            key = (regime, policy)
            if key not in summary.index:
                continue
            row = summary.loc[key]
            lines.append(
                f"  {policy}: old_ghost_auc={row['auc_old_route_ghost_after_switch']:.4f}, "
                f"new_hidden_auc={row['auc_new_route_hidden_after_switch']:.4f}, "
                f"route_g1={row['mean_route_balanced_loss_g1']:.4f}, "
                f"route_g2={row['mean_route_balanced_loss_g2']:.4f}, "
                f"uploaded={row['uploaded_tube_points']:.1f}"
            )
        d = delta[delta["freshness_regime"] == regime]
        if not d.empty:
            dr = d.iloc[0]
            lines.append(
                "  wave_critical - random_budget: "
                f"delta_old_ghost_auc={dr['delta_auc_old_route_ghost_after_switch']:.4f}, "
                f"delta_new_hidden_auc={dr['delta_auc_new_route_hidden_after_switch']:.4f}, "
                f"delta_route_g1={dr['delta_mean_route_balanced_loss_g1']:.4f}, "
                f"delta_route_g2={dr['delta_mean_route_balanced_loss_g2']:.4f}"
            )
        lines.append("")

    if not delta.empty:
        better_g1 = int((delta["delta_mean_route_balanced_loss_g1"] < 0).sum())
        better_hidden = int((delta["delta_auc_new_route_hidden_after_switch"] < 0).sum())
        better_ghost = int((delta["delta_auc_old_route_ghost_after_switch"] < 0).sum())
        lines.extend(
            [
                "Overall:",
                f"  wave_critical beats random_budget on route_balanced_loss_g1 in {better_g1}/{len(delta)} regimes.",
                f"  wave_critical lowers new-route hidden AUC in {better_hidden}/{len(delta)} regimes.",
                f"  wave_critical lowers old-route ghost AUC in {better_ghost}/{len(delta)} regimes.",
            ]
        )
        if better_g1 == len(delta):
            conclusion = "wave_critical is robust across the evaluated freshness regimes under route_balanced_loss_g1."
        elif better_g1 > 0:
            conclusion = "wave_critical is regime-dependent: it helps under some freshness regimes but is not universally best."
        else:
            conclusion = "wave_critical does not beat random_budget under the evaluated route-balanced metric."
        lines.append(f"  conclusion: {conclusion}")
    lines.append("")
    lines.append("No universal freshness_tau is recommended here; tau remains a hidden/ghost risk preference knob.")
    return "\n".join(lines)


def print_e5d_summary(event: pd.DataFrame, delta: pd.DataFrame, cfg: E5DConfig, out_dir: Path) -> None:
    print("\nE5D Dynamic Policy by Tau Regime Summary")
    cols = [
        "auc_old_route_ghost_after_switch",
        "auc_new_route_hidden_after_switch",
        "mean_route_balanced_loss_g1",
        "mean_route_balanced_loss_g2",
        "uploaded_tube_points",
    ]
    table = event.groupby(["freshness_regime", "policy"], dropna=False)[cols].mean().reset_index()
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if not delta.empty:
        print("\nwave_critical minus random_budget deltas:")
        print(delta.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    budgeted = event[event["policy"].isin(["random_budget", "freshness_based", "wave_critical"])]
    fairness = budgeted.groupby(["freshness_regime", "policy"])["uploaded_tube_points"].mean().unstack("policy")
    print("\nBudgeted uploaded_tube_points sanity:")
    print(fairness.to_string(float_format=lambda v: f"{v:.2f}"))
    print("\nNo single tau is universal; policy conclusions are evaluated per freshness regime.")
    print(f"Wrote E5D outputs to: {out_dir}")


def run_e5d_dynamic_policy_by_tau_regime(cfg: E5DConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    total = len(cfg.seeds) * len(cfg.freshness_tau_regimes) * len(cfg.policies)
    run_idx = 0
    for seed in cfg.seeds:
        for regime, tau in cfg.freshness_tau_regimes:
            for policy in cfg.policies:
                run_idx += 1
                print(f"[E5D {run_idx}/{total}] seed={seed} regime={regime} tau={tau:g} policy={policy}", flush=True)
                e5b_cfg = E5BConfig(
                    scenario=cfg.scenario,
                    agents=cfg.agents,
                    seeds=(seed,),
                    sim_steps=cfg.sim_steps,
                    horizon=cfg.horizon,
                    upload_period=cfg.upload_period,
                    route_switch_tick=cfg.route_switch_tick,
                    switch_ratio=cfg.switch_ratio,
                    delay_list=(cfg.delay,),
                    policy=policy,
                    buffer_modes=(cfg.buffer_mode,),
                    budget_points=cfg.budget_points,
                    freshness_tau=float(tau),
                    ttl=cfg.ttl,
                    base_radius=cfg.base_radius,
                    width_growth=cfg.width_growth,
                    force_upload_on_replan=cfg.force_upload_on_replan,
                    plot_level=cfg.plot_level,
                    beta=cfg.beta,
                    top_k=cfg.top_k,
                    peak_tolerance_radius=cfg.peak_tolerance_radius,
                    active_threshold_ratio=cfg.active_threshold_ratio,
                )
                frame = run_e5b_single(seed, cfg.delay, cfg.buffer_mode, e5b_cfg, [])
                frame["freshness_regime"] = regime
                frame["freshness_tau"] = float(tau)
                frames.append(add_e5c_balanced_columns(frame))
    df = pd.concat(frames, ignore_index=True)
    event = make_e5d_event_window_summary(df, cfg)
    merge_cols = [
        "seed",
        "freshness_regime",
        "freshness_tau",
        "policy",
        "auc_old_route_ghost_after_switch",
        "auc_new_route_hidden_after_switch",
        "peak_old_route_ghost_after_switch",
        "peak_new_route_hidden_after_switch",
    ]
    df = df.merge(event[merge_cols], on=["seed", "freshness_regime", "freshness_tau", "policy"], how="left")
    df.to_csv(out_dir / "metrics_tick.csv", index=False)

    summary_cols = [
        "auc_old_route_ghost_after_switch",
        "auc_new_route_hidden_after_switch",
        "peak_old_route_ghost_after_switch",
        "peak_new_route_hidden_after_switch",
        "mean_global_ghost_after_switch",
        "mean_global_hidden_after_switch",
        "mean_field_fidelity_score_window",
        "mean_active_shape_fidelity_score_window",
        "mean_field_mass_ratio_window",
        "mean_top_k_peak_overlap_tolerant_window",
        "mean_entropy_diff_window",
        "mean_balanced_loss_g1",
        "mean_balanced_loss_g2",
        "mean_route_balanced_loss_g1",
        "mean_route_balanced_loss_g2",
        "message_count",
        "uploaded_tube_points",
        "mean_intention_age",
        "old_version_active_count",
        "new_version_active_count",
    ]
    summary = event.groupby(["freshness_regime", "freshness_tau", "policy"], dropna=False)[summary_cols].agg(["mean", "std"]).reset_index()
    summary = flatten_grouped_columns(summary)
    summary.to_csv(out_dir / "summary_by_policy_tau.csv", index=False)

    delta = make_e5d_policy_delta(event)
    delta.to_csv(out_dir / "policy_delta_vs_random.csv", index=False)
    findings = make_e5d_key_findings(event, delta, cfg)
    (out_dir / "e5d_key_findings.txt").write_text(findings + "\n", encoding="utf-8")
    save_e5d_plots(df, event, delta, cfg, plots_dir)
    print_e5d_summary(event, delta, cfg, out_dir)
    return df, event


@dataclass(frozen=True)
class E5EConfig:
    scenario: str = "dual_corridor"
    agents: int = 60
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    sim_steps: int = 80
    horizon: int = 25
    upload_period: int = 5
    route_switch_tick: int = 30
    switch_ratio: float = 0.5
    delay: int = 5
    buffer_mode: str = "overwrite_with_decay_TTL"
    ttl: int = 30
    freshness_tau: float = 20.0
    policies: tuple[str, ...] = ("periodic_full", "random_budget", "wave_critical", "wave_critical_dynamic")
    budget_points: int = 800
    base_radius: float = 1.5
    width_growth: float = 0.0
    force_upload_on_replan: bool = True
    plot_level: str = "minimal"
    analysis_window: int = 40
    beta: float = 0.03
    top_k: int = 20
    peak_tolerance_radius: float = 3.0
    active_threshold_ratio: float = 0.05


def make_e5e_pre_post_summary(df: pd.DataFrame, cfg: E5EConfig) -> pd.DataFrame:
    rows: list[dict] = []
    switch = cfg.route_switch_tick
    delivery = cfg.route_switch_tick + cfg.delay
    post_end = cfg.route_switch_tick + cfg.analysis_window
    event_end = cfg.route_switch_tick + cfg.delay + 20
    for (seed, policy), sub in df.groupby(["seed", "policy"], dropna=False):
        sub = sub.sort_values("tick")
        pre = sub[(sub["tick"] >= switch) & (sub["tick"] < delivery)]
        post = sub[(sub["tick"] >= delivery) & (sub["tick"] <= post_end)]
        event = sub[(sub["tick"] >= switch) & (sub["tick"] <= event_end)]

        def auc(frame: pd.DataFrame, col: str) -> float:
            if frame.empty:
                return 0.0
            return float(np.trapz(frame[col].to_numpy(dtype=float), frame["tick"].to_numpy(dtype=float)))

        rows.append(
            {
                "seed": seed,
                "policy": policy,
                "freshness_tau": cfg.freshness_tau,
                "route_switch_tick": switch,
                "delivery_tick": delivery,
                "pre_delivery_old_route_ghost_auc": auc(pre, "old_route_ghost_pressure"),
                "post_delivery_old_route_ghost_auc": auc(post, "old_route_ghost_pressure"),
                "pre_delivery_new_route_hidden_auc": auc(pre, "new_route_hidden_congestion"),
                "post_delivery_new_route_hidden_auc": auc(post, "new_route_hidden_congestion"),
                "pre_delivery_peak_old_route_ghost": 0.0 if pre.empty else float(pre["old_route_ghost_pressure"].max()),
                "post_delivery_peak_old_route_ghost": 0.0 if post.empty else float(post["old_route_ghost_pressure"].max()),
                "pre_delivery_peak_new_route_hidden": 0.0 if pre.empty else float(pre["new_route_hidden_congestion"].max()),
                "post_delivery_peak_new_route_hidden": 0.0 if post.empty else float(post["new_route_hidden_congestion"].max()),
                "old_route_ghost_auc": auc(event, "old_route_ghost_pressure"),
                "new_route_hidden_auc": auc(event, "new_route_hidden_congestion"),
                "peak_old_route_ghost_after_switch": 0.0 if event.empty else float(event["old_route_ghost_pressure"].max()),
                "peak_new_route_hidden_after_switch": 0.0 if event.empty else float(event["new_route_hidden_congestion"].max()),
                "route_balanced_loss_g1": 0.0 if event.empty else float(event["route_balanced_loss_g1_tick"].mean()),
                "route_balanced_loss_g2": 0.0 if event.empty else float(event["route_balanced_loss_g2_tick"].mean()),
                "balanced_loss_g1": 0.0 if event.empty else float(event["balanced_loss_g1"].mean()),
                "balanced_loss_g2": 0.0 if event.empty else float(event["balanced_loss_g2"].mean()),
                "field_fidelity_score": 0.0 if event.empty else float(event["field_fidelity_score"].mean()),
                "active_shape_fidelity_score": 0.0 if event.empty else float(event["active_shape_fidelity_score"].mean()),
                "field_mass_ratio": 0.0 if event.empty else float(event["field_mass_ratio"].mean()),
                "top_k_peak_overlap_tolerant": 0.0 if event.empty else float(event["top_k_peak_overlap_tolerant"].mean()),
                "entropy_diff": 0.0 if event.empty else float(event["entropy_diff"].mean()),
                "message_count": 0.0 if event.empty else float(event["message_count"].mean()),
                "uploaded_tube_points": 0.0 if event.empty else float(event["uploaded_tube_points"].mean()),
                "mean_intention_age": 0.0 if event.empty else float(event["mean_intention_age"].mean()),
                "old_version_active_count": 0.0 if event.empty else float(event["old_version_active_count"].mean()),
                "new_version_active_count": 0.0 if event.empty else float(event["new_version_active_count"].mean()),
            }
        )
    return pd.DataFrame(rows)


def make_selected_point_summary(selected_df: pd.DataFrame) -> pd.DataFrame:
    if selected_df.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (seed, policy, regime), sub in selected_df.groupby(["seed", "policy", "regime"], dropna=False):
        selected = sub[sub["selected"]].copy()
        per_agent = selected.groupby("agent_id").size()
        time_counts = selected.groupby("time_bin").size().to_dict()
        row = {
            "seed": seed,
            "policy": policy,
            "regime": regime,
            "selected_old_route_points": int(selected["is_old_route"].sum()),
            "selected_new_route_points": int(selected["is_new_route"].sum()),
            "selected_old_version_points": int(selected["is_old_version"].sum()),
            "selected_new_version_points": int(selected["is_new_version"].sum()),
            "selected_points_per_agent_mean": float(per_agent.mean()) if not per_agent.empty else 0.0,
            "selected_points_per_agent_std": float(per_agent.std(ddof=0)) if len(per_agent) > 1 else 0.0,
            "selected_points_per_time_bin": ",".join(f"{int(k)}:{int(v)}" for k, v in sorted(time_counts.items())),
            "selected_points_per_time_bin_std": float(np.std([time_counts.get(b, 0) for b in range(5)])),
        }
        for b in range(5):
            row[f"selected_time_bin_{b}"] = int(time_counts.get(b, 0))
        rows.append(row)
    return pd.DataFrame(rows)


def make_e5e_delta_summary(prepost: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "old_route_ghost_auc",
        "new_route_hidden_auc",
        "post_delivery_new_route_hidden_auc",
        "route_balanced_loss_g1",
        "route_balanced_loss_g2",
    ]
    means = prepost.groupby("policy", dropna=False)[metrics].mean()
    rows: list[dict] = []
    for baseline in ["wave_critical", "random_budget"]:
        if "wave_critical_dynamic" not in means.index or baseline not in means.index:
            continue
        row = {"comparison": f"wave_critical_dynamic_minus_{baseline}"}
        for metric in metrics:
            row[f"delta_{metric}"] = float(means.loc["wave_critical_dynamic", metric] - means.loc[baseline, metric])
        rows.append(row)
    return pd.DataFrame(rows)


def save_e5e_plots(prepost: pd.DataFrame, selected_summary: pd.DataFrame, delta: pd.DataFrame, df: pd.DataFrame, cfg: E5EConfig, plots_dir: Path) -> None:
    policy_order = list(cfg.policies)
    grouped = prepost.groupby("policy", dropna=False).agg(["mean", "std"])

    metrics = [
        "pre_delivery_new_route_hidden_auc",
        "post_delivery_new_route_hidden_auc",
        "pre_delivery_old_route_ghost_auc",
        "post_delivery_old_route_ghost_auc",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, metric in zip(axes.ravel(), metrics):
        labels, means, stds = [], [], []
        for policy in policy_order:
            if policy in grouped.index:
                labels.append(policy)
                means.append(float(grouped.loc[policy, (metric, "mean")]))
                stds.append(float(grouped.loc[policy, (metric, "std")]) if not pd.isna(grouped.loc[policy, (metric, "std")]) else 0.0)
        ax.bar(labels, means, yerr=stds, capsize=3)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "e5e_pre_post_delivery_comparison.png", dpi=150)
    plt.close(fig)

    metrics2 = [
        "old_route_ghost_auc",
        "new_route_hidden_auc",
        "route_balanced_loss_g1",
        "route_balanced_loss_g2",
        "post_delivery_new_route_hidden_auc",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    for ax, metric in zip(axes.ravel(), metrics2):
        labels, means, stds = [], [], []
        for policy in policy_order:
            if policy in grouped.index:
                labels.append(policy)
                means.append(float(grouped.loc[policy, (metric, "mean")]))
                stds.append(float(grouped.loc[policy, (metric, "std")]) if not pd.isna(grouped.loc[policy, (metric, "std")]) else 0.0)
        ax.bar(labels, means, yerr=stds, capsize=3)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    fig.savefig(plots_dir / "e5e_policy_comparison_balanced_regime.png", dpi=150)
    plt.close(fig)

    if not delta.empty:
        fig, axes = plt.subplots(3, 2, figsize=(13, 10))
        delta_metrics = [c for c in delta.columns if c.startswith("delta_")]
        for ax, metric in zip(axes.ravel(), delta_metrics):
            ax.bar(delta["comparison"], delta[metric])
            ax.axhline(0.0, color="k", linewidth=1)
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=20)
            ax.grid(axis="y", alpha=0.25)
        for ax in axes.ravel()[len(delta_metrics):]:
            ax.axis("off")
        fig.suptitle("wave_critical_dynamic deltas; negative is good", y=1.02)
        fig.tight_layout()
        fig.savefig(plots_dir / "e5e_wave_dynamic_vs_static_delta.png", dpi=150)
        plt.close(fig)

    diag = selected_summary[selected_summary["policy"].isin(["wave_critical", "wave_critical_dynamic"])]
    if not diag.empty:
        diag_numeric_cols = diag.select_dtypes(include=[np.number]).columns.tolist()
        diag_group = diag.groupby("policy", dropna=False)[diag_numeric_cols].agg(["mean", "std"])
        diag_metrics = [
            "selected_new_version_points",
            "selected_old_version_points",
            "selected_new_route_points",
            "selected_old_route_points",
            "selected_points_per_agent_std",
        ]
        fig, axes = plt.subplots(3, 2, figsize=(13, 10))
        for ax, metric in zip(axes.ravel(), diag_metrics):
            labels, means, stds = [], [], []
            for policy in ["wave_critical", "wave_critical_dynamic"]:
                if policy in diag_group.index:
                    labels.append(policy)
                    means.append(float(diag_group.loc[policy, (metric, "mean")]))
                    stds.append(float(diag_group.loc[policy, (metric, "std")]) if not pd.isna(diag_group.loc[policy, (metric, "std")]) else 0.0)
            ax.bar(labels, means, yerr=stds, capsize=3)
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=20)
            ax.grid(axis="y", alpha=0.25)
        axes.ravel()[-1].axis("off")
        fig.tight_layout()
        fig.savefig(plots_dir / "e5e_selected_point_diagnostics.png", dpi=150)
        plt.close(fig)


def make_e5e_key_findings(prepost: pd.DataFrame, selected_summary: pd.DataFrame, delta: pd.DataFrame) -> str:
    lines = ["E5E Dynamic Wave-Critical V2 Findings", ""]
    means = prepost.groupby("policy", dropna=False).mean(numeric_only=True)
    for policy, row in means.iterrows():
        lines.append(
            f"{policy}: old_ghost_auc={row['old_route_ghost_auc']:.4f}, "
            f"new_hidden_auc={row['new_route_hidden_auc']:.4f}, "
            f"post_new_hidden_auc={row['post_delivery_new_route_hidden_auc']:.4f}, "
            f"route_g1={row['route_balanced_loss_g1']:.4f}, route_g2={row['route_balanced_loss_g2']:.4f}"
        )
    lines.append("")
    if not delta.empty:
        lines.append("Deltas:")
        for _, row in delta.iterrows():
            lines.append(
                f"{row['comparison']}: "
                f"delta_old_ghost_auc={row['delta_old_route_ghost_auc']:.4f}, "
                f"delta_new_hidden_auc={row['delta_new_route_hidden_auc']:.4f}, "
                f"delta_post_new_hidden_auc={row['delta_post_delivery_new_route_hidden_auc']:.4f}, "
                f"delta_route_g1={row['delta_route_balanced_loss_g1']:.4f}, "
                f"delta_route_g2={row['delta_route_balanced_loss_g2']:.4f}"
            )
    lines.append("")
    if not selected_summary.empty:
        diag = selected_summary.groupby("policy", dropna=False).mean(numeric_only=True)
        for policy in ["wave_critical", "wave_critical_dynamic"]:
            if policy in diag.index:
                row = diag.loc[policy]
                lines.append(
                    f"{policy} selected diagnostics: new_version={row['selected_new_version_points']:.1f}, "
                    f"old_version={row['selected_old_version_points']:.1f}, "
                    f"new_route={row['selected_new_route_points']:.1f}, old_route={row['selected_old_route_points']:.1f}, "
                    f"per_agent_std={row['selected_points_per_agent_std']:.2f}"
                )
    lines.append("")

    useful = False
    if not delta.empty:
        static = delta[delta["comparison"] == "wave_critical_dynamic_minus_wave_critical"]
        random = delta[delta["comparison"] == "wave_critical_dynamic_minus_random_budget"]
        if not static.empty and not random.empty:
            useful = (
                float(static.iloc[0]["delta_post_delivery_new_route_hidden_auc"]) < 0
                and float(static.iloc[0]["delta_old_route_ghost_auc"]) <= 0.05
                and float(random.iloc[0]["delta_route_balanced_loss_g1"]) <= 0.0
                and float(random.iloc[0]["delta_route_balanced_loss_g2"]) <= 0.0
            )
    if useful:
        lines.append("Conclusion: wave_critical_dynamic improves post-delivery new-route recovery without worsening route-balanced losses versus random_budget.")
    else:
        lines.append("Conclusion: Static criticality is insufficient under route-switching rolling intention fields. More explicit coverage or change-aware information sharing is required.")
    return "\n".join(lines)


def run_e5e_dynamic_wave_critical_v2(cfg: E5EConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    total = len(cfg.seeds) * len(cfg.policies)
    run_idx = 0
    for seed in cfg.seeds:
        for policy in cfg.policies:
            run_idx += 1
            print(f"[E5E {run_idx}/{total}] seed={seed} tau={cfg.freshness_tau:g} policy={policy}", flush=True)
            e5b_cfg = E5BConfig(
                scenario=cfg.scenario,
                agents=cfg.agents,
                seeds=(seed,),
                sim_steps=cfg.sim_steps,
                horizon=cfg.horizon,
                upload_period=cfg.upload_period,
                route_switch_tick=cfg.route_switch_tick,
                switch_ratio=cfg.switch_ratio,
                delay_list=(cfg.delay,),
                policy=policy,
                buffer_modes=(cfg.buffer_mode,),
                budget_points=cfg.budget_points,
                freshness_tau=cfg.freshness_tau,
                ttl=cfg.ttl,
                base_radius=cfg.base_radius,
                width_growth=cfg.width_growth,
                force_upload_on_replan=cfg.force_upload_on_replan,
                plot_level=cfg.plot_level,
                beta=cfg.beta,
                top_k=cfg.top_k,
                peak_tolerance_radius=cfg.peak_tolerance_radius,
                active_threshold_ratio=cfg.active_threshold_ratio,
                selected_diagnostics=policy in {"wave_critical", "wave_critical_dynamic"},
                diagnostics_regime="balanced",
            )
            frame = run_e5b_single(seed, cfg.delay, cfg.buffer_mode, e5b_cfg, [], selected_rows)
            frame["freshness_regime"] = "balanced"
            frame["freshness_tau"] = cfg.freshness_tau
            frames.append(add_e5c_balanced_columns(frame))
    df = pd.concat(frames, ignore_index=True)
    prepost = make_e5e_pre_post_summary(df, cfg)
    merge_cols = [
        "seed",
        "policy",
        "pre_delivery_old_route_ghost_auc",
        "post_delivery_old_route_ghost_auc",
        "pre_delivery_new_route_hidden_auc",
        "post_delivery_new_route_hidden_auc",
        "old_route_ghost_auc",
        "new_route_hidden_auc",
        "route_balanced_loss_g1",
        "route_balanced_loss_g2",
    ]
    df = df.merge(prepost[merge_cols], on=["seed", "policy"], how="left")
    df.to_csv(out_dir / "metrics_tick.csv", index=False)
    summary = prepost.groupby("policy", dropna=False).agg(["mean", "std"]).reset_index()
    summary = flatten_grouped_columns(summary)
    summary.to_csv(out_dir / "summary_by_policy.csv", index=False)
    prepost.to_csv(out_dir / "pre_post_delivery_summary.csv", index=False)
    selected_df = pd.DataFrame(selected_rows)
    if selected_df.empty:
        selected_df = pd.DataFrame(
            columns=[
                "seed",
                "tick",
                "policy",
                "regime",
                "agent_id",
                "plan_version",
                "route_label",
                "future_step",
                "x",
                "y",
                "score",
                "selected",
            ]
        )
    selected_df.to_csv(out_dir / "selected_points_by_route.csv", index=False)
    selected_summary = make_selected_point_summary(selected_df)
    selected_summary.to_csv(out_dir / "selected_point_summary.csv", index=False)
    delta = make_e5e_delta_summary(prepost)
    delta.to_csv(out_dir / "policy_delta_vs_baselines.csv", index=False)
    findings = make_e5e_key_findings(prepost, selected_summary, delta)
    (out_dir / "e5e_key_findings.txt").write_text(findings + "\n", encoding="utf-8")
    save_e5e_plots(prepost, selected_summary, delta, df, cfg, plots_dir)
    print("\nE5E Dynamic Wave-Critical V2 Summary")
    print(findings)
    print(f"\nWrote E5E outputs to: {out_dir}")
    return df, prepost


@dataclass(frozen=True)
class E5FConfig:
    scenario: str = "dual_corridor"
    agents: int = 60
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    sim_steps: int = 80
    horizon: int = 25
    upload_period: int = 5
    route_switch_tick: int = 30
    switch_ratio: float = 0.5
    delay: int = 5
    buffer_mode: str = "overwrite_with_decay_TTL"
    ttl: int = 30
    freshness_tau: float = 20.0
    budget_points: int = 800
    regime: str = "balanced"
    base_radius: float = 1.5
    width_growth: float = 0.0
    force_upload_on_replan: bool = True
    plot_level: str = "minimal"
    analysis_window: int = 40
    beta: float = 0.03
    top_k: int = 20
    peak_tolerance_radius: float = 3.0
    active_threshold_ratio: float = 0.05


def e5f_policy_variants() -> list[dict[str, Any]]:
    return [
        {"policy": "periodic_full", "selector_policy": "periodic_full", "critical_budget_ratio": np.nan, "coverage_budget_ratio": np.nan, "new_version_weight": np.nan},
        {"policy": "random_budget", "selector_policy": "random_budget", "critical_budget_ratio": np.nan, "coverage_budget_ratio": np.nan, "new_version_weight": np.nan},
        {"policy": "wave_critical", "selector_policy": "wave_critical", "critical_budget_ratio": np.nan, "coverage_budget_ratio": np.nan, "new_version_weight": np.nan},
        {"policy": "wave_critical_dynamic_default", "selector_policy": "wave_critical_dynamic_default", "critical_budget_ratio": 0.7, "coverage_budget_ratio": 0.3, "new_version_weight": 1.0},
        {"policy": "wave_dynamic_c30", "selector_policy": "wave_dynamic_c30", "critical_budget_ratio": 0.7, "coverage_budget_ratio": 0.3, "new_version_weight": 1.0},
        {"policy": "wave_dynamic_c50", "selector_policy": "wave_dynamic_c50", "critical_budget_ratio": 0.5, "coverage_budget_ratio": 0.5, "new_version_weight": 1.0},
        {"policy": "wave_dynamic_c70", "selector_policy": "wave_dynamic_c70", "critical_budget_ratio": 0.3, "coverage_budget_ratio": 0.7, "new_version_weight": 1.0},
        {"policy": "wave_dynamic_c50_bonus2", "selector_policy": "wave_dynamic_c50_bonus2", "critical_budget_ratio": 0.5, "coverage_budget_ratio": 0.5, "new_version_weight": 2.0},
        {"policy": "wave_dynamic_c70_bonus2", "selector_policy": "wave_dynamic_c70_bonus2", "critical_budget_ratio": 0.3, "coverage_budget_ratio": 0.7, "new_version_weight": 2.0},
    ]


def make_e5f_delta_vs_random(prepost: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "old_route_ghost_auc",
        "new_route_hidden_auc",
        "post_delivery_new_route_hidden_auc",
        "route_balanced_loss_g1",
        "route_balanced_loss_g2",
    ]
    means = prepost.groupby("policy", dropna=False)[metrics].mean()
    if "random_budget" not in means.index:
        return pd.DataFrame()
    rows: list[dict] = []
    for policy in means.index:
        if policy in {"periodic_full", "random_budget", "wave_critical"}:
            continue
        row = {"policy": policy}
        for metric in metrics:
            row[f"delta_{metric}"] = float(means.loc[policy, metric] - means.loc["random_budget", metric])
        rows.append(row)
    return pd.DataFrame(rows)


def save_e5f_plots(prepost: pd.DataFrame, coverage_summary: pd.DataFrame, selected_summary: pd.DataFrame, out_dir: Path) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    policy_order = [v["policy"] for v in e5f_policy_variants()]
    grouped = prepost.groupby("policy", dropna=False).agg(["mean", "std"])
    metrics = [
        "old_route_ghost_auc",
        "new_route_hidden_auc",
        "post_delivery_new_route_hidden_auc",
        "route_balanced_loss_g1",
        "route_balanced_loss_g2",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    for ax, metric in zip(axes.ravel(), metrics):
        labels, means, stds = [], [], []
        for policy in policy_order:
            if policy in grouped.index:
                labels.append(policy)
                means.append(float(grouped.loc[policy, (metric, "mean")]))
                stds.append(float(grouped.loc[policy, (metric, "std")]) if not pd.isna(grouped.loc[policy, (metric, "std")]) else 0.0)
        ax.bar(labels, means, yerr=stds, capsize=3)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=38, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    fig.savefig(plots_dir / "e5f_policy_comparison.png", dpi=150)
    plt.close(fig)

    delta = make_e5f_delta_vs_random(prepost)
    if not delta.empty:
        delta_metrics = [c for c in delta.columns if c.startswith("delta_")]
        fig, axes = plt.subplots(3, 2, figsize=(16, 11))
        for ax, metric in zip(axes.ravel(), delta_metrics):
            ax.bar(delta["policy"], delta[metric])
            ax.axhline(0.0, color="k", linewidth=1)
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=35, labelsize=8)
            ax.grid(axis="y", alpha=0.25)
        for ax in axes.ravel()[len(delta_metrics):]:
            ax.axis("off")
        fig.suptitle("variant minus random_budget; negative is good", y=1.02)
        fig.tight_layout()
        fig.savefig(plots_dir / "e5f_dynamic_vs_random_delta.png", dpi=150)
        plt.close(fig)

    means = prepost.groupby("policy", dropna=False).mean(numeric_only=True)
    pareto_policies = [
        "random_budget",
        "wave_critical",
        "wave_dynamic_c30",
        "wave_dynamic_c50",
        "wave_dynamic_c70",
        "wave_dynamic_c50_bonus2",
        "wave_dynamic_c70_bonus2",
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    for policy in pareto_policies:
        if policy not in means.index:
            continue
        x = float(means.loc[policy, "post_delivery_new_route_hidden_auc"])
        y = float(means.loc[policy, "old_route_ghost_auc"])
        ax.scatter(x, y, s=55)
        ax.annotate(policy, (x, y), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("post_delivery_new_route_hidden_auc")
    ax.set_ylabel("old_route_ghost_auc")
    ax.set_title("E5F post-delivery hidden vs stale ghost")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "e5f_post_delivery_hidden_vs_ghost_pareto.png", dpi=150)
    plt.close(fig)

    diag_metrics = [
        "selected_new_route_points",
        "selected_old_route_points",
        "selected_new_version_points",
        "selected_points_per_agent_std",
        "selected_points_per_time_bin_std",
    ]
    diag = selected_summary.copy()
    if not diag.empty:
        diag_numeric_cols = diag.select_dtypes(include=[np.number]).columns.tolist()
        diag_group = diag.groupby("policy", dropna=False)[diag_numeric_cols].agg(["mean", "std"])
        fig, axes = plt.subplots(3, 2, figsize=(16, 11))
        for ax, metric in zip(axes.ravel(), diag_metrics):
            labels, means, stds = [], [], []
            for policy in policy_order:
                if policy in diag_group.index and (metric, "mean") in diag_group.columns:
                    labels.append(policy)
                    means.append(float(diag_group.loc[policy, (metric, "mean")]))
                    stds.append(float(diag_group.loc[policy, (metric, "std")]) if not pd.isna(diag_group.loc[policy, (metric, "std")]) else 0.0)
            ax.bar(labels, means, yerr=stds, capsize=3)
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=35, labelsize=8)
            ax.grid(axis="y", alpha=0.25)
        for ax in axes.ravel()[len(diag_metrics):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(plots_dir / "e5f_selected_point_diagnostics.png", dpi=150)
        plt.close(fig)


def make_e5f_findings(prepost: pd.DataFrame, coverage_summary: pd.DataFrame, selected_summary: pd.DataFrame) -> str:
    lines = ["E5F Dynamic Wave-Critical Coverage Sweep Findings", ""]
    means = prepost.groupby("policy", dropna=False).mean(numeric_only=True)
    for policy, row in means.iterrows():
        lines.append(
            f"{policy}: old_ghost_auc={row['old_route_ghost_auc']:.4f}, "
            f"new_hidden_auc={row['new_route_hidden_auc']:.4f}, "
            f"post_new_hidden_auc={row['post_delivery_new_route_hidden_auc']:.4f}, "
            f"route_g1={row['route_balanced_loss_g1']:.4f}, route_g2={row['route_balanced_loss_g2']:.4f}"
        )
    lines.append("")
    if "wave_critical" in means.index:
        static = means.loc["wave_critical"]
        better_static = []
        for policy, row in means.iterrows():
            if not str(policy).startswith("wave_dynamic") and policy != "wave_critical_dynamic_default":
                continue
            if (
                row["post_delivery_new_route_hidden_auc"] < static["post_delivery_new_route_hidden_auc"]
                and row["route_balanced_loss_g1"] < static["route_balanced_loss_g1"]
                and row["route_balanced_loss_g2"] < static["route_balanced_loss_g2"]
                and row["old_route_ghost_auc"] <= static["old_route_ghost_auc"] + 0.05
            ):
                better_static.append(str(policy))
        lines.append(f"Variants better than static wave_critical by acceptance rule: {', '.join(better_static) if better_static else 'none'}")
    better_random = []
    if "random_budget" in means.index:
        random = means.loc["random_budget"]
        for policy, row in means.iterrows():
            if not str(policy).startswith("wave_dynamic") and policy != "wave_critical_dynamic_default":
                continue
            if (
                row["post_delivery_new_route_hidden_auc"] < random["post_delivery_new_route_hidden_auc"]
                and (row["route_balanced_loss_g1"] < random["route_balanced_loss_g1"] or row["route_balanced_loss_g2"] < random["route_balanced_loss_g2"])
                and row["old_route_ghost_auc"] <= random["old_route_ghost_auc"]
            ):
                better_random.append(str(policy))
        lines.append(f"Variants better than random_budget by acceptance rule: {', '.join(better_random) if better_random else 'none'}")
    lines.append("")
    if not selected_summary.empty:
        diag = selected_summary.groupby("policy", dropna=False).mean(numeric_only=True)
        for policy in diag.index:
            if str(policy).startswith("wave") and policy in diag.index:
                row = diag.loc[policy]
                lines.append(
                    f"{policy} selected: new_route={row['selected_new_route_points']:.1f}, "
                    f"old_route={row['selected_old_route_points']:.1f}, "
                    f"new_version={row['selected_new_version_points']:.1f}, "
                    f"agent_std={row['selected_points_per_agent_std']:.2f}, "
                    f"time_bin_std={row['selected_points_per_time_bin_std']:.2f}"
                )
    lines.append("")
    if not better_random:
        lines.append("Random coverage remains a strong baseline under rolling route-switching intention fields. Pure criticality and simple dynamic criticality are insufficient. A stronger hybrid method or route-aware coverage mechanism is required.")
    else:
        lines.append(f"At least one dynamic variant beats random_budget: {', '.join(better_random)}.")
    return "\n".join(lines)


def run_e5f_dynamic_wave_coverage_sweep(cfg: E5FConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    variants = e5f_policy_variants()
    frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    total = len(cfg.seeds) * len(variants)
    run_idx = 0
    for seed in cfg.seeds:
        for variant in variants:
            run_idx += 1
            policy = str(variant["policy"])
            print(f"[E5F {run_idx}/{total}] seed={seed} policy={policy}", flush=True)
            e5b_cfg = E5BConfig(
                scenario=cfg.scenario,
                agents=cfg.agents,
                seeds=(seed,),
                sim_steps=cfg.sim_steps,
                horizon=cfg.horizon,
                upload_period=cfg.upload_period,
                route_switch_tick=cfg.route_switch_tick,
                switch_ratio=cfg.switch_ratio,
                delay_list=(cfg.delay,),
                policy=policy,
                buffer_modes=(cfg.buffer_mode,),
                budget_points=cfg.budget_points,
                freshness_tau=cfg.freshness_tau,
                ttl=cfg.ttl,
                base_radius=cfg.base_radius,
                width_growth=cfg.width_growth,
                force_upload_on_replan=cfg.force_upload_on_replan,
                plot_level=cfg.plot_level,
                beta=cfg.beta,
                top_k=cfg.top_k,
                peak_tolerance_radius=cfg.peak_tolerance_radius,
                active_threshold_ratio=cfg.active_threshold_ratio,
                selected_diagnostics=policy.startswith("wave"),
                diagnostics_regime=cfg.regime,
                critical_budget_ratio=float(variant["critical_budget_ratio"]) if not pd.isna(variant["critical_budget_ratio"]) else 0.7,
                coverage_budget_ratio=float(variant["coverage_budget_ratio"]) if not pd.isna(variant["coverage_budget_ratio"]) else 0.3,
                new_version_weight=float(variant["new_version_weight"]) if not pd.isna(variant["new_version_weight"]) else 1.0,
                old_version_weight=2.0,
                new_version_boost_window=10,
            )
            frame = run_e5b_single(seed, cfg.delay, cfg.buffer_mode, e5b_cfg, [], selected_rows)
            frame["freshness_regime"] = cfg.regime
            frame["freshness_tau"] = cfg.freshness_tau
            frame["critical_budget_ratio"] = variant["critical_budget_ratio"]
            frame["coverage_budget_ratio"] = variant["coverage_budget_ratio"]
            frame["new_version_weight"] = variant["new_version_weight"]
            frames.append(add_e5c_balanced_columns(frame))
    df = pd.concat(frames, ignore_index=True)
    e5e_cfg = E5EConfig(
        scenario=cfg.scenario,
        agents=cfg.agents,
        seeds=cfg.seeds,
        sim_steps=cfg.sim_steps,
        horizon=cfg.horizon,
        upload_period=cfg.upload_period,
        route_switch_tick=cfg.route_switch_tick,
        switch_ratio=cfg.switch_ratio,
        delay=cfg.delay,
        buffer_mode=cfg.buffer_mode,
        ttl=cfg.ttl,
        freshness_tau=cfg.freshness_tau,
        policies=tuple(v["policy"] for v in variants),
        budget_points=cfg.budget_points,
        base_radius=cfg.base_radius,
        width_growth=cfg.width_growth,
        force_upload_on_replan=cfg.force_upload_on_replan,
        plot_level=cfg.plot_level,
        analysis_window=cfg.analysis_window,
    )
    prepost = make_e5e_pre_post_summary(df, e5e_cfg)
    variant_df = pd.DataFrame(variants).drop(columns=["selector_policy"])
    prepost = prepost.merge(variant_df, on="policy", how="left")
    merge_cols = [
        "seed",
        "policy",
        "old_route_ghost_auc",
        "new_route_hidden_auc",
        "pre_delivery_old_route_ghost_auc",
        "post_delivery_old_route_ghost_auc",
        "pre_delivery_new_route_hidden_auc",
        "post_delivery_new_route_hidden_auc",
        "route_balanced_loss_g1",
        "route_balanced_loss_g2",
    ]
    df = df.merge(prepost[merge_cols], on=["seed", "policy"], how="left")
    df.to_csv(out_dir / "metrics_tick.csv", index=False)
    summary_cols = prepost.select_dtypes(include=[np.number]).columns.tolist()
    summary = flatten_grouped_columns(prepost.groupby("policy", dropna=False)[summary_cols].agg(["mean", "std"]).reset_index())
    summary.to_csv(out_dir / "summary_by_policy.csv", index=False)
    coverage_summary = summary.copy()
    coverage_summary.to_csv(out_dir / "coverage_variant_summary.csv", index=False)
    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(out_dir / "selected_points_by_route.csv", index=False)
    selected_summary = make_selected_point_summary(selected_df)
    selected_summary.to_csv(out_dir / "selected_point_summary.csv", index=False)
    findings = make_e5f_findings(prepost, coverage_summary, selected_summary)
    (out_dir / "e5f_key_findings.txt").write_text(findings + "\n", encoding="utf-8")
    save_e5f_plots(prepost, coverage_summary, selected_summary, out_dir)
    print("\nE5F Dynamic Wave Coverage Sweep Summary")
    print(findings)
    print(f"\nWrote E5F outputs to: {out_dir}")
    return df, prepost


@dataclass(frozen=True)
class E5GConfig(E5FConfig):
    pass


def e5g_policy_variants() -> list[dict[str, Any]]:
    base = [
        {"policy": "periodic_full", "critical_budget_ratio": np.nan, "coverage_budget_ratio": np.nan, "new_version_weight": np.nan},
        {"policy": "random_budget", "critical_budget_ratio": np.nan, "coverage_budget_ratio": np.nan, "new_version_weight": np.nan},
        {"policy": "wave_critical", "critical_budget_ratio": np.nan, "coverage_budget_ratio": np.nan, "new_version_weight": np.nan},
        {"policy": "wave_critical_dynamic_default", "critical_budget_ratio": 0.7, "coverage_budget_ratio": 0.3, "new_version_weight": 1.0},
        {"policy": "wave_dynamic_c70_bonus2", "critical_budget_ratio": 0.3, "coverage_budget_ratio": 0.7, "new_version_weight": 2.0},
        {"policy": "wave_dynamic_c70_bonus2_time_balanced", "critical_budget_ratio": 0.3, "coverage_budget_ratio": 0.7, "new_version_weight": 2.0, "balance_time_bins": True},
        {"policy": "wave_dynamic_c70_bonus2_agent_balanced", "critical_budget_ratio": 0.3, "coverage_budget_ratio": 0.7, "new_version_weight": 2.0, "balance_agents": True},
        {
            "policy": "wave_dynamic_c70_bonus2_spacetime_balanced",
            "critical_budget_ratio": 0.3,
            "coverage_budget_ratio": 0.7,
            "new_version_weight": 2.0,
            "balance_time_bins": True,
            "balance_agents": True,
            "balance_spatial_bins": True,
        },
        {
            "policy": "wave_dynamic_c70_bonus2_postswitch_boost",
            "critical_budget_ratio": 0.3,
            "coverage_budget_ratio": 0.7,
            "new_version_weight": 2.0,
            "post_switch_new_version_bonus": 2.0,
        },
    ]
    for v in base:
        v.setdefault("balance_time_bins", False)
        v.setdefault("balance_agents", False)
        v.setdefault("balance_spatial_bins", False)
        v.setdefault("post_switch_new_version_bonus", 0.0)
        v.setdefault("use_route_balance", False if str(v["policy"]).startswith("wave_dynamic_c70_bonus2_") else True)
    return base


def make_post_switch_selection_summary(selected_df: pd.DataFrame, metrics_df: pd.DataFrame, cfg: E5GConfig) -> pd.DataFrame:
    if selected_df.empty:
        return pd.DataFrame()
    start = cfg.route_switch_tick + cfg.delay
    end = cfg.route_switch_tick + cfg.analysis_window
    candidates = selected_df[(selected_df["tick"] >= start) & (selected_df["tick"] <= end)].copy()
    selected = selected_df[(selected_df["selected"]) & (selected_df["tick"] >= start) & (selected_df["tick"] <= end)].copy()
    metric_window = metrics_df[(metrics_df["tick"] >= start) & (metrics_df["tick"] <= end)]
    msg = metric_window.groupby(["seed", "policy"], dropna=False)[["uploaded_tube_points", "message_count"]].mean().reset_index()
    rows: list[dict] = []
    for (seed, policy), sub in selected.groupby(["seed", "policy"], dropna=False):
        cand = candidates[(candidates["seed"] == seed) & (candidates["policy"] == policy)]
        n = max(1, len(sub))
        per_agent = sub.groupby("agent_id").size()
        per_time = sub.groupby("time_bin").size()
        per_spatial = sub.groupby(["spatial_bin_y", "spatial_bin_x"]).size()
        new_route_selected_cells = set(zip(sub.loc[sub["is_new_route"], "y"], sub.loc[sub["is_new_route"], "x"]))
        old_route_selected_cells = set(zip(sub.loc[sub["is_old_route"], "y"], sub.loc[sub["is_old_route"], "x"]))
        new_route_candidate_cells = set(zip(cand.loc[cand["is_new_route"], "y"], cand.loc[cand["is_new_route"], "x"]))
        old_route_candidate_cells = set(zip(cand.loc[cand["is_old_route"], "y"], cand.loc[cand["is_old_route"], "x"]))
        rows.append(
            {
                "seed": seed,
                "policy": policy,
                "post_switch_selected_new_version_points": int(sub["is_new_version"].sum()),
                "post_switch_selected_old_version_points": int(sub["is_old_version"].sum()),
                "post_switch_selected_new_version_ratio": float(sub["is_new_version"].sum() / n),
                "post_switch_selected_new_route_points": int(sub["is_new_route"].sum()),
                "post_switch_selected_old_route_points": int(sub["is_old_route"].sum()),
                "post_switch_selected_new_route_ratio": float(sub["is_new_route"].sum() / n),
                "post_switch_selected_points_per_agent_std": float(per_agent.std(ddof=0)) if len(per_agent) > 1 else 0.0,
                "post_switch_selected_points_per_time_bin_std": float(per_time.reindex(range(5), fill_value=0).std(ddof=0)),
                "post_switch_selected_points_per_spatial_bin_std": float(per_spatial.std(ddof=0)) if len(per_spatial) > 1 else 0.0,
                "post_switch_selected_points_per_agent_max": int(per_agent.max()) if not per_agent.empty else 0,
                "post_switch_selected_points_per_time_bin_max": int(per_time.max()) if not per_time.empty else 0,
                "post_switch_selected_points_per_spatial_bin_max": int(per_spatial.max()) if not per_spatial.empty else 0,
                "post_switch_new_route_coverage_ratio": float(len(new_route_selected_cells) / max(1, len(new_route_candidate_cells))),
                "post_switch_old_route_coverage_ratio": float(len(old_route_selected_cells) / max(1, len(old_route_candidate_cells))),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.merge(msg, on=["seed", "policy"], how="left")
        out = out.rename(columns={"uploaded_tube_points": "post_switch_uploaded_tube_points", "message_count": "post_switch_message_count"})
    return out


def make_e5g_delta_vs_random(prepost: pd.DataFrame) -> pd.DataFrame:
    return make_e5f_delta_vs_random(prepost)


def save_e5g_plots(prepost: pd.DataFrame, post_summary: pd.DataFrame, metrics_df: pd.DataFrame, out_dir: Path) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    policy_order = [v["policy"] for v in e5g_policy_variants()]
    grouped = prepost.groupby("policy", dropna=False).agg(["mean", "std"])
    metrics = ["old_route_ghost_auc", "new_route_hidden_auc", "post_delivery_new_route_hidden_auc", "route_balanced_loss_g1", "route_balanced_loss_g2"]
    fig, axes = plt.subplots(3, 2, figsize=(17, 12))
    for ax, metric in zip(axes.ravel(), metrics):
        labels, means, stds = [], [], []
        for policy in policy_order:
            if policy in grouped.index:
                labels.append(policy)
                means.append(float(grouped.loc[policy, (metric, "mean")]))
                stds.append(float(grouped.loc[policy, (metric, "std")]) if not pd.isna(grouped.loc[policy, (metric, "std")]) else 0.0)
        ax.bar(labels, means, yerr=stds, capsize=3)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=38, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    fig.savefig(plots_dir / "e5g_policy_comparison.png", dpi=150)
    plt.close(fig)

    delta = make_e5g_delta_vs_random(prepost)
    if not delta.empty:
        fig, axes = plt.subplots(3, 2, figsize=(17, 11))
        delta_metrics = [c for c in delta.columns if c.startswith("delta_")]
        for ax, metric in zip(axes.ravel(), delta_metrics):
            ax.bar(delta["policy"], delta[metric])
            ax.axhline(0.0, color="k", linewidth=1)
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=35, labelsize=8)
            ax.grid(axis="y", alpha=0.25)
        for ax in axes.ravel()[len(delta_metrics):]:
            ax.axis("off")
        fig.suptitle("variant minus random_budget; negative is better", y=1.02)
        fig.tight_layout()
        fig.savefig(plots_dir / "e5g_dynamic_vs_random_delta.png", dpi=150)
        plt.close(fig)

    means = prepost.groupby("policy", dropna=False).mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy in policy_order:
        if policy == "periodic_full" or policy not in means.index:
            continue
        x = float(means.loc[policy, "post_delivery_new_route_hidden_auc"])
        y = float(means.loc[policy, "old_route_ghost_auc"])
        ax.scatter(x, y, s=55)
        ax.annotate(policy, (x, y), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("post_delivery_new_route_hidden_auc")
    ax.set_ylabel("old_route_ghost_auc")
    ax.set_title("E5G post-delivery hidden vs stale ghost")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "e5g_post_delivery_hidden_vs_ghost_pareto.png", dpi=150)
    plt.close(fig)

    if not post_summary.empty:
        post_group = post_summary.groupby("policy", dropna=False).agg(["mean", "std"])
        diag_metrics = [
            "post_switch_selected_new_version_ratio",
            "post_switch_selected_new_route_ratio",
            "post_switch_selected_points_per_agent_std",
            "post_switch_selected_points_per_time_bin_std",
            "post_switch_selected_points_per_spatial_bin_std",
        ]
        fig, axes = plt.subplots(3, 2, figsize=(17, 11))
        for ax, metric in zip(axes.ravel(), diag_metrics):
            labels, means2, stds = [], [], []
            for policy in policy_order:
                if policy in post_group.index and (metric, "mean") in post_group.columns:
                    labels.append(policy)
                    means2.append(float(post_group.loc[policy, (metric, "mean")]))
                    stds.append(float(post_group.loc[policy, (metric, "std")]) if not pd.isna(post_group.loc[policy, (metric, "std")]) else 0.0)
            ax.bar(labels, means2, yerr=stds, capsize=3)
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=35, labelsize=8)
            ax.grid(axis="y", alpha=0.25)
        for ax in axes.ravel()[len(diag_metrics):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(plots_dir / "e5g_post_switch_selection_diagnostics.png", dpi=150)
        plt.close(fig)

    best_policy = means.drop(index=["periodic_full"], errors="ignore")["post_delivery_new_route_hidden_auc"].idxmin()
    trace_policies = ["random_budget", "wave_dynamic_c70_bonus2", str(best_policy)]
    trace_policies = list(dict.fromkeys(trace_policies))
    ts = metrics_df[(metrics_df["seed"] == 0) & (metrics_df["policy"].isin(trace_policies))]
    trace_metrics = ["old_route_ghost_pressure", "new_route_hidden_congestion", "ghost_pressure", "hidden_congestion", "mean_intention_age"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for ax, metric in zip(axes.ravel(), trace_metrics):
        for policy, sub in ts.groupby("policy", sort=False):
            sub = sub.sort_values("tick")
            ax.plot(sub["tick"], sub[metric], label=policy)
        ax.axvline(30, color="k", linestyle="--", alpha=0.45)
        ax.axvline(35, color="r", linestyle="--", alpha=0.35)
        ax.set_title(metric)
        ax.grid(alpha=0.25)
    axes.ravel()[-1].axis("off")
    axes[0, 1].legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "e5g_time_trace_best_variants.png", dpi=150)
    plt.close(fig)


def make_e5g_findings(prepost: pd.DataFrame, post_summary: pd.DataFrame) -> str:
    lines = ["E5G Post-Switch Coverage Guarantee Findings", ""]
    means = prepost.groupby("policy", dropna=False).mean(numeric_only=True)
    for policy, row in means.iterrows():
        lines.append(
            f"{policy}: old_ghost_auc={row['old_route_ghost_auc']:.4f}, "
            f"new_hidden_auc={row['new_route_hidden_auc']:.4f}, "
            f"post_new_hidden_auc={row['post_delivery_new_route_hidden_auc']:.4f}, "
            f"post_old_ghost_auc={row['post_delivery_old_route_ghost_auc']:.4f}, "
            f"route_g1={row['route_balanced_loss_g1']:.4f}, route_g2={row['route_balanced_loss_g2']:.4f}"
        )
    lines.append("")
    base_name = "wave_dynamic_c70_bonus2"
    better_base = []
    if base_name in means.index:
        base = means.loc[base_name]
        for policy, row in means.iterrows():
            if not str(policy).startswith("wave_dynamic_c70_bonus2_"):
                continue
            if (
                row["post_delivery_new_route_hidden_auc"] < base["post_delivery_new_route_hidden_auc"]
                and row["route_balanced_loss_g1"] <= base["route_balanced_loss_g1"]
                and row["route_balanced_loss_g2"] <= base["route_balanced_loss_g2"]
                and row["old_route_ghost_auc"] <= 1.05 * base["old_route_ghost_auc"]
            ):
                better_base.append(str(policy))
    better_random = []
    if "random_budget" in means.index:
        random = means.loc["random_budget"]
        for policy, row in means.iterrows():
            if not str(policy).startswith("wave_dynamic"):
                continue
            if (
                row["post_delivery_new_route_hidden_auc"] < random["post_delivery_new_route_hidden_auc"]
                and row["old_route_ghost_auc"] <= random["old_route_ghost_auc"]
                and (row["route_balanced_loss_g1"] < random["route_balanced_loss_g1"] or row["route_balanced_loss_g2"] < random["route_balanced_loss_g2"])
            ):
                better_random.append(str(policy))
    lines.append(f"Variants better than wave_dynamic_c70_bonus2 by acceptance rule: {', '.join(better_base) if better_base else 'none'}")
    lines.append(f"Variants better than random_budget by acceptance rule: {', '.join(better_random) if better_random else 'none'}")
    if not post_summary.empty:
        post = post_summary.groupby("policy", dropna=False).mean(numeric_only=True)
        lines.append("")
        for policy, row in post.iterrows():
            lines.append(
                f"{policy} post-switch selection: new_version_ratio={row['post_switch_selected_new_version_ratio']:.3f}, "
                f"new_route_ratio={row['post_switch_selected_new_route_ratio']:.3f}, "
                f"agent_std={row['post_switch_selected_points_per_agent_std']:.2f}, "
                f"time_bin_std={row['post_switch_selected_points_per_time_bin_std']:.2f}, "
                f"spatial_bin_std={row['post_switch_selected_points_per_spatial_bin_std']:.2f}"
            )
    lines.append("")
    if better_random:
        lines.append(f"At least one E5G variant beats random_budget: {', '.join(better_random)}.")
    else:
        lines.append("Random budget remains a strong coverage baseline under rolling route-switch intention fields. Dynamic wave-critical sharing reduces stale ghost but still needs stronger post-switch coverage to match random on new-route hidden.")
    return "\n".join(lines)


def run_e5g_post_switch_coverage_guarantee(cfg: E5GConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    variants = e5g_policy_variants()
    frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    total = len(cfg.seeds) * len(variants)
    run_idx = 0
    for seed in cfg.seeds:
        for variant in variants:
            run_idx += 1
            policy = str(variant["policy"])
            print(f"[E5G {run_idx}/{total}] seed={seed} policy={policy}", flush=True)
            e5b_cfg = E5BConfig(
                scenario=cfg.scenario,
                agents=cfg.agents,
                seeds=(seed,),
                sim_steps=cfg.sim_steps,
                horizon=cfg.horizon,
                upload_period=cfg.upload_period,
                route_switch_tick=cfg.route_switch_tick,
                switch_ratio=cfg.switch_ratio,
                delay_list=(cfg.delay,),
                policy=policy,
                buffer_modes=(cfg.buffer_mode,),
                budget_points=cfg.budget_points,
                freshness_tau=cfg.freshness_tau,
                ttl=cfg.ttl,
                base_radius=cfg.base_radius,
                width_growth=cfg.width_growth,
                force_upload_on_replan=cfg.force_upload_on_replan,
                plot_level=cfg.plot_level,
                beta=cfg.beta,
                top_k=cfg.top_k,
                peak_tolerance_radius=cfg.peak_tolerance_radius,
                active_threshold_ratio=cfg.active_threshold_ratio,
                selected_diagnostics=policy.startswith("wave"),
                diagnostics_regime=cfg.regime,
                critical_budget_ratio=float(variant["critical_budget_ratio"]) if not pd.isna(variant["critical_budget_ratio"]) else 0.7,
                coverage_budget_ratio=float(variant["coverage_budget_ratio"]) if not pd.isna(variant["coverage_budget_ratio"]) else 0.3,
                new_version_weight=float(variant["new_version_weight"]) if not pd.isna(variant["new_version_weight"]) else 1.0,
                old_version_weight=2.0,
                new_version_boost_window=10,
                balance_time_bins=bool(variant["balance_time_bins"]),
                balance_agents=bool(variant["balance_agents"]),
                balance_spatial_bins=bool(variant["balance_spatial_bins"]),
                use_route_balance=bool(variant["use_route_balance"]),
                post_switch_new_version_bonus=float(variant["post_switch_new_version_bonus"]),
            )
            frame = run_e5b_single(seed, cfg.delay, cfg.buffer_mode, e5b_cfg, [], selected_rows)
            frame["freshness_regime"] = cfg.regime
            frame["freshness_tau"] = cfg.freshness_tau
            frame["critical_budget_ratio"] = variant["critical_budget_ratio"]
            frame["coverage_budget_ratio"] = variant["coverage_budget_ratio"]
            frame["new_version_weight"] = variant["new_version_weight"]
            frames.append(add_e5c_balanced_columns(frame))
    df = pd.concat(frames, ignore_index=True)
    e5e_cfg = E5EConfig(
        scenario=cfg.scenario,
        agents=cfg.agents,
        seeds=cfg.seeds,
        sim_steps=cfg.sim_steps,
        horizon=cfg.horizon,
        upload_period=cfg.upload_period,
        route_switch_tick=cfg.route_switch_tick,
        switch_ratio=cfg.switch_ratio,
        delay=cfg.delay,
        buffer_mode=cfg.buffer_mode,
        ttl=cfg.ttl,
        freshness_tau=cfg.freshness_tau,
        policies=tuple(v["policy"] for v in variants),
        budget_points=cfg.budget_points,
        base_radius=cfg.base_radius,
        width_growth=cfg.width_growth,
        force_upload_on_replan=cfg.force_upload_on_replan,
        plot_level=cfg.plot_level,
        analysis_window=cfg.analysis_window,
    )
    prepost = make_e5e_pre_post_summary(df, e5e_cfg)
    selected_df = pd.DataFrame(selected_rows)
    post_summary = make_post_switch_selection_summary(selected_df, df, cfg)
    df.to_csv(out_dir / "metrics_tick.csv", index=False)
    summary_cols = prepost.select_dtypes(include=[np.number]).columns.tolist()
    summary = flatten_grouped_columns(prepost.groupby("policy", dropna=False)[summary_cols].agg(["mean", "std"]).reset_index())
    summary.to_csv(out_dir / "summary_by_policy.csv", index=False)
    post_summary.to_csv(out_dir / "post_switch_selection_summary.csv", index=False)
    selected_df.to_csv(out_dir / "selected_points_by_route.csv", index=False)
    findings = make_e5g_findings(prepost, post_summary)
    (out_dir / "e5g_key_findings.txt").write_text(findings + "\n", encoding="utf-8")
    save_e5g_plots(prepost, post_summary, df, out_dir)
    print("\nE5G Post-Switch Coverage Guarantee Summary")
    print(findings)
    print(f"\nWrote E5G outputs to: {out_dir}")
    return df, prepost


@dataclass(frozen=True)
class E5HConfig(E5GConfig):
    pass


def e5h_policy_variants() -> list[dict[str, Any]]:
    variants = [
        {"policy": "periodic_full"},
        {"policy": "random_budget"},
        {"policy": "wave_critical"},
        {"policy": "wave_critical_dynamic_default", "critical_budget_ratio": 0.7, "coverage_budget_ratio": 0.3, "new_version_weight": 1.0, "use_route_balance": True},
        {"policy": "wave_dynamic_c70_bonus2", "critical_budget_ratio": 0.3, "coverage_budget_ratio": 0.7, "new_version_weight": 2.0, "use_route_balance": True},
        {"policy": "wave_dynamic_c70_bonus2_agent_balanced", "critical_budget_ratio": 0.3, "coverage_budget_ratio": 0.7, "new_version_weight": 2.0, "balance_agents": True, "use_route_balance": False},
        {"policy": "wave_dynamic_soft_agent_v1", "critical_budget_ratio": 0.3, "coverage_budget_ratio": 0.7, "new_version_weight": 2.0, "lambda_agent": 0.5, "use_route_balance": False},
        {"policy": "wave_dynamic_soft_time_v1", "critical_budget_ratio": 0.3, "coverage_budget_ratio": 0.7, "new_version_weight": 2.0, "lambda_time": 0.3, "use_route_balance": False},
        {"policy": "wave_dynamic_soft_space_v1", "critical_budget_ratio": 0.3, "coverage_budget_ratio": 0.7, "new_version_weight": 2.0, "lambda_space": 0.3, "use_route_balance": False},
        {"policy": "wave_dynamic_soft_agent_time_v1", "critical_budget_ratio": 0.3, "coverage_budget_ratio": 0.7, "new_version_weight": 2.0, "lambda_agent": 0.5, "lambda_time": 0.3, "use_route_balance": False},
        {"policy": "wave_dynamic_soft_agent_time_space_v1", "critical_budget_ratio": 0.3, "coverage_budget_ratio": 0.7, "new_version_weight": 2.0, "lambda_agent": 0.5, "lambda_time": 0.3, "lambda_space": 0.3, "use_route_balance": False},
        {"policy": "wave_dynamic_soft_agent_time_space_v2", "critical_budget_ratio": 0.3, "coverage_budget_ratio": 0.7, "new_version_weight": 2.0, "lambda_agent": 0.8, "lambda_time": 0.5, "lambda_space": 0.5, "use_route_balance": False},
    ]
    defaults = {
        "critical_budget_ratio": np.nan,
        "coverage_budget_ratio": np.nan,
        "new_version_weight": np.nan,
        "balance_time_bins": False,
        "balance_agents": False,
        "balance_spatial_bins": False,
        "post_switch_new_version_bonus": 0.0,
        "use_route_balance": True,
        "lambda_agent": 0.0,
        "lambda_time": 0.0,
        "lambda_space": 0.0,
    }
    for variant in variants:
        for key, value in defaults.items():
            variant.setdefault(key, value)
    return variants


def save_e5h_plots(prepost: pd.DataFrame, post_summary: pd.DataFrame, metrics_df: pd.DataFrame, out_dir: Path) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    policy_order = [v["policy"] for v in e5h_policy_variants()]
    grouped = prepost.groupby("policy", dropna=False).agg(["mean", "std"])
    metrics = ["old_route_ghost_auc", "new_route_hidden_auc", "post_delivery_new_route_hidden_auc", "route_balanced_loss_g1", "route_balanced_loss_g2"]
    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    for ax, metric in zip(axes.ravel(), metrics):
        labels, means, stds = [], [], []
        for policy in policy_order:
            if policy in grouped.index:
                labels.append(policy)
                means.append(float(grouped.loc[policy, (metric, "mean")]))
                stds.append(float(grouped.loc[policy, (metric, "std")]) if not pd.isna(grouped.loc[policy, (metric, "std")]) else 0.0)
        ax.bar(labels, means, yerr=stds, capsize=3)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=38, labelsize=7)
        ax.grid(axis="y", alpha=0.25)
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    fig.savefig(plots_dir / "e5h_policy_comparison.png", dpi=150)
    plt.close(fig)

    delta = make_e5g_delta_vs_random(prepost)
    if not delta.empty:
        fig, axes = plt.subplots(3, 2, figsize=(18, 11))
        delta_metrics = [c for c in delta.columns if c.startswith("delta_")]
        for ax, metric in zip(axes.ravel(), delta_metrics):
            ax.bar(delta["policy"], delta[metric])
            ax.axhline(0.0, color="k", linewidth=1)
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=35, labelsize=7)
            ax.grid(axis="y", alpha=0.25)
        for ax in axes.ravel()[len(delta_metrics):]:
            ax.axis("off")
        fig.suptitle("variant minus random_budget; negative is better", y=1.02)
        fig.tight_layout()
        fig.savefig(plots_dir / "e5h_dynamic_vs_random_delta.png", dpi=150)
        plt.close(fig)

    means = prepost.groupby("policy", dropna=False).mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy in policy_order:
        if policy == "periodic_full" or policy not in means.index:
            continue
        x = float(means.loc[policy, "post_delivery_new_route_hidden_auc"])
        y = float(means.loc[policy, "old_route_ghost_auc"])
        ax.scatter(x, y, s=55)
        ax.annotate(policy, (x, y), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("post_delivery_new_route_hidden_auc")
    ax.set_ylabel("old_route_ghost_auc")
    ax.set_title("E5H post-delivery hidden vs stale ghost")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "e5h_post_delivery_hidden_vs_ghost_pareto.png", dpi=150)
    plt.close(fig)

    if not post_summary.empty:
        post_group = post_summary.groupby("policy", dropna=False).agg(["mean", "std"])
        diag_metrics = [
            "post_switch_selected_new_version_ratio",
            "post_switch_selected_new_route_ratio",
            "post_switch_selected_points_per_agent_std",
            "post_switch_selected_points_per_time_bin_std",
            "post_switch_selected_points_per_spatial_bin_std",
            "post_switch_new_route_coverage_ratio",
            "post_switch_old_route_coverage_ratio",
        ]
        fig, axes = plt.subplots(4, 2, figsize=(18, 13))
        for ax, metric in zip(axes.ravel(), diag_metrics):
            labels, means2, stds = [], [], []
            for policy in policy_order:
                if policy in post_group.index and (metric, "mean") in post_group.columns:
                    labels.append(policy)
                    means2.append(float(post_group.loc[policy, (metric, "mean")]))
                    stds.append(float(post_group.loc[policy, (metric, "std")]) if not pd.isna(post_group.loc[policy, (metric, "std")]) else 0.0)
            ax.bar(labels, means2, yerr=stds, capsize=3)
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=35, labelsize=7)
            ax.grid(axis="y", alpha=0.25)
        for ax in axes.ravel()[len(diag_metrics):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(plots_dir / "e5h_post_switch_selection_diagnostics.png", dpi=150)
        plt.close(fig)

    best_policy = means.drop(index=["periodic_full"], errors="ignore")["post_delivery_new_route_hidden_auc"].idxmin()
    trace_policies = ["random_budget", "wave_dynamic_c70_bonus2_agent_balanced", str(best_policy)]
    trace_policies = list(dict.fromkeys(trace_policies))
    ts = metrics_df[(metrics_df["seed"] == 0) & (metrics_df["policy"].isin(trace_policies))]
    trace_metrics = ["old_route_ghost_pressure", "new_route_hidden_congestion", "ghost_pressure", "hidden_congestion", "mean_intention_age"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for ax, metric in zip(axes.ravel(), trace_metrics):
        for policy, sub in ts.groupby("policy", sort=False):
            sub = sub.sort_values("tick")
            ax.plot(sub["tick"], sub[metric], label=policy)
        ax.axvline(30, color="k", linestyle="--", alpha=0.45)
        ax.axvline(35, color="r", linestyle="--", alpha=0.35)
        ax.set_title(metric)
        ax.grid(alpha=0.25)
    axes.ravel()[-1].axis("off")
    axes[0, 1].legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "e5h_time_trace_best_variants.png", dpi=150)
    plt.close(fig)


def make_e5h_findings(prepost: pd.DataFrame, post_summary: pd.DataFrame) -> str:
    lines = ["E5H Soft Coverage Regularized Wave-Critical Findings", ""]
    means = prepost.groupby("policy", dropna=False).mean(numeric_only=True)
    for policy, row in means.iterrows():
        lines.append(
            f"{policy}: old_ghost_auc={row['old_route_ghost_auc']:.4f}, "
            f"new_hidden_auc={row['new_route_hidden_auc']:.4f}, "
            f"post_new_hidden_auc={row['post_delivery_new_route_hidden_auc']:.4f}, "
            f"post_old_ghost_auc={row['post_delivery_old_route_ghost_auc']:.4f}, "
            f"route_g1={row['route_balanced_loss_g1']:.4f}, route_g2={row['route_balanced_loss_g2']:.4f}, "
            f"uploaded={row['uploaded_tube_points']:.1f}"
        )
    dynamic = means[[str(idx).startswith("wave_dynamic") or str(idx).startswith("wave_critical_dynamic") for idx in means.index]]
    if not dynamic.empty:
        best_hidden_policy = str(dynamic["post_delivery_new_route_hidden_auc"].idxmin())
        lines.append("")
        lines.append(f"Lowest post_delivery_new_route_hidden_auc among dynamic policies: {best_hidden_policy}")
    better_random = []
    if "random_budget" in means.index:
        random = means.loc["random_budget"]
        for policy, row in dynamic.iterrows():
            if (
                row["post_delivery_new_route_hidden_auc"] < random["post_delivery_new_route_hidden_auc"]
                and row["old_route_ghost_auc"] <= random["old_route_ghost_auc"]
                and row["uploaded_tube_points"] <= random["uploaded_tube_points"] + 1e-9
            ):
                better_random.append(str(policy))
    lines.append(f"Dynamic policies beating random_budget on both old ghost and post-delivery hidden: {', '.join(better_random) if better_random else 'none'}")

    base_name = "wave_dynamic_c70_bonus2_agent_balanced"
    better_base = []
    if base_name in means.index:
        base = means.loc[base_name]
        for policy, row in dynamic.iterrows():
            if policy == base_name:
                continue
            if (
                row["post_delivery_new_route_hidden_auc"] < base["post_delivery_new_route_hidden_auc"]
                and row["old_route_ghost_auc"] <= 1.05 * base["old_route_ghost_auc"]
                and (row["route_balanced_loss_g1"] < base["route_balanced_loss_g1"] or row["route_balanced_loss_g2"] < base["route_balanced_loss_g2"])
            ):
                better_base.append(str(policy))
    lines.append(f"Dynamic policies better than wave_dynamic_c70_bonus2_agent_balanced: {', '.join(better_base) if better_base else 'none'}")

    if not post_summary.empty:
        post = post_summary.groupby("policy", dropna=False).mean(numeric_only=True)
        lines.append("")
        for policy, row in post.iterrows():
            if str(policy).startswith("wave"):
                lines.append(
                    f"{policy} post-switch selection: new_version_ratio={row['post_switch_selected_new_version_ratio']:.3f}, "
                    f"new_route_ratio={row['post_switch_selected_new_route_ratio']:.3f}, "
                    f"agent_std={row['post_switch_selected_points_per_agent_std']:.2f}, "
                    f"time_bin_std={row['post_switch_selected_points_per_time_bin_std']:.2f}, "
                    f"space_std={row['post_switch_selected_points_per_spatial_bin_std']:.2f}, "
                    f"new_route_coverage={row['post_switch_new_route_coverage_ratio']:.3f}, "
                    f"old_route_coverage={row['post_switch_old_route_coverage_ratio']:.3f}"
                )
    lines.append("")
    if better_random:
        lines.append(f"At least one E5H policy beats random_budget under the requested criteria: {', '.join(better_random)}.")
    else:
        lines.append("Random budget remains a strong coverage baseline. Soft coverage regularization improves dynamic wave critical sharing but does not yet fully match random coverage on new-route hidden. The result suggests that future work should combine wave criticality with explicit route-structure coverage estimation.")
    return "\n".join(lines)


def run_e5h_soft_coverage_regularized_wave_critical(cfg: E5HConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    variants = e5h_policy_variants()
    frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    total = len(cfg.seeds) * len(variants)
    run_idx = 0
    for seed in cfg.seeds:
        for variant in variants:
            run_idx += 1
            policy = str(variant["policy"])
            print(f"[E5H {run_idx}/{total}] seed={seed} policy={policy}", flush=True)
            e5b_cfg = E5BConfig(
                scenario=cfg.scenario,
                agents=cfg.agents,
                seeds=(seed,),
                sim_steps=cfg.sim_steps,
                horizon=cfg.horizon,
                upload_period=cfg.upload_period,
                route_switch_tick=cfg.route_switch_tick,
                switch_ratio=cfg.switch_ratio,
                delay_list=(cfg.delay,),
                policy=policy,
                buffer_modes=(cfg.buffer_mode,),
                budget_points=cfg.budget_points,
                freshness_tau=cfg.freshness_tau,
                ttl=cfg.ttl,
                base_radius=cfg.base_radius,
                width_growth=cfg.width_growth,
                force_upload_on_replan=cfg.force_upload_on_replan,
                plot_level=cfg.plot_level,
                beta=cfg.beta,
                top_k=cfg.top_k,
                peak_tolerance_radius=cfg.peak_tolerance_radius,
                active_threshold_ratio=cfg.active_threshold_ratio,
                selected_diagnostics=policy.startswith("wave"),
                diagnostics_regime=cfg.regime,
                critical_budget_ratio=float(variant["critical_budget_ratio"]) if not pd.isna(variant["critical_budget_ratio"]) else 0.7,
                coverage_budget_ratio=float(variant["coverage_budget_ratio"]) if not pd.isna(variant["coverage_budget_ratio"]) else 0.3,
                new_version_weight=float(variant["new_version_weight"]) if not pd.isna(variant["new_version_weight"]) else 1.0,
                old_version_weight=2.0,
                new_version_boost_window=10,
                balance_time_bins=bool(variant["balance_time_bins"]),
                balance_agents=bool(variant["balance_agents"]),
                balance_spatial_bins=bool(variant["balance_spatial_bins"]),
                use_route_balance=bool(variant["use_route_balance"]),
                post_switch_new_version_bonus=float(variant["post_switch_new_version_bonus"]),
                lambda_agent=float(variant["lambda_agent"]),
                lambda_time=float(variant["lambda_time"]),
                lambda_space=float(variant["lambda_space"]),
            )
            frame = run_e5b_single(seed, cfg.delay, cfg.buffer_mode, e5b_cfg, [], selected_rows)
            frame["freshness_regime"] = cfg.regime
            frame["freshness_tau"] = cfg.freshness_tau
            frame["critical_budget_ratio"] = variant["critical_budget_ratio"]
            frame["coverage_budget_ratio"] = variant["coverage_budget_ratio"]
            frame["new_version_weight"] = variant["new_version_weight"]
            frame["lambda_agent"] = variant["lambda_agent"]
            frame["lambda_time"] = variant["lambda_time"]
            frame["lambda_space"] = variant["lambda_space"]
            frames.append(add_e5c_balanced_columns(frame))
    df = pd.concat(frames, ignore_index=True)
    e5e_cfg = E5EConfig(
        scenario=cfg.scenario,
        agents=cfg.agents,
        seeds=cfg.seeds,
        sim_steps=cfg.sim_steps,
        horizon=cfg.horizon,
        upload_period=cfg.upload_period,
        route_switch_tick=cfg.route_switch_tick,
        switch_ratio=cfg.switch_ratio,
        delay=cfg.delay,
        buffer_mode=cfg.buffer_mode,
        ttl=cfg.ttl,
        freshness_tau=cfg.freshness_tau,
        policies=tuple(v["policy"] for v in variants),
        budget_points=cfg.budget_points,
        base_radius=cfg.base_radius,
        width_growth=cfg.width_growth,
        force_upload_on_replan=cfg.force_upload_on_replan,
        plot_level=cfg.plot_level,
        analysis_window=cfg.analysis_window,
    )
    prepost = make_e5e_pre_post_summary(df, e5e_cfg)
    selected_df = pd.DataFrame(selected_rows)
    post_summary = make_post_switch_selection_summary(selected_df, df, cfg)
    df.to_csv(out_dir / "metrics_tick.csv", index=False)
    summary_cols = prepost.select_dtypes(include=[np.number]).columns.tolist()
    summary = flatten_grouped_columns(prepost.groupby("policy", dropna=False)[summary_cols].agg(["mean", "std"]).reset_index())
    summary.to_csv(out_dir / "summary_by_policy.csv", index=False)
    post_summary.to_csv(out_dir / "post_switch_selection_summary.csv", index=False)
    findings = make_e5h_findings(prepost, post_summary)
    (out_dir / "e5h_key_findings.txt").write_text(findings + "\n", encoding="utf-8")
    save_e5h_plots(prepost, post_summary, df, out_dir)
    print("\nE5H Soft Coverage Regularized Wave-Critical Summary")
    print(findings)
    print(f"\nWrote E5H outputs to: {out_dir}")
    return df, prepost


@dataclass(frozen=True)
class E5IConfig(E5GConfig):
    pass


def e5i_policy_variants() -> list[dict[str, Any]]:
    return [
        {"policy": "periodic_full"},
        {"policy": "random_budget"},
        {"policy": "wave_critical"},
        {"policy": "wave_critical_dynamic_default", "critical_budget_ratio": 0.7, "coverage_budget_ratio": 0.3, "new_version_weight": 1.0, "use_route_balance": True},
        {"policy": "wave_dynamic_c70_bonus2_agent_balanced", "critical_budget_ratio": 0.3, "coverage_budget_ratio": 0.7, "new_version_weight": 2.0, "balance_agents": True, "use_route_balance": False},
        {"policy": "hybrid_random25_wave75"},
        {"policy": "hybrid_random50_wave50"},
        {"policy": "hybrid_random75_wave25"},
        {"policy": "hybrid_newversion_random50_wave50"},
        {"policy": "hybrid_agent_stratified_random50_wave50"},
        {"policy": "hybrid_time_stratified_random50_wave50"},
    ]


def add_variant_defaults(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    defaults = {
        "critical_budget_ratio": np.nan,
        "coverage_budget_ratio": np.nan,
        "new_version_weight": np.nan,
        "balance_time_bins": False,
        "balance_agents": False,
        "balance_spatial_bins": False,
        "post_switch_new_version_bonus": 0.0,
        "use_route_balance": True,
        "lambda_agent": 0.0,
        "lambda_time": 0.0,
        "lambda_space": 0.0,
    }
    for variant in variants:
        for key, value in defaults.items():
            variant.setdefault(key, value)
    return variants


def make_hybrid_quota_summary(selected_df: pd.DataFrame, cfg: E5IConfig) -> pd.DataFrame:
    if selected_df.empty or "selection_source" not in selected_df.columns:
        return pd.DataFrame()
    start = cfg.route_switch_tick + cfg.delay
    end = cfg.route_switch_tick + cfg.analysis_window
    selected = selected_df[(selected_df["selected"]) & (selected_df["tick"] >= start) & (selected_df["tick"] <= end)].copy()
    selected = selected[selected["policy"].str.startswith("hybrid_", na=False)]
    rows: list[dict] = []
    for (seed, policy), sub in selected.groupby(["seed", "policy"], dropna=False):
        random_part = sub[sub["selection_source"] == "random"]
        critical_part = sub[sub["selection_source"] == "critical"]

        def ratio(frame: pd.DataFrame, col: str) -> float:
            return 0.0 if frame.empty else float(frame[col].sum() / len(frame))

        rows.append(
            {
                "seed": seed,
                "policy": policy,
                "hybrid_random_quota_points": int(len(random_part)),
                "hybrid_critical_quota_points": int(len(critical_part)),
                "hybrid_random_selected_new_version_ratio": ratio(random_part, "is_new_version"),
                "hybrid_critical_selected_new_version_ratio": ratio(critical_part, "is_new_version"),
                "hybrid_random_selected_new_route_ratio": ratio(random_part, "is_new_route"),
                "hybrid_critical_selected_new_route_ratio": ratio(critical_part, "is_new_route"),
            }
        )
    return pd.DataFrame(rows)


def save_e5i_plots(prepost: pd.DataFrame, selection_summary: pd.DataFrame, metrics_df: pd.DataFrame, out_dir: Path) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    policy_order = [v["policy"] for v in e5i_policy_variants()]
    grouped = prepost.groupby("policy", dropna=False).agg(["mean", "std"])
    metrics = ["old_route_ghost_auc", "new_route_hidden_auc", "post_delivery_new_route_hidden_auc", "route_balanced_loss_g1", "route_balanced_loss_g2"]
    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    for ax, metric in zip(axes.ravel(), metrics):
        labels, means, stds = [], [], []
        for policy in policy_order:
            if policy in grouped.index:
                labels.append(policy)
                means.append(float(grouped.loc[policy, (metric, "mean")]))
                stds.append(float(grouped.loc[policy, (metric, "std")]) if not pd.isna(grouped.loc[policy, (metric, "std")]) else 0.0)
        ax.bar(labels, means, yerr=stds, capsize=3)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=38, labelsize=7)
        ax.grid(axis="y", alpha=0.25)
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    fig.savefig(plots_dir / "e5i_policy_comparison.png", dpi=150)
    plt.close(fig)

    delta = make_e5g_delta_vs_random(prepost)
    delta = delta[delta["policy"].str.startswith("hybrid_", na=False)] if not delta.empty else delta
    if not delta.empty:
        fig, axes = plt.subplots(3, 2, figsize=(18, 11))
        delta_metrics = [c for c in delta.columns if c.startswith("delta_")]
        for ax, metric in zip(axes.ravel(), delta_metrics):
            ax.bar(delta["policy"], delta[metric])
            ax.axhline(0.0, color="k", linewidth=1)
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=35, labelsize=7)
            ax.grid(axis="y", alpha=0.25)
        for ax in axes.ravel()[len(delta_metrics):]:
            ax.axis("off")
        fig.suptitle("hybrid minus random_budget; negative is better", y=1.02)
        fig.tight_layout()
        fig.savefig(plots_dir / "e5i_hybrid_vs_random_delta.png", dpi=150)
        plt.close(fig)

    means = prepost.groupby("policy", dropna=False).mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy in policy_order:
        if policy == "periodic_full" or policy not in means.index:
            continue
        x = float(means.loc[policy, "post_delivery_new_route_hidden_auc"])
        y = float(means.loc[policy, "old_route_ghost_auc"])
        ax.scatter(x, y, s=55)
        ax.annotate(policy, (x, y), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("post_delivery_new_route_hidden_auc")
    ax.set_ylabel("old_route_ghost_auc")
    ax.set_title("E5I post-delivery hidden vs stale ghost")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "e5i_post_delivery_hidden_vs_ghost_pareto.png", dpi=150)
    plt.close(fig)

    if not selection_summary.empty:
        post_group = selection_summary.groupby("policy", dropna=False).agg(["mean", "std"])
        diag_metrics = [
            "post_switch_selected_new_version_ratio",
            "post_switch_selected_new_route_ratio",
            "post_switch_new_route_coverage_ratio",
            "post_switch_old_route_coverage_ratio",
            "post_switch_selected_points_per_agent_std",
            "post_switch_selected_points_per_time_bin_std",
            "post_switch_selected_points_per_spatial_bin_std",
        ]
        fig, axes = plt.subplots(4, 2, figsize=(18, 13))
        for ax, metric in zip(axes.ravel(), diag_metrics):
            labels, means2, stds = [], [], []
            for policy in policy_order:
                if policy in post_group.index and (metric, "mean") in post_group.columns:
                    labels.append(policy)
                    means2.append(float(post_group.loc[policy, (metric, "mean")]))
                    stds.append(float(post_group.loc[policy, (metric, "std")]) if not pd.isna(post_group.loc[policy, (metric, "std")]) else 0.0)
            ax.bar(labels, means2, yerr=stds, capsize=3)
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=35, labelsize=7)
            ax.grid(axis="y", alpha=0.25)
        for ax in axes.ravel()[len(diag_metrics):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(plots_dir / "e5i_selection_diagnostics.png", dpi=150)
        plt.close(fig)

        hybrid_cols = [
            "hybrid_random_quota_points",
            "hybrid_critical_quota_points",
            "hybrid_random_selected_new_version_ratio",
            "hybrid_critical_selected_new_version_ratio",
            "hybrid_random_selected_new_route_ratio",
            "hybrid_critical_selected_new_route_ratio",
        ]
        hybrid_summary = selection_summary[selection_summary["policy"].str.startswith("hybrid_", na=False)]
        if not hybrid_summary.empty:
            hybrid_group = hybrid_summary.groupby("policy", dropna=False).agg(["mean", "std"])
            fig, axes = plt.subplots(3, 2, figsize=(18, 11))
            for ax, metric in zip(axes.ravel(), hybrid_cols):
                labels, means3, stds = [], [], []
                for policy in policy_order:
                    if policy in hybrid_group.index and (metric, "mean") in hybrid_group.columns:
                        labels.append(policy)
                        means3.append(float(hybrid_group.loc[policy, (metric, "mean")]))
                        stds.append(float(hybrid_group.loc[policy, (metric, "std")]) if not pd.isna(hybrid_group.loc[policy, (metric, "std")]) else 0.0)
                ax.bar(labels, means3, yerr=stds, capsize=3)
                ax.set_title(metric)
                ax.tick_params(axis="x", rotation=35, labelsize=7)
                ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(plots_dir / "e5i_hybrid_quota_diagnostics.png", dpi=150)
            plt.close(fig)

    best_policy = means.drop(index=["periodic_full"], errors="ignore")["post_delivery_new_route_hidden_auc"].idxmin()
    trace_policies = ["random_budget", "wave_dynamic_c70_bonus2_agent_balanced", str(best_policy)]
    trace_policies = list(dict.fromkeys(trace_policies))
    ts = metrics_df[(metrics_df["seed"] == 0) & (metrics_df["policy"].isin(trace_policies))]
    trace_metrics = ["old_route_ghost_pressure", "new_route_hidden_congestion", "ghost_pressure", "hidden_congestion", "mean_intention_age"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for ax, metric in zip(axes.ravel(), trace_metrics):
        for policy, sub in ts.groupby("policy", sort=False):
            sub = sub.sort_values("tick")
            ax.plot(sub["tick"], sub[metric], label=policy)
        ax.axvline(30, color="k", linestyle="--", alpha=0.45)
        ax.axvline(35, color="r", linestyle="--", alpha=0.35)
        ax.set_title(metric)
        ax.grid(alpha=0.25)
    axes.ravel()[-1].axis("off")
    axes[0, 1].legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "e5i_time_trace_best_variants.png", dpi=150)
    plt.close(fig)


def make_e5i_findings(prepost: pd.DataFrame, selection_summary: pd.DataFrame) -> str:
    lines = ["E5I Hybrid Random + Wave-Critical Findings", ""]
    means = prepost.groupby("policy", dropna=False).mean(numeric_only=True)
    for policy, row in means.iterrows():
        lines.append(
            f"{policy}: old_ghost_auc={row['old_route_ghost_auc']:.4f}, "
            f"new_hidden_auc={row['new_route_hidden_auc']:.4f}, "
            f"post_new_hidden_auc={row['post_delivery_new_route_hidden_auc']:.4f}, "
            f"route_g1={row['route_balanced_loss_g1']:.4f}, route_g2={row['route_balanced_loss_g2']:.4f}, "
            f"uploaded={row['uploaded_tube_points']:.1f}"
        )
    hybrids = means[[str(idx).startswith("hybrid_") for idx in means.index]]
    if not hybrids.empty:
        lines.append("")
        lines.append(f"Lowest hybrid post_delivery_new_route_hidden_auc: {hybrids['post_delivery_new_route_hidden_auc'].idxmin()}")
        lines.append(f"Lowest hybrid route_balanced_loss_g1: {hybrids['route_balanced_loss_g1'].idxmin()}")
        lines.append(f"Lowest hybrid route_balanced_loss_g2: {hybrids['route_balanced_loss_g2'].idxmin()}")
    beats_random = []
    useful = []
    if "random_budget" in means.index and not hybrids.empty:
        random = means.loc["random_budget"]
        base = means.loc["wave_dynamic_c70_bonus2_agent_balanced"] if "wave_dynamic_c70_bonus2_agent_balanced" in means.index else None
        for policy, row in hybrids.iterrows():
            if (
                row["post_delivery_new_route_hidden_auc"] <= random["post_delivery_new_route_hidden_auc"]
                and row["old_route_ghost_auc"] <= random["old_route_ghost_auc"]
                and (row["route_balanced_loss_g1"] < random["route_balanced_loss_g1"] or row["route_balanced_loss_g2"] < random["route_balanced_loss_g2"])
                and row["uploaded_tube_points"] <= random["uploaded_tube_points"] + 1e-9
            ):
                beats_random.append(str(policy))
            if base is not None:
                within_5 = row["post_delivery_new_route_hidden_auc"] <= 1.05 * random["post_delivery_new_route_hidden_auc"]
                lower_ghost = row["old_route_ghost_auc"] < random["old_route_ghost_auc"]
                lower_base_loss = row["route_balanced_loss_g1"] < base["route_balanced_loss_g1"] or row["route_balanced_loss_g2"] < base["route_balanced_loss_g2"]
                if within_5 and lower_ghost and lower_base_loss:
                    useful.append(str(policy))
    lines.append(f"Hybrid policies beating random_budget by acceptance rule: {', '.join(beats_random) if beats_random else 'none'}")
    lines.append(f"Useful hybrids by relaxed rule: {', '.join(useful) if useful else 'none'}")

    if not selection_summary.empty:
        sel = selection_summary.groupby("policy", dropna=False).mean(numeric_only=True)
        lines.append("")
        for policy, row in sel.iterrows():
            if str(policy).startswith("hybrid_"):
                lines.append(
                    f"{policy}: random_quota={row.get('hybrid_random_quota_points', np.nan):.1f}, "
                    f"critical_quota={row.get('hybrid_critical_quota_points', np.nan):.1f}, "
                    f"rand_new_version={row.get('hybrid_random_selected_new_version_ratio', np.nan):.3f}, "
                    f"crit_new_version={row.get('hybrid_critical_selected_new_version_ratio', np.nan):.3f}, "
                    f"rand_new_route={row.get('hybrid_random_selected_new_route_ratio', np.nan):.3f}, "
                    f"crit_new_route={row.get('hybrid_critical_selected_new_route_ratio', np.nan):.3f}"
                )
    lines.append("")
    if beats_random:
        lines.append(f"At least one hybrid policy beats random_budget: {', '.join(beats_random)}.")
    else:
        lines.append("Random budget remains a strong coverage baseline under dynamic route switching. Hybrid policies improve the ghost-hidden tradeoff relative to pure wave critical selection, but random coverage still provides the most robust new-route visibility. This suggests that dynamic future-trajectory sharing should combine criticality-aware selection with explicit coverage guarantees.")
    return "\n".join(lines)


def run_e5i_hybrid_random_wave_critical(cfg: E5IConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    variants = add_variant_defaults(e5i_policy_variants())
    frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    total = len(cfg.seeds) * len(variants)
    run_idx = 0
    for seed in cfg.seeds:
        for variant in variants:
            run_idx += 1
            policy = str(variant["policy"])
            print(f"[E5I {run_idx}/{total}] seed={seed} policy={policy}", flush=True)
            e5b_cfg = E5BConfig(
                scenario=cfg.scenario,
                agents=cfg.agents,
                seeds=(seed,),
                sim_steps=cfg.sim_steps,
                horizon=cfg.horizon,
                upload_period=cfg.upload_period,
                route_switch_tick=cfg.route_switch_tick,
                switch_ratio=cfg.switch_ratio,
                delay_list=(cfg.delay,),
                policy=policy,
                buffer_modes=(cfg.buffer_mode,),
                budget_points=cfg.budget_points,
                freshness_tau=cfg.freshness_tau,
                ttl=cfg.ttl,
                base_radius=cfg.base_radius,
                width_growth=cfg.width_growth,
                force_upload_on_replan=cfg.force_upload_on_replan,
                plot_level=cfg.plot_level,
                beta=cfg.beta,
                top_k=cfg.top_k,
                peak_tolerance_radius=cfg.peak_tolerance_radius,
                active_threshold_ratio=cfg.active_threshold_ratio,
                selected_diagnostics=policy.startswith("wave") or policy.startswith("hybrid_"),
                diagnostics_regime=cfg.regime,
                critical_budget_ratio=float(variant["critical_budget_ratio"]) if not pd.isna(variant["critical_budget_ratio"]) else 0.7,
                coverage_budget_ratio=float(variant["coverage_budget_ratio"]) if not pd.isna(variant["coverage_budget_ratio"]) else 0.3,
                new_version_weight=float(variant["new_version_weight"]) if not pd.isna(variant["new_version_weight"]) else 1.0,
                old_version_weight=2.0,
                new_version_boost_window=10,
                balance_time_bins=bool(variant["balance_time_bins"]),
                balance_agents=bool(variant["balance_agents"]),
                balance_spatial_bins=bool(variant["balance_spatial_bins"]),
                use_route_balance=bool(variant["use_route_balance"]),
                post_switch_new_version_bonus=float(variant["post_switch_new_version_bonus"]),
                lambda_agent=float(variant["lambda_agent"]),
                lambda_time=float(variant["lambda_time"]),
                lambda_space=float(variant["lambda_space"]),
            )
            frame = run_e5b_single(seed, cfg.delay, cfg.buffer_mode, e5b_cfg, [], selected_rows)
            frame["freshness_regime"] = cfg.regime
            frame["freshness_tau"] = cfg.freshness_tau
            frames.append(add_e5c_balanced_columns(frame))
    df = pd.concat(frames, ignore_index=True)
    e5e_cfg = E5EConfig(
        scenario=cfg.scenario,
        agents=cfg.agents,
        seeds=cfg.seeds,
        sim_steps=cfg.sim_steps,
        horizon=cfg.horizon,
        upload_period=cfg.upload_period,
        route_switch_tick=cfg.route_switch_tick,
        switch_ratio=cfg.switch_ratio,
        delay=cfg.delay,
        buffer_mode=cfg.buffer_mode,
        ttl=cfg.ttl,
        freshness_tau=cfg.freshness_tau,
        policies=tuple(v["policy"] for v in variants),
        budget_points=cfg.budget_points,
        base_radius=cfg.base_radius,
        width_growth=cfg.width_growth,
        force_upload_on_replan=cfg.force_upload_on_replan,
        plot_level=cfg.plot_level,
        analysis_window=cfg.analysis_window,
    )
    prepost = make_e5e_pre_post_summary(df, e5e_cfg)
    selected_df = pd.DataFrame(selected_rows)
    selection_summary = make_post_switch_selection_summary(selected_df, df, cfg)
    hybrid_summary = make_hybrid_quota_summary(selected_df, cfg)
    if not hybrid_summary.empty:
        selection_summary = selection_summary.merge(hybrid_summary, on=["seed", "policy"], how="left")
    df.to_csv(out_dir / "metrics_tick.csv", index=False)
    summary_cols = prepost.select_dtypes(include=[np.number]).columns.tolist()
    summary = flatten_grouped_columns(prepost.groupby("policy", dropna=False)[summary_cols].agg(["mean", "std"]).reset_index())
    summary.to_csv(out_dir / "summary_by_policy.csv", index=False)
    selection_summary.to_csv(out_dir / "selection_summary.csv", index=False)
    findings = make_e5i_findings(prepost, selection_summary)
    (out_dir / "e5i_key_findings.txt").write_text(findings + "\n", encoding="utf-8")
    save_e5i_plots(prepost, selection_summary, df, out_dir)
    print("\nE5I Hybrid Random + Wave-Critical Summary")
    print(findings)
    print(f"\nWrote E5I outputs to: {out_dir}")
    return df, prepost


@dataclass(frozen=True)
class E5JConfig(E5IConfig):
    pass


def e5j_policy_variants() -> list[dict[str, Any]]:
    return [
        {"policy": "periodic_full"},
        {"policy": "random_budget"},
        {"policy": "wave_critical"},
        {
            "policy": "wave_dynamic_c70_bonus2_agent_balanced",
            "critical_budget_ratio": 0.3,
            "coverage_budget_ratio": 0.7,
            "new_version_weight": 2.0,
            "balance_agents": True,
            "use_route_balance": False,
        },
        {"policy": "hybrid_random25_wave75"},
        {"policy": "hybrid_agent_stratified_random50_wave50"},
        {"policy": "hybrid_time_stratified_random50_wave50"},
        {"policy": "hybrid_random75_wave25"},
    ]


def make_e5j_delta_vs_random(prepost: pd.DataFrame) -> pd.DataFrame:
    delta = make_e5f_delta_vs_random(prepost)
    if delta.empty:
        return delta
    return delta[delta["policy"].str.startswith("hybrid_", na=False)].reset_index(drop=True)


def save_e5j_plots(
    prepost: pd.DataFrame,
    selection_summary: pd.DataFrame,
    metrics_df: pd.DataFrame,
    delta: pd.DataFrame,
    out_dir: Path,
) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    policy_order = [v["policy"] for v in e5j_policy_variants()]
    grouped = prepost.groupby("policy", dropna=False).agg(["mean", "std"])
    metrics = [
        "old_route_ghost_auc",
        "new_route_hidden_auc",
        "post_delivery_new_route_hidden_auc",
        "route_balanced_loss_g1",
        "route_balanced_loss_g2",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(17, 12))
    for ax, metric in zip(axes.ravel(), metrics):
        labels, means, stds = [], [], []
        for policy in policy_order:
            if policy in grouped.index:
                labels.append(policy)
                means.append(float(grouped.loc[policy, (metric, "mean")]))
                stds.append(float(grouped.loc[policy, (metric, "std")]) if not pd.isna(grouped.loc[policy, (metric, "std")]) else 0.0)
        ax.bar(labels, means, yerr=stds, capsize=3)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=38, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    fig.savefig(plots_dir / "e5j_policy_comparison.png", dpi=150)
    plt.close(fig)

    if not delta.empty:
        delta_metrics = [c for c in delta.columns if c.startswith("delta_")]
        fig, axes = plt.subplots(3, 2, figsize=(17, 11))
        for ax, metric in zip(axes.ravel(), delta_metrics):
            ax.bar(delta["policy"], delta[metric])
            ax.axhline(0.0, color="k", linewidth=1)
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=35, labelsize=8)
            ax.grid(axis="y", alpha=0.25)
        for ax in axes.ravel()[len(delta_metrics):]:
            ax.axis("off")
        fig.suptitle("hybrid minus random_budget; negative is better", y=1.02)
        fig.tight_layout()
        fig.savefig(plots_dir / "e5j_hybrid_vs_random_delta.png", dpi=150)
        plt.close(fig)

    means_df = prepost.groupby("policy", dropna=False).mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy in policy_order:
        if policy == "periodic_full" or policy not in means_df.index:
            continue
        x = float(means_df.loc[policy, "post_delivery_new_route_hidden_auc"])
        y = float(means_df.loc[policy, "old_route_ghost_auc"])
        ax.scatter(x, y, s=55)
        ax.annotate(policy, (x, y), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("post_delivery_new_route_hidden_auc")
    ax.set_ylabel("old_route_ghost_auc")
    ax.set_title("E5J hidden vs stale ghost confirmation")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "e5j_pareto_hidden_vs_ghost.png", dpi=150)
    plt.close(fig)

    if not selection_summary.empty:
        post_group = selection_summary.groupby("policy", dropna=False).agg(["mean", "std"])
        diag_metrics = [
            "post_switch_selected_new_version_ratio",
            "post_switch_selected_new_route_ratio",
            "post_switch_new_route_coverage_ratio",
            "post_switch_selected_points_per_agent_std",
            "post_switch_selected_points_per_time_bin_std",
        ]
        fig, axes = plt.subplots(3, 2, figsize=(17, 11))
        for ax, metric in zip(axes.ravel(), diag_metrics):
            labels, means2, stds = [], [], []
            for policy in policy_order:
                if policy in post_group.index and (metric, "mean") in post_group.columns:
                    labels.append(policy)
                    means2.append(float(post_group.loc[policy, (metric, "mean")]))
                    stds.append(float(post_group.loc[policy, (metric, "std")]) if not pd.isna(post_group.loc[policy, (metric, "std")]) else 0.0)
            ax.bar(labels, means2, yerr=stds, capsize=3)
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=38, labelsize=8)
            ax.grid(axis="y", alpha=0.25)
        for ax in axes.ravel()[len(diag_metrics):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(plots_dir / "e5j_selection_diagnostics.png", dpi=150)
        plt.close(fig)

    trace_policies = ["random_budget", "wave_dynamic_c70_bonus2_agent_balanced", "hybrid_random25_wave75"]
    ts = metrics_df[(metrics_df["seed"] == 0) & (metrics_df["policy"].isin(trace_policies))]
    trace_metrics = ["old_route_ghost_pressure", "new_route_hidden_congestion", "ghost_pressure", "hidden_congestion", "mean_intention_age"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for ax, metric in zip(axes.ravel(), trace_metrics):
        for policy, sub in ts.groupby("policy", sort=False):
            sub = sub.sort_values("tick")
            ax.plot(sub["tick"], sub[metric], label=policy)
        ax.axvline(30, color="k", linestyle="--", alpha=0.45)
        ax.axvline(35, color="r", linestyle="--", alpha=0.35)
        ax.set_title(metric)
        ax.grid(alpha=0.25)
    axes.ravel()[-1].axis("off")
    axes[0, 1].legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "e5j_time_trace_best.png", dpi=150)
    plt.close(fig)


def make_e5j_findings(prepost: pd.DataFrame, selection_summary: pd.DataFrame) -> str:
    lines = ["E5J Hybrid Winner Confirmation Findings", ""]
    means = prepost.groupby("policy", dropna=False).mean(numeric_only=True)
    for policy, row in means.iterrows():
        lines.append(
            f"{policy}: old_ghost_auc={row['old_route_ghost_auc']:.4f}, "
            f"new_hidden_auc={row['new_route_hidden_auc']:.4f}, "
            f"post_new_hidden_auc={row['post_delivery_new_route_hidden_auc']:.4f}, "
            f"route_g1={row['route_balanced_loss_g1']:.4f}, route_g2={row['route_balanced_loss_g2']:.4f}, "
            f"uploaded={row['uploaded_tube_points']:.1f}, messages={row['message_count']:.1f}"
        )

    if "random_budget" in means.index and "hybrid_random25_wave75" in means.index:
        random = means.loc["random_budget"]
        hybrid25 = means.loc["hybrid_random25_wave75"]
        lines.extend(
            [
                "",
                "hybrid_random25_wave75 vs random_budget:",
                f"- beats old_route_ghost_auc: {hybrid25['old_route_ghost_auc'] <= random['old_route_ghost_auc']}",
                f"- beats post_delivery_new_route_hidden_auc: {hybrid25['post_delivery_new_route_hidden_auc'] <= random['post_delivery_new_route_hidden_auc']}",
                f"- beats route_balanced_loss_g1: {hybrid25['route_balanced_loss_g1'] <= random['route_balanced_loss_g1']}",
                f"- beats route_balanced_loss_g2: {hybrid25['route_balanced_loss_g2'] <= random['route_balanced_loss_g2']}",
            ]
        )

    best_metrics = {
        "old_route_ghost_auc": "best old_route_ghost_auc",
        "post_delivery_new_route_hidden_auc": "best post_delivery_new_route_hidden_auc",
        "route_balanced_loss_g1": "best route_balanced_loss_g1",
        "route_balanced_loss_g2": "best route_balanced_loss_g2",
    }
    lines.append("")
    for metric, label in best_metrics.items():
        if metric in means.columns:
            lines.append(f"{label}: {means[metric].idxmin()} ({means[metric].min():.4f})")

    hybrids = means[[str(idx).startswith("hybrid_") for idx in means.index]]
    if "wave_critical" in means.index and not hybrids.empty:
        wave = means.loc["wave_critical"]
        better_than_wave = []
        for policy, row in hybrids.iterrows():
            if (
                row["post_delivery_new_route_hidden_auc"] < wave["post_delivery_new_route_hidden_auc"]
                and row["route_balanced_loss_g1"] < wave["route_balanced_loss_g1"]
                and row["route_balanced_loss_g2"] < wave["route_balanced_loss_g2"]
            ):
                better_than_wave.append(str(policy))
        lines.append("")
        lines.append(f"Hybrid policies outperforming pure wave_critical on post-hidden and both route losses: {', '.join(better_than_wave) if better_than_wave else 'none'}")

    if not hybrids.empty:
        per_seed_best_post = (
            prepost[prepost["policy"].str.startswith("hybrid_", na=False)]
            .sort_values(["seed", "post_delivery_new_route_hidden_auc"])
            .groupby("seed", dropna=False)
            .first()["policy"]
            .value_counts()
        )
        per_seed_best_g1 = (
            prepost[prepost["policy"].str.startswith("hybrid_", na=False)]
            .sort_values(["seed", "route_balanced_loss_g1"])
            .groupby("seed", dropna=False)
            .first()["policy"]
            .value_counts()
        )
        lines.append("")
        lines.append("Best hybrid by seed for post_delivery_new_route_hidden_auc:")
        for policy, count in per_seed_best_post.items():
            lines.append(f"- {policy}: {int(count)} / {prepost['seed'].nunique()}")
        lines.append("Best hybrid by seed for route_balanced_loss_g1:")
        for policy, count in per_seed_best_g1.items():
            lines.append(f"- {policy}: {int(count)} / {prepost['seed'].nunique()}")

    if "random_budget" in means.index and not hybrids.empty:
        random_upload = float(means.loc["random_budget", "uploaded_tube_points"])
        max_gap = max(abs(float(row["uploaded_tube_points"]) - random_upload) for _, row in hybrids.iterrows())
        lines.append("")
        lines.append(f"Fair-budget uploaded_tube_points max hybrid/random mean gap: {max_gap:.4f}")
        if max_gap > 1.0:
            lines.append("WARNING: uploaded_tube_points differ by more than one point between random_budget and hybrids.")

    if "periodic_full" in means.index and "hybrid_random25_wave75" in means.index:
        if means.loc["hybrid_random25_wave75", "post_delivery_new_route_hidden_auc"] < means.loc["periodic_full", "post_delivery_new_route_hidden_auc"]:
            lines.append("WARNING: hybrid_random25_wave75 has lower post-delivery hidden than periodic_full; verify metric computation.")

    if not selection_summary.empty:
        sel = selection_summary.groupby("policy", dropna=False).mean(numeric_only=True)
        lines.append("")
        for policy, row in sel.iterrows():
            if str(policy).startswith("hybrid_"):
                lines.append(
                    f"{policy}: new_version_ratio={row.get('post_switch_selected_new_version_ratio', np.nan):.3f}, "
                    f"new_route_ratio={row.get('post_switch_selected_new_route_ratio', np.nan):.3f}, "
                    f"new_route_coverage={row.get('post_switch_new_route_coverage_ratio', np.nan):.3f}"
                )

    lines.append("")
    if "random_budget" in means.index and "hybrid_random25_wave75" in means.index:
        random = means.loc["random_budget"]
        hybrid25 = means.loc["hybrid_random25_wave75"]
        strict = (
            hybrid25["old_route_ghost_auc"] <= random["old_route_ghost_auc"]
            and hybrid25["post_delivery_new_route_hidden_auc"] <= random["post_delivery_new_route_hidden_auc"]
            and (
                hybrid25["route_balanced_loss_g1"] <= random["route_balanced_loss_g1"]
                or hybrid25["route_balanced_loss_g2"] <= random["route_balanced_loss_g2"]
            )
        )
        if strict:
            lines.append("hybrid_random25_wave75 remains a robust winner against random_budget under the E5J acceptance rule.")
        else:
            lines.append("random_budget remains competitive or better than hybrid_random25_wave75 under the 10-seed confirmation.")
    lines.append("E5J is a robustness confirmation only; it does not add new policies or reaction behavior.")
    return "\n".join(lines)


def run_e5j_hybrid_winner_confirmation(cfg: E5JConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    variants = add_variant_defaults(e5j_policy_variants())
    frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    total = len(cfg.seeds) * len(variants)
    run_idx = 0
    for seed in cfg.seeds:
        for variant in variants:
            run_idx += 1
            policy = str(variant["policy"])
            print(f"[E5J {run_idx}/{total}] seed={seed} policy={policy}", flush=True)
            e5b_cfg = E5BConfig(
                scenario=cfg.scenario,
                agents=cfg.agents,
                seeds=(seed,),
                sim_steps=cfg.sim_steps,
                horizon=cfg.horizon,
                upload_period=cfg.upload_period,
                route_switch_tick=cfg.route_switch_tick,
                switch_ratio=cfg.switch_ratio,
                delay_list=(cfg.delay,),
                policy=policy,
                buffer_modes=(cfg.buffer_mode,),
                budget_points=cfg.budget_points,
                freshness_tau=cfg.freshness_tau,
                ttl=cfg.ttl,
                base_radius=cfg.base_radius,
                width_growth=cfg.width_growth,
                force_upload_on_replan=cfg.force_upload_on_replan,
                plot_level=cfg.plot_level,
                beta=cfg.beta,
                top_k=cfg.top_k,
                peak_tolerance_radius=cfg.peak_tolerance_radius,
                active_threshold_ratio=cfg.active_threshold_ratio,
                selected_diagnostics=policy != "periodic_full",
                diagnostics_regime=cfg.regime,
                critical_budget_ratio=float(variant["critical_budget_ratio"]) if not pd.isna(variant["critical_budget_ratio"]) else 0.7,
                coverage_budget_ratio=float(variant["coverage_budget_ratio"]) if not pd.isna(variant["coverage_budget_ratio"]) else 0.3,
                new_version_weight=float(variant["new_version_weight"]) if not pd.isna(variant["new_version_weight"]) else 1.0,
                old_version_weight=2.0,
                new_version_boost_window=10,
                balance_time_bins=bool(variant["balance_time_bins"]),
                balance_agents=bool(variant["balance_agents"]),
                balance_spatial_bins=bool(variant["balance_spatial_bins"]),
                use_route_balance=bool(variant["use_route_balance"]),
                post_switch_new_version_bonus=float(variant["post_switch_new_version_bonus"]),
                lambda_agent=float(variant["lambda_agent"]),
                lambda_time=float(variant["lambda_time"]),
                lambda_space=float(variant["lambda_space"]),
            )
            frame = run_e5b_single(seed, cfg.delay, cfg.buffer_mode, e5b_cfg, [], selected_rows)
            frame["freshness_regime"] = cfg.regime
            frame["freshness_tau"] = cfg.freshness_tau
            frames.append(add_e5c_balanced_columns(frame))

    df = pd.concat(frames, ignore_index=True)
    e5e_cfg = E5EConfig(
        scenario=cfg.scenario,
        agents=cfg.agents,
        seeds=cfg.seeds,
        sim_steps=cfg.sim_steps,
        horizon=cfg.horizon,
        upload_period=cfg.upload_period,
        route_switch_tick=cfg.route_switch_tick,
        switch_ratio=cfg.switch_ratio,
        delay=cfg.delay,
        buffer_mode=cfg.buffer_mode,
        ttl=cfg.ttl,
        freshness_tau=cfg.freshness_tau,
        policies=tuple(v["policy"] for v in variants),
        budget_points=cfg.budget_points,
        base_radius=cfg.base_radius,
        width_growth=cfg.width_growth,
        force_upload_on_replan=cfg.force_upload_on_replan,
        plot_level=cfg.plot_level,
        analysis_window=cfg.analysis_window,
    )
    prepost = make_e5e_pre_post_summary(df, e5e_cfg)
    selected_df = pd.DataFrame(selected_rows)
    selection_summary = make_post_switch_selection_summary(selected_df, df, cfg)
    hybrid_summary = make_hybrid_quota_summary(selected_df, cfg)
    if not hybrid_summary.empty:
        selection_summary = selection_summary.merge(hybrid_summary, on=["seed", "policy"], how="left")
    delta = make_e5j_delta_vs_random(prepost)

    df.to_csv(out_dir / "metrics_tick.csv", index=False)
    summary_cols = prepost.select_dtypes(include=[np.number]).columns.tolist()
    summary = flatten_grouped_columns(prepost.groupby("policy", dropna=False)[summary_cols].agg(["mean", "std"]).reset_index())
    summary.to_csv(out_dir / "summary_by_policy.csv", index=False)
    selection_summary.to_csv(out_dir / "selection_summary.csv", index=False)
    delta.to_csv(out_dir / "hybrid_delta_vs_random.csv", index=False)
    findings = make_e5j_findings(prepost, selection_summary)
    (out_dir / "e5j_key_findings.txt").write_text(findings + "\n", encoding="utf-8")
    save_e5j_plots(prepost, selection_summary, df, delta, out_dir)

    print("\nE5J Hybrid Winner Confirmation Summary")
    print(findings)
    print(f"\nWrote E5J outputs to: {out_dir}")
    return df, prepost


@dataclass(frozen=True)
class E5KConfig(E5JConfig):
    pass


def e5k_policy_variants() -> list[dict[str, Any]]:
    return [
        {"policy": "periodic_full"},
        {"policy": "random_budget"},
        {"policy": "wave_critical"},
        {
            "policy": "wave_dynamic_c70_bonus2_agent_balanced",
            "critical_budget_ratio": 0.3,
            "coverage_budget_ratio": 0.7,
            "new_version_weight": 2.0,
            "balance_agents": True,
            "use_route_balance": False,
        },
        {"policy": "hybrid_random10_wave90"},
        {"policy": "hybrid_random25_wave75"},
        {"policy": "hybrid_random50_wave50"},
        {"policy": "hybrid_random75_wave25"},
        {"policy": "hybrid_agent_stratified_random50_wave50"},
        {"policy": "hybrid_time_stratified_random50_wave50"},
    ]


def e5k_short_label(policy: str) -> str:
    labels = {
        "periodic_full": "periodic_full",
        "random_budget": "random",
        "wave_critical": "wave_static",
        "wave_dynamic_c70_bonus2_agent_balanced": "wave_dynamic",
        "hybrid_random10_wave90": "hybrid_10_90",
        "hybrid_random25_wave75": "hybrid_25_75",
        "hybrid_random50_wave50": "hybrid_50_50",
        "hybrid_random75_wave25": "hybrid_75_25",
        "hybrid_agent_stratified_random50_wave50": "hybrid_agent_50_50",
        "hybrid_time_stratified_random50_wave50": "hybrid_time_50_50",
    }
    return labels.get(policy, policy)


def e5k_ratio_for_policy(policy: str) -> float | None:
    ratios = {
        "hybrid_random10_wave90": 0.10,
        "hybrid_random25_wave75": 0.25,
        "hybrid_random50_wave50": 0.50,
        "hybrid_random75_wave25": 0.75,
    }
    return ratios.get(policy)


def make_e5k_delta_vs_random(prepost: pd.DataFrame) -> pd.DataFrame:
    return make_e5j_delta_vs_random(prepost)


def make_e5k_ratio_summary(prepost: pd.DataFrame) -> pd.DataFrame:
    metrics = ["old_route_ghost_auc", "post_delivery_new_route_hidden_auc", "route_balanced_loss_g1"]
    rows: list[dict[str, Any]] = []
    for policy, sub in prepost.groupby("policy", dropna=False):
        ratio = e5k_ratio_for_policy(str(policy))
        if ratio is None:
            continue
        row: dict[str, Any] = {"policy": policy, "random_quota_ratio": ratio, "wave_quota_ratio": 1.0 - ratio}
        for metric in metrics:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_std"] = float(sub[metric].std(ddof=1)) if len(sub) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("random_quota_ratio").reset_index(drop=True)


def save_e5k_plots(
    prepost: pd.DataFrame,
    selection_summary: pd.DataFrame,
    metrics_df: pd.DataFrame,
    delta: pd.DataFrame,
    ratio_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    policy_order = [v["policy"] for v in e5k_policy_variants()]
    grouped = prepost.groupby("policy", dropna=False).agg(["mean", "std"])
    metrics = [
        "old_route_ghost_auc",
        "new_route_hidden_auc",
        "post_delivery_new_route_hidden_auc",
        "route_balanced_loss_g1",
        "route_balanced_loss_g2",
    ]

    means_df = prepost.groupby("policy", dropna=False).mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy in policy_order:
        if policy not in means_df.index:
            continue
        x = float(means_df.loc[policy, "post_delivery_new_route_hidden_auc"])
        y = float(means_df.loc[policy, "old_route_ghost_auc"])
        ax.scatter(x, y, s=55)
        ax.annotate(e5k_short_label(policy), (x, y), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("post_delivery_new_route_hidden_auc")
    ax.set_ylabel("old_route_ghost_auc")
    ax.set_title("E5K hidden vs stale ghost; lower left is better")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "e5k_pareto_hidden_vs_ghost.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    for ax, metric in zip(axes.ravel(), metrics):
        labels, means, stds = [], [], []
        for policy in policy_order:
            if policy in grouped.index:
                labels.append(e5k_short_label(policy))
                means.append(float(grouped.loc[policy, (metric, "mean")]))
                stds.append(float(grouped.loc[policy, (metric, "std")]) if not pd.isna(grouped.loc[policy, (metric, "std")]) else 0.0)
        ax.bar(labels, means, yerr=stds, capsize=3)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=38, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    fig.savefig(plots_dir / "e5k_policy_comparison.png", dpi=150)
    plt.close(fig)

    if not delta.empty:
        delta_metrics = [c for c in delta.columns if c.startswith("delta_")]
        fig, axes = plt.subplots(3, 2, figsize=(18, 11))
        for ax, metric in zip(axes.ravel(), delta_metrics):
            ax.bar([e5k_short_label(p) for p in delta["policy"]], delta[metric])
            ax.axhline(0.0, color="k", linewidth=1)
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=35, labelsize=8)
            ax.grid(axis="y", alpha=0.25)
        for ax in axes.ravel()[len(delta_metrics):]:
            ax.axis("off")
        fig.suptitle("hybrid minus random_budget; negative is better", y=1.02)
        fig.tight_layout()
        fig.savefig(plots_dir / "e5k_hybrid_vs_random_delta.png", dpi=150)
        plt.close(fig)

    if not ratio_summary.empty:
        ratio_metrics = ["old_route_ghost_auc_mean", "post_delivery_new_route_hidden_auc_mean", "route_balanced_loss_g1_mean"]
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        for ax, metric in zip(axes, ratio_metrics):
            ax.plot(ratio_summary["random_quota_ratio"], ratio_summary[metric], marker="o", label="hybrid ratios")
            if "random_budget" in means_df.index:
                ref_metric = metric.replace("_mean", "")
                ax.axhline(float(means_df.loc["random_budget", ref_metric]), color="tab:gray", linestyle="--", label="random")
            if "wave_critical" in means_df.index:
                ref_metric = metric.replace("_mean", "")
                ax.axhline(float(means_df.loc["wave_critical", ref_metric]), color="tab:red", linestyle=":", label="wave_static")
            ax.set_xlabel("random quota ratio")
            ax.set_title(metric.replace("_mean", ""))
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / "e5k_ratio_sweep.png", dpi=150)
        plt.close(fig)

    if not selection_summary.empty:
        selected_policies = [
            "random_budget",
            "wave_critical",
            "wave_dynamic_c70_bonus2_agent_balanced",
            "hybrid_random25_wave75",
            "hybrid_agent_stratified_random50_wave50",
            "hybrid_time_stratified_random50_wave50",
        ]
        post_group = selection_summary.groupby("policy", dropna=False).agg(["mean", "std"])
        diag_metrics = [
            "post_switch_selected_new_version_ratio",
            "post_switch_selected_new_route_ratio",
            "post_switch_new_route_coverage_ratio",
            "post_switch_old_route_coverage_ratio",
            "post_switch_selected_points_per_agent_std",
            "post_switch_selected_points_per_time_bin_std",
            "post_switch_selected_points_per_spatial_bin_std",
        ]
        fig, axes = plt.subplots(4, 2, figsize=(17, 13))
        for ax, metric in zip(axes.ravel(), diag_metrics):
            labels, means2, stds = [], [], []
            for policy in selected_policies:
                if policy in post_group.index and (metric, "mean") in post_group.columns:
                    labels.append(e5k_short_label(policy))
                    means2.append(float(post_group.loc[policy, (metric, "mean")]))
                    stds.append(float(post_group.loc[policy, (metric, "std")]) if not pd.isna(post_group.loc[policy, (metric, "std")]) else 0.0)
            ax.bar(labels, means2, yerr=stds, capsize=3)
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=35, labelsize=8)
            ax.grid(axis="y", alpha=0.25)
        for ax in axes.ravel()[len(diag_metrics):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(plots_dir / "e5k_selection_diagnostics.png", dpi=150)
        plt.close(fig)

    trace_policies = ["random_budget", "wave_dynamic_c70_bonus2_agent_balanced", "hybrid_random25_wave75"]
    ts = metrics_df[(metrics_df["seed"] == 0) & (metrics_df["policy"].isin(trace_policies))]
    trace_metrics = ["old_route_ghost_pressure", "new_route_hidden_congestion", "ghost_pressure", "hidden_congestion", "mean_intention_age"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for ax, metric in zip(axes.ravel(), trace_metrics):
        for policy, sub in ts.groupby("policy", sort=False):
            sub = sub.sort_values("tick")
            ax.plot(sub["tick"], sub[metric], label=e5k_short_label(policy))
        ax.axvline(30, color="k", linestyle="--", alpha=0.45)
        ax.axvline(35, color="r", linestyle="--", alpha=0.35)
        ax.set_title(metric)
        ax.grid(alpha=0.25)
    axes.ravel()[-1].axis("off")
    axes[0, 1].legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "e5k_time_trace_best.png", dpi=150)
    plt.close(fig)


def make_e5k_winner_summary(prepost: pd.DataFrame) -> str:
    means = prepost.groupby("policy", dropna=False).mean(numeric_only=True)
    lines = ["E5K Hybrid Ratio Confirmation Winner Summary", ""]
    for policy, row in means.iterrows():
        lines.append(
            f"{policy}: old_ghost={row['old_route_ghost_auc']:.4f}, "
            f"new_hidden={row['new_route_hidden_auc']:.4f}, "
            f"post_hidden={row['post_delivery_new_route_hidden_auc']:.4f}, "
            f"route_g1={row['route_balanced_loss_g1']:.4f}, "
            f"route_g2={row['route_balanced_loss_g2']:.4f}, "
            f"uploaded={row['uploaded_tube_points']:.1f}, messages={row['message_count']:.1f}"
        )
    best_metrics = [
        "old_route_ghost_auc",
        "post_delivery_new_route_hidden_auc",
        "route_balanced_loss_g1",
        "route_balanced_loss_g2",
    ]
    lines.append("")
    for metric in best_metrics:
        lines.append(f"Best policy by {metric}: {means[metric].idxmin()} ({means[metric].min():.4f})")

    confirmed = False
    if "random_budget" in means.index and "hybrid_random25_wave75" in means.index:
        random = means.loc["random_budget"]
        hybrid25 = means.loc["hybrid_random25_wave75"]
        checks = {
            "old_route_ghost_auc": hybrid25["old_route_ghost_auc"] <= random["old_route_ghost_auc"],
            "post_delivery_new_route_hidden_auc": hybrid25["post_delivery_new_route_hidden_auc"] <= random["post_delivery_new_route_hidden_auc"],
            "route_balanced_loss_g1": hybrid25["route_balanced_loss_g1"] < random["route_balanced_loss_g1"],
            "route_balanced_loss_g2": hybrid25["route_balanced_loss_g2"] < random["route_balanced_loss_g2"],
        }
        lines.append("")
        lines.append("hybrid_random25_wave75 vs random_budget:")
        for metric, ok in checks.items():
            lines.append(f"- beats {metric}: {ok}")
        seed_rows = prepost[prepost["policy"].isin(["random_budget", "hybrid_random25_wave75"])]
        pivot = seed_rows.pivot(index="seed", columns="policy", values=best_metrics)
        lines.append("")
        lines.append("Per-seed win rate of hybrid_random25_wave75 vs random_budget:")
        for metric in best_metrics:
            if (metric, "hybrid_random25_wave75") in pivot.columns and (metric, "random_budget") in pivot.columns:
                wins = (pivot[(metric, "hybrid_random25_wave75")] <= pivot[(metric, "random_budget")]).sum()
                lines.append(f"- {metric}: {int(wins)} / {len(pivot)}")
        confirmed = all(checks.values())

    conclusion = "not confirmed"
    if confirmed:
        conclusion = "confirmed"
    elif "random_budget" in means.index and "hybrid_random25_wave75" in means.index:
        random = means.loc["random_budget"]
        hybrid25 = means.loc["hybrid_random25_wave75"]
        if (
            hybrid25["route_balanced_loss_g1"] < random["route_balanced_loss_g1"]
            or hybrid25["route_balanced_loss_g2"] < random["route_balanced_loss_g2"]
            or hybrid25["old_route_ghost_auc"] <= random["old_route_ghost_auc"]
            or hybrid25["post_delivery_new_route_hidden_auc"] <= random["post_delivery_new_route_hidden_auc"]
        ):
            conclusion = "partially confirmed"
    lines.append("")
    lines.append(f"Plain English conclusion: {conclusion}.")
    if conclusion == "confirmed":
        lines.append("hybrid_random25_wave75 improves both route-balanced losses and does not worsen old-route ghost or post-delivery new-route hidden relative to random_budget.")
    elif conclusion == "partially confirmed":
        lines.append("hybrid_random25_wave75 improves some metrics but does not satisfy the full confirmation rule.")
    else:
        lines.append("hybrid_random25_wave75 does not beat random_budget under the E5K confirmation rule.")
    return "\n".join(lines)


def run_e5k_hybrid_ratio_confirm(cfg: E5KConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    variants = add_variant_defaults(e5k_policy_variants())
    print("E5K_HYBRID_RATIO_CONFIRM settings")
    print(f"scenario={cfg.scenario}")
    print(f"agents={cfg.agents}")
    print(f"seeds={list(cfg.seeds)}")
    print(f"sim_steps={cfg.sim_steps}")
    print(f"horizon={cfg.horizon}")
    print(f"upload_period={cfg.upload_period}")
    print(f"route_switch_tick={cfg.route_switch_tick}")
    print(f"switch_ratio={cfg.switch_ratio}")
    print(f"delay={cfg.delay}")
    print(f"buffer_mode={cfg.buffer_mode}")
    print(f"TTL={cfg.ttl}")
    print(f"freshness_tau={cfg.freshness_tau}")
    print(f"budget_points={cfg.budget_points}")
    print(f"regime={cfg.regime}")
    print(f"policies={[v['policy'] for v in variants]}")

    frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    total = len(cfg.seeds) * len(variants)
    run_idx = 0
    for seed in cfg.seeds:
        for variant in variants:
            run_idx += 1
            policy = str(variant["policy"])
            print(f"[E5K {run_idx}/{total}] seed={seed} policy={policy}", flush=True)
            e5b_cfg = E5BConfig(
                scenario=cfg.scenario,
                agents=cfg.agents,
                seeds=(seed,),
                sim_steps=cfg.sim_steps,
                horizon=cfg.horizon,
                upload_period=cfg.upload_period,
                route_switch_tick=cfg.route_switch_tick,
                switch_ratio=cfg.switch_ratio,
                delay_list=(cfg.delay,),
                policy=policy,
                buffer_modes=(cfg.buffer_mode,),
                budget_points=cfg.budget_points,
                freshness_tau=cfg.freshness_tau,
                ttl=cfg.ttl,
                base_radius=cfg.base_radius,
                width_growth=cfg.width_growth,
                force_upload_on_replan=cfg.force_upload_on_replan,
                plot_level=cfg.plot_level,
                beta=cfg.beta,
                top_k=cfg.top_k,
                peak_tolerance_radius=cfg.peak_tolerance_radius,
                active_threshold_ratio=cfg.active_threshold_ratio,
                selected_diagnostics=policy != "periodic_full",
                diagnostics_regime=cfg.regime,
                critical_budget_ratio=float(variant["critical_budget_ratio"]) if not pd.isna(variant["critical_budget_ratio"]) else 0.7,
                coverage_budget_ratio=float(variant["coverage_budget_ratio"]) if not pd.isna(variant["coverage_budget_ratio"]) else 0.3,
                new_version_weight=float(variant["new_version_weight"]) if not pd.isna(variant["new_version_weight"]) else 1.0,
                old_version_weight=2.0,
                new_version_boost_window=10,
                balance_time_bins=bool(variant["balance_time_bins"]),
                balance_agents=bool(variant["balance_agents"]),
                balance_spatial_bins=bool(variant["balance_spatial_bins"]),
                use_route_balance=bool(variant["use_route_balance"]),
                post_switch_new_version_bonus=float(variant["post_switch_new_version_bonus"]),
                lambda_agent=float(variant["lambda_agent"]),
                lambda_time=float(variant["lambda_time"]),
                lambda_space=float(variant["lambda_space"]),
            )
            frame = run_e5b_single(seed, cfg.delay, cfg.buffer_mode, e5b_cfg, [], selected_rows)
            frame["freshness_regime"] = cfg.regime
            frame["freshness_tau"] = cfg.freshness_tau
            frames.append(add_e5c_balanced_columns(frame))

    df = pd.concat(frames, ignore_index=True)
    e5e_cfg = E5EConfig(
        scenario=cfg.scenario,
        agents=cfg.agents,
        seeds=cfg.seeds,
        sim_steps=cfg.sim_steps,
        horizon=cfg.horizon,
        upload_period=cfg.upload_period,
        route_switch_tick=cfg.route_switch_tick,
        switch_ratio=cfg.switch_ratio,
        delay=cfg.delay,
        buffer_mode=cfg.buffer_mode,
        ttl=cfg.ttl,
        freshness_tau=cfg.freshness_tau,
        policies=tuple(v["policy"] for v in variants),
        budget_points=cfg.budget_points,
        base_radius=cfg.base_radius,
        width_growth=cfg.width_growth,
        force_upload_on_replan=cfg.force_upload_on_replan,
        plot_level=cfg.plot_level,
        analysis_window=cfg.analysis_window,
    )
    prepost = make_e5e_pre_post_summary(df, e5e_cfg)
    selected_df = pd.DataFrame(selected_rows)
    selection_summary = make_post_switch_selection_summary(selected_df, df, cfg)
    hybrid_summary = make_hybrid_quota_summary(selected_df, cfg)
    if not hybrid_summary.empty:
        selection_summary = selection_summary.merge(hybrid_summary, on=["seed", "policy"], how="left")
    delta = make_e5k_delta_vs_random(prepost)
    ratio_summary = make_e5k_ratio_summary(prepost)

    metric_cols = [
        "seed",
        "policy",
        "old_route_ghost_auc",
        "new_route_hidden_auc",
        "post_delivery_new_route_hidden_auc",
        "pre_delivery_new_route_hidden_auc",
        "pre_delivery_old_route_ghost_auc",
        "post_delivery_old_route_ghost_auc",
        "route_balanced_loss_g1",
        "route_balanced_loss_g2",
        "field_fidelity_score",
        "message_count",
        "uploaded_tube_points",
    ]
    prepost[metric_cols].to_csv(out_dir / "e5k_metrics_by_seed.csv", index=False)
    summary_cols = prepost.select_dtypes(include=[np.number]).columns.tolist()
    policy_summary = flatten_grouped_columns(prepost.groupby("policy", dropna=False)[summary_cols].agg(["mean", "std"]).reset_index())
    policy_summary.to_csv(out_dir / "e5k_policy_summary.csv", index=False)
    ratio_summary.to_csv(out_dir / "e5k_hybrid_ratio_summary.csv", index=False)
    selection_summary.to_csv(out_dir / "selection_summary.csv", index=False)
    delta.to_csv(out_dir / "hybrid_delta_vs_random.csv", index=False)
    findings = make_e5k_winner_summary(prepost)
    (out_dir / "e5k_winner_summary.txt").write_text(findings + "\n", encoding="utf-8")
    save_e5k_plots(prepost, selection_summary, df, delta, ratio_summary, out_dir)

    print("\nE5K Hybrid Ratio Confirmation Summary")
    cols = [
        "policy",
        "old_route_ghost_auc_mean",
        "post_delivery_new_route_hidden_auc_mean",
        "route_balanced_loss_g1_mean",
        "route_balanced_loss_g2_mean",
        "uploaded_tube_points_mean",
    ]
    print(policy_summary[cols].to_string(index=False))
    print("")
    print(findings)
    print(f"\nOutput folder: {out_dir}")
    if "random_budget" in prepost["policy"].values and "hybrid_random25_wave75" in prepost["policy"].values:
        means = prepost.groupby("policy", dropna=False).mean(numeric_only=True)
        random = means.loc["random_budget"]
        hybrid25 = means.loc["hybrid_random25_wave75"]
        confirmed = (
            hybrid25["route_balanced_loss_g1"] < random["route_balanced_loss_g1"]
            and hybrid25["route_balanced_loss_g2"] < random["route_balanced_loss_g2"]
            and hybrid25["old_route_ghost_auc"] <= random["old_route_ghost_auc"]
            and hybrid25["post_delivery_new_route_hidden_auc"] <= random["post_delivery_new_route_hidden_auc"]
        )
        print(f"hybrid_random25_wave75 confirmed against random_budget: {confirmed}")
    return df, prepost


@dataclass(frozen=True)
class E5LConfig(E5KConfig):
    budget_points_list: tuple[int, ...] = (400, 600, 800, 1000, 1200)


def e5l_policy_variants() -> list[dict[str, Any]]:
    return [
        {"policy": "periodic_full"},
        {"policy": "random_budget"},
        {"policy": "wave_critical"},
        {
            "policy": "wave_dynamic_c70_bonus2_agent_balanced",
            "critical_budget_ratio": 0.3,
            "coverage_budget_ratio": 0.7,
            "new_version_weight": 2.0,
            "balance_agents": True,
            "use_route_balance": False,
        },
        {"policy": "hybrid_random25_wave75"},
    ]


def make_e5l_delta_vs_random(prepost: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "old_route_ghost_auc",
        "new_route_hidden_auc",
        "post_delivery_new_route_hidden_auc",
        "route_balanced_loss_g1",
        "route_balanced_loss_g2",
    ]
    means = prepost.groupby(["budget_points", "policy"], dropna=False)[metrics].mean()
    rows: list[dict[str, Any]] = []
    for budget in sorted(prepost["budget_points"].unique()):
        if (budget, "random_budget") not in means.index:
            continue
        random = means.loc[(budget, "random_budget")]
        for policy in sorted(prepost["policy"].unique()):
            if not str(policy).startswith("hybrid_"):
                continue
            if (budget, policy) not in means.index:
                continue
            row = {"budget_points": int(budget), "policy": policy}
            for metric in metrics:
                row[f"delta_{metric}"] = float(means.loc[(budget, policy), metric] - random[metric])
            rows.append(row)
    return pd.DataFrame(rows)


def make_e5l_winner_summary(prepost: pd.DataFrame) -> str:
    lines = ["E5L Budget Robustness Summary", ""]
    metrics = [
        "old_route_ghost_auc",
        "post_delivery_new_route_hidden_auc",
        "route_balanced_loss_g1",
        "route_balanced_loss_g2",
        "uploaded_tube_points",
    ]
    grouped = prepost.groupby(["budget_points", "policy"], dropna=False)[metrics].mean().reset_index()
    for budget, sub in grouped.groupby("budget_points", sort=True):
        lines.append(f"budget_points={int(budget)}")
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['policy']}: old_ghost={row['old_route_ghost_auc']:.4f}, "
                f"post_hidden={row['post_delivery_new_route_hidden_auc']:.4f}, "
                f"g1={row['route_balanced_loss_g1']:.4f}, g2={row['route_balanced_loss_g2']:.4f}, "
                f"uploaded={row['uploaded_tube_points']:.1f}"
            )
        if "random_budget" in set(sub["policy"]) and "hybrid_random25_wave75" in set(sub["policy"]):
            random = sub[sub["policy"] == "random_budget"].iloc[0]
            hybrid = sub[sub["policy"] == "hybrid_random25_wave75"].iloc[0]
            confirmed = (
                hybrid["old_route_ghost_auc"] <= random["old_route_ghost_auc"]
                and hybrid["post_delivery_new_route_hidden_auc"] <= random["post_delivery_new_route_hidden_auc"]
                and hybrid["route_balanced_loss_g1"] < random["route_balanced_loss_g1"]
                and hybrid["route_balanced_loss_g2"] < random["route_balanced_loss_g2"]
            )
            lines.append(f"  hybrid_random25_wave75 beats random_budget: {confirmed}")
        lines.append("")

    beat_rows = []
    for budget, sub in grouped.groupby("budget_points", sort=True):
        if "random_budget" not in set(sub["policy"]) or "hybrid_random25_wave75" not in set(sub["policy"]):
            continue
        random = sub[sub["policy"] == "random_budget"].iloc[0]
        hybrid = sub[sub["policy"] == "hybrid_random25_wave75"].iloc[0]
        beat_rows.append(
            bool(
                hybrid["old_route_ghost_auc"] <= random["old_route_ghost_auc"]
                and hybrid["post_delivery_new_route_hidden_auc"] <= random["post_delivery_new_route_hidden_auc"]
                and hybrid["route_balanced_loss_g1"] < random["route_balanced_loss_g1"]
                and hybrid["route_balanced_loss_g2"] < random["route_balanced_loss_g2"]
            )
        )
    lines.append(f"hybrid_random25_wave75 wins {sum(beat_rows)} / {len(beat_rows)} evaluated budget levels by the strict E5K rule.")
    if beat_rows and all(beat_rows):
        lines.append("Conclusion: confirmed across evaluated budgets.")
    elif beat_rows and any(beat_rows):
        lines.append("Conclusion: partially confirmed; the hybrid advantage is budget-dependent.")
    else:
        lines.append("Conclusion: not confirmed across the evaluated budgets.")
    return "\n".join(lines)


def save_e5l_plots(prepost: pd.DataFrame, delta: pd.DataFrame, out_dir: Path) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    policy_order = [v["policy"] for v in e5l_policy_variants()]
    labels = {p: e5k_short_label(p) for p in policy_order}
    grouped = prepost.groupby(["budget_points", "policy"], dropna=False).agg(["mean", "std"])
    metrics = [
        "old_route_ghost_auc",
        "post_delivery_new_route_hidden_auc",
        "route_balanced_loss_g1",
        "route_balanced_loss_g2",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, metric in zip(axes.ravel(), metrics):
        for policy in policy_order:
            xs, ys, yerr = [], [], []
            for budget in sorted(prepost["budget_points"].unique()):
                idx = (budget, policy)
                if idx in grouped.index:
                    xs.append(int(budget))
                    ys.append(float(grouped.loc[idx, (metric, "mean")]))
                    std = grouped.loc[idx, (metric, "std")]
                    yerr.append(0.0 if pd.isna(std) else float(std))
            if xs:
                ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=labels.get(policy, policy))
        ax.set_title(metric)
        ax.set_xlabel("budget_points")
        ax.grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "e5l_budget_curves.png", dpi=150)
    plt.close(fig)

    if not delta.empty:
        delta_metrics = [c for c in delta.columns if c.startswith("delta_")]
        fig, axes = plt.subplots(3, 2, figsize=(14, 11))
        for ax, metric in zip(axes.ravel(), delta_metrics):
            for policy, sub in delta.groupby("policy", sort=False):
                sub = sub.sort_values("budget_points")
                ax.plot(sub["budget_points"], sub[metric], marker="o", label=e5k_short_label(policy))
            ax.axhline(0.0, color="k", linewidth=1)
            ax.set_title(metric)
            ax.set_xlabel("budget_points")
            ax.grid(alpha=0.25)
        for ax in axes.ravel()[len(delta_metrics):]:
            ax.axis("off")
        axes[0, 1].legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left")
        fig.suptitle("hybrid minus random_budget; negative is better", y=1.02)
        fig.tight_layout()
        fig.savefig(plots_dir / "e5l_hybrid_delta_vs_random.png", dpi=150)
        plt.close(fig)

    means = prepost.groupby(["budget_points", "policy"], dropna=False).mean(numeric_only=True).reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in means.iterrows():
        policy = str(row["policy"])
        if policy == "periodic_full":
            continue
        ax.scatter(row["post_delivery_new_route_hidden_auc"], row["old_route_ghost_auc"], s=45)
        ax.annotate(f"{e5k_short_label(policy)}:{int(row['budget_points'])}", (row["post_delivery_new_route_hidden_auc"], row["old_route_ghost_auc"]), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("post_delivery_new_route_hidden_auc")
    ax.set_ylabel("old_route_ghost_auc")
    ax.set_title("E5L budget Pareto points")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "e5l_pareto_by_budget.png", dpi=150)
    plt.close(fig)


def run_e5l_budget_robustness(cfg: E5LConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    variants = add_variant_defaults(e5l_policy_variants())
    print("E5L_BUDGET_ROBUSTNESS settings")
    print(f"scenario={cfg.scenario}")
    print(f"agents={cfg.agents}")
    print(f"seeds={list(cfg.seeds)}")
    print(f"sim_steps={cfg.sim_steps}")
    print(f"horizon={cfg.horizon}")
    print(f"upload_period={cfg.upload_period}")
    print(f"route_switch_tick={cfg.route_switch_tick}")
    print(f"switch_ratio={cfg.switch_ratio}")
    print(f"delay={cfg.delay}")
    print(f"buffer_mode={cfg.buffer_mode}")
    print(f"TTL={cfg.ttl}")
    print(f"freshness_tau={cfg.freshness_tau}")
    print(f"budget_points_list={list(cfg.budget_points_list)}")
    print(f"regime={cfg.regime}")
    print(f"policies={[v['policy'] for v in variants]}")

    frames: list[pd.DataFrame] = []
    total = len(cfg.budget_points_list) * len(cfg.seeds) * len(variants)
    run_idx = 0
    for budget in cfg.budget_points_list:
        for seed in cfg.seeds:
            for variant in variants:
                run_idx += 1
                policy = str(variant["policy"])
                print(f"[E5L {run_idx}/{total}] budget={budget} seed={seed} policy={policy}", flush=True)
                e5b_cfg = E5BConfig(
                    scenario=cfg.scenario,
                    agents=cfg.agents,
                    seeds=(seed,),
                    sim_steps=cfg.sim_steps,
                    horizon=cfg.horizon,
                    upload_period=cfg.upload_period,
                    route_switch_tick=cfg.route_switch_tick,
                    switch_ratio=cfg.switch_ratio,
                    delay_list=(cfg.delay,),
                    policy=policy,
                    buffer_modes=(cfg.buffer_mode,),
                    budget_points=int(budget),
                    freshness_tau=cfg.freshness_tau,
                    ttl=cfg.ttl,
                    base_radius=cfg.base_radius,
                    width_growth=cfg.width_growth,
                    force_upload_on_replan=cfg.force_upload_on_replan,
                    plot_level=cfg.plot_level,
                    beta=cfg.beta,
                    top_k=cfg.top_k,
                    peak_tolerance_radius=cfg.peak_tolerance_radius,
                    active_threshold_ratio=cfg.active_threshold_ratio,
                    selected_diagnostics=False,
                    diagnostics_regime=cfg.regime,
                    critical_budget_ratio=float(variant["critical_budget_ratio"]) if not pd.isna(variant["critical_budget_ratio"]) else 0.7,
                    coverage_budget_ratio=float(variant["coverage_budget_ratio"]) if not pd.isna(variant["coverage_budget_ratio"]) else 0.3,
                    new_version_weight=float(variant["new_version_weight"]) if not pd.isna(variant["new_version_weight"]) else 1.0,
                    old_version_weight=2.0,
                    new_version_boost_window=10,
                    balance_time_bins=bool(variant["balance_time_bins"]),
                    balance_agents=bool(variant["balance_agents"]),
                    balance_spatial_bins=bool(variant["balance_spatial_bins"]),
                    use_route_balance=bool(variant["use_route_balance"]),
                    post_switch_new_version_bonus=float(variant["post_switch_new_version_bonus"]),
                    lambda_agent=float(variant["lambda_agent"]),
                    lambda_time=float(variant["lambda_time"]),
                    lambda_space=float(variant["lambda_space"]),
                )
                frame = run_e5b_single(seed, cfg.delay, cfg.buffer_mode, e5b_cfg, [], None)
                frame["freshness_regime"] = cfg.regime
                frame["freshness_tau"] = cfg.freshness_tau
                frame["budget_points"] = int(budget)
                frames.append(add_e5c_balanced_columns(frame))

    df = pd.concat(frames, ignore_index=True)
    preposts: list[pd.DataFrame] = []
    for budget, sub in df.groupby("budget_points", sort=True):
        e5e_cfg = E5EConfig(
            scenario=cfg.scenario,
            agents=cfg.agents,
            seeds=cfg.seeds,
            sim_steps=cfg.sim_steps,
            horizon=cfg.horizon,
            upload_period=cfg.upload_period,
            route_switch_tick=cfg.route_switch_tick,
            switch_ratio=cfg.switch_ratio,
            delay=cfg.delay,
            buffer_mode=cfg.buffer_mode,
            ttl=cfg.ttl,
            freshness_tau=cfg.freshness_tau,
            policies=tuple(v["policy"] for v in variants),
            budget_points=int(budget),
            base_radius=cfg.base_radius,
            width_growth=cfg.width_growth,
            force_upload_on_replan=cfg.force_upload_on_replan,
            plot_level=cfg.plot_level,
            analysis_window=cfg.analysis_window,
        )
        pp = make_e5e_pre_post_summary(sub, e5e_cfg)
        pp["budget_points"] = int(budget)
        preposts.append(pp)
    prepost = pd.concat(preposts, ignore_index=True)
    delta = make_e5l_delta_vs_random(prepost)
    summary_cols = prepost.select_dtypes(include=[np.number]).columns.tolist()
    policy_budget_summary = flatten_grouped_columns(prepost.groupby(["budget_points", "policy"], dropna=False)[summary_cols].agg(["mean", "std"]).reset_index())
    findings = make_e5l_winner_summary(prepost)

    df.to_csv(out_dir / "e5l_metrics_tick.csv", index=False)
    prepost.to_csv(out_dir / "e5l_metrics_by_seed.csv", index=False)
    policy_budget_summary.to_csv(out_dir / "e5l_policy_budget_summary.csv", index=False)
    delta.to_csv(out_dir / "e5l_hybrid_delta_vs_random.csv", index=False)
    (out_dir / "e5l_winner_summary.txt").write_text(findings + "\n", encoding="utf-8")
    save_e5l_plots(prepost, delta, out_dir)

    print("\nE5L Budget Robustness Summary")
    compact_cols = [
        "budget_points",
        "policy",
        "old_route_ghost_auc_mean",
        "post_delivery_new_route_hidden_auc_mean",
        "route_balanced_loss_g1_mean",
        "route_balanced_loss_g2_mean",
        "uploaded_tube_points_mean",
    ]
    print(policy_budget_summary[compact_cols].to_string(index=False))
    print("")
    print(findings)
    print(f"\nOutput folder: {out_dir}")
    return df, prepost


@dataclass(frozen=True)
class E5MConfig(E5LConfig):
    hybrid_random_ratios: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)


def e5m_policy_name_for_ratio(ratio: float) -> str:
    pct = int(round(ratio * 100))
    return f"hybrid_random{pct}_wave{100 - pct}"


def e5m_ratio_for_policy(policy: str) -> float | None:
    if policy.startswith("hybrid_random") and "_wave" in policy:
        try:
            pct = int(policy.split("hybrid_random", 1)[1].split("_wave", 1)[0])
        except ValueError:
            return None
        return pct / 100.0
    return None


def e5m_short_label(policy: str) -> str:
    base = {
        "periodic_full": "full",
        "random_budget": "random",
        "wave_critical": "wave_static",
        "wave_dynamic_c70_bonus2_agent_balanced": "wave_dynamic",
    }
    if policy in base:
        return base[policy]
    ratio = e5m_ratio_for_policy(policy)
    if ratio is not None:
        pct = int(round(ratio * 100))
        return f"h{pct}_{100 - pct}"
    return e5k_short_label(policy)


def e5m_policy_variants(ratios: tuple[float, ...]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = [
        {"policy": "periodic_full"},
        {"policy": "random_budget"},
        {"policy": "wave_critical"},
        {
            "policy": "wave_dynamic_c70_bonus2_agent_balanced",
            "critical_budget_ratio": 0.3,
            "coverage_budget_ratio": 0.7,
            "new_version_weight": 2.0,
            "balance_agents": True,
            "use_route_balance": False,
        },
    ]
    for ratio in ratios:
        variants.append({"policy": e5m_policy_name_for_ratio(ratio)})
    return variants


def make_e5m_summary_by_budget_ratio(prepost: pd.DataFrame, selection_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "old_route_ghost_auc",
        "post_delivery_new_route_hidden_auc",
        "new_route_hidden_auc",
        "route_balanced_loss_g1",
        "route_balanced_loss_g2",
        "uploaded_tube_points",
        "message_count",
    ]
    summary = flatten_grouped_columns(prepost.groupby(["budget_points", "policy"], dropna=False)[metrics].agg(["mean", "std"]).reset_index())
    summary["random_quota_ratio"] = summary["policy"].map(lambda p: e5m_ratio_for_policy(str(p)))
    if not selection_summary.empty:
        diag_cols = [
            "post_switch_selected_new_version_ratio",
            "post_switch_selected_new_route_ratio",
            "post_switch_new_route_coverage_ratio",
            "post_switch_old_route_coverage_ratio",
            "post_switch_selected_points_per_agent_std",
            "post_switch_selected_points_per_time_bin_std",
            "post_switch_selected_points_per_spatial_bin_std",
        ]
        diag = flatten_grouped_columns(selection_summary.groupby(["budget_points", "policy"], dropna=False)[diag_cols].agg(["mean", "std"]).reset_index())
        summary = summary.merge(diag, on=["budget_points", "policy"], how="left")
    return summary


def make_e5m_best_ratio_by_budget(prepost: pd.DataFrame) -> pd.DataFrame:
    means = prepost.groupby(["budget_points", "policy"], dropna=False).mean(numeric_only=True).reset_index()
    rows: list[dict[str, Any]] = []
    for budget, sub in means.groupby("budget_points", sort=True):
        hybrids = sub[sub["policy"].map(lambda p: e5m_ratio_for_policy(str(p)) is not None)].copy()
        if hybrids.empty:
            continue
        row: dict[str, Any] = {"budget_points": int(budget)}
        for metric, out_name in [
            ("route_balanced_loss_g1", "g1"),
            ("route_balanced_loss_g2", "g2"),
            ("post_delivery_new_route_hidden_auc", "hidden"),
            ("old_route_ghost_auc", "ghost"),
        ]:
            best_h = hybrids.loc[hybrids[metric].idxmin()]
            row[f"best_ratio_by_{out_name}"] = e5m_ratio_for_policy(str(best_h["policy"]))
            row[f"best_hybrid_policy_by_{out_name}"] = best_h["policy"]
            row[f"best_{out_name}_value"] = float(best_h[metric])
        row["best_policy_by_g1"] = sub.loc[sub["route_balanced_loss_g1"].idxmin(), "policy"]
        row["best_policy_by_g2"] = sub.loc[sub["route_balanced_loss_g2"].idxmin(), "policy"]
        rows.append(row)
    return pd.DataFrame(rows)


def make_e5m_adaptive_policy_summary(prepost: pd.DataFrame, best_table: pd.DataFrame) -> pd.DataFrame:
    means = prepost.groupby(["budget_points", "policy"], dropna=False).mean(numeric_only=True).reset_index()
    rows: list[dict[str, Any]] = []
    metric_cols = ["old_route_ghost_auc", "post_delivery_new_route_hidden_auc", "route_balanced_loss_g1", "route_balanced_loss_g2"]
    for _, best in best_table.iterrows():
        budget = int(best["budget_points"])
        sub = means[means["budget_points"] == budget]
        entries = [
            ("random_budget", "random_budget"),
            ("fixed_h25_75", "hybrid_random25_wave75"),
            ("adaptive_hybrid_g1", best["best_hybrid_policy_by_g1"]),
            ("adaptive_hybrid_g2", best["best_hybrid_policy_by_g2"]),
        ]
        for label, policy in entries:
            match = sub[sub["policy"] == policy]
            if match.empty:
                continue
            row = {"budget_points": budget, "adaptive_policy": label, "source_policy": policy}
            for metric in metric_cols:
                row[metric] = float(match.iloc[0][metric])
            rows.append(row)
    return pd.DataFrame(rows)


def save_e5m_heatmap(matrix_df: pd.DataFrame, value_col: str, title: str, path: Path) -> None:
    budgets = sorted(matrix_df["budget_points"].unique())
    ratios = sorted(matrix_df["random_quota_ratio"].dropna().unique())
    mat = np.full((len(ratios), len(budgets)), np.nan)
    for i, ratio in enumerate(ratios):
        for j, budget in enumerate(budgets):
            match = matrix_df[(matrix_df["budget_points"] == budget) & (np.isclose(matrix_df["random_quota_ratio"], ratio))]
            if not match.empty:
                mat[i, j] = float(match.iloc[0][value_col])
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, aspect="auto", origin="lower")
    ax.set_xticks(range(len(budgets)), [str(int(b)) for b in budgets])
    ax.set_yticks(range(len(ratios)), [f"{r:.2f}" for r in ratios])
    ax.set_xlabel("budget_points")
    ax.set_ylabel("random quota ratio")
    ax.set_title(title)
    for i in range(len(ratios)):
        for j in range(len(budgets)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_e5m_plots(prepost: pd.DataFrame, summary: pd.DataFrame, best_table: pd.DataFrame, adaptive: pd.DataFrame, out_dir: Path) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    hybrids = summary[summary["random_quota_ratio"].notna()].copy()
    save_e5m_heatmap(hybrids, "route_balanced_loss_g1_mean", "route_balanced_loss_g1; lower is better", plots_dir / "e5m_ratio_heatmap_g1.png")
    save_e5m_heatmap(hybrids, "route_balanced_loss_g2_mean", "route_balanced_loss_g2; lower is better", plots_dir / "e5m_ratio_heatmap_g2.png")
    save_e5m_heatmap(hybrids, "post_delivery_new_route_hidden_auc_mean", "post_delivery_new_route_hidden_auc", plots_dir / "e5m_ratio_heatmap_hidden.png")
    save_e5m_heatmap(hybrids, "old_route_ghost_auc_mean", "old_route_ghost_auc", plots_dir / "e5m_ratio_heatmap_ghost.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    for col, label in [
        ("best_ratio_by_g1", "g1"),
        ("best_ratio_by_g2", "g2"),
        ("best_ratio_by_hidden", "hidden"),
        ("best_ratio_by_ghost", "ghost"),
    ]:
        ax.plot(best_table["budget_points"], best_table[col], marker="o", label=label)
    ax.set_xlabel("budget_points")
    ax.set_ylabel("best random quota ratio")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("E5M best ratio by budget")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "e5m_best_ratio_by_budget.png", dpi=150)
    plt.close(fig)

    metrics = ["old_route_ghost_auc", "post_delivery_new_route_hidden_auc", "route_balanced_loss_g1", "route_balanced_loss_g2"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, metric in zip(axes.ravel(), metrics):
        for policy, sub in adaptive.groupby("adaptive_policy", sort=False):
            sub = sub.sort_values("budget_points")
            ax.plot(sub["budget_points"], sub[metric], marker="o", label=policy)
        ax.set_title(metric)
        ax.set_xlabel("budget_points")
        ax.grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "e5m_adaptive_vs_fixed_vs_random.png", dpi=150)
    plt.close(fig)

    means = prepost.groupby(["budget_points", "policy"], dropna=False).mean(numeric_only=True).reset_index()
    budgets = sorted(means["budget_points"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, budget in zip(axes.ravel(), budgets):
        sub = means[means["budget_points"] == budget]
        for _, row in sub.iterrows():
            policy = str(row["policy"])
            if policy == "periodic_full":
                continue
            ax.scatter(row["post_delivery_new_route_hidden_auc"], row["old_route_ghost_auc"], s=38)
            ax.annotate(e5m_short_label(policy), (row["post_delivery_new_route_hidden_auc"], row["old_route_ghost_auc"]), fontsize=7, xytext=(3, 2), textcoords="offset points")
        ax.set_title(f"B={int(budget)}")
        ax.set_xlabel("post hidden")
        ax.set_ylabel("old ghost")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(budgets):]:
        ax.axis("off")
    fig.suptitle("E5M Pareto by budget; lower left is better", y=1.02)
    fig.tight_layout()
    fig.savefig(plots_dir / "e5m_pareto_by_budget.png", dpi=150)
    plt.close(fig)


def make_e5m_findings(best_table: pd.DataFrame, adaptive: pd.DataFrame) -> str:
    lines = ["E5M Budget-Adaptive Ratio Sweep Summary", ""]
    lines.append("Best ratio by budget:")
    for _, row in best_table.iterrows():
        lines.append(
            f"B={int(row['budget_points'])}: "
            f"g1={row['best_ratio_by_g1']:.2f}, g2={row['best_ratio_by_g2']:.2f}, "
            f"hidden={row['best_ratio_by_hidden']:.2f}, ghost={row['best_ratio_by_ghost']:.2f}, "
            f"best_policy_g1={row['best_policy_by_g1']}, best_policy_g2={row['best_policy_by_g2']}"
        )
    ratio_set = set(float(v) for v in best_table["best_ratio_by_g1"].dropna().unique())
    ratio_set.update(float(v) for v in best_table["best_ratio_by_g2"].dropna().unique())
    lines.append("")
    if len(ratio_set) > 1:
        lines.append("Interpretation: the optimal hybrid ratio is budget dependent.")
    else:
        only = next(iter(ratio_set)) if ratio_set else np.nan
        lines.append(f"Interpretation: one ratio dominates the evaluated balanced losses in this sweep: {only:.2f}.")
    wins_g1 = 0
    wins_g2 = 0
    budgets = sorted(adaptive["budget_points"].unique()) if not adaptive.empty else []
    for budget in budgets:
        sub = adaptive[adaptive["budget_points"] == budget]
        random = sub[sub["adaptive_policy"] == "random_budget"]
        g1 = sub[sub["adaptive_policy"] == "adaptive_hybrid_g1"]
        g2 = sub[sub["adaptive_policy"] == "adaptive_hybrid_g2"]
        if not random.empty and not g1.empty:
            wins_g1 += int(g1.iloc[0]["route_balanced_loss_g1"] < random.iloc[0]["route_balanced_loss_g1"])
        if not random.empty and not g2.empty:
            wins_g2 += int(g2.iloc[0]["route_balanced_loss_g2"] < random.iloc[0]["route_balanced_loss_g2"])
    lines.append(f"adaptive_hybrid_g1 beats random_budget on g1 at {wins_g1} / {len(budgets)} budgets.")
    lines.append(f"adaptive_hybrid_g2 beats random_budget on g2 at {wins_g2} / {len(budgets)} budgets.")
    fixed = adaptive[adaptive["adaptive_policy"] == "fixed_h25_75"]
    if not fixed.empty:
        fixed_mean_g1 = float(fixed["route_balanced_loss_g1"].mean())
        adaptive_g1_mean = float(adaptive[adaptive["adaptive_policy"] == "adaptive_hybrid_g1"]["route_balanced_loss_g1"].mean())
        lines.append(f"Mean g1 fixed h25_75={fixed_mean_g1:.4f}; adaptive_g1={adaptive_g1_mean:.4f}.")
    if wins_g1 >= 4 or wins_g2 >= 4:
        lines.append("Acceptance: adaptive hybrid beats random_budget at at least 4 / 5 budgets.")
    else:
        lines.append("Acceptance: adaptive hybrid does not beat random_budget at 4 / 5 budgets under the balanced-loss criterion.")
    return "\n".join(lines)


def run_e5m_budget_adaptive_ratio_sweep(cfg: E5MConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    variants = add_variant_defaults(e5m_policy_variants(cfg.hybrid_random_ratios))
    print("E5M_BUDGET_ADAPTIVE_RATIO_SWEEP settings")
    print(f"scenario={cfg.scenario}")
    print(f"agents={cfg.agents}")
    print(f"seeds={list(cfg.seeds)}")
    print(f"sim_steps={cfg.sim_steps}")
    print(f"horizon={cfg.horizon}")
    print(f"upload_period={cfg.upload_period}")
    print(f"route_switch_tick={cfg.route_switch_tick}")
    print(f"switch_ratio={cfg.switch_ratio}")
    print(f"delay={cfg.delay}")
    print(f"buffer_mode={cfg.buffer_mode}")
    print(f"TTL={cfg.ttl}")
    print(f"freshness_tau={cfg.freshness_tau}")
    print(f"budgets={list(cfg.budget_points_list)}")
    print(f"hybrid_random_ratios={list(cfg.hybrid_random_ratios)}")
    print(f"policies={[v['policy'] for v in variants]}")

    frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    total = len(cfg.budget_points_list) * len(cfg.seeds) * len(variants)
    run_idx = 0
    for budget in cfg.budget_points_list:
        for seed in cfg.seeds:
            for variant in variants:
                run_idx += 1
                policy = str(variant["policy"])
                print(f"[E5M {run_idx}/{total}] budget={budget} seed={seed} policy={policy}", flush=True)
                e5b_cfg = E5BConfig(
                    scenario=cfg.scenario,
                    agents=cfg.agents,
                    seeds=(seed,),
                    sim_steps=cfg.sim_steps,
                    horizon=cfg.horizon,
                    upload_period=cfg.upload_period,
                    route_switch_tick=cfg.route_switch_tick,
                    switch_ratio=cfg.switch_ratio,
                    delay_list=(cfg.delay,),
                    policy=policy,
                    buffer_modes=(cfg.buffer_mode,),
                    budget_points=int(budget),
                    freshness_tau=cfg.freshness_tau,
                    ttl=cfg.ttl,
                    base_radius=cfg.base_radius,
                    width_growth=cfg.width_growth,
                    force_upload_on_replan=cfg.force_upload_on_replan,
                    plot_level=cfg.plot_level,
                    beta=cfg.beta,
                    top_k=cfg.top_k,
                    peak_tolerance_radius=cfg.peak_tolerance_radius,
                    active_threshold_ratio=cfg.active_threshold_ratio,
                    selected_diagnostics=policy != "periodic_full",
                    selected_diagnostics_selected_only=True,
                    diagnostics_regime=cfg.regime,
                    critical_budget_ratio=float(variant["critical_budget_ratio"]) if not pd.isna(variant["critical_budget_ratio"]) else 0.7,
                    coverage_budget_ratio=float(variant["coverage_budget_ratio"]) if not pd.isna(variant["coverage_budget_ratio"]) else 0.3,
                    new_version_weight=float(variant["new_version_weight"]) if not pd.isna(variant["new_version_weight"]) else 1.0,
                    old_version_weight=2.0,
                    new_version_boost_window=10,
                    balance_time_bins=bool(variant["balance_time_bins"]),
                    balance_agents=bool(variant["balance_agents"]),
                    balance_spatial_bins=bool(variant["balance_spatial_bins"]),
                    use_route_balance=bool(variant["use_route_balance"]),
                    post_switch_new_version_bonus=float(variant["post_switch_new_version_bonus"]),
                    lambda_agent=float(variant["lambda_agent"]),
                    lambda_time=float(variant["lambda_time"]),
                    lambda_space=float(variant["lambda_space"]),
                )
                before_selected = len(selected_rows)
                frame = run_e5b_single(seed, cfg.delay, cfg.buffer_mode, e5b_cfg, [], selected_rows)
                for row in selected_rows[before_selected:]:
                    row["budget_points"] = int(budget)
                frame["freshness_regime"] = cfg.regime
                frame["freshness_tau"] = cfg.freshness_tau
                frame["budget_points"] = int(budget)
                frames.append(add_e5c_balanced_columns(frame))

    df = pd.concat(frames, ignore_index=True)
    preposts: list[pd.DataFrame] = []
    for budget, sub in df.groupby("budget_points", sort=True):
        e5e_cfg = E5EConfig(
            scenario=cfg.scenario,
            agents=cfg.agents,
            seeds=cfg.seeds,
            sim_steps=cfg.sim_steps,
            horizon=cfg.horizon,
            upload_period=cfg.upload_period,
            route_switch_tick=cfg.route_switch_tick,
            switch_ratio=cfg.switch_ratio,
            delay=cfg.delay,
            buffer_mode=cfg.buffer_mode,
            ttl=cfg.ttl,
            freshness_tau=cfg.freshness_tau,
            policies=tuple(v["policy"] for v in variants),
            budget_points=int(budget),
            base_radius=cfg.base_radius,
            width_growth=cfg.width_growth,
            force_upload_on_replan=cfg.force_upload_on_replan,
            plot_level=cfg.plot_level,
            analysis_window=cfg.analysis_window,
        )
        pp = make_e5e_pre_post_summary(sub, e5e_cfg)
        pp["budget_points"] = int(budget)
        preposts.append(pp)
    prepost = pd.concat(preposts, ignore_index=True)
    selection_df = pd.DataFrame(selected_rows)
    summaries = []
    if not selection_df.empty and "budget_points" in selection_df.columns:
        for budget, sub_sel in selection_df.groupby("budget_points", dropna=False):
            sub_metrics = df[df["budget_points"] == budget]
            part = make_post_switch_selection_summary(sub_sel, sub_metrics, cfg)
            if not part.empty:
                part["budget_points"] = int(budget)
                summaries.append(part)
    selection_summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()

    summary = make_e5m_summary_by_budget_ratio(prepost, selection_summary)
    best_table = make_e5m_best_ratio_by_budget(prepost)
    adaptive = make_e5m_adaptive_policy_summary(prepost, best_table)
    findings = make_e5m_findings(best_table, adaptive)

    metric_cols = [
        "seed",
        "budget_points",
        "policy",
        "old_route_ghost_auc",
        "post_delivery_new_route_hidden_auc",
        "new_route_hidden_auc",
        "route_balanced_loss_g1",
        "route_balanced_loss_g2",
        "uploaded_tube_points",
        "message_count",
    ]
    prepost[metric_cols].to_csv(out_dir / "e5m_metrics_by_seed.csv", index=False)
    summary.to_csv(out_dir / "e5m_summary_by_budget_ratio.csv", index=False)
    best_table.to_csv(out_dir / "e5m_best_ratio_by_budget.csv", index=False)
    adaptive.to_csv(out_dir / "e5m_adaptive_policy_summary.csv", index=False)
    if not selection_summary.empty:
        selection_summary.to_csv(out_dir / "e5m_selection_diagnostics.csv", index=False)
    (out_dir / "e5m_findings.txt").write_text(findings + "\n", encoding="utf-8")
    save_e5m_plots(prepost, summary, best_table, adaptive, out_dir)

    print("\nE5M Budget-Adaptive Ratio Sweep Summary")
    print(best_table.to_string(index=False))
    print("")
    print(findings)
    print(f"\nOutput folder: {out_dir}")
    return df, prepost


@dataclass(frozen=True)
class E5QConfig(E5MConfig):
    distribution_modes: tuple[str, ...] = ("random_uniform", "clustered_start", "balanced_corridor", "skewed_route", "deterministic_grid")


def e5q1_policy_variants() -> list[dict[str, Any]]:
    return add_variant_defaults(
        [
            {"policy": "random_budget"},
            {"policy": "wave_critical"},
            {
                "policy": "wave_dynamic_c70_bonus2_agent_balanced",
                "critical_budget_ratio": 0.3,
                "coverage_budget_ratio": 0.7,
                "new_version_weight": 2.0,
                "balance_agents": True,
                "use_route_balance": False,
            },
            {"policy": "fixed_hybrid_25_75"},
            {"policy": "adaptive_ratio_rule"},
            {"policy": "periodic_full"},
        ]
    )


def e5q2_policy_variants() -> list[dict[str, Any]]:
    return add_variant_defaults(
        [
            {"policy": "random_budget"},
            {"policy": "wave_critical"},
            {
                "policy": "wave_dynamic_c70_bonus2_agent_balanced",
                "critical_budget_ratio": 0.3,
                "coverage_budget_ratio": 0.7,
                "new_version_weight": 2.0,
                "balance_agents": True,
                "use_route_balance": False,
            },
            {"policy": "fixed_hybrid_25_75"},
            {"policy": "hybrid_random50_wave50"},
            {"policy": "adaptive_ratio_rule"},
        ]
    )


def e5q_short_label(policy: str) -> str:
    return {
        "periodic_full": "full",
        "random_budget": "random",
        "wave_critical": "wave_static",
        "wave_dynamic_c70_bonus2_agent_balanced": "wave_dynamic",
        "fixed_hybrid_25_75": "h25_75",
        "adaptive_ratio_rule": "adaptive",
        "hybrid_random50_wave50": "h50_50",
    }.get(policy, e5m_short_label(policy))


def add_candidate_budget_stats(prepost: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    upload = df[df["candidate_tube_points_per_upload"] > 0].copy()
    if upload.empty:
        return prepost
    group_cols = [c for c in ["distribution_mode", "budget_points", "seed", "policy"] if c in prepost.columns and c in upload.columns]
    stats = (
        upload.groupby(group_cols, dropna=False)
        .agg(
            full_candidate_points_mean=("candidate_tube_points_per_upload", "mean"),
            full_candidate_points_min=("candidate_tube_points_per_upload", "min"),
            full_candidate_points_max=("candidate_tube_points_per_upload", "max"),
            actual_uploaded_tube_points=("uploaded_tube_points", "mean"),
        )
        .reset_index()
    )
    out = prepost.merge(stats, on=group_cols, how="left")
    out["budget_ratio_mean"] = out["budget_points"] / out["full_candidate_points_mean"].replace(0, np.nan)
    out["actual_upload_ratio"] = out["actual_uploaded_tube_points"] / out["full_candidate_points_mean"].replace(0, np.nan)
    return out


def run_e5q_block(
    cfg: E5QConfig,
    variants: list[dict[str, Any]],
    budgets: tuple[int, ...],
    out_dir: Path,
    *,
    distribution_modes: tuple[str, ...] = ("random_uniform",),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    selection_summaries: list[pd.DataFrame] = []
    total = len(distribution_modes) * len(budgets) * len(cfg.seeds) * len(variants)
    run_idx = 0
    for distribution_mode in distribution_modes:
        for budget in budgets:
            for seed in cfg.seeds:
                for variant in variants:
                    run_idx += 1
                    policy = str(variant["policy"])
                    print(f"[E5Q {run_idx}/{total}] dist={distribution_mode} budget={budget} seed={seed} policy={policy}", flush=True)
                    e5b_cfg = E5BConfig(
                        scenario=cfg.scenario,
                        agents=cfg.agents,
                        seeds=(seed,),
                        sim_steps=cfg.sim_steps,
                        horizon=cfg.horizon,
                        upload_period=cfg.upload_period,
                        route_switch_tick=cfg.route_switch_tick,
                        switch_ratio=cfg.switch_ratio,
                        delay_list=(cfg.delay,),
                        policy=policy,
                        buffer_modes=(cfg.buffer_mode,),
                        budget_points=int(budget),
                        freshness_tau=cfg.freshness_tau,
                        ttl=cfg.ttl,
                        base_radius=cfg.base_radius,
                        width_growth=cfg.width_growth,
                        force_upload_on_replan=cfg.force_upload_on_replan,
                        plot_level=cfg.plot_level,
                        beta=cfg.beta,
                        top_k=cfg.top_k,
                        peak_tolerance_radius=cfg.peak_tolerance_radius,
                        active_threshold_ratio=cfg.active_threshold_ratio,
                        selected_diagnostics=policy != "periodic_full",
                        selected_diagnostics_selected_only=True,
                        diagnostics_regime=cfg.regime,
                        critical_budget_ratio=float(variant["critical_budget_ratio"]) if not pd.isna(variant["critical_budget_ratio"]) else 0.7,
                        coverage_budget_ratio=float(variant["coverage_budget_ratio"]) if not pd.isna(variant["coverage_budget_ratio"]) else 0.3,
                        new_version_weight=float(variant["new_version_weight"]) if not pd.isna(variant["new_version_weight"]) else 1.0,
                        old_version_weight=2.0,
                        new_version_boost_window=10,
                        balance_time_bins=bool(variant["balance_time_bins"]),
                        balance_agents=bool(variant["balance_agents"]),
                        balance_spatial_bins=bool(variant["balance_spatial_bins"]),
                        use_route_balance=bool(variant["use_route_balance"]),
                        post_switch_new_version_bonus=float(variant["post_switch_new_version_bonus"]),
                        lambda_agent=float(variant["lambda_agent"]),
                        lambda_time=float(variant["lambda_time"]),
                        lambda_space=float(variant["lambda_space"]),
                        distribution_mode=distribution_mode,
                    )
                    selected_rows: list[dict[str, Any]] = []
                    frame = run_e5b_single(seed, cfg.delay, cfg.buffer_mode, e5b_cfg, [], selected_rows)
                    frame["freshness_regime"] = cfg.regime
                    frame["freshness_tau"] = cfg.freshness_tau
                    frame["budget_points"] = int(budget)
                    frame["distribution_mode"] = distribution_mode
                    if selected_rows:
                        selected_df = pd.DataFrame(selected_rows)
                        selected_df["budget_points"] = int(budget)
                        selected_df["distribution_mode"] = distribution_mode
                        summary_cfg = E5EConfig(
                            scenario=cfg.scenario,
                            agents=cfg.agents,
                            seeds=(seed,),
                            sim_steps=cfg.sim_steps,
                            horizon=cfg.horizon,
                            upload_period=cfg.upload_period,
                            route_switch_tick=cfg.route_switch_tick,
                            switch_ratio=cfg.switch_ratio,
                            delay=cfg.delay,
                            buffer_mode=cfg.buffer_mode,
                            ttl=cfg.ttl,
                            freshness_tau=cfg.freshness_tau,
                            policies=(policy,),
                            budget_points=int(budget),
                            base_radius=cfg.base_radius,
                            width_growth=cfg.width_growth,
                            force_upload_on_replan=cfg.force_upload_on_replan,
                            plot_level=cfg.plot_level,
                            analysis_window=cfg.analysis_window,
                        )
                        part = make_post_switch_selection_summary(selected_df, frame, summary_cfg)
                        if not part.empty:
                            part["budget_points"] = int(budget)
                            part["distribution_mode"] = distribution_mode
                            selection_summaries.append(part)
                    frames.append(add_e5c_balanced_columns(frame))
    df = pd.concat(frames, ignore_index=True)
    preposts: list[pd.DataFrame] = []
    for (distribution_mode, budget), sub in df.groupby(["distribution_mode", "budget_points"], sort=True):
        e5e_cfg = E5EConfig(
            scenario=cfg.scenario,
            agents=cfg.agents,
            seeds=cfg.seeds,
            sim_steps=cfg.sim_steps,
            horizon=cfg.horizon,
            upload_period=cfg.upload_period,
            route_switch_tick=cfg.route_switch_tick,
            switch_ratio=cfg.switch_ratio,
            delay=cfg.delay,
            buffer_mode=cfg.buffer_mode,
            ttl=cfg.ttl,
            freshness_tau=cfg.freshness_tau,
            policies=tuple(v["policy"] for v in variants),
            budget_points=int(budget),
            base_radius=cfg.base_radius,
            width_growth=cfg.width_growth,
            force_upload_on_replan=cfg.force_upload_on_replan,
            plot_level=cfg.plot_level,
            analysis_window=cfg.analysis_window,
        )
        pp = make_e5e_pre_post_summary(sub, e5e_cfg)
        pp["budget_points"] = int(budget)
        pp["distribution_mode"] = distribution_mode
        preposts.append(pp)
    prepost = add_candidate_budget_stats(pd.concat(preposts, ignore_index=True), df)
    selection_summary = pd.concat(selection_summaries, ignore_index=True) if selection_summaries else pd.DataFrame()
    return df, prepost, selection_summary


def save_e5q1_plots(prepost: pd.DataFrame, out_dir: Path) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    metrics = ["old_route_ghost_auc", "post_delivery_new_route_hidden_auc", "route_balanced_loss_g1", "route_balanced_loss_g2"]
    means = prepost.groupby(["budget_points", "policy"], dropna=False).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, metric in zip(axes.ravel(), metrics):
        for policy in ["random_budget", "fixed_hybrid_25_75", "adaptive_ratio_rule", "periodic_full"]:
            sub = means[means["policy"] == policy].sort_values("budget_ratio_mean")
            if not sub.empty:
                ax.plot(sub["budget_ratio_mean"], sub[metric], marker="o", label=e5q_short_label(policy))
        ax.set_title(metric)
        ax.set_xlabel("budget_ratio_mean")
        ax.grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "e5q1_budget_ratio_curves.png", dpi=150)
    plt.close(fig)

    budgets = sorted(means["budget_points"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, budget in zip(axes.ravel(), budgets):
        sub = means[means["budget_points"] == budget]
        for _, row in sub.iterrows():
            ax.scatter(row["post_delivery_new_route_hidden_auc"], row["old_route_ghost_auc"], s=40)
            ax.annotate(e5q_short_label(str(row["policy"])), (row["post_delivery_new_route_hidden_auc"], row["old_route_ghost_auc"]), fontsize=7, xytext=(3, 2), textcoords="offset points")
        ax.set_title(f"B={int(budget)}")
        ax.set_xlabel("post hidden")
        ax.set_ylabel("old ghost")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(budgets):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(plots / "e5q1_near_full_pareto.png", dpi=150)
    plt.close(fig)

    full = means[means["policy"] == "periodic_full"].set_index("budget_points")
    rows = []
    for policy in ["adaptive_ratio_rule", "fixed_hybrid_25_75"]:
        for _, row in means[means["policy"] == policy].iterrows():
            budget = row["budget_points"]
            if budget not in full.index:
                continue
            out = {"budget_points": budget, "policy": policy}
            for metric in metrics:
                out[f"delta_{metric}"] = row[metric] - full.loc[budget, metric]
            rows.append(out)
    delta = pd.DataFrame(rows)
    if not delta.empty:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for ax, metric in zip(axes.ravel(), metrics):
            dcol = f"delta_{metric}"
            for policy, sub in delta.groupby("policy"):
                ax.plot(sub["budget_points"], sub[dcol], marker="o", label=e5q_short_label(policy))
            ax.axhline(0, color="k", linewidth=1)
            ax.set_title(dcol)
            ax.grid(alpha=0.25)
        axes[0, 1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots / "e5q1_hybrid_vs_full_delta.png", dpi=150)
        plt.close(fig)

    ratio_table = means[["budget_points", "policy", "full_candidate_points_mean", "budget_ratio_mean", "actual_uploaded_tube_points", "actual_upload_ratio"]]
    fig, ax = plt.subplots(figsize=(12, max(4, 0.28 * len(ratio_table))))
    ax.axis("off")
    ax.table(cellText=ratio_table.round(3).astype(str).values, colLabels=ratio_table.columns, loc="center", cellLoc="center")
    fig.tight_layout()
    fig.savefig(plots / "e5q1_budget_ratio_table.png", dpi=150)
    plt.close(fig)


def run_e5q1_high_budget_saturation(cfg: E5QConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = e5q1_policy_variants()
    df, prepost, selection = run_e5q_block(cfg, variants, cfg.budget_points_list, out_dir, distribution_modes=("random_uniform",))
    numeric_cols = [c for c in prepost.select_dtypes(include=[np.number]).columns if c not in {"budget_points"}]
    summary = flatten_grouped_columns(prepost.groupby(["budget_points", "policy"], dropna=False)[numeric_cols].agg(["mean", "std"]).reset_index())
    ratio_summary = summary[[
        "budget_points",
        "policy",
        "full_candidate_points_mean_mean",
        "full_candidate_points_min_mean",
        "full_candidate_points_max_mean",
        "actual_uploaded_tube_points_mean",
        "budget_ratio_mean_mean",
        "actual_upload_ratio_mean",
    ]]
    best_rows = []
    means = prepost.groupby(["budget_points", "policy"], dropna=False).mean(numeric_only=True).reset_index()
    for budget, sub in means.groupby("budget_points", sort=True):
        best_rows.append({
            "budget_points": budget,
            "best_policy_by_g1": sub.loc[sub["route_balanced_loss_g1"].idxmin(), "policy"],
            "best_policy_by_g2": sub.loc[sub["route_balanced_loss_g2"].idxmin(), "policy"],
            "best_policy_by_hidden": sub.loc[sub["post_delivery_new_route_hidden_auc"].idxmin(), "policy"],
            "best_policy_by_ghost": sub.loc[sub["old_route_ghost_auc"].idxmin(), "policy"],
        })
    best = pd.DataFrame(best_rows)
    prepost.to_csv(out_dir / "e5q1_metrics_by_seed.csv", index=False)
    summary.to_csv(out_dir / "e5q1_summary_by_budget_policy.csv", index=False)
    ratio_summary.to_csv(out_dir / "e5q1_budget_ratio_summary.csv", index=False)
    best.to_csv(out_dir / "e5q1_best_policy_by_budget_ratio.csv", index=False)
    if not selection.empty:
        selection.to_csv(out_dir / "e5q1_selection_diagnostics.csv", index=False)
    save_e5q1_plots(prepost, out_dir)
    print("\nE5Q1 High Budget Saturation Summary")
    print(ratio_summary.to_string(index=False))
    print(best.to_string(index=False))
    write_e5q_combined_interpretation(out_dir.parent)
    return df, prepost


def save_e5q2_plots(prepost: pd.DataFrame, selection: pd.DataFrame, random_vs_adaptive: pd.DataFrame, out_dir: Path) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    metrics = ["old_route_ghost_auc", "post_delivery_new_route_hidden_auc", "route_balanced_loss_g1", "route_balanced_loss_g2"]
    means = prepost.groupby(["distribution_mode", "budget_points", "policy"], dropna=False).mean(numeric_only=True).reset_index()

    dist_modes = list(dict.fromkeys(means["distribution_mode"].tolist()))
    fig, axes = plt.subplots(len(dist_modes), len(metrics), figsize=(5 * len(metrics), 3.2 * len(dist_modes)), squeeze=False)
    for r, dist in enumerate(dist_modes):
        sub_dist = means[means["distribution_mode"] == dist].groupby("policy", dropna=False)[metrics].mean().reset_index()
        order = [p for p in ["random_budget", "wave_critical", "wave_dynamic_c70_bonus2_agent_balanced", "fixed_hybrid_25_75", "hybrid_random50_wave50", "adaptive_ratio_rule"] if p in set(sub_dist["policy"])]
        for c, metric in enumerate(metrics):
            ax = axes[r, c]
            vals = [float(sub_dist.loc[sub_dist["policy"] == p, metric].iloc[0]) for p in order]
            ax.bar(range(len(order)), vals)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([e5q_short_label(p) for p in order], rotation=40, ha="right", fontsize=8)
            ax.set_title(metric if r == 0 else "")
            ax.grid(axis="y", alpha=0.25)
            if c == 0:
                ax.set_ylabel(dist)
    fig.tight_layout()
    fig.savefig(plots / "e5q2_policy_comparison_by_distribution.png", dpi=150)
    plt.close(fig)

    low = means[means["budget_points"].isin([400, 600])]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, dist in zip(axes.ravel(), dist_modes):
        sub = low[low["distribution_mode"] == dist]
        for _, row in sub.iterrows():
            ax.scatter(row["post_delivery_new_route_hidden_auc"], row["old_route_ghost_auc"], s=36)
            ax.annotate(f"{e5q_short_label(str(row['policy']))}:{int(row['budget_points'])}", (row["post_delivery_new_route_hidden_auc"], row["old_route_ghost_auc"]), fontsize=7, xytext=(3, 2), textcoords="offset points")
        ax.set_title(dist)
        ax.set_xlabel("post hidden")
        ax.set_ylabel("old ghost")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(dist_modes):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(plots / "e5q2_low_budget_pareto_by_distribution.png", dpi=150)
    plt.close(fig)

    if not random_vs_adaptive.empty:
        fig, axes = plt.subplots(2, 2, figsize=(16, 9))
        labels = [f"{r.distribution_mode}\nB{int(r.budget_points)}" for r in random_vs_adaptive.itertuples()]
        for ax, metric in zip(axes.ravel(), metrics):
            col = f"delta_{metric}"
            ax.bar(range(len(random_vs_adaptive)), random_vs_adaptive[col])
            ax.axhline(0, color="k", linewidth=1)
            ax.set_xticks(range(len(random_vs_adaptive)))
            ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
            ax.set_title(f"adaptive - random: {metric}")
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots / "e5q2_adaptive_vs_random_delta.png", dpi=150)
        plt.close(fig)

        budgets = sorted(random_vs_adaptive["budget_points"].unique())
        mat = np.full((len(dist_modes), len(budgets)), np.nan)
        for i, dist in enumerate(dist_modes):
            for j, budget in enumerate(budgets):
                match = random_vs_adaptive[(random_vs_adaptive["distribution_mode"] == dist) & (random_vs_adaptive["budget_points"] == budget)]
                if not match.empty:
                    mat[i, j] = float(match["random_advantage_g1"].iloc[0])
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm")
        ax.set_xticks(range(len(budgets)))
        ax.set_xticklabels([str(int(b)) for b in budgets])
        ax.set_yticks(range(len(dist_modes)))
        ax.set_yticklabels(dist_modes)
        ax.set_title("random g1 - adaptive g1; positive means adaptive better")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(plots / "e5q2_random_advantage_heatmap.png", dpi=150)
        plt.close(fig)

    if not selection.empty:
        diag_metrics = [
            "post_switch_new_route_coverage_ratio",
            "post_switch_selected_points_per_agent_std",
            "post_switch_selected_points_per_time_bin_std",
            "post_switch_selected_points_per_spatial_bin_std",
        ]
        available = [m for m in diag_metrics if m in selection.columns]
        if available:
            sel = selection[selection["policy"].isin(["random_budget", "adaptive_ratio_rule"])].groupby(["distribution_mode", "policy"], dropna=False)[available].mean().reset_index()
            fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4), squeeze=False)
            for ax, metric in zip(axes.ravel(), available):
                labels = []
                values = []
                for _, row in sel.iterrows():
                    labels.append(f"{row['distribution_mode']}\n{e5q_short_label(str(row['policy']))}")
                    values.append(row[metric])
                ax.bar(range(len(labels)), values)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
                ax.set_title(metric)
                ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(plots / "e5q2_selected_coverage_by_distribution.png", dpi=150)
            plt.close(fig)


def make_e5q2_random_vs_adaptive(prepost: pd.DataFrame) -> pd.DataFrame:
    metrics = ["old_route_ghost_auc", "post_delivery_new_route_hidden_auc", "route_balanced_loss_g1", "route_balanced_loss_g2"]
    means = prepost.groupby(["distribution_mode", "budget_points", "policy"], dropna=False)[metrics].mean().reset_index()
    rows: list[dict[str, Any]] = []
    for (dist, budget), sub in means.groupby(["distribution_mode", "budget_points"], sort=True):
        rand = sub[sub["policy"] == "random_budget"]
        adap = sub[sub["policy"] == "adaptive_ratio_rule"]
        if rand.empty or adap.empty:
            continue
        r = rand.iloc[0]
        a = adap.iloc[0]
        row: dict[str, Any] = {"distribution_mode": dist, "budget_points": int(budget)}
        for metric in metrics:
            row[f"random_{metric}"] = float(r[metric])
            row[f"adaptive_{metric}"] = float(a[metric])
            row[f"delta_{metric}"] = float(a[metric] - r[metric])
        row["random_advantage_g1"] = float(r["route_balanced_loss_g1"] - a["route_balanced_loss_g1"])
        row["random_advantage_g2"] = float(r["route_balanced_loss_g2"] - a["route_balanced_loss_g2"])
        rows.append(row)
    return pd.DataFrame(rows)


def make_e5q2_win_rate(random_vs_adaptive: pd.DataFrame) -> pd.DataFrame:
    if random_vs_adaptive.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for dist, sub in random_vs_adaptive.groupby("distribution_mode", sort=True):
        n = max(1, len(sub))
        rows.append(
            {
                "distribution_mode": dist,
                "budgets_tested": int(n),
                "adaptive_beats_random_g1_budgets": int((sub["delta_route_balanced_loss_g1"] < 0).sum()),
                "adaptive_beats_random_g2_budgets": int((sub["delta_route_balanced_loss_g2"] < 0).sum()),
                "adaptive_beats_random_hidden_budgets": int((sub["delta_post_delivery_new_route_hidden_auc"] < 0).sum()),
                "adaptive_beats_random_ghost_budgets": int((sub["delta_old_route_ghost_auc"] < 0).sum()),
                "adaptive_beats_random_g1_rate": float((sub["delta_route_balanced_loss_g1"] < 0).mean()),
                "adaptive_beats_random_g2_rate": float((sub["delta_route_balanced_loss_g2"] < 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def run_e5q2_agent_distribution_ablation(cfg: E5QConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = e5q2_policy_variants()
    df, prepost, selection = run_e5q_block(cfg, variants, cfg.budget_points_list, out_dir, distribution_modes=cfg.distribution_modes)
    numeric_cols = [c for c in prepost.select_dtypes(include=[np.number]).columns if c not in {"budget_points"}]
    summary = flatten_grouped_columns(prepost.groupby(["distribution_mode", "budget_points", "policy"], dropna=False)[numeric_cols].agg(["mean", "std"]).reset_index())
    random_vs_adaptive = make_e5q2_random_vs_adaptive(prepost)
    win_rate = make_e5q2_win_rate(random_vs_adaptive)
    prepost.to_csv(out_dir / "e5q2_metrics_by_seed.csv", index=False)
    summary.to_csv(out_dir / "e5q2_summary_by_distribution_budget_policy.csv", index=False)
    random_vs_adaptive.to_csv(out_dir / "e5q2_random_vs_adaptive_by_distribution.csv", index=False)
    win_rate.to_csv(out_dir / "e5q2_distribution_win_rate_summary.csv", index=False)
    if not selection.empty:
        selection.to_csv(out_dir / "e5q2_selection_diagnostics.csv", index=False)
    save_e5q2_plots(prepost, selection, random_vs_adaptive, out_dir)
    print("\nE5Q2 Agent Distribution Ablation Summary")
    if not random_vs_adaptive.empty:
        print(random_vs_adaptive[["distribution_mode", "budget_points", "delta_route_balanced_loss_g1", "delta_route_balanced_loss_g2"]].to_string(index=False))
    if not win_rate.empty:
        print(win_rate.to_string(index=False))
    write_e5q_combined_interpretation(out_dir.parent)
    return df, prepost


def write_e5q_combined_interpretation(root: Path) -> None:
    q1_dir = root / "e5q1_high_budget_saturation"
    q2_dir = root / "e5q2_agent_distribution_ablation"
    q1_summary_path = q1_dir / "e5q1_summary_by_budget_policy.csv"
    q1_ratio_path = q1_dir / "e5q1_budget_ratio_summary.csv"
    q2_delta_path = q2_dir / "e5q2_random_vs_adaptive_by_distribution.csv"
    q2_win_path = q2_dir / "e5q2_distribution_win_rate_summary.csv"
    if not (q1_summary_path.exists() and q1_ratio_path.exists() and q2_delta_path.exists()):
        return
    q1_summary = pd.read_csv(q1_summary_path)
    q1_ratio = pd.read_csv(q1_ratio_path)
    q2_delta = pd.read_csv(q2_delta_path)
    q2_win = pd.read_csv(q2_win_path) if q2_win_path.exists() else pd.DataFrame()

    lines = ["E5Q Combined Interpretation", ""]
    b1200 = q1_ratio[(q1_ratio["budget_points"] == 1200) & (q1_ratio["policy"] != "periodic_full")]
    if not b1200.empty:
        ratio = float(b1200["budget_ratio_mean_mean"].mean())
        lines.append(f"1. B=1200 coverage: mean budget_ratio={ratio:.3f}.")
        if ratio >= 0.98:
            lines.append("   B=1200 is effectively full coverage in this setup.")
        elif ratio >= 0.75:
            lines.append("   B=1200 is high budget but not full; it should not be described as unconstrained sharing.")
        else:
            lines.append("   B=1200 is not close to full coverage.")
    else:
        lines.append("1. B=1200 coverage: unavailable.")

    means_cols = {
        "old_route_ghost_auc": "old_route_ghost_auc_mean",
        "post_delivery_new_route_hidden_auc": "post_delivery_new_route_hidden_auc_mean",
        "route_balanced_loss_g1": "route_balanced_loss_g1_mean",
        "route_balanced_loss_g2": "route_balanced_loss_g2_mean",
    }
    full_dominates = True
    selective_better = False
    for budget, sub in q1_summary.groupby("budget_points", sort=True):
        full = sub[sub["policy"] == "periodic_full"]
        if full.empty:
            continue
        f = full.iloc[0]
        for policy in ["fixed_hybrid_25_75", "adaptive_ratio_rule"]:
            row = sub[sub["policy"] == policy]
            if row.empty:
                continue
            r = row.iloc[0]
            if any(float(r[col]) < float(f[col]) for col in means_cols.values() if col in r.index and col in f.index):
                selective_better = True
            if any(float(f[col]) > float(r[col]) for col in means_cols.values() if col in r.index and col in f.index):
                full_dominates = False
    if full_dominates:
        lines.append("2. Full sharing dominates the checked field metrics; the advantage of selection diminishes when communication is effectively unconstrained.")
    elif selective_better:
        lines.append("2. Selective sharing remains meaningful near high budget because at least one selective policy improves some ghost/hidden balanced metrics versus periodic_full.")
    else:
        lines.append("2. Full sharing is competitive, but no single dominance pattern is visible across all high-budget metrics.")

    random_rows = q1_summary[q1_summary["policy"] == "random_budget"].sort_values("budget_points")
    if len(random_rows) >= 2:
        hidden_first = float(random_rows.iloc[0]["post_delivery_new_route_hidden_auc_mean"])
        hidden_last = float(random_rows.iloc[-1]["post_delivery_new_route_hidden_auc_mean"])
        ghost_first = float(random_rows.iloc[0]["old_route_ghost_auc_mean"])
        ghost_last = float(random_rows.iloc[-1]["old_route_ghost_auc_mean"])
        lines.append(f"3. High-budget trend for random_budget: post-delivery hidden {hidden_first:.4f}->{hidden_last:.4f}, old-route ghost {ghost_first:.4f}->{ghost_last:.4f}.")

    if not q2_delta.empty:
        adaptive_g1_wins = int((q2_delta["delta_route_balanced_loss_g1"] < 0).sum())
        total = len(q2_delta)
        random_uniform = q2_delta[q2_delta["distribution_mode"] == "random_uniform"]
        non_uniform = q2_delta[q2_delta["distribution_mode"] != "random_uniform"]
        random_uniform_win = int((random_uniform["delta_route_balanced_loss_g1"] < 0).sum()) if not random_uniform.empty else 0
        non_uniform_win = int((non_uniform["delta_route_balanced_loss_g1"] < 0).sum()) if not non_uniform.empty else 0
        lines.append(f"4. Adaptive hybrid beats random_budget on g1 in {adaptive_g1_wins}/{total} distribution-budget settings.")
        if non_uniform_win > 0 and adaptive_g1_wins >= total / 2:
            lines.append("   The low-budget random baseline is not only a random_uniform artifact; adaptive hybrid gains appear across multiple placement distributions.")
        elif random_uniform_win > 0 and non_uniform_win == 0:
            lines.append("   The random-budget comparison is sensitive to the random_uniform setup; distribution structure changes the conclusion.")
        else:
            lines.append("   Random_budget remains a strong low-budget coverage baseline across the checked distributions.")

    if not q2_win.empty:
        mean_rate = float(q2_win["adaptive_beats_random_g1_rate"].mean())
        lines.append(f"5. Mean adaptive-vs-random g1 win rate across distribution modes: {mean_rate:.3f}.")

    lines.append("")
    lines.append("Final claim:")
    lines.append("The E5Q validation should be reported as a saturation and distribution-sensitivity check. Do not call high budgets full unless actual_upload_ratio is near 1.0, and do not claim random_budget is weak unless adaptive_ratio_rule wins across distribution modes.")
    root.mkdir(parents=True, exist_ok=True)
    (root / "e5q_combined_interpretation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class E6AConfig:
    scenario: str = "dual_corridor"
    agents: int = 60
    seeds: tuple[int, ...] = (0, 1, 2)
    sim_steps: int = 100
    horizon: int = 25
    upload_period: int = 5
    route_switch_tick: int = 30
    switch_ratio: float = 0.5
    delay: int = 5
    budget_points: int = 800
    freshness_tau: float = 20.0
    ttl: int = 30
    buffer_mode: str = "overwrite_with_decay_TTL"
    communication_policies: tuple[str, ...] = ("random_budget", "fixed_hybrid_25_75", "adaptive_ratio_rule", "periodic_full_reference")
    reaction_modes: tuple[str, ...] = ("no_reaction", "fixed_threshold", "adaptive_threshold")
    base_radius: float = 1.5
    width_growth: float = 0.0
    beta: float = 0.03
    top_k: int = 20
    peak_tolerance_radius: float = 3.0
    active_threshold_ratio: float = 0.05
    plot_level: str = "minimal"
    h_eval: int = 15
    theta_fixed: float = 1.0
    theta_base: float = 1.0
    alpha_age: float = 0.05
    alpha_stale: float = 0.5
    alpha_replan: float = 0.3
    alpha_persist: float = 0.2
    cooldown_ticks: int = 5
    margin: float = 0.1
    risk_persistence_window: int = 3
    distribution_mode: str = "random_uniform"
    risk_gap_threshold: float = 0.5
    stale_mass_threshold: float = 0.2
    exogenous_route_switch: bool = False
    force_upload_on_replan: bool = True
    collect_agent_diagnostics: bool = False


def e6a_policy_to_selector(policy: str, budget_points: int) -> str:
    if policy == "periodic_full_reference":
        return "periodic_full"
    if policy == "fixed_hybrid_25_75":
        return "hybrid_random25_wave75"
    if policy == "adaptive_ratio_rule":
        if budget_points <= 500:
            return "hybrid_random50_wave50"
        if budget_points < 900:
            return "hybrid_random25_wave75"
        return "hybrid_random10_wave90"
    return policy


def e6a_route_path(agent: RollingAgent, scenario, route_variant: str, horizon: int) -> list[tuple[int, int]]:
    path = generate_dual_corridor_path(scenario, agent.current_position, agent.goal, route_variant)
    if not path:
        return agent.get_future_path(0, horizon)
    return [path[min(k, len(path) - 1)] for k in range(1, horizon + 1)]


def e6a_field_risk(field: np.ndarray, path: list[tuple[int, int]], horizon: int) -> tuple[float, float]:
    vals: list[float] = []
    if field.size == 0 or not path:
        return 0.0, 0.0
    for k, (y, x) in enumerate(path[:horizon], start=1):
        if 0 <= k - 1 < field.shape[0] and 0 <= y < field.shape[1] and 0 <= x < field.shape[2]:
            vals.append(float(field[k - 1, y, x]))
    if not vals:
        return 0.0, 0.0
    arr = np.asarray(vals, dtype=float)
    return float(np.max(arr)), float(np.mean(arr))


def e6a_route_stale_mass_ratio(buffer: IntentionBuffer, current_time: int, path: list[tuple[int, int]], stale_age: int = 10) -> float:
    if not path:
        return 0.0
    cells = set(path)
    total = 0.0
    stale = 0.0
    for msg in buffer.get_active_messages():
        age = max(0, current_time - msg.upload_time)
        for p in msg.tube_points:
            if (p.y, p.x) in cells:
                total += 1.0
                if age > stale_age:
                    stale += 1.0
    return float(stale / max(1.0, total))


def e6a_route_buffer_diagnostics(
    buffer: IntentionBuffer,
    current_time: int,
    path: list[tuple[int, int]],
    stale_age: int = 10,
) -> tuple[float, float]:
    """Return point-weighted stale fraction and intention age near a route."""
    if not path:
        return 0.0, 0.0
    cells = set(path)
    ages: list[float] = []
    stale = 0
    for msg in buffer.get_active_messages():
        age = float(max(0, current_time - msg.upload_time))
        matches = sum(1 for point in msg.tube_points if (point.y, point.x) in cells)
        if matches:
            ages.extend([age] * matches)
            if age > stale_age:
                stale += matches
    if not ages:
        return 0.0, 0.0
    return float(stale / len(ages)), float(np.mean(ages))


def e6a_select_upload_points(
    policy: str,
    candidates: list[MotionTubePoint],
    rng: np.random.Generator,
    cfg: E6AConfig,
    scenario,
    current_time: int,
    new_version_by_agent: dict[int, int],
) -> list[MotionTubePoint]:
    selector = e6a_policy_to_selector(policy, cfg.budget_points)
    if selector.startswith("hybrid_"):
        selected, _, _ = select_hybrid_random_wave(
            selector,
            candidates,
            rng,
            cfg.budget_points,
            cfg.horizon,
            scenario.critical_mask,
            current_time,
            cfg.freshness_tau,
            current_time,
            current_time + cfg.delay,
            new_version_by_agent,
            scenario.grid.height,
        )
        return selected
    return select_tube_points(selector, candidates, rng, cfg.budget_points, cfg.horizon, scenario.critical_mask)


def e6a_enqueue_points(buffer: IntentionBuffer, points: list[MotionTubePoint], current_time: int, delay: int, horizon: int, policy: str) -> tuple[int, int]:
    message_count = 0
    uploaded = 0
    for msg in make_messages_from_points(points, current_time, delay, horizon, policy):
        buffer.enqueue_message(msg)
        message_count += 1
        uploaded += len(msg.tube_points)
    return message_count, uploaded


def run_e6a_single(seed: int, communication_policy: str, reaction_mode: str, cfg: E6AConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario = get_scenario(cfg.scenario)
    rng = np.random.default_rng(stable_seed(seed, "e6a", communication_policy, reaction_mode))
    agents = generate_distribution_agents(scenario, cfg.agents, rng, cfg.distribution_mode)
    route_variants = initialize_route_variants(agents, scenario, cfg.distribution_mode)
    buffer = IntentionBuffer(mode=cfg.buffer_mode, freshness_tau=cfg.freshness_tau, ttl=cfg.ttl)
    cooldown = defaultdict(int)
    replan_ticks: dict[int, list[int]] = defaultdict(list)
    risk_history: dict[int, list[float]] = defaultdict(list)
    old_route_mask = np.zeros_like(scenario.grid.obstacles, dtype=bool)
    new_route_mask = np.zeros_like(scenario.grid.obstacles, dtype=bool)
    new_version_by_agent: dict[int, int] = {}
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    agent_diagnostics: list[dict[str, Any]] = []

    for current_time in range(cfg.sim_steps):
        for agent in agents:
            if agent.active:
                agent.step()
            cooldown[agent.agent_id] = max(0, cooldown[agent.agent_id] - 1)

        forced_upload_points: list[MotionTubePoint] = []
        if cfg.exogenous_route_switch and current_time == cfg.route_switch_tick:
            n_switch = int(round(len(agents) * np.clip(cfg.switch_ratio, 0.0, 1.0)))
            if n_switch > 0:
                switch_idx = rng.choice(len(agents), size=n_switch, replace=False)
                old_points_all: list[MotionTubePoint] = []
                new_points_all: list[MotionTubePoint] = []
                for idx in switch_idx:
                    agent = agents[int(idx)]
                    old_points, new_points, _ = switch_agent_route(agent, scenario, current_time, cfg.horizon, cfg, route_variants)
                    old_points_all.extend(old_points)
                    new_points_all.extend(new_points)
                    new_version_by_agent[agent.agent_id] = agent.version_id
                    if cfg.force_upload_on_replan:
                        forced_upload_points.extend(new_points)
                old_route_mask |= tube_points_mask(old_points_all, old_route_mask.shape, radius=2)
                new_route_mask |= tube_points_mask(new_points_all, new_route_mask.shape, radius=2)

        oracle_points: list[MotionTubePoint] = []
        for agent in agents:
            oracle_points.extend(make_tube_points(agent, current_time, cfg.horizon, cfg.base_radius, "thin", 0.0, 2.0, cfg.beta, agent.version_id))
        R_full = build_rolling_field(oracle_points, scenario.grid, current_time, cfg.horizon)

        message_count = 0
        uploaded_tube_points = 0
        if current_time % max(1, cfg.upload_period) == 0 or forced_upload_points:
            candidates: list[MotionTubePoint] = list(forced_upload_points)
            skipped = {p.agent_id for p in forced_upload_points} if forced_upload_points else set()
            for agent in agents:
                if current_time % max(1, cfg.upload_period) != 0:
                    continue
                if agent.agent_id in skipped:
                    continue
                agent.version_id += 1
                candidates.extend(make_tube_points(agent, current_time, cfg.horizon, cfg.base_radius, "thin", cfg.width_growth, 2.0, cfg.beta, agent.version_id))
            selected = e6a_select_upload_points(communication_policy, candidates, rng, cfg, scenario, current_time, new_version_by_agent)
            msg_count, uploaded = e6a_enqueue_points(buffer, selected, current_time, cfg.delay, cfg.horizon, communication_policy)
            message_count += msg_count
            uploaded_tube_points += uploaded

        buffer.deliver_messages(current_time)
        buffer.update_buffer(current_time)
        stats = buffer.get_buffer_stats(current_time)
        comm_points = buffer.get_active_intentions(current_time)
        R_comm = build_rolling_field(comm_points, scenario.grid, current_time, cfg.horizon)
        field_metrics = compute_metrics(R_full, R_comm, communication_policy, top_k=cfg.top_k, peak_tolerance_radius=cfg.peak_tolerance_radius, active_threshold_ratio=cfg.active_threshold_ratio)
        old_route_ghost, new_route_hidden = route_specific_metrics(R_full, R_comm, old_route_mask, new_route_mask)

        tick_events: list[dict[str, Any]] = []
        theta_values: list[float] = []
        cur_comm_vals: list[float] = []
        alt_comm_vals: list[float] = []
        cur_full_vals: list[float] = []
        alt_full_vals: list[float] = []
        stale_vals: list[float] = []
        replanned_agents: list[RollingAgent] = []
        replan_new_points: list[MotionTubePoint] = []

        if reaction_mode != "no_reaction":
            for agent in agents:
                current_route = route_variants.get(agent.agent_id, "upper")
                alt_route = "lower" if current_route == "upper" else "upper"
                current_path = agent.get_future_path(current_time, cfg.h_eval)
                alt_path = e6a_route_path(agent, scenario, alt_route, cfg.h_eval)
                cur_comm_max, cur_comm_mean = e6a_field_risk(R_comm, current_path, cfg.h_eval)
                alt_comm_max, alt_comm_mean = e6a_field_risk(R_comm, alt_path, cfg.h_eval)
                cur_full_max, cur_full_mean = e6a_field_risk(R_full, current_path, cfg.h_eval)
                alt_full_max, alt_full_mean = e6a_field_risk(R_full, alt_path, cfg.h_eval)
                persistent = float(np.mean(risk_history[agent.agent_id][-cfg.risk_persistence_window:])) if risk_history[agent.agent_id] else 0.0
                if cfg.collect_agent_diagnostics:
                    stale_ratio, current_route_age = e6a_route_buffer_diagnostics(buffer, current_time, current_path)
                    alt_stale_ratio, alt_route_age = e6a_route_buffer_diagnostics(buffer, current_time, alt_path)
                else:
                    stale_ratio = e6a_route_stale_mass_ratio(buffer, current_time, current_path)
                    current_route_age = 0.0
                    alt_stale_ratio = 0.0
                    alt_route_age = 0.0
                recent_replans = sum(1 for t in replan_ticks[agent.agent_id] if current_time - t <= cfg.cooldown_ticks * 2)
                if reaction_mode == "fixed_threshold":
                    theta = cfg.theta_fixed
                else:
                    theta = (
                        cfg.theta_base
                        + cfg.alpha_age * stats.mean_intention_age
                        + cfg.alpha_stale * stale_ratio
                        + cfg.alpha_replan * recent_replans
                        - cfg.alpha_persist * persistent
                    )
                    theta = max(0.05, float(theta))
                risk_history[agent.agent_id].append(cur_comm_max)
                theta_values.append(theta)
                cur_comm_vals.append(cur_comm_max)
                alt_comm_vals.append(alt_comm_max)
                cur_full_vals.append(cur_full_max)
                alt_full_vals.append(alt_full_max)
                stale_vals.append(stale_ratio)
                did_replan = False
                if cur_comm_max > theta and alt_comm_max + cfg.margin < cur_comm_max and cooldown[agent.agent_id] == 0:
                    old_points = make_tube_points(agent, current_time, cfg.horizon, cfg.base_radius, "thin", 0.0, 2.0, cfg.beta, agent.version_id)
                    new_path = generate_dual_corridor_path(scenario, agent.current_position, agent.goal, alt_route)
                    if not new_path:
                        continue
                    old_route = current_route
                    agent.current_path = new_path
                    agent.path_index = 0
                    route_variants[agent.agent_id] = alt_route
                    agent.version_id += 1
                    new_version_by_agent[agent.agent_id] = agent.version_id
                    new_points = make_tube_points(agent, current_time, cfg.horizon, cfg.base_radius, "thin", 0.0, 2.0, cfg.beta, agent.version_id)
                    old_route_mask |= tube_points_mask(old_points, old_route_mask.shape, radius=2)
                    new_route_mask |= tube_points_mask(new_points, new_route_mask.shape, radius=2)
                    replan_new_points.extend(new_points)
                    replanned_agents.append(agent)
                    cooldown[agent.agent_id] = cfg.cooldown_ticks
                    oscillation = any(current_time - t <= cfg.cooldown_ticks * 2 for t in replan_ticks[agent.agent_id])
                    replan_ticks[agent.agent_id].append(current_time)
                    did_replan = True
                    risk_gap = cur_comm_max - cur_full_max
                    is_ghost = cur_full_max < cfg.theta_fixed
                    is_useful = cur_full_max > cfg.theta_fixed and alt_full_max + cfg.margin < cur_full_max
                    false_positive = cur_comm_max > theta and risk_gap > cfg.risk_gap_threshold
                    stale_window = cfg.route_switch_tick <= current_time < cfg.route_switch_tick + cfg.delay
                    stale_field = stale_ratio > cfg.stale_mass_threshold
                    event_type = "ghost_induced" if is_ghost else "useful" if is_useful else "unnecessary"
                    tick_events.append(
                        {
                            "seed": seed,
                            "tick": current_time,
                            "agent_id": agent.agent_id,
                            "old_route": old_route,
                            "new_route": alt_route,
                            "reaction_mode": reaction_mode,
                            "communication_policy": communication_policy,
                            "old_plan_version": agent.version_id - 1,
                            "new_plan_version": agent.version_id,
                            "current_route_risk_comm": cur_comm_max,
                            "alternative_route_risk_comm": alt_comm_max,
                            "current_route_risk_full": cur_full_max,
                            "alternative_route_risk_full": alt_full_max,
                            "current_route_mean_risk_comm": cur_comm_mean,
                            "alternative_route_mean_risk_comm": alt_comm_mean,
                            "current_route_mean_risk_full": cur_full_mean,
                            "alternative_route_mean_risk_full": alt_full_mean,
                            "threshold": theta,
                            "risk_gap": risk_gap,
                            "stale_mass_ratio_near_route": stale_ratio,
                            "false_positive_replan": false_positive,
                            "stale_window_replan": stale_window,
                            "stale_field_replan": stale_field,
                            "event_type": event_type,
                            "oscillation": oscillation,
                            "pre_replan_current_route_risk_full": cur_full_max,
                            "pre_replan_alternative_route_risk_full": alt_full_max,
                            "post_replan_realized_route_cost": alt_full_max,
                            "full_risk_reduction": cur_full_max - alt_full_max,
                            "comm_risk_reduction": cur_comm_max - alt_comm_max,
                            "overreaction_score": (cur_comm_max - alt_comm_max) - (cur_full_max - alt_full_max),
                            "was_replan_useful": is_useful,
                            "was_replan_unnecessary": event_type == "unnecessary",
                            "was_false_positive": false_positive,
                            "was_ghost_induced": is_ghost,
                        }
                    )
                if cfg.collect_agent_diagnostics:
                    full_scale_current = max(abs(cur_full_max), 1e-9)
                    full_scale_alt = max(abs(alt_full_max), 1e-9)
                    agent_diagnostics.append(
                        {
                            "seed": seed,
                            "tick": current_time,
                            "agent_id": agent.agent_id,
                            "communication_policy": communication_policy,
                            "reaction_mode": reaction_mode,
                            "current_route_risk_comm": cur_comm_max,
                            "alternative_route_risk_comm": alt_comm_max,
                            "current_route_risk_full": cur_full_max,
                            "alternative_route_risk_full": alt_full_max,
                            "risk_gap_current": cur_comm_max - cur_full_max,
                            "risk_gap_alternative": alt_comm_max - alt_full_max,
                            "risk_advantage_comm": cur_comm_max - alt_comm_max,
                            "risk_advantage_full": cur_full_max - alt_full_max,
                            "threshold": theta,
                            "stale_mass_ratio_near_current_route": stale_ratio,
                            "stale_mass_ratio_near_alternative_route": alt_stale_ratio,
                            "intention_age_near_current_route": current_route_age,
                            "intention_age_near_alternative_route": alt_route_age,
                            "local_route_risk_coverage_current": float(np.clip(1.0 - abs(cur_comm_max - cur_full_max) / full_scale_current, 0.0, 1.0)),
                            "local_route_risk_coverage_alternative": float(np.clip(1.0 - abs(alt_comm_max - alt_full_max) / full_scale_alt, 0.0, 1.0)),
                            "whether_replanned": did_replan,
                        }
                    )
        if replan_new_points:
            msg_count, uploaded = e6a_enqueue_points(buffer, replan_new_points, current_time, cfg.delay, cfg.horizon, communication_policy)
            message_count += msg_count
            uploaded_tube_points += uploaded
        events.extend(tick_events)

        total_replans = len(tick_events)
        useful = sum(1 for e in tick_events if e["event_type"] == "useful")
        ghost = sum(1 for e in tick_events if e["event_type"] == "ghost_induced")
        unnecessary = sum(1 for e in tick_events if e["event_type"] == "unnecessary")
        oscillations = sum(1 for e in tick_events if e["oscillation"])
        false_positive_replans = sum(1 for e in tick_events if e["false_positive_replan"])
        stale_window_replans = sum(1 for e in tick_events if e["stale_window_replan"])
        stale_field_replans = sum(1 for e in tick_events if e["stale_field_replan"])
        risk_gaps = [float(e["risk_gap"]) for e in tick_events]
        realized_risks = [e6a_field_risk(R_full, agent.get_future_path(current_time, cfg.h_eval), cfg.h_eval)[0] for agent in agents]
        row = {
            "seed": seed,
            "tick": current_time,
            "scenario": cfg.scenario,
            "communication_policy": communication_policy,
            "reaction_mode": reaction_mode,
            "horizon": cfg.horizon,
            "upload_period": cfg.upload_period,
            "delay": cfg.delay,
            "budget_points": cfg.budget_points,
            "total_replans": total_replans,
            "useful_replans": useful,
            "ghost_induced_replans": ghost,
            "false_positive_replans": false_positive_replans,
            "stale_window_replans": stale_window_replans,
            "stale_field_replans": stale_field_replans,
            "unnecessary_replans": unnecessary,
            "oscillation_count": oscillations,
            "mean_risk_gap_on_replans": float(np.mean(risk_gaps)) if risk_gaps else 0.0,
            "mean_theta": float(np.mean(theta_values)) if theta_values else 0.0,
            "mean_current_route_risk_comm": float(np.mean(cur_comm_vals)) if cur_comm_vals else 0.0,
            "mean_alternative_route_risk_comm": float(np.mean(alt_comm_vals)) if alt_comm_vals else 0.0,
            "mean_current_route_risk_full": float(np.mean(cur_full_vals)) if cur_full_vals else 0.0,
            "mean_alternative_route_risk_full": float(np.mean(alt_full_vals)) if alt_full_vals else 0.0,
            "mean_intention_age": stats.mean_intention_age,
            "stale_mass_ratio": float(np.mean(stale_vals)) if stale_vals else stats.stale_intention_ratio,
            "uploaded_tube_points": uploaded_tube_points,
            "message_count": message_count,
            "old_route_ghost_pressure": old_route_ghost,
            "new_route_hidden_congestion": new_route_hidden,
            "average_route_cost": float(np.mean(realized_risks)) if realized_risks else 0.0,
            "realized_congestion_count": int(np.sum(np.asarray(realized_risks) > cfg.theta_fixed)) if realized_risks else 0,
            "active_intention_count": stats.active_intention_count,
        }
        row.update(field_metrics)
        rows.append(row)

    tick_frame = pd.DataFrame(rows)
    if cfg.collect_agent_diagnostics:
        tick_frame.attrs["agent_diagnostics"] = pd.DataFrame(agent_diagnostics)
    return tick_frame, pd.DataFrame(events)


def summarize_e6a(tick_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (seed, policy, reaction), sub in tick_df.groupby(["seed", "communication_policy", "reaction_mode"], sort=True):
        rows.append(
            {
                "seed": seed,
                "communication_policy": policy,
                "reaction_mode": reaction,
                "total_replans": float(sub["total_replans"].sum()),
                "useful_replans": float(sub["useful_replans"].sum()),
                "ghost_induced_replans": float(sub["ghost_induced_replans"].sum()),
                "false_positive_replans": float(sub.get("false_positive_replans", pd.Series(dtype=float)).sum()),
                "stale_window_replans": float(sub.get("stale_window_replans", pd.Series(dtype=float)).sum()),
                "stale_field_replans": float(sub.get("stale_field_replans", pd.Series(dtype=float)).sum()),
                "unnecessary_replans": float(sub["unnecessary_replans"].sum()),
                "oscillation_count": float(sub["oscillation_count"].sum()),
                "mean_risk_gap_on_replans": float(sub.get("mean_risk_gap_on_replans", pd.Series([0.0])).mean()),
                "old_route_ghost_auc": float(sub["old_route_ghost_pressure"].sum()),
                "post_delivery_new_route_hidden_auc": float(sub["new_route_hidden_congestion"].sum()),
                "route_balanced_loss_g1": float((sub["hidden_congestion"] + sub["ghost_pressure"] / (1.0 + sub["ghost_pressure"])).mean()),
                "route_balanced_loss_g2": float((sub["hidden_congestion"] + 2.0 * sub["ghost_pressure"] / (1.0 + sub["ghost_pressure"])).mean()),
                "average_route_cost": float(sub["average_route_cost"].mean()),
                "realized_congestion_count": float(sub["realized_congestion_count"].sum()),
                "communication_points_used": float(sub["uploaded_tube_points"].sum()),
                "mean_field_fidelity_score": float(sub["field_fidelity_score"].mean()),
                "mean_theta_std": float(sub["mean_theta"].std(ddof=0)),
                "new_plan_versions_entered": float((sub["message_count"] > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def save_e6a_plots(tick_df: pd.DataFrame, events: pd.DataFrame, summary_seed: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, cfg: E6AConfig) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    mean_ticks = tick_df.groupby(["tick", "reaction_mode"], dropna=False)[["total_replans", "useful_replans", "ghost_induced_replans", "unnecessary_replans"]].mean().reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, metric in zip(axes.ravel(), ["total_replans", "useful_replans", "ghost_induced_replans", "unnecessary_replans"]):
        for mode, sub in mean_ticks.groupby("reaction_mode"):
            ax.plot(sub["tick"], sub[metric], label=mode)
        ax.set_title(metric)
        ax.grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "e6a_replan_time_series.png", dpi=150)
    plt.close(fig)

    mode_summary = summary_seed.groupby("reaction_mode", dropna=False)[["useful_replans", "ghost_induced_replans", "unnecessary_replans", "oscillation_count"]].agg(["mean", "std"]).reset_index()
    mode_summary = flatten_grouped_columns(mode_summary)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    modes = mode_summary["reaction_mode"].tolist()
    for ax, metric in zip(axes, ["useful_replans", "ghost_induced_replans", "unnecessary_replans", "oscillation_count"]):
        ax.bar(range(len(modes)), mode_summary[f"{metric}_mean"], yerr=mode_summary.get(f"{metric}_std"))
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels(modes, rotation=35, ha="right")
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "e6a_reaction_mode_comparison.png", dpi=150)
    plt.close(fig)

    field_metrics = ["old_route_ghost_auc", "post_delivery_new_route_hidden_auc", "route_balanced_loss_g1", "route_balanced_loss_g2", "average_route_cost"]
    fig, axes = plt.subplots(1, len(field_metrics), figsize=(5 * len(field_metrics), 4))
    labels = [f"{r.communication_policy}\n{r.reaction_mode}" for r in summary.itertuples()]
    for ax, metric in zip(axes, field_metrics):
        ax.bar(range(len(summary)), summary[f"{metric}_mean"])
        ax.set_xticks(range(len(summary)))
        ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "e6a_field_vs_reaction_summary.png", dpi=150)
    plt.close(fig)

    candidates = tick_df[(tick_df["reaction_mode"] == "adaptive_threshold") & (tick_df["communication_policy"] == "adaptive_ratio_rule")]
    if not candidates.empty:
        seed = int(candidates["seed"].iloc[0])
        sub = candidates[candidates["seed"] == seed]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(sub["tick"], sub["mean_current_route_risk_comm"], label="current risk comm")
        ax.plot(sub["tick"], sub["mean_alternative_route_risk_comm"], label="alternative risk comm")
        ax.plot(sub["tick"], sub["mean_theta"], label="threshold")
        if not events.empty:
            ev = events[(events["seed"] == seed) & (events["reaction_mode"] == "adaptive_threshold") & (events["communication_policy"] == "adaptive_ratio_rule")]
            for tick in sorted(ev["tick"].unique())[:30]:
                ax.axvline(tick, color="r", alpha=0.15)
        ax.set_title("E6A representative adaptive threshold trace")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots / "e6a_threshold_example_agent.png", dpi=150)
        plt.close(fig)


def write_e6a_findings(tick_df: pd.DataFrame, events: pd.DataFrame, summary_seed: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> str:
    lines = ["E6A Quick Closed-Loop Reaction Summary", ""]
    no_reaction_replans = float(summary_seed[summary_seed["reaction_mode"] == "no_reaction"]["total_replans"].sum())
    fixed_replans = float(summary_seed[summary_seed["reaction_mode"] == "fixed_threshold"]["total_replans"].sum())
    adaptive_theta_std = float(tick_df[tick_df["reaction_mode"] == "adaptive_threshold"]["mean_theta"].std(ddof=0))
    version_entered = int((events["new_plan_version"] > events["old_plan_version"]).sum()) if not events.empty else 0
    lines.append(f"no_reaction_zero_replans={no_reaction_replans == 0.0} total={no_reaction_replans:.0f}")
    lines.append(f"fixed_threshold_triggers_replans={fixed_replans > 0.0} total={fixed_replans:.0f}")
    lines.append(f"adaptive_threshold_changes_over_time={adaptive_theta_std > 1e-6} theta_std={adaptive_theta_std:.4f}")
    lines.append(f"new_plan_versions_created={version_entered}")
    means = summary_seed.groupby(["communication_policy", "reaction_mode"], dropna=False).mean(numeric_only=True)
    key_a = ("adaptive_ratio_rule", "adaptive_threshold")
    key_b = ("random_budget", "fixed_threshold")
    if key_a in means.index and key_b in means.index:
        a = means.loc[key_a, "ghost_induced_replans"]
        b = means.loc[key_b, "ghost_induced_replans"]
        lines.append(f"adaptive_ratio_rule+adaptive_threshold ghost_induced_replans_mean={a:.3f}")
        lines.append(f"random_budget+fixed_threshold ghost_induced_replans_mean={b:.3f}")
        lines.append(f"adaptive_ratio_rule+adaptive_threshold reduces ghost-induced replans vs random_budget+fixed_threshold: {a < b}")
    out = "\n".join(lines) + "\n"
    (out_dir / "e6a_key_findings.txt").write_text(out, encoding="utf-8")
    return out


def run_e6a_quick_closed_loop_reaction(cfg: E6AConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("E6A_QUICK_CLOSED_LOOP_REACTION settings")
    print(f"scenario={cfg.scenario} agents={cfg.agents} sim_steps={cfg.sim_steps} horizon={cfg.horizon}")
    print(f"upload_period={cfg.upload_period} delay={cfg.delay} budget_points={cfg.budget_points}")
    print(f"freshness_tau={cfg.freshness_tau} TTL={cfg.ttl} buffer_mode={cfg.buffer_mode}")
    print(f"communication_policies={list(cfg.communication_policies)}")
    print(f"reaction_modes={list(cfg.reaction_modes)}")
    frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    total = len(cfg.seeds) * len(cfg.communication_policies) * len(cfg.reaction_modes)
    run_idx = 0
    for seed in cfg.seeds:
        for policy in cfg.communication_policies:
            for reaction in cfg.reaction_modes:
                run_idx += 1
                print(f"[E6A {run_idx}/{total}] seed={seed} policy={policy} reaction={reaction}", flush=True)
                tick, events = run_e6a_single(seed, policy, reaction, cfg)
                frames.append(tick)
                if not events.empty:
                    event_frames.append(events)
    tick_df = pd.concat(frames, ignore_index=True)
    events_df = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    summary_seed = summarize_e6a(tick_df)
    summary = flatten_grouped_columns(summary_seed.groupby(["communication_policy", "reaction_mode"], dropna=False).agg(["mean", "std"]).reset_index())
    tick_df.to_csv(out_dir / "e6a_tick_metrics.csv", index=False)
    events_df.to_csv(out_dir / "e6a_agent_replan_events.csv", index=False)
    summary.to_csv(out_dir / "e6a_summary_by_policy_reaction.csv", index=False)
    save_e6a_plots(tick_df, events_df, summary_seed, summary, out_dir, cfg)
    findings = write_e6a_findings(tick_df, events_df, summary_seed, summary, out_dir)
    print("\n" + findings)
    print("E6A policy/reaction summary:")
    cols = [
        "communication_policy",
        "reaction_mode",
        "total_replans_mean",
        "useful_replans_mean",
        "ghost_induced_replans_mean",
        "unnecessary_replans_mean",
        "oscillation_count_mean",
    ]
    print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))
    print(f"Output folder: {out_dir}")
    return tick_df, events_df, summary


@dataclass(frozen=True)
class E6BConfig(E6AConfig):
    pass


def e6b_run_matrix(
    base_cfg: E6BConfig,
    *,
    phase: str,
    policies: tuple[str, ...],
    reaction_modes: tuple[str, ...],
    theta_fixed: float,
    margin: float,
    cooldown_ticks: int,
    delay: int,
    budget_points: int,
    freshness_tau: float,
    exogenous_route_switch: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    total = len(base_cfg.seeds) * len(policies) * len(reaction_modes)
    run_idx = 0
    cfg = replace(
        base_cfg,
        communication_policies=policies,
        reaction_modes=reaction_modes,
        theta_fixed=theta_fixed,
        theta_base=theta_fixed,
        margin=margin,
        cooldown_ticks=cooldown_ticks,
        delay=delay,
        budget_points=budget_points,
        freshness_tau=freshness_tau,
        exogenous_route_switch=exogenous_route_switch,
    )
    for seed in cfg.seeds:
        for policy in policies:
            for reaction in reaction_modes:
                run_idx += 1
                print(
                    f"[E6B {phase} {run_idx}/{total}] seed={seed} policy={policy} reaction={reaction} "
                    f"theta={theta_fixed} margin={margin} cooldown={cooldown_ticks} delay={delay} B={budget_points} tau={freshness_tau}",
                    flush=True,
                )
                tick, events = run_e6a_single(seed, policy, reaction, cfg)
                for frame in (tick, events):
                    if not frame.empty:
                        frame["phase"] = phase
                        frame["theta_fixed"] = theta_fixed
                        frame["margin"] = margin
                        frame["cooldown_ticks"] = cooldown_ticks
                        frame["freshness_tau"] = freshness_tau
                frames.append(tick)
                if not events.empty:
                    event_frames.append(events)
    tick_df = pd.concat(frames, ignore_index=True)
    events_df = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    seed_summary = summarize_e6a(tick_df)
    for col, value in [
        ("phase", phase),
        ("theta_fixed", theta_fixed),
        ("margin", margin),
        ("cooldown_ticks", cooldown_ticks),
        ("delay", delay),
        ("budget_points", budget_points),
        ("freshness_tau", freshness_tau),
    ]:
        seed_summary[col] = value
    return tick_df, events_df, seed_summary


def e6b_calibration_score(summary: pd.DataFrame) -> pd.DataFrame:
    work = summary[summary["reaction_mode"].isin(["fixed_threshold", "adaptive_threshold"])].copy()
    grouped = (
        work.groupby(["theta_fixed", "margin", "cooldown_ticks"], dropna=False)
        .agg(
            useful_replans=("useful_replans", "mean"),
            unnecessary_replans=("unnecessary_replans", "mean"),
            oscillation_count=("oscillation_count", "mean"),
            average_route_cost=("average_route_cost", "mean"),
        )
        .reset_index()
    )
    if grouped.empty:
        return grouped
    for col in ["unnecessary_replans", "oscillation_count", "average_route_cost"]:
        denom = max(float(grouped[col].max() - grouped[col].min()), 1e-9)
        grouped[f"{col}_norm"] = (grouped[col] - grouped[col].min()) / denom
    useful_denom = max(float(grouped["useful_replans"].max() - grouped["useful_replans"].min()), 1e-9)
    grouped["useful_loss_norm"] = (grouped["useful_replans"].max() - grouped["useful_replans"]) / useful_denom
    grouped["selection_score"] = (
        grouped["oscillation_count_norm"]
        + 0.7 * grouped["unnecessary_replans_norm"]
        + 0.3 * grouped["average_route_cost_norm"]
        + 0.4 * grouped["useful_loss_norm"]
    )
    return grouped.sort_values("selection_score").reset_index(drop=True)


def e6b_load_e6a_default_oscillation() -> float | None:
    path = Path("results/e6a_quick_closed_loop_reaction/e6a_summary_by_policy_reaction.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    col = "oscillation_count_mean"
    if col not in df.columns:
        return None
    return float(df[df["reaction_mode"] != "no_reaction"][col].mean())


def save_e6b_plots(calibration: pd.DataFrame, ghost: pd.DataFrame, events: pd.DataFrame, tick_df: pd.DataFrame, out_dir: Path) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    cal_mean = calibration.groupby(["theta_fixed", "margin", "cooldown_ticks"], dropna=False).mean(numeric_only=True).reset_index()
    if not cal_mean.empty:
        labels = [f"t{r.theta_fixed:g}/m{r.margin:g}/c{int(r.cooldown_ticks)}" for r in cal_mean.itertuples()]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(cal_mean["oscillation_count"], cal_mean["useful_replans"], s=50)
        for label, row in zip(labels, cal_mean.itertuples()):
            ax.annotate(label, (row.oscillation_count, row.useful_replans), fontsize=7, xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel("oscillation_count")
        ax.set_ylabel("useful_replans")
        ax.set_title("E6B calibration: lower x, higher y is better")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots / "e6b_calibration_oscillation_vs_useful.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(cal_mean["unnecessary_replans"], cal_mean["average_route_cost"], s=50)
        for label, row in zip(labels, cal_mean.itertuples()):
            ax.annotate(label, (row.unnecessary_replans, row.average_route_cost), fontsize=7, xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel("unnecessary_replans")
        ax.set_ylabel("average_route_cost")
        ax.set_title("E6B calibration: lower left is better")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots / "e6b_calibration_unnecessary_vs_cost.png", dpi=150)
        plt.close(fig)

    ghost_mean = ghost[ghost["reaction_mode"].isin(["fixed_threshold", "adaptive_threshold"])].groupby(["communication_policy", "reaction_mode"], dropna=False).mean(numeric_only=True).reset_index()
    if not ghost_mean.empty:
        metrics = ["false_positive_replans", "ghost_induced_replans", "stale_window_replans", "unnecessary_replans", "oscillation_count"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
        labels = [f"{r.communication_policy}\n{r.reaction_mode}" for r in ghost_mean.itertuples()]
        for ax, metric in zip(axes, metrics):
            ax.bar(range(len(ghost_mean)), ghost_mean[metric])
            ax.set_xticks(range(len(ghost_mean)))
            ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
            ax.set_title(metric)
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots / "e6b_ghost_stress_reaction_comparison.png", dpi=150)
        plt.close(fig)

    if not events.empty and "risk_gap" in events.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        data = [sub["risk_gap"].to_numpy() for _, sub in events.groupby("reaction_mode") if len(sub)]
        labels = [str(mode) for mode, sub in events.groupby("reaction_mode") if len(sub)]
        if data:
            ax.boxplot(data, labels=labels, showfliers=False)
        ax.axhline(0.5, color="r", linestyle="--", label="risk_gap_threshold")
        ax.set_title("E6B risk_gap distribution for replan events")
        ax.set_ylabel("R_comm current risk - R_full current risk")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "e6b_risk_gap_distribution.png", dpi=150)
        plt.close(fig)

    trace = tick_df[(tick_df["phase"] == "ghost_stress_tau20") & (tick_df["communication_policy"] == "adaptive_ratio_rule") & (tick_df["reaction_mode"] == "adaptive_threshold")]
    if not trace.empty:
        sub = trace[trace["seed"] == trace["seed"].iloc[0]]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(sub["tick"], sub["mean_theta"], label="threshold")
        ax.plot(sub["tick"], sub["stale_mass_ratio"], label="stale_mass_ratio")
        ax.plot(sub["tick"], sub["mean_current_route_risk_comm"], label="current_route_risk_comm")
        ax.plot(sub["tick"], sub["mean_current_route_risk_full"], label="current_route_risk_full")
        replans = sub[sub["total_replans"] > 0]
        for tick in replans["tick"].head(40):
            ax.axvline(tick, color="r", alpha=0.12)
        ax.set_title("E6B adaptive threshold and stale-mass trace")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots / "e6b_threshold_stale_mass_trace.png", dpi=150)
        plt.close(fig)


def run_e6b_reaction_calibration_and_ghost_stress(cfg: E6BConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("E6B_REACTION_CALIBRATION_AND_GHOST_STRESS settings")
    print(f"scenario={cfg.scenario} agents={cfg.agents} sim_steps={cfg.sim_steps} horizon={cfg.horizon}")
    print(f"policies={list(cfg.communication_policies)} reaction_modes={list(cfg.reaction_modes)} seeds={list(cfg.seeds)}")
    policies = cfg.communication_policies
    modes = cfg.reaction_modes
    tick_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    calibration_summaries: list[pd.DataFrame] = []

    for cooldown in (5, 10, 15):
        tick, events, summary = e6b_run_matrix(
            cfg,
            phase="calibration_cooldown",
            policies=policies,
            reaction_modes=modes,
            theta_fixed=1.5,
            margin=0.3,
            cooldown_ticks=cooldown,
            delay=5,
            budget_points=800,
            freshness_tau=cfg.freshness_tau,
            exogenous_route_switch=False,
        )
        tick_frames.append(tick)
        if not events.empty:
            event_frames.append(events)
        calibration_summaries.append(summary)
    phase1 = pd.concat(calibration_summaries, ignore_index=True)
    phase1_scores = e6b_calibration_score(phase1)
    best_cooldown = int(phase1_scores.iloc[0]["cooldown_ticks"]) if not phase1_scores.empty else 10

    for theta in (1.0, 1.5, 2.0):
        for margin in (0.1, 0.3, 0.5):
            tick, events, summary = e6b_run_matrix(
                cfg,
                phase="calibration_threshold_margin",
                policies=policies,
                reaction_modes=modes,
                theta_fixed=theta,
                margin=margin,
                cooldown_ticks=best_cooldown,
                delay=5,
                budget_points=800,
                freshness_tau=cfg.freshness_tau,
                exogenous_route_switch=False,
            )
            tick_frames.append(tick)
            if not events.empty:
                event_frames.append(events)
            calibration_summaries.append(summary)

    calibration_summary = pd.concat(calibration_summaries, ignore_index=True)
    calibration_scores = e6b_calibration_score(calibration_summary)
    best = calibration_scores.iloc[0] if not calibration_scores.empty else pd.Series({"theta_fixed": 1.5, "margin": 0.3, "cooldown_ticks": best_cooldown})
    selected_theta = float(best["theta_fixed"])
    selected_margin = float(best["margin"])
    selected_cooldown = int(best["cooldown_ticks"])

    ghost_summaries: list[pd.DataFrame] = []
    for tau, phase in [(20.0, "ghost_stress_tau20"), (9999.0, "ghost_stress_no_decay")]:
        tick, events, summary = e6b_run_matrix(
            cfg,
            phase=phase,
            policies=policies,
            reaction_modes=modes,
            theta_fixed=selected_theta,
            margin=selected_margin,
            cooldown_ticks=selected_cooldown,
            delay=10,
            budget_points=1200,
            freshness_tau=tau,
            exogenous_route_switch=True,
        )
        tick_frames.append(tick)
        if not events.empty:
            event_frames.append(events)
        ghost_summaries.append(summary)

    tick_df = pd.concat(tick_frames, ignore_index=True)
    events_df = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    ghost_summary = pd.concat(ghost_summaries, ignore_index=True)
    calibration_summary.to_csv(out_dir / "e6b_calibration_summary.csv", index=False)
    ghost_summary.to_csv(out_dir / "e6b_ghost_stress_summary.csv", index=False)
    events_df.to_csv(out_dir / "e6b_replan_events.csv", index=False)
    tick_df.to_csv(out_dir / "e6b_tick_metrics.csv", index=False)

    baseline_osc = e6b_load_e6a_default_oscillation()
    selected_mean = calibration_summary[
        (np.isclose(calibration_summary["theta_fixed"], selected_theta))
        & (np.isclose(calibration_summary["margin"], selected_margin))
        & (calibration_summary["cooldown_ticks"] == selected_cooldown)
        & (calibration_summary["reaction_mode"].isin(["fixed_threshold", "adaptive_threshold"]))
    ]["oscillation_count"].mean()
    reduction = np.nan if baseline_osc is None or baseline_osc <= 0 else 1.0 - float(selected_mean) / baseline_osc
    selected_text = [
        "E6B Selected Parameters",
        f"theta_fixed={selected_theta}",
        f"theta_base={selected_theta}",
        f"margin={selected_margin}",
        f"cooldown_ticks={selected_cooldown}",
        f"baseline_e6a_oscillation_mean={baseline_osc}",
        f"selected_oscillation_mean={selected_mean}",
        f"oscillation_reduction_vs_e6a={reduction}",
    ]
    (out_dir / "e6b_selected_parameters.txt").write_text("\n".join(selected_text) + "\n", encoding="utf-8")
    save_e6b_plots(calibration_summary, ghost_summary, events_df, tick_df, out_dir)

    ghost_means = ghost_summary.groupby(["phase", "communication_policy", "reaction_mode"], dropna=False).mean(numeric_only=True).reset_index()
    fixed_false = ghost_means[ghost_means["reaction_mode"] == "fixed_threshold"]["false_positive_replans"].sum()
    fixed_ghost = ghost_means[ghost_means["reaction_mode"] == "fixed_threshold"]["ghost_induced_replans"].sum()
    adapt_false = ghost_means[ghost_means["reaction_mode"] == "adaptive_threshold"]["false_positive_replans"].sum()
    adapt_ghost = ghost_means[ghost_means["reaction_mode"] == "adaptive_threshold"]["ghost_induced_replans"].sum()
    key_a = ghost_means[(ghost_means["phase"] == "ghost_stress_tau20") & (ghost_means["communication_policy"] == "adaptive_ratio_rule") & (ghost_means["reaction_mode"] == "adaptive_threshold")]
    key_b = ghost_means[(ghost_means["phase"] == "ghost_stress_tau20") & (ghost_means["communication_policy"] == "random_budget") & (ghost_means["reaction_mode"] == "fixed_threshold")]
    adaptive_helps = False
    if not key_a.empty and not key_b.empty:
        adaptive_helps = bool(
            float(key_a["false_positive_replans"].iloc[0] + key_a["unnecessary_replans"].iloc[0])
            < float(key_b["false_positive_replans"].iloc[0] + key_b["unnecessary_replans"].iloc[0])
        )
    print("\nE6B Reaction Calibration and Ghost Stress Summary")
    print(f"selected theta={selected_theta}, margin={selected_margin}, cooldown={selected_cooldown}")
    print(f"oscillation reduction compared with E6A={reduction}")
    print(f"fixed_threshold false_positive_sum={fixed_false:.3f}, ghost_induced_sum={fixed_ghost:.3f}")
    print(f"adaptive_threshold false_positive_sum={adapt_false:.3f}, ghost_induced_sum={adapt_ghost:.3f}")
    print(f"false reactions activated={(fixed_false + fixed_ghost + adapt_false + adapt_ghost) > 0}")
    print(f"adaptive_ratio_rule helps under closed-loop reaction={adaptive_helps}")
    print(f"Output folder: {out_dir}")
    return tick_df, events_df, ghost_summary


@dataclass(frozen=True)
class E6CConfig(E6BConfig):
    """Frozen E6B winner used for the E6C confirmation matrix."""

    regimes: tuple[str, ...] = ("normal", "ghost_stress")
    theta_fixed: float = 2.0
    theta_base: float = 2.0
    margin: float = 0.5
    cooldown_ticks: int = 15
    risk_gap_threshold: float = 0.5
    stale_mass_threshold: float = 0.2


E6C_METRICS = [
    "total_replans", "useful_replans", "ghost_induced_replans", "false_positive_replans",
    "stale_window_replans", "stale_field_replans", "unnecessary_replans", "oscillation_count",
    "mean_risk_gap_on_replans", "average_route_cost", "realized_congestion_count",
    "old_route_ghost_auc", "post_delivery_new_route_hidden_auc", "new_route_hidden_auc",
    "route_balanced_loss_g1", "route_balanced_loss_g2", "mean_intention_age", "stale_mass_ratio",
    "communication_points_used",
]


def _e6c_run_job(job: tuple[E6CConfig, str, int, str, str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_cfg, regime, seed, policy, reaction, settings = job
    run_cfg = replace(
        base_cfg,
        seeds=(seed,),
        delay=int(settings["delay"]),
        budget_points=int(settings["budget_points"]),
        freshness_tau=float(settings["freshness_tau"]),
        exogenous_route_switch=bool(settings["exogenous_route_switch"]),
        theta_fixed=2.0,
        theta_base=2.0,
        margin=0.5,
        cooldown_ticks=15,
    )
    ticks, events = run_e6a_single(seed, policy, reaction, run_cfg)
    ticks["phase"] = regime
    ticks["theta_fixed"] = 2.0
    ticks["margin"] = 0.5
    ticks["cooldown_ticks"] = 15
    ticks["freshness_tau"] = float(settings["freshness_tau"])
    if not events.empty:
        events["phase"] = regime
        events["theta_fixed"] = 2.0
        events["margin"] = 0.5
        events["cooldown_ticks"] = 15
        events["freshness_tau"] = float(settings["freshness_tau"])
    return ticks, events


def _e6c_seed_metrics(ticks: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    out = summarize_e6a(ticks)
    out["mean_intention_age"] = ticks.groupby(
        ["seed", "communication_policy", "reaction_mode"]
    )["mean_intention_age"].mean().reindex(
        pd.MultiIndex.from_frame(out[["seed", "communication_policy", "reaction_mode"]])
    ).to_numpy()
    out["stale_mass_ratio"] = ticks.groupby(
        ["seed", "communication_policy", "reaction_mode"]
    )["stale_mass_ratio"].mean().reindex(
        pd.MultiIndex.from_frame(out[["seed", "communication_policy", "reaction_mode"]])
    ).to_numpy()
    out["new_route_hidden_auc"] = out["post_delivery_new_route_hidden_auc"]
    if not events.empty:
        gaps = events.groupby(["seed", "communication_policy", "reaction_mode"])["risk_gap"].mean()
        keys = pd.MultiIndex.from_frame(out[["seed", "communication_policy", "reaction_mode"]])
        out["mean_risk_gap_on_replans"] = gaps.reindex(keys).fillna(0.0).to_numpy()
    return out


def _e6c_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    grouped = seed_df.groupby(["regime", "communication_policy", "reaction_mode"], dropna=False)[E6C_METRICS].agg(["mean", "std"]).reset_index()
    return flatten_grouped_columns(grouped)


def _e6c_win_rates(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reaction_metrics = ["false_positive_replans", "ghost_induced_replans", "unnecessary_replans", "oscillation_count"]
    for regime, reg in seed_df.groupby("regime"):
        paired = reg.pivot_table(index=["seed", "communication_policy"], columns="reaction_mode", values=reaction_metrics)
        for metric in reaction_metrics:
            if (metric, "adaptive_threshold") not in paired or (metric, "fixed_threshold") not in paired:
                continue
            delta = paired[(metric, "adaptive_threshold")] - paired[(metric, "fixed_threshold")]
            rows.append({
                "regime": regime, "comparison": "adaptive_threshold_vs_fixed_threshold", "metric": metric,
                "n_pairs": int(delta.notna().sum()), "wins": int((delta < 0).sum()), "ties": int((delta == 0).sum()),
                "losses": int((delta > 0).sum()), "win_rate": float((delta < 0).mean()), "mean_delta": float(delta.mean()),
            })
        adaptive = reg[reg["reaction_mode"] == "adaptive_threshold"]
        policy_pair = adaptive.pivot_table(index="seed", columns="communication_policy", values=["average_route_cost", "route_balanced_loss_g1", "route_balanced_loss_g2"])
        for metric in ["average_route_cost", "route_balanced_loss_g1", "route_balanced_loss_g2"]:
            if (metric, "adaptive_ratio_rule") not in policy_pair or (metric, "random_budget") not in policy_pair:
                continue
            delta = policy_pair[(metric, "adaptive_ratio_rule")] - policy_pair[(metric, "random_budget")]
            rows.append({
                "regime": regime, "comparison": "adaptive_ratio_rule_vs_random_budget_under_adaptive_threshold", "metric": metric,
                "n_pairs": int(delta.notna().sum()), "wins": int((delta < 0).sum()), "ties": int((delta == 0).sum()),
                "losses": int((delta > 0).sum()), "win_rate": float((delta < 0).mean()), "mean_delta": float(delta.mean()),
            })
    return pd.DataFrame(rows)


def _e6c_bar_panels(data: pd.DataFrame, metrics: list[str], path: Path, title: str) -> None:
    regimes = list(data["regime"].drop_duplicates())
    fig, axes = plt.subplots(len(regimes), len(metrics), figsize=(4.5 * len(metrics), 4 * len(regimes)), squeeze=False)
    for i, regime in enumerate(regimes):
        sub = data[data["regime"] == regime]
        labels = [f"{r.communication_policy}\n{r.reaction_mode}" for r in sub.itertuples()]
        for j, metric in enumerate(metrics):
            axes[i, j].bar(range(len(sub)), sub[f"{metric}_mean"])
            axes[i, j].set_xticks(range(len(sub)))
            axes[i, j].set_xticklabels(labels, rotation=65, ha="right", fontsize=6)
            axes[i, j].set_title(f"{regime}: {metric}")
            axes[i, j].grid(axis="y", alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_e6c_plots(seed_df: pd.DataFrame, summary: pd.DataFrame, events: pd.DataFrame, out_dir: Path) -> None:
    _e6c_bar_panels(summary, ["false_positive_replans", "ghost_induced_replans", "unnecessary_replans", "oscillation_count"], out_dir / "e6c_reaction_comparison_by_regime.png", "E6C reaction comparison")
    _e6c_bar_panels(summary, ["average_route_cost", "realized_congestion_count", "useful_replans", "total_replans"], out_dir / "e6c_task_performance_by_regime.png", "E6C task performance")

    fixed_adaptive = seed_df[seed_df["reaction_mode"].isin(["fixed_threshold", "adaptive_threshold"])]
    delta_metrics = ["false_positive_replans", "ghost_induced_replans", "unnecessary_replans", "oscillation_count", "average_route_cost"]
    p = fixed_adaptive.pivot_table(index=["regime", "seed", "communication_policy"], columns="reaction_mode", values=delta_metrics)
    deltas = pd.DataFrame({m: p[(m, "adaptive_threshold")] - p[(m, "fixed_threshold")] for m in delta_metrics}).reset_index()
    dm = deltas.groupby(["regime", "communication_policy"])[delta_metrics].mean().reset_index()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9)); axes = axes.ravel()
    labels = [f"{r.regime}\n{r.communication_policy}" for r in dm.itertuples()]
    for ax, metric in zip(axes, delta_metrics):
        ax.bar(range(len(dm)), dm[metric]); ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(dm))); ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7); ax.set_title(metric)
    axes[-1].axis("off"); fig.suptitle("adaptive_threshold minus fixed_threshold (negative is better)"); fig.tight_layout()
    fig.savefig(out_dir / "e6c_adaptive_vs_fixed_delta.png", dpi=150); plt.close(fig)

    adaptive_summary = summary[summary["reaction_mode"] == "adaptive_threshold"]
    _e6c_bar_panels(adaptive_summary, ["average_route_cost", "route_balanced_loss_g1", "route_balanced_loss_g2", "false_positive_replans"], out_dir / "e6c_policy_comparison_adaptive_threshold.png", "Communication policies under adaptive_threshold")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    for ax, (regime, sub) in zip(axes.ravel(), summary.groupby("regime")):
        for row in sub.itertuples():
            ax.scatter(row.false_positive_replans_mean, row.average_route_cost_mean)
            ax.annotate(f"{row.communication_policy}\n{row.reaction_mode}", (row.false_positive_replans_mean, row.average_route_cost_mean), fontsize=6)
        ax.set_title(regime); ax.set_xlabel("false_positive_replans"); ax.set_ylabel("average_route_cost"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out_dir / "e6c_pareto_false_positive_vs_cost.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    for ax, regime in zip(axes.ravel(), seed_df["regime"].drop_duplicates()):
        has_event_data = not events.empty and {"regime", "reaction_mode", "risk_gap"}.issubset(events.columns)
        sub = events[(events["regime"] == regime) & events["reaction_mode"].isin(["fixed_threshold", "adaptive_threshold"])] if has_event_data else pd.DataFrame()
        groups = [(mode, grp["risk_gap"].to_numpy()) for mode, grp in sub.groupby("reaction_mode") if len(grp)] if "reaction_mode" in sub else []
        if groups: ax.boxplot([v for _, v in groups], labels=[k for k, _ in groups], showfliers=False)
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1); ax.set_title(regime); ax.set_ylabel("risk_gap"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out_dir / "e6c_risk_gap_boxplot.png", dpi=150); plt.close(fig)


def run_e6c_frozen_closed_loop_verification(cfg: E6CConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    allowed = {"normal", "ghost_stress"}
    unknown = set(cfg.regimes) - allowed
    if unknown:
        raise ValueError(f"Unknown E6C regimes: {sorted(unknown)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("E6C_FROZEN_CLOSED_LOOP_VERIFICATION settings")
    print(f"frozen theta_fixed={cfg.theta_fixed} theta_base={cfg.theta_base} margin={cfg.margin} cooldown_ticks={cfg.cooldown_ticks}")
    print(f"frozen alpha_age={cfg.alpha_age} alpha_stale={cfg.alpha_stale} alpha_replan={cfg.alpha_replan} alpha_persist={cfg.alpha_persist} persistence_window={cfg.risk_persistence_window}")
    tick_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    seed_frames: list[pd.DataFrame] = []
    settings = {
        "normal": dict(delay=5, budget_points=800, freshness_tau=20.0, exogenous_route_switch=False),
        "ghost_stress": dict(delay=10, budget_points=1200, freshness_tau=9999.0, exogenous_route_switch=True),
    }
    jobs = [
        (cfg, regime, seed, policy, reaction, settings[regime])
        for regime in cfg.regimes
        for seed in cfg.seeds
        for policy in cfg.communication_policies
        for reaction in cfg.reaction_modes
    ]
    completed = 0
    results_by_regime: dict[str, list[tuple[pd.DataFrame, pd.DataFrame]]] = defaultdict(list)
    with ProcessPoolExecutor(max_workers=min(4, len(jobs))) as executor:
        for job, result in zip(jobs, executor.map(_e6c_run_job, jobs)):
            completed += 1
            regime, seed, policy, reaction = job[1:5]
            print(f"[E6C {completed}/{len(jobs)}] regime={regime} seed={seed} policy={policy} reaction={reaction}", flush=True)
            results_by_regime[regime].append(result)
    for regime in cfg.regimes:
        regime_results = results_by_regime[regime]
        ticks = pd.concat([result[0] for result in regime_results], ignore_index=True)
        event_parts = [result[1] for result in regime_results if not result[1].empty]
        events = pd.concat(event_parts, ignore_index=True) if event_parts else pd.DataFrame()
        ticks["regime"] = regime
        if not events.empty:
            events["regime"] = regime
            event_frames.append(events)
        per_seed = _e6c_seed_metrics(ticks, events)
        per_seed["regime"] = regime
        tick_frames.append(ticks)
        seed_frames.append(per_seed)
    ticks_all = pd.concat(tick_frames, ignore_index=True)
    events_all = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    seed_all = pd.concat(seed_frames, ignore_index=True)
    summary = _e6c_summary(seed_all)
    wins = _e6c_win_rates(seed_all)
    seed_all.to_csv(out_dir / "e6c_metrics_by_seed.csv", index=False)
    summary.to_csv(out_dir / "e6c_summary_by_regime_policy_reaction.csv", index=False)
    wins.to_csv(out_dir / "e6c_win_rate_summary.csv", index=False)
    events_all.to_csv(out_dir / "e6c_event_metrics.csv", index=False)
    save_e6c_plots(seed_all, summary, events_all, out_dir)

    print("\nSummary table by regime")
    print(summary[["regime", "communication_policy", "reaction_mode", "false_positive_replans_mean", "ghost_induced_replans_mean", "unnecessary_replans_mean", "oscillation_count_mean", "useful_replans_mean", "average_route_cost_mean"]].to_string(index=False))
    print("\nAdaptive vs fixed threshold win rates")
    print(wins[wins["comparison"] == "adaptive_threshold_vs_fixed_threshold"].to_string(index=False))
    print("\nAdaptive ratio rule vs random budget under adaptive_threshold")
    print(wins[wins["comparison"].str.startswith("adaptive_ratio_rule")].to_string(index=False))

    ghost = wins[(wins["regime"] == "ghost_stress") & (wins["comparison"] == "adaptive_threshold_vs_fixed_threshold")]
    reaction_ok = bool((ghost[ghost["metric"].isin(["false_positive_replans", "ghost_induced_replans"])]["win_rate"] > 0.5).any())
    stability_ok = bool((ghost[ghost["metric"].isin(["unnecessary_replans", "oscillation_count"])]["mean_delta"] < 0).all())
    policy = wins[wins["comparison"].str.startswith("adaptive_ratio_rule")]
    policy_cost = seed_all[seed_all["reaction_mode"] == "adaptive_threshold"].groupby(
        ["regime", "communication_policy"]
    )["average_route_cost"].mean().unstack()
    # E6C defines "competitive" as average route cost within 5% of random_budget
    # in both regimes; superiority still requires a negative paired delta.
    policy_ok = bool(
        {"adaptive_ratio_rule", "random_budget"}.issubset(policy_cost.columns)
        and (policy_cost["adaptive_ratio_rule"] <= 1.05 * policy_cost["random_budget"]).all()
    )
    supports = reaction_ok and stability_ok and policy_ok
    print(f"\nE6C supports a closed-loop performance claim: {supports}")
    if supports and bool((policy[policy["metric"] == "average_route_cost"]["mean_delta"] < 0).all()):
        print("Budget-adaptive sharing and freshness-aware reaction jointly improve closed-loop coordination.")
    elif reaction_ok:
        print("The main closed-loop benefit comes from freshness-aware reaction stabilization, while budget-adaptive sharing remains important for field-level hidden-ghost control.")
    print(f"Output folder: {out_dir}")
    return seed_all, events_all, summary


@dataclass(frozen=True)
class E6DConfig(E6CConfig):
    reaction_modes: tuple[str, ...] = ("adaptive_threshold",)
    collect_agent_diagnostics: bool = True


def _e6d_run_job(job: tuple[E6DConfig, str, int, str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg, regime, seed, policy, setting = job
    run_cfg = replace(
        cfg,
        seeds=(seed,), delay=int(setting["delay"]), budget_points=int(setting["budget_points"]),
        freshness_tau=float(setting["freshness_tau"]),
        exogenous_route_switch=bool(setting["exogenous_route_switch"]),
        theta_fixed=2.0, theta_base=2.0, margin=0.5, cooldown_ticks=15,
        collect_agent_diagnostics=True,
    )
    ticks, events = run_e6a_single(seed, policy, "adaptive_threshold", run_cfg)
    agents = ticks.attrs.pop("agent_diagnostics", pd.DataFrame())
    for frame in (ticks, events, agents):
        if not frame.empty:
            frame["regime"] = regime
            frame["delay"] = int(setting["delay"])
            frame["budget_points"] = int(setting["budget_points"])
            frame["freshness_tau"] = float(setting["freshness_tau"])
    return ticks, events, agents


def _e6d_seed_diagnostics(ticks: pd.DataFrame, events: pd.DataFrame, agents: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["regime", "seed", "communication_policy"]
    for key, agent_sub in agents.groupby(keys, sort=True):
        regime, seed, policy = key
        tick_sub = ticks[(ticks["regime"] == regime) & (ticks["seed"] == seed) & (ticks["communication_policy"] == policy)]
        event_sub = events[(events["regime"] == regime) & (events["seed"] == seed) & (events["communication_policy"] == policy)] if not events.empty else pd.DataFrame()
        task_sub = summarize_e6a(tick_sub)
        event_count = len(event_sub)
        rows.append({
            "regime": regime,
            "seed": seed,
            "communication_policy": policy,
            "reaction_mode": "adaptive_threshold",
            "mean_current_route_risk_comm": float(agent_sub["current_route_risk_comm"].mean()),
            "mean_current_route_risk_full": float(agent_sub["current_route_risk_full"].mean()),
            "mean_risk_gap_current": float(agent_sub["risk_gap_current"].mean()),
            "mean_abs_risk_gap_current": float(agent_sub["risk_gap_current"].abs().mean()),
            "mean_overreaction_score": float(event_sub["overreaction_score"].mean()) if event_count else 0.0,
            "fraction_ticks_current_risk_above_threshold": float((agent_sub["current_route_risk_comm"] > agent_sub["threshold"]).mean()),
            "fraction_replans_that_reduce_full_risk": float((event_sub["full_risk_reduction"] > 0).mean()) if event_count else 0.0,
            "average_full_risk_reduction_per_replan": float(event_sub["full_risk_reduction"].mean()) if event_count else 0.0,
            "average_comm_risk_reduction_per_replan": float(event_sub["comm_risk_reduction"].mean()) if event_count else 0.0,
            "average_route_cost": float(task_sub["average_route_cost"].iloc[0]),
            "realized_congestion_count": float(task_sub["realized_congestion_count"].iloc[0]),
            "false_positive_replans": float(task_sub["false_positive_replans"].iloc[0]),
            "unnecessary_replans": float(task_sub["unnecessary_replans"].iloc[0]),
            "oscillation_count": float(task_sub["oscillation_count"].iloc[0]),
            "route_balanced_loss_g1": float(task_sub["route_balanced_loss_g1"].iloc[0]),
            "route_balanced_loss_g2": float(task_sub["route_balanced_loss_g2"].iloc[0]),
            "local_route_risk_coverage_current": float(agent_sub["local_route_risk_coverage_current"].mean()),
            "local_route_risk_coverage_alternative": float(agent_sub["local_route_risk_coverage_alternative"].mean()),
            "stale_mass_ratio_near_current_route": float(agent_sub["stale_mass_ratio_near_current_route"].mean()),
            "stale_mass_ratio_near_alternative_route": float(agent_sub["stale_mass_ratio_near_alternative_route"].mean()),
            **_decision_utility_metrics(agent_sub),
        })
    return pd.DataFrame(rows)


def _decision_utility_metrics(agent_sub: pd.DataFrame) -> dict[str, float]:
    """Compute per-seed route decision agreement and regret from agent diagnostics."""
    if agent_sub.empty:
        return {
            "mean_route_decision_agreement": float("nan"),
            "fraction_route_decision_disagreement": float("nan"),
            "mean_route_decision_regret": float("nan"),
            "p90_route_decision_regret": float("nan"),
            "fraction_positive_decision_regret": float("nan"),
        }
    comm_adv = agent_sub["risk_advantage_comm"].to_numpy(dtype=float)
    full_adv = agent_sub["risk_advantage_full"].to_numpy(dtype=float)
    agreement = (np.sign(comm_adv) == np.sign(full_adv)).astype(float)
    cur_comm = agent_sub["current_route_risk_comm"].to_numpy(dtype=float)
    alt_comm = agent_sub["alternative_route_risk_comm"].to_numpy(dtype=float)
    cur_full = agent_sub["current_route_risk_full"].to_numpy(dtype=float)
    alt_full = agent_sub["alternative_route_risk_full"].to_numpy(dtype=float)
    comm_selects_alt = alt_comm < cur_comm
    comm_selected_full = np.where(comm_selects_alt, alt_full, cur_full)
    full_best = np.minimum(cur_full, alt_full)
    regret = comm_selected_full - full_best
    mean_agr = float(agreement.mean())
    return {
        "mean_route_decision_agreement": mean_agr,
        "fraction_route_decision_disagreement": 1.0 - mean_agr,
        "mean_route_decision_regret": float(regret.mean()),
        "p90_route_decision_regret": float(np.percentile(regret, 90)),
        "fraction_positive_decision_regret": float((regret > 0).mean()),
    }


def _e6d_correlations(seed_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "false_positive_replans", "unnecessary_replans", "oscillation_count",
        "average_full_risk_reduction_per_replan", "mean_abs_risk_gap_current",
        "route_balanced_loss_g1", "route_balanced_loss_g2",
    ]
    rows: list[dict[str, Any]] = []
    groups = [(str(regime), sub) for regime, sub in seed_summary.groupby("regime")]
    groups.append(("all", seed_summary))
    for regime, sub in groups:
        for target in ["average_route_cost", "realized_congestion_count"]:
            for metric in metrics:
                valid = sub[[target, metric]].dropna()
                corr = float(valid[target].corr(valid[metric])) if len(valid) > 1 and valid[target].std() > 0 and valid[metric].std() > 0 else np.nan
                rows.append({"regime": regime, "target": target, "metric": metric, "pearson_r": corr, "abs_pearson_r": abs(corr), "n_observations": len(valid)})
    return pd.DataFrame(rows)


def save_e6d_plots(agents: pd.DataFrame, events: pd.DataFrame, policy_summary: pd.DataFrame, correlations: pd.DataFrame, out_dir: Path) -> None:
    policies = ["random_budget", "adaptive_ratio_rule", "fixed_hybrid_25_75", "periodic_full_reference"]
    regimes = [r for r in ["normal", "ghost_stress"] if r in set(agents["regime"])]
    fig, axes = plt.subplots(len(regimes), len(policies), figsize=(5 * len(policies), 4.5 * len(regimes)), squeeze=False)
    for i, regime in enumerate(regimes):
        for j, policy in enumerate(policies):
            sub = agents[(agents["regime"] == regime) & (agents["communication_policy"] == policy)]
            if len(sub) > 5000:
                sub = sub.sample(5000, random_state=0)
            axes[i, j].scatter(sub["current_route_risk_full"], sub["current_route_risk_comm"], s=3, alpha=0.15)
            limit = float(max(sub["current_route_risk_full"].max(), sub["current_route_risk_comm"].max())) if len(sub) else 1.0
            axes[i, j].plot([0, limit], [0, limit], "k--", linewidth=0.8)
            axes[i, j].set_title(f"{regime}: {policy}"); axes[i, j].set_xlabel("full risk"); axes[i, j].set_ylabel("comm risk")
    fig.tight_layout(); fig.savefig(out_dir / "e6d_comm_vs_full_risk_scatter.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, len(regimes), figsize=(7 * len(regimes), 5), squeeze=False)
    for ax, regime in zip(axes.ravel(), regimes):
        data = [agents[(agents["regime"] == regime) & (agents["communication_policy"] == p)]["risk_gap_current"].to_numpy() for p in policies]
        ax.boxplot(data, labels=policies, showfliers=False); ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(regime); ax.tick_params(axis="x", rotation=35); ax.set_ylabel("risk_gap_current")
    fig.tight_layout(); fig.savefig(out_dir / "e6d_risk_gap_by_policy.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, len(regimes), figsize=(7 * len(regimes), 5), squeeze=False)
    for ax, regime in zip(axes.ravel(), regimes):
        has_events = not events.empty and {"regime", "communication_policy", "overreaction_score"}.issubset(events.columns)
        data = [events[(events["regime"] == regime) & (events["communication_policy"] == p)]["overreaction_score"].to_numpy() for p in policies] if has_events else [np.asarray([np.nan])] * len(policies)
        ax.boxplot(data, labels=policies, showfliers=False); ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(regime); ax.tick_params(axis="x", rotation=35); ax.set_ylabel("overreaction_score")
    fig.tight_layout(); fig.savefig(out_dir / "e6d_overreaction_score_by_policy.png", dpi=150); plt.close(fig)

    quality = ["fraction_replans_that_reduce_full_risk", "average_full_risk_reduction_per_replan", "unnecessary_replans", "false_positive_replans"]
    quality_summary = policy_summary.rename(columns={m: f"{m}_mean" for m in quality}).copy()
    quality_summary["reaction_mode"] = "adaptive_threshold"
    _e6c_bar_panels(quality_summary, quality, out_dir / "e6d_replan_quality_by_policy.png", "E6D replan quality")

    cost_corr = correlations[(correlations["target"] == "average_route_cost") & (correlations["regime"] != "all")]
    fig, axes = plt.subplots(1, len(regimes), figsize=(8 * len(regimes), 5), squeeze=False)
    for ax, regime in zip(axes.ravel(), regimes):
        sub = cost_corr[cost_corr["regime"] == regime]
        ax.bar(sub["metric"], sub["pearson_r"]); ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(regime); ax.set_ylabel("Pearson r with average_route_cost"); ax.tick_params(axis="x", rotation=55)
    fig.tight_layout(); fig.savefig(out_dir / "e6d_task_cost_explainer.png", dpi=150); plt.close(fig)

    local_metrics = ["local_route_risk_coverage_current", "local_route_risk_coverage_alternative", "stale_mass_ratio_near_current_route", "stale_mass_ratio_near_alternative_route"]
    compare = policy_summary[policy_summary["communication_policy"].isin(["random_budget", "adaptive_ratio_rule"])]
    fig, axes = plt.subplots(len(regimes), len(local_metrics), figsize=(5 * len(local_metrics), 4 * len(regimes)), squeeze=False)
    for i, regime in enumerate(regimes):
        sub = compare[compare["regime"] == regime].set_index("communication_policy")
        for j, metric in enumerate(local_metrics):
            axes[i, j].bar(["random_budget", "adaptive_ratio_rule"], sub.reindex(["random_budget", "adaptive_ratio_rule"])[metric])
            axes[i, j].set_title(f"{regime}: {metric}"); axes[i, j].tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(out_dir / "e6d_random_vs_adaptive_ratio_local_coverage.png", dpi=150); plt.close(fig)


def run_e6d_closed_loop_diagnostic(cfg: E6DConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "normal": dict(delay=5, budget_points=800, freshness_tau=20.0, exogenous_route_switch=False),
        "ghost_stress": dict(delay=10, budget_points=1200, freshness_tau=9999.0, exogenous_route_switch=True),
    }
    unknown = set(cfg.regimes) - set(settings)
    if unknown:
        raise ValueError(f"Unknown E6D regimes: {sorted(unknown)}")
    print("E6D_CLOSED_LOOP_DIAGNOSTIC settings")
    print("reaction_mode=adaptive_threshold; frozen theta=2.0 margin=0.5 cooldown=15")
    jobs = [(cfg, regime, seed, policy, settings[regime]) for regime in cfg.regimes for seed in cfg.seeds for policy in cfg.communication_policies]
    tick_parts: list[pd.DataFrame] = []
    event_parts: list[pd.DataFrame] = []
    agent_parts: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=min(4, len(jobs))) as executor:
        for idx, (job, result) in enumerate(zip(jobs, executor.map(_e6d_run_job, jobs)), start=1):
            ticks, events, agents = result
            tick_parts.append(ticks); agent_parts.append(agents)
            if not events.empty: event_parts.append(events)
            print(f"[E6D {idx}/{len(jobs)}] regime={job[1]} seed={job[2]} policy={job[3]}", flush=True)
    ticks_all = pd.concat(tick_parts, ignore_index=True)
    events_all = pd.concat(event_parts, ignore_index=True) if event_parts else pd.DataFrame()
    agents_all = pd.concat(agent_parts, ignore_index=True)
    seed_summary = _e6d_seed_diagnostics(ticks_all, events_all, agents_all)
    diagnostic_cols = [c for c in seed_summary.columns if c not in {"seed", "reaction_mode"}]
    policy_summary = seed_summary.groupby(["regime", "communication_policy"], dropna=False)[[c for c in diagnostic_cols if c not in {"regime", "communication_policy"}]].mean().reset_index()
    correlations = _e6d_correlations(seed_summary)
    agents_all.to_csv(out_dir / "e6d_agent_tick_diagnostics.csv", index=False)
    events_all.to_csv(out_dir / "e6d_replan_event_diagnostics.csv", index=False)
    policy_summary.to_csv(out_dir / "e6d_policy_diagnostics_summary.csv", index=False)
    correlations.to_csv(out_dir / "e6d_metric_correlation_summary.csv", index=False)
    save_e6d_plots(agents_all, events_all, policy_summary, correlations, out_dir)

    print("\nE6D policy diagnostics")
    print(policy_summary[["regime", "communication_policy", "mean_current_route_risk_comm", "mean_current_route_risk_full", "mean_abs_risk_gap_current", "mean_overreaction_score", "fraction_replans_that_reduce_full_risk", "average_route_cost", "unnecessary_replans", "false_positive_replans", "oscillation_count", "route_balanced_loss_g1", "route_balanced_loss_g2"]].to_string(index=False))
    paired = policy_summary.set_index(["regime", "communication_policy"])
    adaptive_overreacts = bool(np.mean([paired.loc[(r, "adaptive_ratio_rule"), "mean_overreaction_score"] - paired.loc[(r, "random_budget"), "mean_overreaction_score"] for r in cfg.regimes]) > 0)
    random_lower_gap = bool(all(paired.loc[(r, "random_budget"), "mean_abs_risk_gap_current"] < paired.loc[(r, "adaptive_ratio_rule"), "mean_abs_risk_gap_current"] for r in cfg.regimes))
    field_corr = correlations[(correlations["target"] == "average_route_cost") & correlations["metric"].isin(["route_balanced_loss_g1", "route_balanced_loss_g2"])]
    field_predicts = bool(field_corr["abs_pearson_r"].mean() >= 0.5)
    fixed_best = {r: paired.loc[r]["average_route_cost"].idxmin() == "fixed_hybrid_25_75" for r in cfg.regimes}
    print("\nDiagnostic interpretation")
    print(f"adaptive_ratio_rule causes higher mean overreaction_score: {adaptive_overreacts}")
    print(f"random_budget has lower mean_abs_risk_gap in both regimes: {random_lower_gap}")
    print(f"route balanced loss strongly predicts task cost (mean |r| >= 0.5): {field_predicts}")
    print(f"fixed_hybrid_25_75 has best task cost by regime: {fixed_best}")
    if adaptive_overreacts:
        print("adaptive_ratio_rule may overemphasize critical regions, causing stronger perceived route risk and more overreaction in closed-loop decisions.")
    if random_lower_gap:
        print("random_budget may provide a better local route-risk estimator for simple dual-corridor closed-loop decisions.")
    if not field_predicts:
        print("field-level fidelity metrics do not fully predict closed-loop task performance.")
    if any(fixed_best.values()):
        print("a fixed hybrid may provide a better closed-loop compromise than the budget-adaptive field-level rule in this scenario.")
    print(f"Output folder: {out_dir}")
    return agents_all, events_all, policy_summary
