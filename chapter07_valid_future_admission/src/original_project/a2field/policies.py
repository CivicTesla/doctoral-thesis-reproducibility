from __future__ import annotations

from collections import Counter

import numpy as np

from .grid import dilate_mask
from .messages import MotionTubePoint


def select_tube_points(
    policy_name: str,
    points: list[MotionTubePoint],
    rng: np.random.Generator,
    budget_points: int,
    horizon: int,
    critical_mask: np.ndarray,
    fixed_horizon: int = 10,
) -> list[MotionTubePoint]:
    budget = min(max(0, int(budget_points)), len(points))
    if policy_name in {"periodic_full", "full_upload"}:
        return list(points)
    if not points or budget <= 0:
        return []
    if policy_name == "random_budget":
        if budget >= len(points):
            return list(points)
        idx = rng.choice(len(points), size=budget, replace=False)
        return [points[int(i)] for i in idx]
    if policy_name == "freshness_based":
        scored = [(p.freshness * np.exp(-p.k / max(1, horizon)), -i, p) for i, p in enumerate(points)]
        scored.sort(reverse=True)
        return [p for _, _, p in scored[:budget]]
    if policy_name == "fixed_horizon_upload":
        ordered = sorted(points, key=lambda p: (p.k, p.agent_id, p.future_time_abs, p.y, p.x))
        short = [p for p in ordered if p.k <= fixed_horizon]
        if len(short) >= budget:
            return short[:budget]
        return ordered[:budget]
    if policy_name == "wave_critical":
        return wave_critical_upload(points, budget, horizon, critical_mask)
    raise ValueError(f"Unknown sharing policy {policy_name!r}")


def wave_critical_upload(
    points: list[MotionTubePoint],
    budget_points: int,
    horizon: int,
    critical_mask: np.ndarray,
    critical_radius: int = 3,
) -> list[MotionTubePoint]:
    if budget_points >= len(points):
        return list(points)
    if budget_points <= 0 or not points:
        return []

    dilated = dilate_mask(critical_mask, critical_radius)
    bins = Counter((p.future_time_abs, p.y // 2, p.x // 2) for p in points)
    max_overlap = max(bins.values()) if bins else 1
    scored: list[tuple[float, int, MotionTubePoint]] = []
    for i, p in enumerate(points):
        local_overlap = bins[(p.future_time_abs, p.y // 2, p.x // 2)] / max_overlap
        critical = 1.0 if 0 <= p.y < dilated.shape[0] and 0 <= p.x < dilated.shape[1] and dilated[p.y, p.x] else 0.0
        temporal = float(np.exp(-p.k / max(1, horizon)))
        width_penalty = max(0.0, p.radius - 2.0)
        score = (
            1.0 * local_overlap
            + 1.0 * critical
            + 0.5 * temporal
            + 0.5 * p.freshness
            - 0.2 * width_penalty
        )
        scored.append((score, -i, p))
    scored.sort(reverse=True)
    return [p for _, _, p in scored[:budget_points]]
