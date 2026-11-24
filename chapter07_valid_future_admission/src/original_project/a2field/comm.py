from __future__ import annotations

from collections import Counter

import numpy as np

from .grid import dilate_mask
from .tubes import TubePoint, with_freshness


def fixed_horizon(points: list[TubePoint], h_upload: int) -> list[TubePoint]:
    return [p for p in points if p.k < h_upload]


def random_budget(points: list[TubePoint], budget_points: int, rng: np.random.Generator) -> list[TubePoint]:
    if budget_points >= len(points):
        return list(points)
    idx = rng.choice(len(points), size=max(0, budget_points), replace=False)
    return [points[int(i)] for i in idx]


def freshness_based(points: list[TubePoint], budget_points: int, horizon: int) -> list[TubePoint]:
    """Select reliable near-future points under the same point budget."""

    if budget_points >= len(points):
        return list(points)
    if budget_points <= 0:
        return []
    scored = [
        (p.q * np.exp(-p.k / max(1, horizon)), -i, p)
        for i, p in enumerate(points)
    ]
    scored.sort(reverse=True)
    return [p for _, _, p in scored[:budget_points]]


def fixed_budget_horizon(points: list[TubePoint], budget_points: int) -> list[TubePoint]:
    """Budget-matched near-horizon upload sorted by future step then agent id."""

    if budget_points >= len(points):
        return list(points)
    if budget_points <= 0:
        return []
    ordered = sorted(points, key=lambda p: (p.k, p.agent_id, p.y, p.x))
    return ordered[:budget_points]


def partial_access_budget(
    points: list[TubePoint],
    budget_points: int,
    horizon: int,
    rng: np.random.Generator,
) -> list[TubePoint]:
    """Budget-matched partial access: complete agents plus one partial agent."""

    if budget_points >= len(points):
        return list(points)
    if budget_points <= 0:
        return []
    by_agent: dict[int, list[TubePoint]] = {}
    for p in points:
        by_agent.setdefault(p.agent_id, []).append(p)
    agent_ids = np.array(sorted(by_agent), dtype=int)
    rng.shuffle(agent_ids)
    full_agents = min(len(agent_ids), budget_points // max(1, horizon))
    remainder = budget_points - full_agents * horizon
    chosen: list[TubePoint] = []
    for agent_id in agent_ids[:full_agents]:
        chosen.extend(sorted(by_agent[int(agent_id)], key=lambda p: p.k)[:horizon])
    if remainder > 0 and full_agents < len(agent_ids):
        agent_id = int(agent_ids[full_agents])
        chosen.extend(sorted(by_agent[agent_id], key=lambda p: p.k)[:remainder])
    return chosen[:budget_points]


def freshness_decay(points: list[TubePoint], delay: float, freshness_tau: float = 5.0) -> list[TubePoint]:
    return with_freshness(points, float(np.exp(-delay / freshness_tau)))


def partial_access(points: list[TubePoint], fraction: float, rng: np.random.Generator) -> list[TubePoint]:
    agent_ids = sorted({p.agent_id for p in points})
    if not agent_ids:
        return []
    n_upload = int(np.ceil(np.clip(fraction, 0.0, 1.0) * len(agent_ids)))
    chosen = set(rng.choice(agent_ids, size=n_upload, replace=False).tolist())
    return [p for p in points if p.agent_id in chosen]


def wave_critical(
    points: list[TubePoint],
    budget_points: int,
    critical_mask: np.ndarray,
    horizon: int,
    critical_radius: int = 3,
    weights: dict[str, float] | None = None,
) -> list[TubePoint]:
    """Oracle-inspired criticality ranking over all candidate tube points."""

    if budget_points >= len(points):
        return list(points)
    if budget_points <= 0 or not points:
        return []

    dilated = dilate_mask(critical_mask, critical_radius)
    bins = Counter((p.k, p.y // 2, p.x // 2) for p in points)
    max_overlap = max(bins.values()) if bins else 1
    w = {
        "local_overlap": 1.0,
        "bottleneck": 1.0,
        "temporal": 0.5,
        "freshness": 0.5,
        "width_penalty": 0.2,
    }
    if weights:
        w.update(weights)

    scored: list[tuple[float, int, TubePoint]] = []
    for i, p in enumerate(points):
        local_overlap = bins[(p.k, p.y // 2, p.x // 2)] / max_overlap
        critical = 1.0 if 0 <= p.y < dilated.shape[0] and 0 <= p.x < dilated.shape[1] and dilated[p.y, p.x] else 0.0
        temporal = float(np.exp(-p.k / max(1, horizon)))
        excessive_width = max(0.0, p.tube_width - 2.0)
        score = (
            w["local_overlap"] * local_overlap
            + w["bottleneck"] * critical
            + w["temporal"] * temporal
            + w["freshness"] * p.q
            - w["width_penalty"] * excessive_width
        )
        scored.append((score, -i, p))
    scored.sort(reverse=True)
    return [p for _, _, p in scored[:budget_points]]


def _criticality_scores(
    points: list[TubePoint],
    critical_mask: np.ndarray,
    horizon: int,
    critical_radius: int = 3,
    weights: dict[str, float] | None = None,
) -> list[tuple[float, int, TubePoint]]:
    dilated = dilate_mask(critical_mask, critical_radius)
    bins = Counter((p.k, p.y // 2, p.x // 2) for p in points)
    max_overlap = max(bins.values()) if bins else 1
    w = {
        "overlap": 1.0,
        "bottleneck": 1.0,
        "temporal": 0.5,
        "freshness": 0.5,
        "width": 0.2,
    }
    if weights:
        w.update(weights)

    scored: list[tuple[float, int, TubePoint]] = []
    for i, p in enumerate(points):
        local_overlap = bins[(p.k, p.y // 2, p.x // 2)] / max_overlap
        bottleneck = 1.0 if 0 <= p.y < dilated.shape[0] and 0 <= p.x < dilated.shape[1] and dilated[p.y, p.x] else 0.0
        temporal = float(np.exp(-p.k / max(1, horizon)))
        width_penalty = max(0.0, p.tube_width - 2.0)
        score = (
            w["overlap"] * local_overlap
            + w["bottleneck"] * bottleneck
            + w["temporal"] * temporal
            + w["freshness"] * p.q
            - w["width"] * width_penalty
        )
        scored.append((score, i, p))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return scored


def _region_id(p: TubePoint, critical_mask: np.ndarray) -> str:
    h = critical_mask.shape[0]
    if 0 <= p.y < critical_mask.shape[0] and 0 <= p.x < critical_mask.shape[1] and critical_mask[p.y, p.x]:
        return "bottleneck"
    if p.y < h / 3:
        return "upper"
    if p.y > 2 * h / 3:
        return "lower"
    return "middle"


def _nms_select(
    candidates: list[tuple[float, int, TubePoint]],
    quota: int,
    selected_ids: set[tuple[int, int]],
    agent_counts: dict[int, int],
    max_per_agent: int,
    nms_space_radius: float,
    nms_time_radius: int,
) -> list[TubePoint]:
    out: list[TubePoint] = []
    for _, original_idx, point in candidates:
        key = (point.agent_id, point.k)
        if key in selected_ids:
            continue
        if agent_counts.get(point.agent_id, 0) >= max_per_agent:
            continue
        too_close = False
        for chosen in out:
            if abs(chosen.k - point.k) <= nms_time_radius and float(np.hypot(chosen.y - point.y, chosen.x - point.x)) <= nms_space_radius:
                too_close = True
                break
        if too_close:
            continue
        selected_ids.add(key)
        agent_counts[point.agent_id] = agent_counts.get(point.agent_id, 0) + 1
        out.append(point)
        if len(out) >= quota:
            break
    return out


def wave_critical_diverse(
    points: list[TubePoint],
    budget_points: int,
    critical_mask: np.ndarray,
    horizon: int,
    rng: np.random.Generator | None = None,
    bin_count: int = 5,
    nms_space_radius: float = 4.0,
    nms_time_radius: int = 3,
    diversity_lambda: float = 0.5,
    coverage_lambda: float = 0.2,
) -> list[TubePoint]:
    """Soft diversity greedy selection over criticality-ranked tube points."""

    if budget_points >= len(points):
        return list(points)
    if budget_points <= 0 or not points:
        return []

    scored = _criticality_scores(points, critical_mask, horizon)
    bin_count = max(1, min(bin_count, horizon))
    selected: list[TubePoint] = []
    selected_ids: set[tuple[int, int]] = set()
    remaining: dict[int, tuple[float, TubePoint]] = {idx: (score, point) for score, idx, point in scored}
    redundancy_counts: dict[int, float] = {idx: 0.0 for idx in remaining}
    bin_count = max(1, min(bin_count, horizon))
    region_names = ["bottleneck", "upper", "middle", "lower"]
    bin_counts = np.zeros(bin_count, dtype=float)
    region_counts = {name: 0.0 for name in region_names}

    def time_bin(point: TubePoint) -> int:
        return min(bin_count - 1, int(point.k * bin_count / max(1, horizon)))

    while remaining and len(selected) < budget_points:
        min_bin = float(np.min(bin_counts)) if len(selected) else 0.0
        min_region = min(region_counts.values()) if len(selected) else 0.0
        best_key: int | None = None
        best_score = -np.inf
        for key, (base_score, point) in remaining.items():
            nearby_count = redundancy_counts.get(key, 0.0)
            redundancy = nearby_count / (1.0 + nearby_count)
            b = time_bin(point)
            region = _region_id(point, critical_mask)
            coverage_bonus = (min_bin + 1.0) / (bin_counts[b] + 1.0)
            coverage_bonus += (min_region + 1.0) / (region_counts.get(region, 0.0) + 1.0)
            adjusted_score = base_score - diversity_lambda * redundancy + coverage_lambda * coverage_bonus
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_key = key
        if best_key is None:
            break
        _, point = remaining.pop(best_key)
        redundancy_counts.pop(best_key, None)
        selected_ids.add((point.agent_id, point.k))
        selected.append(point)
        bin_counts[time_bin(point)] += 1.0
        region_counts[_region_id(point, critical_mask)] = region_counts.get(_region_id(point, critical_mask), 0.0) + 1.0
        for key, (_, candidate) in remaining.items():
            if abs(candidate.k - point.k) <= nms_time_radius and float(np.hypot(candidate.y - point.y, candidate.x - point.x)) <= nms_space_radius:
                redundancy_counts[key] = redundancy_counts.get(key, 0.0) + 1.0
    return selected[:budget_points]


def apply_policy(
    policy: str,
    points: list[TubePoint],
    rng: np.random.Generator,
    budget_points: int,
    horizon: int,
    critical_mask: np.ndarray,
    bin_count: int = 5,
    nms_space_radius: float = 4.0,
    nms_time_radius: int = 3,
    diversity_lambda: float = 0.5,
    coverage_lambda: float = 0.2,
) -> list[TubePoint]:
    if policy == "full":
        return list(points)
    if policy.startswith("fixed_horizon_"):
        return fixed_horizon(points, int(policy.rsplit("_", 1)[1]))
    if policy == "random_budget":
        return random_budget(points, budget_points, rng)
    if policy == "freshness_based":
        return freshness_based(points, budget_points, horizon)
    if policy.startswith("freshness_delay_"):
        return freshness_decay(points, float(policy.rsplit("_", 1)[1]))
    if policy.startswith("partial_access_"):
        return partial_access(points, float(policy.rsplit("_", 1)[1]), rng)
    if policy == "wave_critical":
        return wave_critical(points, budget_points, critical_mask, horizon)
    if policy == "wave_critical_diverse":
        return wave_critical_diverse(
            points,
            budget_points,
            critical_mask,
            horizon,
            rng=rng,
            bin_count=bin_count,
            nms_space_radius=nms_space_radius,
            nms_time_radius=nms_time_radius,
            diversity_lambda=diversity_lambda,
            coverage_lambda=coverage_lambda,
        )
    raise ValueError(f"Unknown communication policy {policy!r}")
