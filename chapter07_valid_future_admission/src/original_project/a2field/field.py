from __future__ import annotations

import numpy as np

from .grid import Grid
from .messages import MotionTubePoint
from .tubes import TubePoint


def build_field(
    tube_points: list[TubePoint],
    grid: Grid,
    horizon: int,
    beta: float = 0.03,
) -> np.ndarray:
    """Aggregate tube points into R[t, y, x] via local Gaussian kernels."""

    field = np.zeros((horizon, grid.height, grid.width), dtype=np.float64)
    if not tube_points:
        return field

    for p in tube_points:
        if p.k < 0 or p.k >= horizon:
            continue
        r = max(0.2, float(p.tube_width))
        radius = int(np.ceil(3.0 * r))
        y0, y1 = max(0, p.y - radius), min(grid.height, p.y + radius + 1)
        x0, x1 = max(0, p.x - radius), min(grid.width, p.x + radius + 1)
        ys = np.arange(y0, y1)[:, None]
        xs = np.arange(x0, x1)[None, :]
        d2 = (ys - p.y) ** 2 + (xs - p.x) ** 2
        kernel = np.exp(-0.5 * d2 / (r * r))
        free = ~grid.obstacles[y0:y1, x0:x1]
        weight = p.q * p.base_weight
        field[p.k, y0:y1, x0:x1] += weight * kernel * free
    return field


def project_sum(field: np.ndarray) -> np.ndarray:
    return np.sum(field, axis=0) if field.size else np.zeros((0, 0))


def project_max(field: np.ndarray) -> np.ndarray:
    return np.max(field, axis=0) if field.size else np.zeros((0, 0))


def build_rolling_field(
    tube_points: list[MotionTubePoint],
    grid: Grid,
    current_time: int,
    horizon: int,
) -> np.ndarray:
    """Aggregate absolute-time intention points into R[t_rel-1, y, x]."""

    field = np.zeros((horizon, grid.height, grid.width), dtype=np.float64)
    if not tube_points:
        return field

    for p in tube_points:
        t_rel = int(p.future_time_abs - current_time)
        if t_rel <= 0 or t_rel > horizon:
            continue
        r = max(0.2, float(p.radius))
        radius = int(np.ceil(3.0 * r))
        y0, y1 = max(0, p.y - radius), min(grid.height, p.y + radius + 1)
        x0, x1 = max(0, p.x - radius), min(grid.width, p.x + radius + 1)
        ys = np.arange(y0, y1)[:, None]
        xs = np.arange(x0, x1)[None, :]
        d2 = (ys - p.y) ** 2 + (xs - p.x) ** 2
        kernel = np.exp(-0.5 * d2 / (r * r))
        free = ~grid.obstacles[y0:y1, x0:x1]
        field[t_rel - 1, y0:y1, x0:x1] += p.freshness * p.weight * kernel * free
    return field
