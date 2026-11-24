from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .agents import Agent
from .grid import Grid


@dataclass(frozen=True)
class TubePoint:
    agent_id: int
    k: int
    y: int
    x: int
    dy: int
    dx: int
    tube_width: float
    q: float
    base_weight: float


def _direction(path: list[tuple[int, int]], idx: int) -> tuple[int, int]:
    if not path:
        return (0, 0)
    cur = path[min(idx, len(path) - 1)]
    nxt = path[min(idx + 1, len(path) - 1)]
    return (nxt[0] - cur[0], nxt[1] - cur[1])


def _open_cell_width(grid: Grid, y: int, x: int, default: float) -> float:
    free_neighbors = sum(1 for ny, nx in grid.neighbors4(y, x) if grid.is_free(ny, nx))
    return default + 0.15 * max(0, free_neighbors - 2)


def build_tubes(
    agents: list[Agent],
    grid: Grid,
    horizon: int = 25,
    beta: float = 0.03,
    tube_width: float = 1.5,
    variable_width: bool = False,
    path_offset: int = 0,
    width_growth: float = 0.0,
    freshness_factor: float = 1.0,
    max_radius: float | None = None,
) -> list[TubePoint]:
    points: list[TubePoint] = []
    for agent in agents:
        if not agent.path:
            continue
        for k in range(horizon):
            idx = min(max(0, path_offset) + k, len(agent.path) - 1)
            y, x = agent.path[idx]
            dy, dx = _direction(agent.path, idx)
            width = _open_cell_width(grid, y, x, tube_width) if variable_width else tube_width
            width += max(0.0, width_growth) * k
            if max_radius is not None:
                width = min(width, max_radius)
            points.append(
                TubePoint(
                    agent_id=agent.agent_id,
                    k=k,
                    y=y,
                    x=x,
                    dy=dy,
                    dx=dx,
                    tube_width=width,
                    q=freshness_factor,
                    base_weight=float(np.exp(-beta * k)),
                )
            )
    return points


def with_freshness(points: list[TubePoint], factor: float) -> list[TubePoint]:
    return [replace(p, q=p.q * factor) for p in points]
