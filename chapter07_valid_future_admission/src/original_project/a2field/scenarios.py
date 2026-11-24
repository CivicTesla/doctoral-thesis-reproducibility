from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .grid import Grid, empty_obstacles

Cell = tuple[int, int]
Sampler = Callable[[np.random.Generator], Cell]


@dataclass(frozen=True)
class Scenario:
    name: str
    grid: Grid
    sample_start: Sampler
    sample_goal: Sampler
    critical_mask: np.ndarray
    metadata: dict


def _sample_from(cells: list[Cell]) -> Sampler:
    def sampler(rng: np.random.Generator) -> Cell:
        idx = int(rng.integers(0, len(cells)))
        return cells[idx]

    return sampler


def _free_cells_in_rect(obstacles: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> list[Cell]:
    cells: list[Cell] = []
    for y in range(y0, y1):
        for x in range(x0, x1):
            if 0 <= y < obstacles.shape[0] and 0 <= x < obstacles.shape[1] and not obstacles[y, x]:
                cells.append((y, x))
    return cells


def dual_corridor(height: int = 50, width: int = 80) -> Scenario:
    obstacles = empty_obstacles(height, width)
    obstacles[:3, :] = True
    obstacles[-3:, :] = True
    obstacles[:, :2] = True
    obstacles[:, -2:] = True

    # Central block leaves upper/lower corridors and a narrow middle gate.
    mid_y, mid_x = height // 2, width // 2
    obstacles[14:36, 28:53] = True
    obstacles[mid_y - 3 : mid_y + 4, 28:53] = False

    # Pinch both entrances of the gate to create a bottleneck pressure point.
    obstacles[mid_y - 6 : mid_y - 3, 28:32] = True
    obstacles[mid_y + 4 : mid_y + 7, 28:32] = True
    obstacles[mid_y - 6 : mid_y - 3, 49:53] = True
    obstacles[mid_y + 4 : mid_y + 7, 49:53] = True

    critical = np.zeros_like(obstacles)
    critical[mid_y - 5 : mid_y + 6, 24:32] = True
    critical[mid_y - 5 : mid_y + 6, 49:57] = True
    critical &= ~obstacles

    starts = _free_cells_in_rect(obstacles, 6, height - 6, 3, 9)
    goals = _free_cells_in_rect(obstacles, 6, height - 6, width - 9, width - 3)
    return Scenario(
        "dual_corridor",
        Grid(obstacles),
        _sample_from(starts),
        _sample_from(goals),
        critical,
        {"description": "left-to-right flow through dual corridors and a central gate"},
    )


def asymmetric_dual_corridor(height: int = 50, width: int = 80) -> Scenario:
    """Dual corridor with endpoints biased toward the upper route.

    Geometry and all simulation parameters remain unchanged.  The endpoint bias
    makes the upper route shorter initially while leaving the lower route as the
    spatial alternative after congestion or a route switch.
    """
    base = dual_corridor(height, width)
    obstacles = base.grid.obstacles
    starts = _free_cells_in_rect(obstacles, 6, height // 2 - 5, 3, 9)
    goals = _free_cells_in_rect(obstacles, 6, height // 2 - 5, width - 9, width - 3)
    return Scenario(
        "asymmetric_dual_corridor",
        base.grid,
        _sample_from(starts),
        _sample_from(goals),
        base.critical_mask.copy(),
        {
            "description": "dual corridor with upper-biased endpoints and an initially shorter upper route",
            "variant_of": "dual_corridor",
            "changed_parameter": "endpoint sampling region",
        },
    )


def bottleneck_shift(height: int = 50, width: int = 80) -> Scenario:
    """Dual corridor with endpoints biased toward the lower route.

    Geometry and simulation parameters are unchanged from dual_corridor.  The
    lower-half endpoint bias concentrates traffic in the lower corridor,
    shifting the primary congestion bottleneck there and creating a different
    spatial pressure pattern from both original_dual_corridor (balanced) and
    asymmetric_dual_corridor (upper-biased).
    """
    base = dual_corridor(height, width)
    obstacles = base.grid.obstacles
    starts = _free_cells_in_rect(obstacles, height // 2 + 5, height - 6, 3, 9)
    goals = _free_cells_in_rect(obstacles, height // 2 + 5, height - 6, width - 9, width - 3)
    mid_y = height // 2
    critical = np.zeros_like(obstacles)
    # Critical region covers the gate entrance and the lower corridor approach.
    critical[mid_y - 5 : mid_y + 6, 24:32] = True
    critical[mid_y - 5 : mid_y + 6, 49:57] = True
    critical[mid_y + 8 : min(height - 3, mid_y + 18), 18:35] = True
    critical &= ~obstacles
    return Scenario(
        "bottleneck_shift",
        base.grid,
        _sample_from(starts),
        _sample_from(goals),
        critical,
        {
            "description": "dual corridor with lower-half-biased endpoints concentrating traffic in the lower corridor",
            "variant_of": "dual_corridor",
            "changed_parameter": "endpoint sampling region (lower half, rows h//2+5 to h-6)",
        },
    )


def cross_intersection(height: int = 55, width: int = 55) -> Scenario:
    obstacles = np.ones((height, width), dtype=bool)
    cy, cx = height // 2, width // 2
    half = 5
    obstacles[cy - half : cy + half + 1, :] = False
    obstacles[:, cx - half : cx + half + 1] = False
    obstacles[:2, :] = True
    obstacles[-2:, :] = True
    obstacles[:, :2] = True
    obstacles[:, -2:] = True

    critical = np.zeros_like(obstacles)
    critical[cy - 7 : cy + 8, cx - 7 : cx + 8] = True
    critical &= ~obstacles

    left_starts = _free_cells_in_rect(obstacles, cy - half, cy + half + 1, 3, 8)
    bottom_starts = _free_cells_in_rect(obstacles, height - 8, height - 3, cx - half, cx + half + 1)
    right_goals = _free_cells_in_rect(obstacles, cy - half, cy + half + 1, width - 8, width - 3)
    top_goals = _free_cells_in_rect(obstacles, 3, 8, cx - half, cx + half + 1)

    def sample_start(rng: np.random.Generator) -> Cell:
        return _sample_from(left_starts if rng.random() < 0.5 else bottom_starts)(rng)

    def sample_goal(rng: np.random.Generator) -> Cell:
        return _sample_from(right_goals if rng.random() < 0.5 else top_goals)(rng)

    return Scenario(
        "cross_intersection",
        Grid(obstacles),
        sample_start,
        sample_goal,
        critical,
        {"description": "orthogonal corridor flows crossing at a center intersection"},
    )


def wide_corridor(height: int = 45, width: int = 85) -> Scenario:
    obstacles = np.ones((height, width), dtype=bool)
    obstacles[10:35, 2 : width - 2] = False
    obstacles[:2, :] = True
    obstacles[-2:, :] = True
    obstacles[:, :2] = True
    obstacles[:, -2:] = True

    # Local islands create mild narrow passages while leaving lateral alternatives.
    obstacles[17:28, 31:36] = True
    obstacles[10:20, 51:56] = True
    obstacles[25:35, 51:56] = True
    obstacles[16:29, 66:70] = True

    critical = np.zeros_like(obstacles)
    critical[14:31, 27:40] = True
    critical[8:37, 48:59] = True
    critical[14:31, 62:74] = True
    critical &= ~obstacles

    starts = _free_cells_in_rect(obstacles, 12, 33, 3, 9)
    goals = _free_cells_in_rect(obstacles, 12, 33, width - 9, width - 3)
    return Scenario(
        "wide_corridor",
        Grid(obstacles),
        _sample_from(starts),
        _sample_from(goals),
        critical,
        {"description": "wide corridor with local obstacles and lateral shift options"},
    )


def hourglass_bottleneck(height: int = 50, width: int = 80) -> Scenario:
    """Two open chambers connected by a single narrow vertical gate.

    All agents must pass through the gate regardless of which side of the left or
    right chamber they originate from.  The spatial diversity of approach paths
    creates different ghost and hidden field exposures even though the physical
    bottleneck is shared.
    """
    obstacles = empty_obstacles(height, width)
    obstacles[:3, :] = True
    obstacles[-3:, :] = True
    obstacles[:, :2] = True
    obstacles[:, -2:] = True

    mid_y = height // 2       # 25
    gate_y_half = 5            # gate opening = rows 20-29 (10 rows)
    gate_col0, gate_col1 = 33, 48   # cols forming the middle walls

    # Wall above the gate in the central column band.
    obstacles[3 : mid_y - gate_y_half, gate_col0:gate_col1] = True
    # Wall below the gate in the central column band.
    obstacles[mid_y + gate_y_half : height - 3, gate_col0:gate_col1] = True

    critical = np.zeros_like(obstacles)
    critical[mid_y - gate_y_half - 4 : mid_y + gate_y_half + 4, gate_col0 - 5 : gate_col1 + 5] = True
    critical &= ~obstacles

    starts = _free_cells_in_rect(obstacles, 6, height - 6, 3, 12)
    goals  = _free_cells_in_rect(obstacles, 6, height - 6, width - 12, width - 3)
    return Scenario(
        "hourglass_bottleneck",
        Grid(obstacles),
        _sample_from(starts),
        _sample_from(goals),
        critical,
        {"description": "two open chambers connected by a single narrow vertical gate"},
    )


def warehouse_grid(height: int = 55, width: int = 90) -> Scenario:
    """Warehouse-like topology with three parallel horizontal aisles and three
    vertical cross-aisles forming a grid of local bottlenecks and alternative routes.

    Upper route: agents travel through the top aisle.
    Lower route: agents travel through the bottom aisle.
    The cross-aisles are shared local bottlenecks and form the critical region.
    """
    obstacles = np.ones((height, width), dtype=bool)
    # Boundary walls.
    obstacles[:3, :] = True
    obstacles[-3:, :] = True
    obstacles[:, :2] = True
    obstacles[:, -2:] = True

    # Carve three horizontal aisles.
    obstacles[8:13, 2 : width - 2] = False    # top aisle
    obstacles[25:30, 2 : width - 2] = False   # middle aisle
    obstacles[42:47, 2 : width - 2] = False   # bottom aisle

    # Carve three vertical cross-aisles connecting all horizontal aisles.
    for x0 in (18, 42, 66):
        obstacles[8:47, x0 : x0 + 4] = False

    critical = np.zeros_like(obstacles)
    for x0 in (18, 42, 66):
        critical[8:47, x0 : x0 + 4] = True
    critical &= ~obstacles

    starts = _free_cells_in_rect(obstacles, 8, 47, 3, 9)
    goals  = _free_cells_in_rect(obstacles, 8, 47, width - 9, width - 3)
    return Scenario(
        "warehouse_grid",
        Grid(obstacles),
        _sample_from(starts),
        _sample_from(goals),
        critical,
        {"description": "three parallel aisles connected by three vertical cross-aisles"},
    )


def get_scenario(name: str) -> Scenario:
    scenarios = {
        "dual_corridor": dual_corridor,
        "original_dual_corridor": dual_corridor,
        "dual_corridor_current": dual_corridor,
        "asymmetric_dual_corridor": asymmetric_dual_corridor,
        "bottleneck_shift": bottleneck_shift,
        "cross_intersection": cross_intersection,
        "wide_corridor": wide_corridor,
        "hourglass_bottleneck": hourglass_bottleneck,
        "warehouse_grid": warehouse_grid,
    }
    if name not in scenarios:
        raise ValueError(f"Unknown scenario {name!r}. Choices: {sorted(scenarios)}")
    return scenarios[name]()
