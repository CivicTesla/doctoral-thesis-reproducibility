"""Small MovingAI-style MAPF map loader and A* helper for E19 trace replay."""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path


@dataclass(frozen=True)
class GridMap:
    name: str
    width: int
    height: int
    cells: tuple[str, ...]

    def passable(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height and self.cells[y][x] in ".G"


def load_movingai_map(path: Path) -> GridMap:
    lines = [ln.rstrip("\n") for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    width = height = None
    grid_start = 0
    for i, ln in enumerate(lines):
        low = ln.lower()
        if low.startswith("height"):
            height = int(ln.split()[1])
        elif low.startswith("width"):
            width = int(ln.split()[1])
        elif low == "map":
            grid_start = i + 1
            break
    cells = tuple(lines[grid_start:grid_start + int(height)])
    return GridMap(path.stem, int(width), int(height), cells)


def write_generated_maps(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    maps = {
        "warehouse_shelves.map": [
            "................",
            "..@@@@..@@@@....",
            "..@@@@..@@@@....",
            "................",
            "....@@@@..@@@@..",
            "....@@@@..@@@@..",
            "................",
            "..@@@@..@@@@....",
            "..@@@@..@@@@....",
            "................",
        ],
        "room_bottleneck.map": [
            "................",
            "......@@........",
            "......@@........",
            "@@@@@@..@@@@@@..",
            "................",
            "..@@@@@@..@@@@@@",
            "........@@......",
            "........@@......",
            "................",
            "................",
        ],
    }
    paths = []
    for name, grid in maps.items():
        path = out_dir / name
        text = "\n".join(["type octile", f"height {len(grid)}", f"width {len(grid[0])}", "map", *grid])
        path.write_text(text + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def astar(grid: GridMap, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    def h(p: tuple[int, int]) -> int:
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    openq: list[tuple[int, tuple[int, int]]] = []
    heappush(openq, (h(start), start))
    came: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost: dict[tuple[int, int], int] = {start: 0}
    while openq:
        _, cur = heappop(openq)
        if cur == goal:
            break
        x, y = cur
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if not grid.passable(*nb):
                continue
            nc = cost[cur] + 1
            if nb not in cost or nc < cost[nb]:
                cost[nb] = nc
                came[nb] = cur
                heappush(openq, (nc + h(nb), nb))
    if goal not in came:
        return [start]
    path = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = came[cur]
    return list(reversed(path))
