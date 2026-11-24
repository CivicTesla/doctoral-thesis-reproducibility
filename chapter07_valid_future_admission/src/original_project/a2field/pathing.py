from __future__ import annotations

import heapq

from .grid import Grid

Cell = tuple[int, int]


def _manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid: Grid, start: Cell, goal: Cell) -> list[Cell]:
    """A small 4-neighbor A* planner for prototype path generation."""

    if not grid.is_free(*start) or not grid.is_free(*goal):
        return []
    frontier: list[tuple[int, int, Cell]] = []
    heapq.heappush(frontier, (_manhattan(start, goal), 0, start))
    came_from: dict[Cell, Cell | None] = {start: None}
    cost_so_far: dict[Cell, int] = {start: 0}
    counter = 0

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for nxt in grid.neighbors4(*current):
            new_cost = cost_so_far[current] + 1
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                counter += 1
                priority = new_cost + _manhattan(nxt, goal)
                heapq.heappush(frontier, (priority, counter, nxt))
                came_from[nxt] = current

    if goal not in came_from:
        return []
    path: list[Cell] = []
    cur: Cell | None = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path
