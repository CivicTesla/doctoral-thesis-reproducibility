from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """2D occupancy grid. True obstacle cells are blocked."""

    obstacles: np.ndarray

    @property
    def height(self) -> int:
        return int(self.obstacles.shape[0])

    @property
    def width(self) -> int:
        return int(self.obstacles.shape[1])

    def in_bounds(self, y: int, x: int) -> bool:
        return 0 <= y < self.height and 0 <= x < self.width

    def is_free(self, y: int, x: int) -> bool:
        return self.in_bounds(y, x) and not bool(self.obstacles[y, x])

    def neighbors4(self, y: int, x: int) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if self.is_free(ny, nx):
                out.append((ny, nx))
        return out


def empty_obstacles(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=bool)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Small dependency-free square dilation for critical regions."""

    if radius <= 0:
        return mask.copy()
    ys, xs = np.where(mask)
    out = mask.copy()
    h, w = mask.shape
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        out[y0:y1, x0:x1] = True
    return out
