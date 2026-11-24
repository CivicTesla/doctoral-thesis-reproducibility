"""Trace replay definitions for E19 topology-realism rebuttal runs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from a2field.mapf_loader import load_movingai_map, write_generated_maps


@dataclass(frozen=True)
class TraceSpec:
    trace_family: str
    trace_name: str
    scenario: str
    description: str


DEFAULT_TRACES = (
    TraceSpec(
        "sumo_inspired_grid",
        "directed_road_grid",
        "cross_intersection",
        "Generated lane-like directed grid with multiple origin-destination flows and route switching around intersections.",
    ),
    TraceSpec(
        "mapf_warehouse",
        "warehouse_shelves",
        "warehouse_grid",
        "Generated MovingAI-style warehouse shelves map converted into future-intention replay points.",
    ),
    TraceSpec(
        "mapf_room_bottleneck",
        "room_bottleneck",
        "hourglass",
        "Generated room map with a narrow bottleneck; replay uses shortest-path style future intentions.",
    ),
)


def ensure_trace_assets(data_dir: Path) -> list[Path]:
    return write_generated_maps(data_dir)


def write_trace_manifest(path: Path, traces: tuple[TraceSpec, ...], data_dir: Path) -> None:
    lines = [
        "# E19 trace manifest",
        "",
        "This is trace-derived replay for topology realism, not a SUMO/ROS/ns-3 stack.",
        "The online scheduler still observes only future-intention candidate points, estimated communicated field state, and online-safe proxies.",
        "",
        "| trace_family | trace_name | mapped simulator scenario | description |",
        "|---|---|---|---|",
    ]
    for t in traces:
        lines.append(f"| {t.trace_family} | {t.trace_name} | {t.scenario} | {t.description} |")
    lines.extend(["", f"Generated MAPF map files: `{data_dir}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def draw_trace_maps(data_dir: Path, out_dir: Path) -> None:
    paths = ensure_trace_assets(data_dir)
    fig, axes = plt.subplots(1, len(paths) + 1, figsize=(8.2, 2.7))
    axes[0].set_title("road grid")
    for x in range(6):
        axes[0].plot([x, x], [0, 5], color="#666666", lw=1.2)
        axes[0].plot([0, 5], [x, x], color="#666666", lw=1.2)
    axes[0].scatter([0, 5, 1, 4], [0, 5, 5, 0], color="#2f5f8f", s=18)
    axes[0].set_aspect("equal")
    axes[0].axis("off")
    for ax, path in zip(axes[1:], paths):
        grid = load_movingai_map(path)
        arr = [[0 if c in ".G" else 1 for c in row] for row in grid.cells]
        ax.imshow(arr, cmap="Greys", vmin=0, vmax=1)
        ax.set_title(grid.name.replace("_", " "))
        ax.axis("off")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "e19_fig_trace_maps.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "e19_fig_trace_maps.pdf", bbox_inches="tight")
    plt.close(fig)
