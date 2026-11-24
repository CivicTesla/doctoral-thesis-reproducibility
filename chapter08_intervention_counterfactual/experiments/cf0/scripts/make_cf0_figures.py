#!/usr/bin/env python3
"""CF0 diagnostic figures. Pale/muted palette, no truncated axes without an
explicit annotation. Reads only already-written CF0 CSVs; no simulation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve()
CF0 = HERE.parents[1]
FIG = CF0 / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CAPACITY = "#7FA6C7"   # pale blue
SYNC = "#8FBF8F"       # pale green
NO_ACTION = "#C9A9A6"  # pale rose
GRID = "#DDDDDD"
TEXT = "#333333"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#999999", "axes.labelcolor": TEXT,
    "text.color": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "font.size": 10, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def paired_plot(outcomes, mechanism, path, title):
    subset = outcomes[outcomes.mechanism == mechanism].sort_values("seed")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(subset))
    ax.plot(x, subset.utility_no_action, "o-", color=NO_ACTION,
            label="NO_ACTION", markersize=4)
    ax.plot(x, subset.utility_capacity, "o-", color=CAPACITY,
            label="CAPACITY_ACTION", markersize=4)
    ax.plot(x, subset.utility_sync, "o-", color=SYNC,
            label="SYNC_ACTION", markersize=4)
    ax.set_xlabel("checkpoint (one per seed, sorted)")
    ax.set_ylabel("utility (higher is better)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fraction_bar(within_df, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    mechs = within_df.mechanism.tolist()
    labels = ["H1\n(capacity shortage)", "H2\n(distributed suppression)"]
    width = 0.25
    x = np.arange(len(mechs))
    ax.bar(x - width, within_df.CAPACITY_ACTION, width, color=CAPACITY,
           label="CAPACITY_ACTION")
    ax.bar(x, within_df.SYNC_ACTION, width, color=SYNC, label="SYNC_ACTION")
    ax.bar(x + width, within_df.NO_ACTION, width, color=NO_ACTION,
           label="NO_ACTION")
    ax.set_xticks(x, labels)
    ax.set_ylabel("fraction of checkpoints where action is oracle-best")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def advantage_hist(outcomes, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    for ax, mech, color, name in zip(
            axes, ["H1_CAPACITY_SHORTAGE",
                   "H2_DISTRIBUTED_CAPACITY_SUPPRESSION"],
            [CAPACITY, SYNC], ["H1", "H2"]):
        subset = outcomes[outcomes.mechanism == mech]
        ax.hist(subset.best_vs_second_advantage, bins=12, color=color,
                edgecolor="white")
        ax.set_title(f"{name} action-advantage distribution")
        ax.set_xlabel("best - second-best utility")
    axes[0].set_ylabel("checkpoints")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def per_seed_majority(outcomes, path):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    color_map = {
        "CAPACITY_ACTION": CAPACITY, "SYNC_ACTION": SYNC,
        "NO_ACTION": NO_ACTION}
    for i, mech in enumerate(
            ["H1_CAPACITY_SHORTAGE", "H2_DISTRIBUTED_CAPACITY_SUPPRESSION"]):
        subset = outcomes[outcomes.mechanism == mech].sort_values("seed")
        colors = [color_map[a] for a in subset.oracle_best_action]
        ax.scatter(subset.seed, [i] * len(subset), c=colors, s=40)
    ax.set_yticks([0, 1], ["H1", "H2"])
    ax.set_xlabel("seed")
    ax.set_title("per-seed oracle-best action (one checkpoint per seed)")
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=c, markersize=8, label=a)
               for a, c in color_map.items()]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right",
              bbox_to_anchor=(1.15, 1.3), ncol=1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def advantage_vs_load(outcomes, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, mech, color, name in zip(
            axes, ["H1_CAPACITY_SHORTAGE",
                   "H2_DISTRIBUTED_CAPACITY_SUPPRESSION"],
            [CAPACITY, SYNC], ["H1", "H2"]):
        subset = outcomes[outcomes.mechanism == mech]
        ax.scatter(subset.arrivals, subset.best_vs_second_advantage,
                   color=color, s=30)
        ax.set_xlabel("realized total arrivals (load proxy; see 15_*.md)")
        ax.set_title(f"{name}: advantage vs. realized load")
    axes[0].set_ylabel("action advantage")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    outcomes = pd.read_csv(CF0 / "08_per_checkpoint_action_outcomes.csv")
    within_df = pd.read_csv(CF0 / "11_within_mechanism_action_distribution.csv")

    paired_plot(outcomes, "H1_CAPACITY_SHORTAGE",
                FIG / "01_h1_three_action_paired.png",
                "H1: three-action paired outcomes (identical exogenous history)")
    paired_plot(outcomes, "H2_DISTRIBUTED_CAPACITY_SUPPRESSION",
                FIG / "02_h2_three_action_paired.png",
                "H2: three-action paired outcomes (identical exogenous history)")
    fraction_bar(within_df, FIG / "03_oracle_action_fraction.png")
    advantage_hist(outcomes, FIG / "04_action_advantage_distribution.png")
    per_seed_majority(outcomes, FIG / "05_per_seed_majority_action.png")
    advantage_vs_load(outcomes, FIG / "06_advantage_vs_state_proxy.png")
    print("wrote 6 figures to", FIG)


if __name__ == "__main__":
    main()
