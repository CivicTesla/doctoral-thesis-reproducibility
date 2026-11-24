#!/usr/bin/env python3
"""CF1 diagnostic figures. Pale/muted palette, no truncated axes without an
explicit annotation. Reads only already-written CF1 CSVs; no simulation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve()
CF1 = HERE.parents[1]
FIG = CF1 / "figures"
FIG.mkdir(parents=True, exist_ok=True)

T0_C = "#C9A9A6"
T4_C = "#7FA6C7"
R4_C = "#8FBF8F"
H1_C = "#7FA6C7"
H2_C = "#D6A96B"
GRID = "#DDDDDD"
TEXT = "#333333"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#999999", "axes.labelcolor": TEXT,
    "text.color": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "font.size": 10, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def fig1_ba_vs_regret():
    m = pd.read_csv(CF1 / "09_per_method_strict_regret.csv")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"T0": T0_C, "T4": T4_C, "R4": R4_C}
    for _, row in m.iterrows():
        ax.scatter(row.balanced_accuracy, row.total_regret,
                   color=colors[row.method], s=140, zorder=3)
        ax.annotate(row.method, (row.balanced_accuracy, row.total_regret),
                    xytext=(8, 6), textcoords="offset points", fontsize=10)
    ax.set_xlabel("balanced accuracy")
    ax.set_ylabel("strict total regret")
    ax.set_title("Balanced accuracy vs. strict counterfactual total regret")
    fig.tight_layout()
    fig.savefig(FIG / "01_ba_vs_strict_total_regret.png", dpi=150)
    plt.close(fig)


def fig2_h1_h2_regret():
    m = pd.read_csv(CF1 / "09_per_method_strict_regret.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(m))
    width = 0.35
    ax.bar(x - width / 2, m.h1_regret, width, color=H1_C, label="H1 regret")
    ax.bar(x + width / 2, m.h2_regret, width, color=H2_C, label="H2 regret")
    ax.set_xticks(x, m.method)
    ax.set_ylabel("strict regret")
    ax.set_title("T0/T4/R4 strict H1 and H2 regret")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "02_h1_h2_regret_by_method.png", dpi=150)
    plt.close(fig)


def fig3_error_cost():
    d = pd.read_csv(CF1 / "_error_direction_cost_ratio.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(d))
    ax.bar(x, d.mean_cost_ratio_h2_to_h1_over_h1_to_h2, color=H2_C)
    ax.axhline(1.0, color="#999999", linewidth=1, linestyle="--")
    ax.set_xticks(x, d.method)
    ax.set_ylabel("mean cost ratio (H2->H1 / H1->H2)")
    ax.set_title("H2->H1 vs. H1->H2 error cost ratio (strict)")
    fig.tight_layout()
    fig.savefig(FIG / "03_error_direction_cost_ratio.png", dpi=150)
    plt.close(fig)


def fig4_coverage_regret():
    m = pd.read_csv(CF1 / "09_per_method_strict_regret.csv")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"T0": T0_C, "T4": T4_C, "R4": R4_C}
    for _, row in m.iterrows():
        ax.scatter(row.coverage, row.total_regret, color=colors[row.method],
                   s=140, zorder=3)
        ax.annotate(row.method, (row.coverage, row.total_regret),
                    xytext=(8, 6), textcoords="offset points", fontsize=10)
    ax.set_xlabel("coverage")
    ax.set_ylabel("strict total regret")
    ax.set_title("Coverage vs. strict total regret")
    fig.tight_layout()
    fig.savefig(FIG / "04_coverage_vs_strict_regret.png", dpi=150)
    plt.close(fig)


def fig5_matched_coverage():
    d = pd.read_csv(CF1 / "16_matched_coverage_results.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(d))
    width = 0.35
    ax.bar(x - width / 2, d.h1_regret, width, color=H1_C, label="H1 regret")
    ax.bar(x + width / 2, d.h2_regret, width, color=H2_C, label="H2 regret")
    ax.set_xticks(x, ["T4 matched\nto R4 coverage", "R4 actual"])
    ax.set_ylabel("strict regret")
    ax.set_title(f"Matched-coverage comparison (coverage={d.coverage.iloc[0]:.3f})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "05_matched_coverage_comparison.png", dpi=150)
    plt.close(fig)


def fig6_old_vs_new():
    d = pd.read_csv(CF1 / "19_old_vs_new_regret_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(len(d))
    width = 0.35
    axes[0].bar(x - width / 2, d.old_total_regret_HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY,
                width, color="#BBBBBB", label="old (non-causal, historical)")
    axes[0].bar(x + width / 2, d.new_strict_total_regret, width, color=T4_C,
                label="new (strict)")
    axes[0].set_xticks(x, d.method)
    axes[0].set_title("Total regret: old vs. strict")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(x - width / 2, d.old_h2_regret_HISTORICAL_NONCAUSAL_DIAGNOSTIC_ONLY,
                width, color="#BBBBBB", label="old (non-causal, historical)")
    axes[1].bar(x + width / 2, d.new_strict_h2_regret, width, color=H2_C,
                label="new (strict)")
    axes[1].set_xticks(x, d.method)
    axes[1].set_title("H2 regret: old vs. strict")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "06_old_vs_strict_regret.png", dpi=150)
    plt.close(fig)


def main():
    fig1_ba_vs_regret()
    fig2_h1_h2_regret()
    fig3_error_cost()
    fig4_coverage_regret()
    fig5_matched_coverage()
    fig6_old_vs_new()
    print("wrote 6 figures to", FIG)


if __name__ == "__main__":
    main()
