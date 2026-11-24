from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_DIR = Path("Output/05HPC_results")
PLOT_DIR = RESULT_DIR / "plots_summary"
AGENT_COUNTS = [20, 40, 60, 80, 100]
PREFERRED_TAUS = [0, 1, 2, 3, 4, 5]


def available_taus():
    taus = []
    for tau in PREFERRED_TAUS:
        if all((RESULT_DIR / f"DelayTau{tau}-Agents{agents}-LoopCovMoves.npy").exists() for agents in AGENT_COUNTS):
            taus.append(tau)

    if 4 not in taus:
        return [tau for tau in [0, 1, 2, 3, 5] if tau in taus]
    return taus


def load_successful_covmoves(tau, agents):
    cov_path = RESULT_DIR / f"DelayTau{tau}-Agents{agents}-LoopCovMoves.npy"
    success_path = RESULT_DIR / f"DelayTau{tau}-Agents{agents}-Success.npy"

    if not cov_path.exists():
        raise FileNotFoundError(cov_path)
    if not success_path.exists():
        raise FileNotFoundError(success_path)

    covmoves = np.load(cov_path).astype(float)
    success = np.load(success_path).astype(float)

    if len(covmoves) != len(success):
        raise ValueError(f"Mismatched lengths for tau={tau}, agents={agents}")

    values = covmoves[success == 1]
    if len(values) == 0:
        raise ValueError(f"No successful coverage-move samples for tau={tau}, agents={agents}")
    return values


def collect_coverage_statistics(taus):
    # This figure shows that coverage moves remain stable under bounded
    # DT synchronization delay across different swarm sizes.
    stats = {}

    for agents in AGENT_COUNTS:
        means = []
        stds = []
        counts = []

        for tau in taus:
            covmoves = load_successful_covmoves(tau, agents)
            means.append(float(np.mean(covmoves)))
            stds.append(float(np.std(covmoves, ddof=1)) if len(covmoves) > 1 else 0.0)
            counts.append(int(len(covmoves)))

        stats[agents] = {
            "means": np.array(means),
            "stds": np.array(stds),
            "counts": counts,
        }

    return stats


def draw_figure(taus, stats):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.85,
            "lines.linewidth": 2.0,
        }
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)

    styles = {
        20: {"color": "#0072B2", "marker": "o"},
        40: {"color": "#D55E00", "marker": "s"},
        60: {"color": "#009E73", "marker": "^"},
        80: {"color": "#CC79A7", "marker": "D"},
        100: {"color": "#E69F00", "marker": "v"},
    }

    for agents in AGENT_COUNTS:
        style = styles[agents]
        ax.errorbar(
            taus,
            stats[agents]["means"],
            yerr=stats[agents]["stds"],
            label=f"{agents} agents",
            color=style["color"],
            marker=style["marker"],
            markersize=5.5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            linewidth=2.0,
            elinewidth=1.0,
            capsize=3.5,
        )

    ax.set_title("Impact of DT Synchronization Delay", pad=8)
    ax.set_xlabel("Synchronization Delay $\\tau$ (iterations)")
    ax.set_ylabel("Coverage Moves")
    ax.set_xticks(taus)
    ax.grid(True, which="major", linestyle=":", linewidth=0.65, alpha=0.75)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=5, frameon=True)

    lower_bounds = []
    upper_bounds = []
    for agents in AGENT_COUNTS:
        lower_bounds.extend(stats[agents]["means"] - stats[agents]["stds"])
        upper_bounds.extend(stats[agents]["means"] + stats[agents]["stds"])

    y_min = np.min(lower_bounds)
    y_max = np.max(upper_bounds)
    padding = max((y_max - y_min) * 0.12, 100.0)
    ax.set_ylim(max(0, y_min - padding), y_max + padding)

    for spine in ax.spines.values():
        spine.set_color("#333333")

    png_path = PLOT_DIR / "dt_delay_navigation_cost_impact.png"
    pdf_path = PLOT_DIR / "dt_delay_navigation_cost_impact.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def main():
    taus = available_taus()
    if 0 not in taus:
        raise RuntimeError("tau=0 baseline data is required")

    stats = collect_coverage_statistics(taus)
    png_path, pdf_path = draw_figure(taus, stats)

    print("Taus used:", taus)
    print("Coverage move statistics by agent count:")
    for agents in AGENT_COUNTS:
        values = ", ".join(
            f"tau={tau}: mean={mean:.2f}, std={std:.2f}"
            for tau, mean, std in zip(taus, stats[agents]["means"], stats[agents]["stds"])
        )
        print(f"{agents} agents: {values}")

    print("Generated:")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
