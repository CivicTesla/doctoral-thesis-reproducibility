from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_DIR = Path("Output/05HPC_results")
PLOT_DIR = RESULT_DIR / "plots_summary"
TAUS = [0, 1, 2, 3, 4, 5]
AGENT_COUNTS = [20, 40, 60, 80, 100]


def load_successful_rounds(tau, agents):
    rounds_path = RESULT_DIR / f"DelayTau{tau}-Agents{agents}-LoopiNumber.npy"
    success_path = RESULT_DIR / f"DelayTau{tau}-Agents{agents}-Success.npy"

    if not rounds_path.exists():
        raise FileNotFoundError(rounds_path)
    if not success_path.exists():
        raise FileNotFoundError(success_path)

    rounds = np.load(rounds_path).astype(float)
    success = np.load(success_path).astype(float)

    if len(rounds) != len(success):
        raise ValueError(f"Mismatched lengths for tau={tau}, agents={agents}")

    return rounds[success == 1]


def collect_statistics():
    means = {}
    stds = {}
    counts = {}

    for agents in AGENT_COUNTS:
        agent_means = []
        agent_stds = []
        agent_counts = []

        for tau in TAUS:
            values = load_successful_rounds(tau, agents)
            if len(values) == 0:
                agent_means.append(np.nan)
                agent_stds.append(np.nan)
                agent_counts.append(0)
                continue

            agent_means.append(float(np.mean(values)))
            agent_stds.append(float(np.std(values)))
            agent_counts.append(int(len(values)))

        means[agents] = np.array(agent_means)
        stds[agents] = np.array(agent_stds)
        counts[agents] = agent_counts

    return means, stds, counts


def draw_figure(means, stds):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
        }
    )

    fig, ax = plt.subplots(figsize=(3.6, 2.7))

    styles = {
        20: {"color": "#1f1f1f", "marker": "o", "linestyle": "-"},
        40: {"color": "#3a3a3a", "marker": "s", "linestyle": "--"},
        60: {"color": "#5a5a5a", "marker": "^", "linestyle": "-."},
        80: {"color": "#767676", "marker": "D", "linestyle": ":"},
        100: {"color": "#929292", "marker": "v", "linestyle": "-"},
    }

    for agents in AGENT_COUNTS:
        style = styles[agents]
        ax.errorbar(
            TAUS,
            means[agents],
            yerr=stds[agents],
            label=f"{agents} agents",
            capsize=3,
            elinewidth=0.9,
            markersize=4,
            markerfacecolor="white",
            markeredgewidth=0.9,
            **style,
        )

    ax.set_title("Impact of DT Synchronization Delay on Convergence Performance")
    ax.set_xlabel("Synchronization Delay $\\tau$ (iterations)")
    ax.set_ylabel("Rounds Until 90% Convergence")
    ax.set_xticks(TAUS)
    ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.75)
    ax.legend(loc="upper left", frameon=True, edgecolor="#444444")

    for spine in ax.spines.values():
        spine.set_color("#333333")

    fig.tight_layout()

    png_path = PLOT_DIR / "dt_delay_convergence_impact.png"
    pdf_path = PLOT_DIR / "dt_delay_convergence_impact.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def main():
    means, stds, counts = collect_statistics()
    png_path, pdf_path = draw_figure(means, stds)

    print("Successful-run counts used per line:")
    for agents in AGENT_COUNTS:
        print(f"{agents} agents: {counts[agents]}")

    print("Generated:")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
