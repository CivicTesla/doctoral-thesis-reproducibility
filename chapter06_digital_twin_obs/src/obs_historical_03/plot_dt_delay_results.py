import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


TAUS = list(range(6))
METRICS = {
    "LoopiNumber": "Rounds",
    "LoopCovMoves": "Coverage moves",
    "LoopUpLinkNumber": "Uplink attempts",
    "LoopDownLinkNumber": "Downlink attempts",
    "Success": "Success rate",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot delay experiment npy results.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="Output/DTDelay",
        help="Directory containing DelayTau*-Agents*-*.npy files.",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Directory for generated plots. Defaults to <output_dir>/plots.",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="For non-Success metrics, plot only runs whose matching Success value is 1.",
    )
    parser.add_argument(
        "--drop-outliers",
        action="store_true",
        help="Drop outliers from each tau/agents/metric series before plotting.",
    )
    return parser.parse_args()


def load_results(output_dir):
    pattern = re.compile(r"DelayTau(\d+)-Agents(\d+)-(.+)\.npy$")
    results = {}

    for path in output_dir.glob("DelayTau*-Agents*-*.npy"):
        match = pattern.match(path.name)
        if not match:
            continue

        tau = int(match.group(1))
        agents = int(match.group(2))
        metric = match.group(3)
        values = np.load(path)
        results[(tau, agents, metric)] = values.astype(float)

    agents = sorted({agents for _, agents, _ in results})
    return results, agents


def filter_successful_runs(results):
    filtered = {}
    dropped = {}

    for (tau, agents, metric), values in results.items():
        if metric == "Success":
            filtered[(tau, agents, metric)] = values
            continue

        success_values = results.get((tau, agents, "Success"))
        if success_values is None or len(success_values) != len(values):
            filtered[(tau, agents, metric)] = values
            continue

        mask = success_values == 1
        filtered[(tau, agents, metric)] = values[mask]
        dropped[(tau, agents, metric)] = int(np.sum(~mask))

    return filtered, dropped


def drop_outliers_iqr(results):
    filtered = {}
    dropped = {}

    for key, values in results.items():
        if len(values) < 4:
            filtered[key] = values
            dropped[key] = 0
            continue

        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        if iqr == 0:
            filtered[key] = values
            dropped[key] = 0
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (values >= lower) & (values <= upper)
        filtered[key] = values[mask]
        dropped[key] = int(np.sum(~mask))

    return filtered, dropped


def count_failed_runs(results):
    failed = 0
    for (_, _, metric), values in results.items():
        if metric == "Success":
            failed += int(np.sum(values == 0))
    return failed


def metric_value(values, metric):
    if values.size == 0:
        return np.nan
    if metric == "Success":
        return float(np.mean(values))
    return float(np.median(values))


def build_matrix(results, agents, metric):
    matrix = np.full((len(TAUS), len(agents)), np.nan)
    for row, tau in enumerate(TAUS):
        for col, agent_count in enumerate(agents):
            values = results.get((tau, agent_count, metric))
            if values is not None:
                matrix[row, col] = metric_value(values, metric)
    return matrix


def draw_heatmap(results, agents, metric, plot_dir):
    matrix = build_matrix(results, agents, metric)
    title = METRICS[metric]
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, len(agents) * 1.2), 5.2))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#eeeeee")

    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, aspect="auto")

    ax.set_xticks(np.arange(len(agents)))
    ax.set_xticklabels(agents)
    ax.set_yticks(np.arange(len(TAUS)))
    ax.set_yticklabels(TAUS)
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Tau")
    ax.set_title(f"{title} by agents and tau")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            label = "NA" if np.isnan(value) else f"{value:.0f}" if metric != "Success" else f"{value:.2f}"
            ax.text(col, row, label, ha="center", va="center", color="white" if not np.isnan(value) else "#555555", fontsize=8)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean" if metric == "Success" else "Median")
    fig.tight_layout()

    out_path = plot_dir / f"heatmap_{metric}.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def draw_metric_violin(results, agents, metric, plot_dir):
    title = METRICS[metric]
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(9, len(agents) * 1.6), 5.8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(TAUS)))
    offsets = np.linspace(-0.36, 0.36, len(TAUS))
    width = 0.11

    legend_handles = []
    base_positions = np.arange(1, len(agents) + 1)

    for tau_index, (tau, color, offset) in enumerate(zip(TAUS, colors, offsets)):
        data = []
        positions = []
        for base_position, agent_count in zip(base_positions, agents):
            values = results.get((tau, agent_count, metric))
            if values is None or len(values) == 0:
                continue
            data.append(values)
            positions.append(base_position + offset)

        if not data:
            continue

        parts = ax.violinplot(
            data,
            positions=positions,
            showmeans=False,
            showmedians=True,
            widths=width,
        )
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.58)
        parts["cmedians"].set_color("#222222")
        parts["cmedians"].set_linewidth(1.2)

        legend_handles.append(
            plt.Line2D([0], [0], color=color, lw=6, alpha=0.58, label=f"tau={tau}")
        )

    ax.set_xticks(base_positions)
    ax.set_xticklabels(agents)
    ax.set_xlabel("Number of agents")
    ax.set_ylabel(title)
    ax.set_title(f"{title} distribution by agents and tau")
    ax.grid(axis="y", alpha=0.25)
    if legend_handles:
        ax.legend(handles=legend_handles, title="Tau", ncols=3, loc="upper left")
    fig.tight_layout()

    out_path = plot_dir / f"violin_{metric}_by_tau_agents.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def draw_metric_points(results, agents, metric, plot_dir):
    title = METRICS[metric]
    plot_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(len(TAUS), 1, figsize=(max(7, len(agents) * 1.15), 14), sharex=True)

    for ax, tau in zip(axes, TAUS):
        mean_x = []
        mean_y = []
        std_y = []

        for idx, agent_count in enumerate(agents, start=1):
            values = results.get((tau, agent_count, metric))
            if values is None or len(values) == 0:
                continue

            jitter = rng.uniform(-0.14, 0.14, size=len(values))
            ax.scatter(
                np.full(len(values), idx) + jitter,
                values,
                s=14,
                alpha=0.42,
                color="#4C78A8",
                edgecolors="none",
            )
            mean_x.append(idx)
            mean_y.append(float(np.mean(values)))
            std_y.append(float(np.std(values)))

        if mean_x:
            ax.errorbar(
                mean_x,
                mean_y,
                yerr=std_y,
                color="#D62728",
                marker="o",
                linewidth=1.5,
                capsize=3,
                label="mean +/- std",
            )
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(axis="y", alpha=0.25)
        else:
            ax.text(0.5, 0.5, f"No data for tau={tau}", ha="center", va="center", transform=ax.transAxes)

        ax.set_ylabel(f"tau={tau}")

    axes[-1].set_xticks(range(1, len(agents) + 1))
    axes[-1].set_xticklabels(agents)
    axes[-1].set_xlabel("Number of agents")
    fig.suptitle(f"All runs for {title}: points with mean +/- std", y=0.995)
    fig.tight_layout()

    out_path = plot_dir / f"points_{metric}_by_tau_agents.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def draw_metric_lines(results, agents, metric, plot_dir):
    title = METRICS[metric]
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    colors = plt.cm.tab10(np.linspace(0, 1, len(TAUS)))

    for color, tau in zip(colors, TAUS):
        x_values = []
        means = []
        stds = []

        for agent_count in agents:
            values = results.get((tau, agent_count, metric))
            if values is None or len(values) == 0:
                continue

            x_values.append(agent_count)
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values)))

        if not x_values:
            continue

        x_values = np.array(x_values)
        means = np.array(means)
        stds = np.array(stds)

        ax.plot(
            x_values,
            means,
            marker="o",
            linewidth=1.8,
            color=color,
            label=f"tau={tau}",
        )
        ax.fill_between(
            x_values,
            means - stds,
            means + stds,
            color=color,
            alpha=0.15,
            linewidth=0,
        )

    ax.set_xlabel("Number of agents")
    ax.set_ylabel(title)
    ax.set_title(f"{title}: mean with +/- std band")
    ax.set_xticks(agents)
    ax.grid(alpha=0.25)
    ax.legend(title="Tau", ncols=2)
    fig.tight_layout()

    out_path = plot_dir / f"lines_{metric}_mean_std_by_agents.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    default_plot_dir = "plots_success_only" if args.success_only else "plots"
    plot_dir = Path(args.plot_dir) if args.plot_dir else output_dir / default_plot_dir

    results, agents = load_results(output_dir)
    if not results:
        raise RuntimeError(f"No DelayTau*-Agents*-*.npy files found in {output_dir}")

    if args.success_only:
        failed_runs = count_failed_runs(results)
        results, dropped = filter_successful_runs(results)
        total_dropped = sum(dropped.values())
        print(
            "Using successful runs only for non-Success metrics; "
            f"found {failed_runs} failed Monte Carlo runs and dropped "
            f"{total_dropped} aligned metric samples."
        )

    if args.drop_outliers:
        results, outliers = drop_outliers_iqr(results)
        total_outliers = sum(outliers.values())
        print(f"Dropped {total_outliers} outlier samples by per-series IQR rule.")

    written = []
    for metric in METRICS:
        if any(key[2] == metric for key in results):
            written.append(draw_heatmap(results, agents, metric, plot_dir))

    for metric in METRICS:
        if metric == "Success":
            continue
        if any(key[2] == metric for key in results):
            written.append(draw_metric_lines(results, agents, metric, plot_dir))
            written.append(draw_metric_violin(results, agents, metric, plot_dir))
            written.append(draw_metric_points(results, agents, metric, plot_dir))

    if any(key[2] == "Success" for key in results):
        written.append(draw_metric_lines(results, agents, "Success", plot_dir))

    print("Generated plots:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
