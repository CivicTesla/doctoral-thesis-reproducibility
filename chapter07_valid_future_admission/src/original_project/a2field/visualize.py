from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .field import project_max, project_sum
from .grid import Grid


def save_field_heatmap(
    field2d: np.ndarray,
    grid: Grid,
    out_path: Path,
    title: str,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    masked = np.ma.array(field2d, mask=grid.obstacles)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(grid.obstacles, cmap="gray_r", alpha=0.35, interpolation="nearest")
    im = ax.imshow(masked, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, shrink=0.82, label="pressure")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_field_projections(field: np.ndarray, grid: Grid, out_dir: Path, prefix: str, title: str) -> None:
    save_field_heatmap(project_sum(field), grid, out_dir / f"{prefix}_sum.png", f"{title} time-sum")
    save_field_heatmap(project_max(field), grid, out_dir / f"{prefix}_max.png", f"{title} time-max")


def save_comm_diagnostics(
    R_full: np.ndarray,
    R_comm: np.ndarray,
    grid: Grid,
    out_dir: Path,
    policy: str,
    title: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_policy = policy.replace(".", "p")
    full_sum = project_sum(R_full)
    full_max = project_max(R_full)
    comm_sum = project_sum(R_comm)
    comm_max = project_max(R_comm)
    save_field_heatmap(
        comm_sum,
        grid,
        out_dir / f"R_comm_{safe_policy}_sum_common.png",
        f"{title} sum common",
        vmax=max(float(np.max(full_sum)), 1e-12),
    )
    save_field_heatmap(
        comm_max,
        grid,
        out_dir / f"R_comm_{safe_policy}_max_common.png",
        f"{title} max common",
        vmax=max(float(np.max(full_max)), 1e-12),
    )
    save_field_heatmap(comm_sum, grid, out_dir / f"R_comm_{safe_policy}_sum_auto.png", f"{title} sum auto")
    save_field_heatmap(comm_max, grid, out_dir / f"R_comm_{safe_policy}_max_auto.png", f"{title} max auto")
    save_field_heatmap(
        project_sum(np.maximum(R_full - R_comm, 0.0)),
        grid,
        out_dir / f"hidden_{safe_policy}.png",
        f"{title} hidden map",
        cmap="viridis",
    )
    save_field_heatmap(
        project_sum(np.maximum(R_comm - R_full, 0.0)),
        grid,
        out_dir / f"ghost_{safe_policy}.png",
        f"{title} ghost map",
        cmap="inferno",
    )


def save_comparison_bars(df, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["field_fidelity_score", "ghost_pressure", "hidden_congestion", "entropy_diff"]
    means = df.groupby("policy", sort=False)[metrics].mean()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.ravel(), metrics):
        means[metric].plot(kind="bar", ax=ax, color="#4c78a8")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_comparison_bars_mean_std(df, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["field_fidelity_score", "ghost_pressure", "hidden_congestion", "top_k_peak_overlap"]
    grouped = df.groupby("policy", sort=False)[metrics].agg(["mean", "std"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, metric in zip(axes.ravel(), metrics):
        means = grouped[(metric, "mean")]
        stds = grouped[(metric, "std")].fillna(0.0)
        ax.bar(means.index, means.values, yerr=stds.values, capsize=3, color="#4c78a8")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_structure_bars_mean_std(df, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["active_shape_fidelity_score", "hidden_congestion", "ghost_pressure", "top_k_peak_overlap_tolerant"]
    grouped = df.groupby("policy", sort=False)[metrics].agg(["mean", "std"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, metric in zip(axes.ravel(), metrics):
        means = grouped[(metric, "mean")]
        stds = grouped[(metric, "std")].fillna(0.0)
        ax.bar(means.index, means.values, yerr=stds.values, capsize=3, color="#4c78a8")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=40)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_mass_bars_mean_std(df, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["field_mass_ratio", "uploaded_tube_points"]
    grouped = df.groupby("policy", sort=False)[metrics].agg(["mean", "std"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, metric in zip(axes, metrics):
        means = grouped[(metric, "mean")]
        stds = grouped[(metric, "std")].fillna(0.0)
        ax.bar(means.index, means.values, yerr=stds.values, capsize=3, color="#72b7b2")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=40)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_representative_fields_common(
    fields: dict[str, np.ndarray],
    uploaded_points: dict[str, int],
    mass_ratios: dict[str, float],
    grid: Grid,
    out_path: Path,
) -> None:
    full_key = "full_sharing" if "full_sharing" in fields else "full"
    wanted = [full_key, "random_budget", "wave_critical", "wave_critical_diverse"]
    available = [p for p in wanted if p in fields]
    if not available:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_sum = project_sum(fields[full_key]) if full_key in fields else project_sum(next(iter(fields.values())))
    vmax = max(float(np.max(full_sum)), 1e-12)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, policy in zip(axes.ravel(), available):
        data = np.ma.array(project_sum(fields[policy]), mask=grid.obstacles)
        ax.imshow(grid.obstacles, cmap="gray_r", alpha=0.35, interpolation="nearest")
        im = ax.imshow(data, cmap="magma", interpolation="nearest", vmin=0.0, vmax=vmax)
        label = "R_full" if policy == full_key else policy
        ax.set_title(f"{label} uploaded={uploaded_points.get(policy, 0)} mass={mass_ratios.get(policy, 0.0):.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    for ax in axes.ravel()[len(available):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="time-sum pressure")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_debug_difference_maps(
    fields: dict[str, np.ndarray],
    grid: Grid,
    out_dir: Path,
    policies: list[str],
) -> None:
    if "full" not in fields:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    R_full = fields["full"]
    for policy in policies:
        if policy not in fields:
            continue
        save_field_heatmap(
            project_sum(np.maximum(R_full - fields[policy], 0.0)),
            grid,
            out_dir / f"hidden_map_{policy}.png",
            f"hidden map {policy}",
            cmap="viridis",
        )
        save_field_heatmap(
            project_sum(np.maximum(fields[policy] - R_full, 0.0)),
            grid,
            out_dir / f"ghost_map_{policy}.png",
            f"ghost map {policy}",
            cmap="inferno",
        )


def save_sweep_plot(df, x_col: str, out_path: Path, title: str) -> None:
    if x_col not in df.columns or df.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped = df.groupby([x_col, "policy"])["field_fidelity_score"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy, sub in grouped.groupby("policy"):
        sub = sub.sort_values(x_col)
        ax.plot(sub[x_col], sub["field_fidelity_score"], marker="o", label=policy)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel("mean field_fidelity_score")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_sweep_pair(df, x_col: str, out_path: Path, title: str, metrics: tuple[str, str]) -> None:
    if x_col not in df.columns or df.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric in zip(axes, metrics):
        grouped = df.groupby([x_col, "policy"])[metric].mean().reset_index()
        for policy, sub in grouped.groupby("policy"):
            sub = sub.sort_values(x_col)
            ax.plot(sub[x_col], sub[metric], marker="o", label=policy)
        ax.set_title(metric)
        ax.set_xlabel(x_col)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mean")
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_budget_sweep_summary(df, out_path: Path) -> None:
    if "budget_ratio" not in df.columns or df.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["shape_fidelity_score", "hidden_congestion", "ghost_pressure", "top_k_peak_overlap_tolerant"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, metric in zip(axes.ravel(), metrics):
        grouped = df.groupby(["budget_ratio", "policy"])[metric].mean().reset_index()
        for policy, sub in grouped.groupby("policy"):
            sub = sub.sort_values("budget_ratio")
            ax.plot(sub["budget_ratio"], sub[metric], marker="o", label=policy)
        ax.set_title(metric)
        ax.set_xlabel("budget_ratio")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("mean")
    axes[1, 0].set_ylabel("mean")
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_e3_budget_structure(df, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        "active_shape_fidelity_score",
        "hidden_congestion",
        "ghost_pressure",
        "top_k_peak_overlap_tolerant",
        "entropy_diff",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))]
    for ax, metric in zip(axes.ravel(), metrics):
        grouped = df.groupby(["budget_points", "policy"])[metric].agg(["mean", "std"]).reset_index()
        for idx, (policy, sub) in enumerate(grouped.groupby("policy")):
            sub = sub.sort_values("budget_points")
            ax.errorbar(
                sub["budget_points"],
                sub["mean"],
                yerr=sub["std"].fillna(0.0),
                marker="o",
                markersize=4,
                linewidth=1.5,
                capsize=2,
                linestyle=linestyles[idx % len(linestyles)],
                label=policy,
            )
        ax.set_title(metric)
        ax.set_xlabel("budget_points")
        ax.grid(alpha=0.25)
    axes.ravel()[-1].axis("off")
    axes[0, 1].legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_e3_wave_vs_random_delta(df, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "delta_hidden": "hidden_congestion",
        "delta_peak_overlap": "top_k_peak_overlap_tolerant",
        "delta_active_shape": "active_shape_fidelity_score",
        "delta_mass": "field_mass_ratio",
        "delta_entropy": "entropy_diff",
    }
    grouped = df.groupby(["budget_points", "policy"]).agg({v: ["mean", "std"] for v in metrics.values()})
    budgets = sorted(df["budget_points"].unique())
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    for ax, (delta_name, metric) in zip(axes.ravel(), metrics.items()):
        values = []
        errors = []
        for budget in budgets:
            if (budget, "random_budget") not in grouped.index or (budget, "wave_critical") not in grouped.index:
                values.append(np.nan)
                errors.append(0.0)
                continue
            rb_mean = float(grouped.loc[(budget, "random_budget"), (metric, "mean")])
            wc_mean = float(grouped.loc[(budget, "wave_critical"), (metric, "mean")])
            rb_std = float(grouped.loc[(budget, "random_budget"), (metric, "std")])
            wc_std = float(grouped.loc[(budget, "wave_critical"), (metric, "std")])
            rb_std = 0.0 if np.isnan(rb_std) else rb_std
            wc_std = 0.0 if np.isnan(wc_std) else wc_std
            values.append(wc_mean - rb_mean)
            errors.append(float(np.sqrt(rb_std * rb_std + wc_std * wc_std)))
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
        ax.errorbar(budgets, values, yerr=errors, marker="o", capsize=3, linewidth=1.5)
        ax.set_title(delta_name)
        ax.set_xlabel("budget_points")
        ax.grid(alpha=0.25)
    axes.ravel()[-1].axis("off")
    fig.suptitle("wave_critical - random_budget deltas")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_e3_budget_mass(df, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    grouped = df.groupby(["budget_points", "policy"])["field_mass_ratio"].mean().reset_index()
    for idx, (policy, sub) in enumerate(grouped.groupby("policy")):
        sub = sub.sort_values("budget_points")
        axes[0].plot(
            sub["budget_points"],
            sub["field_mass_ratio"],
            marker="o",
            linestyle=linestyles[idx % len(linestyles)],
            label=policy,
        )
    axes[0].set_title("field_mass_ratio")
    axes[0].set_xlabel("budget_points")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left", borderaxespad=0)

    budget_points = np.array(sorted(df["budget_points"].unique()), dtype=float)
    total_tube_points = float(df["total_tube_points"].max()) if "total_tube_points" in df.columns else float(budget_points.max())
    axes[1].plot(budget_points, np.zeros_like(budget_points), marker="o", linestyle="-", label="no_sharing")
    axes[1].plot(
        budget_points,
        np.full_like(budget_points, total_tube_points),
        marker="o",
        linestyle="--",
        label="full_sharing",
    )
    axes[1].plot(budget_points, budget_points, marker="o", linestyle="-.", label="fair_budget_target")
    axes[1].set_title("uploaded_tube_points sanity check")
    axes[1].set_xlabel("budget_points")
    axes[1].set_ylabel("uploaded_tube_points")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left", borderaxespad=0)
    axes[1].text(
        0.0,
        -0.22,
        "fixed_horizon, random_budget, freshness_based, partial_access, and wave_critical\n"
        "are budget matched and overlap with fair_budget_target.",
        transform=axes[1].transAxes,
        fontsize=8,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    fair_policies = ["fixed_horizon", "random_budget", "freshness_based", "partial_access", "wave_critical"]
    fair = df[df["policy"].isin(fair_policies)]
    if not fair.empty:
        fair_out = out_path.with_name(f"{out_path.stem}_fair_only{out_path.suffix}")
        fig, ax = plt.subplots(figsize=(8, 5))
        grouped = fair.groupby(["budget_points", "policy"])["field_mass_ratio"].mean().reset_index()
        for idx, (policy, sub) in enumerate(grouped.groupby("policy")):
            sub = sub.sort_values("budget_points")
            ax.plot(
                sub["budget_points"],
                sub["field_mass_ratio"],
                marker="o",
                linestyle=linestyles[idx % len(linestyles)],
                label=policy,
            )
        ax.set_title("field_mass_ratio fair-budget policies")
        ax.set_xlabel("budget_points")
        ax.set_ylabel("field_mass_ratio")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left", borderaxespad=0)
        fig.tight_layout()
        fig.savefig(fair_out, dpi=150, bbox_inches="tight")
        plt.close(fig)


def save_e3_representative_fields(
    fields: dict[str, np.ndarray],
    uploaded_points: dict[str, int],
    mass_ratios: dict[str, float],
    grid: Grid,
    out_path: Path,
) -> None:
    wanted = ["full_sharing", "random_budget", "freshness_based", "fixed_horizon", "partial_access", "wave_critical"]
    available = [p for p in wanted if p in fields]
    if not available:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_sum = project_sum(fields["full_sharing"] if "full_sharing" in fields else next(iter(fields.values())))
    vmax = max(float(np.max(full_sum)), 1e-12)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, policy in zip(axes.ravel(), available):
        data = np.ma.array(project_sum(fields[policy]), mask=grid.obstacles)
        ax.imshow(grid.obstacles, cmap="gray_r", alpha=0.35, interpolation="nearest")
        im = ax.imshow(data, cmap="magma", interpolation="nearest", vmin=0.0, vmax=vmax)
        label = "R_full" if policy == "full_sharing" else policy
        ax.set_title(f"{label}\nuploaded={uploaded_points.get(policy, 0)} mass={mass_ratios.get(policy, 0.0):.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    for ax in axes.ravel()[len(available):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="time-sum pressure")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_e2b_ghost_width_sweep(df, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["ghost_pressure", "hidden_congestion", "field_mass_ratio", "entropy_diff"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    linestyles = ["-", "--", "-."]
    for ax, metric in zip(axes.ravel(), metrics):
        grouped = df.groupby(["width_growth", "policy"])[metric].agg(["mean", "std"]).reset_index()
        for idx, (policy, sub) in enumerate(grouped.groupby("policy")):
            sub = sub.sort_values("width_growth")
            ax.errorbar(
                sub["width_growth"],
                sub["mean"],
                yerr=sub["std"].fillna(0.0),
                marker="o",
                capsize=3,
                linestyle=linestyles[idx % len(linestyles)],
                label=policy,
            )
        ax.set_title(metric)
        ax.set_xlabel("width_growth")
        ax.grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_e2b_representative_ghost_maps(
    cases: list[tuple[float, np.ndarray, np.ndarray, Grid]],
    out_path: Path,
) -> None:
    if not cases:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(cases), 4, figsize=(17, 4.2 * len(cases)))
    if len(cases) == 1:
        axes = np.array([axes])
    for row, (width_growth, R_full, R_comm, grid) in enumerate(cases):
        full_sum = project_sum(R_full)
        comm_sum = project_sum(R_comm)
        ghost_sum = project_sum(np.maximum(R_comm - R_full, 0.0))
        hidden_sum = project_sum(np.maximum(R_full - R_comm, 0.0))
        common_vmax = max(float(np.max(full_sum)), 1e-12)
        panels = [
            ("R_full", full_sum, "magma", 0.0, common_vmax),
            ("R_comm", comm_sum, "magma", 0.0, common_vmax),
            ("ghost_map", ghost_sum, "inferno", None, None),
            ("hidden_map", hidden_sum, "viridis", None, None),
        ]
        for col, (title, data, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            masked = np.ma.array(data, mask=grid.obstacles)
            ax.imshow(grid.obstacles, cmap="gray_r", alpha=0.35, interpolation="nearest")
            im = ax.imshow(masked, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
            ax.set_title(f"{title}\nwidth_growth={width_growth}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_e2b_ghost_hidden_tradeoff(df, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped = df.groupby(["width_growth", "policy"])[["hidden_congestion", "ghost_pressure"]].mean().reset_index()
    markers = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, (policy, sub) in enumerate(grouped.groupby("policy")):
        sub = sub.sort_values("width_growth")
        ax.plot(
            sub["hidden_congestion"],
            sub["ghost_pressure"],
            marker=markers[idx % len(markers)],
            label=policy,
        )
        for _, row in sub.iterrows():
            ax.annotate(f"{row['width_growth']:.2f}", (row["hidden_congestion"], row["ghost_pressure"]), fontsize=7)
    ax.set_xlabel("hidden_congestion")
    ax.set_ylabel("ghost_pressure")
    ax.set_title("ghost vs hidden tradeoff")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_e2c_width_tradeoff_summary(df, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        "hidden_congestion",
        "ghost_pressure",
        "balanced_loss_g1",
        "balanced_loss_g2",
        "field_mass_ratio",
        "entropy_diff",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    linestyles = ["-", "--", "-."]
    for ax, metric in zip(axes.ravel(), metrics):
        grouped = df.groupby(["width_growth", "policy"])[metric].agg(["mean", "std"]).reset_index()
        for idx, (policy, sub) in enumerate(grouped.groupby("policy")):
            sub = sub.sort_values("width_growth")
            ax.errorbar(
                sub["width_growth"],
                sub["mean"],
                yerr=sub["std"].fillna(0.0),
                marker="o",
                capsize=3,
                linestyle=linestyles[idx % len(linestyles)],
                label=policy,
            )
        ax.set_title(metric)
        ax.set_xlabel("width_growth")
        ax.grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_e2c_balanced_fidelity(df, best_df, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["balanced_fidelity_score_g1", "balanced_fidelity_score_g2"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, metrics):
        grouped = df.groupby(["width_growth", "policy"])[metric].mean().reset_index()
        for policy, sub in grouped.groupby("policy"):
            sub = sub.sort_values("width_growth")
            ax.plot(sub["width_growth"], sub[metric], marker="o", label=policy)
            loss_col = "best_width_growth_g1" if metric.endswith("g1") else "best_width_growth_g2"
            row = best_df[best_df["policy"] == policy]
            if not row.empty:
                ax.axvline(float(row[loss_col].iloc[0]), alpha=0.25, linestyle="--")
        ax.set_title(metric)
        ax.set_xlabel("width_growth")
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_e2c_pareto_ghost_hidden(df, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped = df.groupby(["width_growth", "policy"])[["hidden_congestion", "ghost_pressure"]].mean().reset_index()
    markers = ["o", "s", "^"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, (policy, sub) in enumerate(grouped.groupby("policy")):
        sub = sub.sort_values("width_growth")
        ax.plot(sub["hidden_congestion"], sub["ghost_pressure"], marker=markers[idx % len(markers)], label=policy)
        for _, row in sub.iterrows():
            ax.annotate(f"{row['width_growth']:.2f}", (row["hidden_congestion"], row["ghost_pressure"]), fontsize=7)
    ax.set_xlabel("hidden_congestion")
    ax.set_ylabel("ghost_pressure")
    ax.set_title("E2C ghost-hidden tradeoff")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, bbox_to_anchor=(1.04, 1.0), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_e2c_representative_maps(cases: list[tuple[str, float, np.ndarray, np.ndarray, Grid]], out_path: Path) -> None:
    if not cases:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(cases), 4, figsize=(17, 4.2 * len(cases)))
    if len(cases) == 1:
        axes = np.array([axes])
    for row, (label, width_growth, R_full, R_comm, grid) in enumerate(cases):
        full_sum = project_sum(R_full)
        comm_sum = project_sum(R_comm)
        ghost_sum = project_sum(np.maximum(R_comm - R_full, 0.0))
        hidden_sum = project_sum(np.maximum(R_full - R_comm, 0.0))
        common_vmax = max(float(np.max(full_sum)), 1e-12)
        panels = [
            ("R_full", full_sum, "magma", 0.0, common_vmax),
            ("R_comm", comm_sum, "magma", 0.0, common_vmax),
            ("ghost_map", ghost_sum, "inferno", None, None),
            ("hidden_map", hidden_sum, "viridis", None, None),
        ]
        for col, (title, data, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            masked = np.ma.array(data, mask=grid.obstacles)
            ax.imshow(grid.obstacles, cmap="gray_r", alpha=0.35, interpolation="nearest")
            im = ax.imshow(masked, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
            ax.set_title(f"{title}\n{label}, width={width_growth}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _curve_label(row) -> str:
    policy = row["policy"]
    mode = row["radius_mode"]
    group = row.get("group", "")
    if group:
        return f"{group}:{mode}"
    return f"{policy}:{mode}"


def save_e4_horizon_tradeoff(df, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        "hidden_congestion",
        "ghost_pressure",
        "balanced_loss_g1",
        "balanced_loss_g2",
        "field_mass_ratio",
        "entropy_diff",
    ]
    work = df.copy()
    work["curve"] = work.apply(_curve_label, axis=1)
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "P"]
    for ax, metric in zip(axes.ravel(), metrics):
        grouped = work.groupby(["upload_horizon", "curve"])[metric].agg(["mean", "std"]).reset_index()
        for idx, (curve, sub) in enumerate(grouped.groupby("curve")):
            sub = sub.sort_values("upload_horizon")
            ax.errorbar(
                sub["upload_horizon"],
                sub["mean"],
                yerr=sub["std"].fillna(0.0),
                marker=markers[idx % len(markers)],
                capsize=3,
                linestyle=linestyles[idx % len(linestyles)],
                label=curve,
            )
        ax.set_title(metric)
        ax.set_xlabel("upload_horizon")
        ax.grid(alpha=0.25)
    axes[0, 1].legend(fontsize=7, bbox_to_anchor=(1.04, 1.0), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_e4_balanced_fidelity(df, best_df, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = df.copy()
    work["curve"] = work.apply(_curve_label, axis=1)
    best = best_df.copy()
    best["curve"] = best.apply(_curve_label, axis=1)
    metrics = ["balanced_fidelity_score_g1", "balanced_fidelity_score_g2"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, metric in zip(axes, metrics):
        grouped = work.groupby(["upload_horizon", "curve"])[metric].mean().reset_index()
        for curve, sub in grouped.groupby("curve"):
            sub = sub.sort_values("upload_horizon")
            ax.plot(sub["upload_horizon"], sub[metric], marker="o", label=curve)
            col = "best_upload_horizon_g1" if metric.endswith("g1") else "best_upload_horizon_g2"
            row = best[best["curve"] == curve]
            if not row.empty:
                ax.axvline(float(row[col].iloc[0]), alpha=0.22, linestyle="--")
        ax.set_title(metric)
        ax.set_xlabel("upload_horizon")
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.04, 1.0), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_e4_pareto_ghost_hidden(df, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = df.copy()
    work["curve"] = work.apply(_curve_label, axis=1)
    grouped = work.groupby(["upload_horizon", "curve"])[["hidden_congestion", "ghost_pressure"]].mean().reset_index()
    markers = ["o", "s", "^", "D", "v", "P"]
    fig, ax = plt.subplots(figsize=(9, 6))
    for idx, (curve, sub) in enumerate(grouped.groupby("curve")):
        sub = sub.sort_values("upload_horizon")
        ax.plot(sub["hidden_congestion"], sub["ghost_pressure"], marker=markers[idx % len(markers)], label=curve)
        for _, row in sub.iterrows():
            ax.annotate(f"H{int(row['upload_horizon'])}", (row["hidden_congestion"], row["ghost_pressure"]), fontsize=7)
    ax.set_xlabel("hidden_congestion")
    ax.set_ylabel("ghost_pressure")
    ax.set_title("E4 ghost-hidden horizon tradeoff")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, bbox_to_anchor=(1.04, 1.0), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_e4_representative_maps(cases: list[tuple[str, int, np.ndarray, np.ndarray, Grid]], out_path: Path) -> None:
    if not cases:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(cases), 4, figsize=(17, 3.6 * len(cases)))
    if len(cases) == 1:
        axes = np.array([axes])
    for row, (label, horizon, R_full, R_comm, grid) in enumerate(cases):
        full_sum = project_sum(R_full)
        comm_sum = project_sum(R_comm)
        ghost_sum = project_sum(np.maximum(R_comm - R_full, 0.0))
        hidden_sum = project_sum(np.maximum(R_full - R_comm, 0.0))
        common_vmax = max(float(np.max(full_sum)), 1e-12)
        panels = [
            ("R_full", full_sum, "magma", 0.0, common_vmax),
            ("R_comm", comm_sum, "magma", 0.0, common_vmax),
            ("ghost_map", ghost_sum, "inferno", None, None),
            ("hidden_map", hidden_sum, "viridis", None, None),
        ]
        for col, (title, data, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            masked = np.ma.array(data, mask=grid.obstacles)
            ax.imshow(grid.obstacles, cmap="gray_r", alpha=0.35, interpolation="nearest")
            im = ax.imshow(masked, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
            ax.set_title(f"{title}\n{label}, H={horizon}", fontsize=9)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, shrink=0.72)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
