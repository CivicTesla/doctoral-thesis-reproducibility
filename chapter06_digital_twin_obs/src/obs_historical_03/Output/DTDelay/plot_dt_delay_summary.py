#!/usr/bin/env python3
"""Plot DT synchronization delay summary figures."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def find_paths():
    script_dir = Path(__file__).resolve().parent

    if (script_dir / "DTDelay-summary.json").exists():
        input_path = script_dir / "DTDelay-summary.json"
        output_dir = script_dir
    else:
        output_dir = script_dir / "Output" / "DTDelay"
        input_path = output_dir / "DTDelay-summary.json"

    return input_path, output_dir


def print_summary_table(rows):
    columns = ["tau", "rounds_median", "covmoves_median", "success_rate"]
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }

    header = "  ".join(column.rjust(widths[column]) for column in columns)
    separator = "  ".join("-" * widths[column] for column in columns)

    print(header)
    print(separator)
    for row in rows:
        print("  ".join(str(row[column]).rjust(widths[column]) for column in columns))


def save_line_plot(x_values, y_values, title, ylabel, output_path, ylim=None):
    plt.figure()
    plt.plot(x_values, y_values, marker="o")
    plt.xlabel("tau")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    input_path, output_dir = find_paths()

    with input_path.open("r", encoding="utf-8") as file:
        rows = json.load(file)

    rows = sorted(rows, key=lambda row: row["tau"])

    tau = [row["tau"] for row in rows]
    rounds_median = [row["rounds_median"] for row in rows]
    covmoves_median = [row["covmoves_median"] for row in rows]
    success_rate = [row["success_rate"] for row in rows]

    print_summary_table(rows)

    output_dir.mkdir(parents=True, exist_ok=True)

    save_line_plot(
        tau,
        rounds_median,
        "DT Synchronization Delay vs Median Convergence Rounds",
        "rounds_median",
        output_dir / "DTDelay-MedianRounds.png",
    )
    save_line_plot(
        tau,
        covmoves_median,
        "DT Synchronization Delay vs Median Coverage Moves",
        "covmoves_median",
        output_dir / "DTDelay-MedianCovMoves.png",
    )
    save_line_plot(
        tau,
        success_rate,
        "DT Synchronization Delay vs Success Rate",
        "success_rate",
        output_dir / "DTDelay-SuccessRate.png",
        ylim=(0, 1.05),
    )


if __name__ == "__main__":
    main()
