"""Plot fold-paired RRR-minus-PCA gaps for selected reconstruction runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd


METRICS = {
    "multivariate_r2": "Multivariate R² gap",
    "mean_coordinate_correlation": "Mean coordinate correlation gap",
    "linear_cka": "Linear CKA gap",
}


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Runs must use LABEL=PATH syntax")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Runs must use non-empty LABEL=PATH values")
    path = Path(raw_path).expanduser()
    if path.is_dir() and not (path / "fold_results.csv").exists():
        matches = list(path.glob("language_reconstruction/within_dataset/*/fold_results.csv"))
        if len(matches) == 1:
            path = matches[0].parent
    if not (path / "fold_results.csv").exists():
        raise argparse.ArgumentTypeError(f"No fold_results.csv found under {path}")
    return label.strip(), path.resolve()


def build_gap_tables(runs: list[tuple[str, Path]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for task, path in runs:
        values = pd.read_csv(path / "fold_results.csv")
        if "model" in values:
            values = values[values["model"] == "linear"]
        if "predictor_set" not in values:
            values["predictor_set"] = "tasks"
        for metric in METRICS:
            paired = values.pivot_table(
                index=["predictor_set", "target_view", "rank", "fold"],
                columns="subspace",
                values=metric,
            ).dropna(subset=["rrr", "pca"])
            for (predictor_set, target_view, rank, fold), row in paired.iterrows():
                rows.append(
                    {
                        "task": task,
                        "predictor_set": predictor_set,
                        "target_view": target_view,
                        "rank": int(rank),
                        "fold": int(fold),
                        "metric": metric,
                        "rrr": float(row["rrr"]),
                        "pca": float(row["pca"]),
                        "gap": float(row["rrr"] - row["pca"]),
                    }
                )
    grid = pd.DataFrame(rows)
    summary = (
        grid.groupby(
            ["task", "predictor_set", "target_view", "rank", "metric"],
            sort=False,
        ).gap
        .agg(["mean", "sem", "count"])
        .reset_index()
        .rename(columns={"count": "n_folds"})
    )
    return grid, summary


def plot_gap_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    series_order = list(
        dict.fromkeys(zip(summary["task"], summary["predictor_set"]))
    )
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(series_order), 2)))
    for metric, ylabel in METRICS.items():
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
        for axis, target_view in zip(axes, ("weighted", "span")):
            subset = summary[
                (summary["metric"] == metric)
                & (summary["target_view"] == target_view)
            ]
            for color, (task, predictor_set) in zip(colors, series_order):
                group = subset[
                    (subset["task"] == task)
                    & (subset["predictor_set"] == predictor_set)
                ].sort_values("rank")
                if group.empty:
                    continue
                ranks = group["rank"].to_numpy(dtype=float)
                means = group["mean"].to_numpy(dtype=float)
                sems = group["sem"].fillna(0).to_numpy(dtype=float)
                label = task if predictor_set == "tasks" else f"{task}/{predictor_set}"
                axis.plot(ranks, means, marker="o", color=color, label=label)
                axis.fill_between(ranks, means - sems, means + sems, color=color, alpha=0.16)
            axis.axhline(0, color="black", linewidth=1, linestyle="--")
            axis.set_xscale("log", base=2)
            axis.set_xticks(sorted(subset["rank"].unique()))
            axis.get_xaxis().set_major_formatter(ScalarFormatter())
            axis.get_xaxis().set_minor_formatter(NullFormatter())
            axis.set_title(f"{target_view.capitalize()} target")
            axis.set_xlabel("Rank")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel(f"RRR − PCA {ylabel}")
        axes[-1].legend(fontsize=8, frameon=False)
        figure.tight_layout()
        figure.savefig(output_dir / f"selected_task_gap_{metric}.png", dpi=180)
        figure.savefig(output_dir / f"selected_task_gap_{metric}.svg")
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=_parse_run)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    grid, summary = build_gap_tables(args.run)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.output_dir / "selected_task_gap_grid.csv", index=False)
    summary.to_csv(args.output_dir / "selected_task_gap_summary.csv", index=False)
    plot_gap_summary(summary, args.output_dir)
    print(f"Done: {args.output_dir}")


if __name__ == "__main__":
    main()
