#!/usr/bin/env python
"""Plot smooth RRR/PCA preference for one task from bootstrap summaries."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.plot_helpers import apply_publication_style


COLORS = {"podcast": "#0072B2", "cars-2": "#D55E00", "fmri": "#009E73"}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--datasets", nargs="+", default=["podcast", "fmri"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--normalization-tau", type=float, default=0.05)
    parser.add_argument(
        "--require-models", type=int, default=6,
        help="Retain only cells with this many contributing models; use 0 to disable",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def smooth_relative_preference(rrr, pca, tau: float):
    rrr = np.asarray(rrr, dtype=float)
    pca = np.asarray(pca, dtype=float)
    return 2 * (rrr - pca) / (
        np.sqrt(rrr ** 2 + tau ** 2) + np.sqrt(pca ** 2 + tau ** 2)
    )


def main() -> None:
    args = _args()
    if args.normalization_tau <= 0:
        raise ValueError("--normalization-tau must be positive")
    unknown = set(args.datasets) - set(COLORS)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")

    rows = []
    for dataset in args.datasets:
        root = Path("results") / dataset / "language_decoding_bootstrap"
        if args.run_name:
            root /= args.run_name
        source = root / "cross_model_rank.csv"
        if not source.is_file():
            raise FileNotFoundError(f"Missing bootstrap summary: {source}")
        frame = pd.read_csv(source)
        frame = frame.loc[frame["task"] == args.task].copy()
        if args.require_models:
            frame = frame.loc[frame["n_models"] == args.require_models]
        if frame.empty:
            raise ValueError(f"No retained {args.task} rows for {dataset}")
        frame["dataset"] = dataset
        frame["smooth_relative_preference"] = smooth_relative_preference(
            frame["rrr_score"], frame["pca_score"], args.normalization_tau
        )
        rows.append(frame)
    combined = pd.concat(rows, ignore_index=True)

    output = Path(
        args.output_dir
        or f"results/language_decoding_task_preference/{args.task}"
    )
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.task}_smooth_relative_preference"
    combined.to_csv(output / f"{stem}.csv", index=False)

    apply_publication_style()
    fig, axis = plt.subplots(figsize=(5.5, 3.6))
    for dataset in args.datasets:
        selected = combined.loc[combined["dataset"] == dataset].sort_values("rank")
        axis.semilogx(
            selected["rank"], selected["smooth_relative_preference"],
            marker="o", markersize=4, linewidth=1.8,
            color=COLORS[dataset], label=dataset,
        )
        significant = selected["significant_rrr_gt_pca"].astype(bool)
        axis.scatter(
            selected.loc[significant, "rank"],
            selected.loc[significant, "smooth_relative_preference"],
            marker="*", s=55, color=COLORS[dataset], edgecolor="black",
            linewidth=.35, zorder=8,
        )
    axis.axhline(0, color="black", linewidth=.8, alpha=.6)
    axis.set_xlabel("Rank $k$ (log scale)")
    axis.set_ylabel("Smooth relative preference")
    axis.set_title(f"{args.task.replace('_', ' ').title()}: RRR versus PCA")
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=.05)
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight", pad_inches=.05)
    plt.close(fig)
    print(f"Wrote {len(combined)} rows and plot to {output}")


if __name__ == "__main__":
    main()
