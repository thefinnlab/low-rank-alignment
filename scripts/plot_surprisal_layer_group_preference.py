#!/usr/bin/env python
"""Compare language-task RRR/PCA preference across causal-model layer groups."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.plot_mean_language_decoding import _relative_preference
from src.utils.plot_helpers import apply_publication_style


DATASETS = ("podcast", "cars-2", "fmri")
CAUSAL_FINAL_LAYERS = {
    "gpt2": 12,
    "gpt2-xl": 48,
    "gpt-j-6b": 28,
    "llama3-8b": 32,
}
DATASET_COLORS = {
    "podcast": "#0072B2",
    "cars-2": "#D55E00",
    "fmri": "#009E73",
}
GROUP_STYLES = {
    "first four depths": {"linestyle": "-", "marker": "o"},
    "final layer": {"linestyle": "--", "marker": "s"},
    "first three depths": {"linestyle": "-", "marker": "o"},
    "final two depths": {"linestyle": "--", "marker": "s"},
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grouping",
        choices=("first-four-vs-final", "first-three-vs-final-two"),
        default="first-four-vs-final",
    )
    parser.add_argument(
        "--metric",
        choices=("relative_preference", "smooth_relative_preference", "delta"),
        default="relative_preference",
        help="Y-axis quantity; delta is robust when scores are near or below zero",
    )
    parser.add_argument(
        "--normalization-tau", type=float, default=0.05,
        help="Stabilizer for smooth_relative_preference (default: 0.05)",
    )
    parser.add_argument(
        "--run-name", default="predictability-context-quick-noiu",
        help="Bootstrap result subdirectory to read within each dataset",
    )
    parser.add_argument(
        "--task", choices=("self_surprisal", "sentence_position"),
        default="self_surprisal",
    )
    parser.add_argument(
        "--output-dir",
        default="results/language_decoding_surprisal_layer_groups",
        help="Directory for the combined CSV, metadata, PNG, and SVG",
    )
    parser.add_argument(
        "--low-performance-threshold", type=float, default=0.05,
        help="Set preference to zero when both averaged scores are below this value",
    )
    return parser.parse_args()


def summarize_layer_groups(
    frame: pd.DataFrame, threshold: float,
    grouping: str = "first-four-vs-final",
    normalization_tau: float = 0.05,
    task: str = "self_surprisal",
) -> pd.DataFrame:
    """Average configured sampled-depth groups within paired causal models."""
    required = {"model", "layer", "rank", "task", "pca_score", "rrr_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Bootstrap summary is missing columns: {sorted(missing)}")
    selected = frame.loc[
        (frame["task"] == task)
        & frame["model"].isin(CAUSAL_FINAL_LAYERS)
    ].copy()
    if selected.empty:
        raise ValueError(f"Bootstrap summary has no causal-model {task} rows")

    complete_models = []
    for model, final_layer in CAUSAL_FINAL_LAYERS.items():
        layers = set(selected.loc[selected["model"] == model, "layer"])
        expected = {final_layer * step // 4 for step in range(5)}
        if expected <= layers:
            complete_models.append(model)
    if not complete_models:
        raise ValueError("No causal model has both non-final and final-layer results")
    selected = selected.loc[selected["model"].isin(complete_models)]
    if grouping == "first-four-vs-final":
        selected["layer_group"] = [
            "final layer" if layer == CAUSAL_FINAL_LAYERS[model]
            else "first four depths"
            for model, layer in zip(selected["model"], selected["layer"])
        ]
    elif grouping == "first-three-vs-final-two":
        selected["layer_group"] = [
            "first three depths" if layer <= CAUSAL_FINAL_LAYERS[model] // 2
            else "final two depths"
            for model, layer in zip(selected["model"], selected["layer"])
        ]
    else:
        raise ValueError(f"Unknown grouping: {grouping}")

    model_rank = selected.groupby(
        ["model", "layer_group", "rank"], as_index=False
    ).agg(
        pca_score=("pca_score", "mean"),
        rrr_score=("rrr_score", "mean"),
        n_layers=("layer", "nunique"),
    )
    grouped = model_rank.groupby(["layer_group", "rank"], as_index=False).agg(
        pca_score=("pca_score", "mean"),
        rrr_score=("rrr_score", "mean"),
        n_models=("model", "nunique"),
        n_layers=("n_layers", "sum"),
    )
    grouped["relative_preference"] = [
        _relative_preference(rrr, pca, threshold)
        for rrr, pca in zip(grouped["rrr_score"], grouped["pca_score"])
    ]
    grouped["delta"] = grouped["rrr_score"] - grouped["pca_score"]
    grouped["smooth_relative_preference"] = (
        2 * grouped["delta"]
        / (
            np.sqrt(grouped["rrr_score"] ** 2 + normalization_tau ** 2)
            + np.sqrt(grouped["pca_score"] ** 2 + normalization_tau ** 2)
        )
    )
    grouped["models"] = ",".join(sorted(complete_models))
    return grouped


def _plot(
    frame: pd.DataFrame, task: str, grouping: str, metric: str,
    destination: Path,
) -> None:
    apply_publication_style()
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    groups = (
        ("first four depths", "final layer")
        if grouping == "first-four-vs-final"
        else ("first three depths", "final two depths")
    )
    for dataset in DATASETS:
        for layer_group in groups:
            selected = frame.loc[
                (frame["dataset"] == dataset)
                & (frame["layer_group"] == layer_group)
            ].sort_values("rank")
            style = GROUP_STYLES[layer_group]
            axis.semilogx(
                selected["rank"], selected[metric],
                color=DATASET_COLORS[dataset],
                linestyle=style["linestyle"], marker=style["marker"],
                markersize=3.5, linewidth=1.7,
                label=f"{dataset} · {layer_group}",
            )
    values = frame[metric].to_numpy(float)
    readable = values[np.isfinite(values) & (np.abs(values) <= 5)]
    if len(readable):
        span = max(float(np.ptp(readable)), .2)
        lower = float(np.min(readable) - .08 * span)
        upper = float(np.max(readable) + .08 * span)
        axis.set_ylim(lower, upper)
        off_scale = frame.loc[
            (frame[metric] < lower) | (frame[metric] > upper)
        ]
        notes = []
        for row in off_scale.itertuples(index=False):
            value = getattr(row, metric)
            boundary = lower if value < lower else upper
            marker = "v" if value < lower else "^"
            axis.scatter(
                row.rank, boundary, marker=marker, s=28,
                color=DATASET_COLORS[row.dataset], zorder=8, clip_on=False,
            )
            notes.append(
                f"{row.dataset} {row.layer_group}, k={row.rank}: "
                f"{value:.1f} (off-scale)"
            )
        if notes:
            axis.text(
                .01, .015, "\n".join(notes), transform=axis.transAxes,
                fontsize=6.5, va="bottom", ha="left",
            )
    axis.axhline(0, color="black", linewidth=.8, alpha=.6)
    axis.set_xlabel("Rank $k$ (log scale)")
    axis.set_ylabel(
        "RRR - PCA held-out correlation"
        if metric == "delta"
        else (
            "Smooth relative preference"
            if metric == "smooth_relative_preference"
            else "Relative preference: (RRR - PCA) / min(RRR, PCA)"
        )
    )
    title_groups = (
        "first four depths versus final layer"
        if grouping == "first-four-vs-final"
        else "first three depths versus final two depths"
    )
    task_title = task.replace("_", " ").title()
    axis.set_title(f"{task_title}: {title_groups}")
    axis.legend(loc="best", ncol=2)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=.05)
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight", pad_inches=.05)
    plt.close(fig)


def main() -> None:
    args = _args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_name):
        raise ValueError("--run-name may contain only letters, numbers, '.', '_', and '-'")
    if args.low_performance_threshold < 0:
        raise ValueError("--low-performance-threshold must be nonnegative")
    if args.normalization_tau <= 0:
        raise ValueError("--normalization-tau must be positive")

    rows = []
    inputs = {}
    for dataset in DATASETS:
        source = (
            Path("results") / dataset / "language_decoding_bootstrap"
            / args.run_name / "per_model_layer_rank.csv"
        )
        if not source.is_file():
            raise FileNotFoundError(f"Missing bootstrap summary: {source}")
        inputs[dataset] = str(source)
        summary = summarize_layer_groups(
            pd.read_csv(source), args.low_performance_threshold, args.grouping,
            args.normalization_tau, args.task,
        )
        summary.insert(0, "dataset", dataset)
        rows.append(summary)

    frame = pd.concat(rows, ignore_index=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.task}_{args.grouping.replace('-', '_')}_{args.metric}"
    frame.to_csv(output / f"{stem}.csv", index=False)
    _plot(frame, args.task, args.grouping, args.metric, output / stem)
    models_by_dataset = {
        dataset: sorted(frame.loc[frame["dataset"] == dataset, "models"].iloc[0].split(","))
        for dataset in DATASETS
    }
    (output / "metadata.json").write_text(
        json.dumps({
            "task": args.task,
            "datasets": list(DATASETS),
            "inputs": inputs,
            "models_by_dataset": models_by_dataset,
            "excluded_models": ["random-gpt2-xl", "roberta-large"],
            "grouping": args.grouping,
            "plotted_metric": args.metric,
            "smooth_relative_preference": (
                "2 * (RRR - PCA) / "
                "(sqrt(RRR^2 + tau^2) + sqrt(PCA^2 + tau^2))"
            ),
            "normalization_tau": args.normalization_tau,
            "model_inclusion": "paired: model must have both layer groups",
            "averaging": "layers within model/group, then equal-weighted models",
            "relative_preference": (
                "(mean RRR - mean PCA) / min(mean RRR, mean PCA), computed after averaging"
            ),
            "low_performance_threshold": args.low_performance_threshold,
            "inference": None,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(frame)} dataset/layer-group/rank rows and plot to {output}")


if __name__ == "__main__":
    main()
