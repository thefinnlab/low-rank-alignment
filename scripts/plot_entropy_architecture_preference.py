#!/usr/bin/env python
"""Plot predictability-task RRR/PCA preference for causal models and RoBERTa."""

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
CAUSAL_MODELS = ("gpt2", "gpt2-xl", "gpt-j-6b", "llama3-8b")
ROBERTA_MODEL = "roberta-large"
DATASET_COLORS = {
    "podcast": "#0072B2",
    "cars-2": "#D55E00",
    "fmri": "#009E73",
}
ARCHITECTURE_STYLES = {
    "causal": {"linestyle": "-", "marker": "o"},
    "roberta": {"linestyle": "--", "marker": "s"},
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name", default="predictability-context-quick-noiu",
        help="Bootstrap result subdirectory to read within each dataset",
    )
    parser.add_argument(
        "--task", choices=("self_entropy", "self_surprisal"),
        default="self_entropy",
        help="Model-specific predictability target to plot",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: results/language_decoding_TASK_architecture)",
    )
    parser.add_argument(
        "--low-performance-threshold", type=float, default=0.05,
        help="Set preference to zero when both averaged scores are below this value",
    )
    parser.add_argument(
        "--metric", choices=("relative_preference", "smooth_relative_preference"),
        default="relative_preference",
    )
    parser.add_argument(
        "--normalization-tau", type=float, default=0.05,
        help="Stabilizer for smooth_relative_preference (default: 0.05)",
    )
    return parser.parse_args()


def summarize_architectures(
    frame: pd.DataFrame, task: str, threshold: float,
    normalization_tau: float = 0.05,
) -> pd.DataFrame:
    """Average layers within models, then models within architecture groups."""
    required = {"model", "layer", "rank", "task", "pca_score", "rrr_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Bootstrap summary is missing columns: {sorted(missing)}")
    selected = frame.loc[frame["task"] == task].copy()
    if selected.empty:
        raise ValueError(f"Bootstrap summary has no {task} rows")
    allowed = {*CAUSAL_MODELS, ROBERTA_MODEL}
    selected = selected.loc[selected["model"].isin(allowed)]
    present = set(selected["model"])
    missing_models = allowed - present
    if missing_models:
        raise ValueError(f"Missing trained models: {sorted(missing_models)}")

    model_rank = selected.groupby(["model", "rank"], as_index=False).agg(
        pca_score=("pca_score", "mean"),
        rrr_score=("rrr_score", "mean"),
        n_layers=("layer", "nunique"),
    )
    model_rank["architecture"] = np.where(
        model_rank["model"] == ROBERTA_MODEL, "roberta", "causal"
    )
    grouped = model_rank.groupby(["architecture", "rank"], as_index=False).agg(
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
    return grouped


def summarize_entropy_architectures(
    frame: pd.DataFrame, threshold: float,
) -> pd.DataFrame:
    """Backward-compatible self-entropy summary helper."""
    return summarize_architectures(frame, "self_entropy", threshold)


def _plot(frame: pd.DataFrame, task: str, metric: str, destination: Path) -> None:
    apply_publication_style()
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    for dataset in DATASETS:
        for architecture in ("causal", "roberta"):
            selected = frame.loc[
                (frame["dataset"] == dataset)
                & (frame["architecture"] == architecture)
            ].sort_values("rank")
            style = ARCHITECTURE_STYLES[architecture]
            axis.semilogx(
                selected["rank"], selected[metric],
                color=DATASET_COLORS[dataset],
                linestyle=style["linestyle"], marker=style["marker"],
                markersize=3.5, linewidth=1.7,
                label=f"{dataset} · {architecture}",
            )
    axis.axhline(0, color="black", linewidth=.8, alpha=.6)
    axis.set_xlabel("Rank $k$ (log scale)")
    axis.set_ylabel(
        "Smooth relative preference"
        if metric == "smooth_relative_preference"
        else "Relative preference: (RRR - PCA) / min(RRR, PCA)"
    )
    title = {"self_entropy": "Self-entropy", "self_surprisal": "Self-surprisal"}[task]
    axis.set_title(f"{title}: causal models versus RoBERTa")
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
        summary = summarize_architectures(
            pd.read_csv(source), args.task, args.low_performance_threshold,
            args.normalization_tau,
        )
        summary.insert(0, "dataset", dataset)
        rows.append(summary)

    frame = pd.concat(rows, ignore_index=True)
    task_slug = args.task.removeprefix("self_")
    output = Path(
        args.output_dir or f"results/language_decoding_{task_slug}_architecture"
    )
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.task}_architecture_{args.metric}"
    frame.to_csv(output / f"{stem}.csv", index=False)
    _plot(frame, args.task, args.metric, output / stem)
    (output / "metadata.json").write_text(
        json.dumps({
            "task": args.task,
            "datasets": list(DATASETS),
            "inputs": inputs,
            "causal_models": list(CAUSAL_MODELS),
            "roberta_model": ROBERTA_MODEL,
            "excluded_models": ["random-gpt2-xl"],
            "averaging": "layers within model, then equal-weighted models within architecture",
            "relative_preference": (
                "(mean RRR - mean PCA) / min(mean RRR, mean PCA), computed after averaging"
            ),
            "low_performance_threshold": args.low_performance_threshold,
            "plotted_metric": args.metric,
            "smooth_relative_preference": (
                "2 * (RRR - PCA) / "
                "(sqrt(RRR^2 + tau^2) + sqrt(PCA^2 + tau^2))"
            ),
            "normalization_tau": args.normalization_tau,
            "inference": None,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(frame)} dataset/architecture/rank rows and plot to {output}")


if __name__ == "__main__":
    main()
