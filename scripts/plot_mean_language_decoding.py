#!/usr/bin/env python
"""Plot fixed mean language-decoding performance across all datasets."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.fmri.bootstrap_language_decoding import (
    _cross_dataset_groups,
    _plot_curve,
    _plot_cross_dataset_curve,
    _score,
    _slug,
    cross_dataset_mean_scores,
)
from src.utils.plot_helpers import RRR_PCA_COLORS, apply_publication_style


DATASETS = ("podcast", "cars-2", "fmri")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name", default=None,
        help="Select decoding outputs created with this run name",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: results/language_decoding_mean[/RUN_NAME])",
    )
    parser.add_argument(
        "--low-performance-threshold", type=float, default=0.05,
        help="Set preference to zero when both method scores are below this value",
    )
    return parser.parse_args()


def _relative_preference(rrr: float, pca: float, threshold: float) -> float:
    if rrr < threshold and pca < threshold:
        return 0.0
    denominator = min(rrr, pca)
    if denominator == 0:
        return np.nan
    return (rrr - pca) / denominator


def _relative_means(by_dataset: dict, kind: str, threshold: float) -> dict[str, float]:
    common_models = sorted(
        set.intersection(*(set(models) for models in by_dataset.values()))
    )
    dataset_means = {}
    for dataset, models in by_dataset.items():
        model_means = []
        for model in common_models:
            layer_values = []
            for record in models[model]:
                pair = record["pair"]
                rrr = _score(kind, pair["y_true"], pair["rrr"])
                pca = _score(kind, pair["y_true"], pair["pca"])
                layer_values.append(_relative_preference(rrr, pca, threshold))
            model_means.append(float(np.nanmean(layer_values)))
        dataset_means[dataset] = float(np.nanmean(model_means))
    dataset_means["all"] = float(np.nanmean(list(dataset_means.values())))
    return dataset_means


def _plot_preference(frame: pd.DataFrame, title: str, destination: Path) -> None:
    frame = frame.sort_values("rank")
    apply_publication_style()
    fig, axis = plt.subplots(figsize=(5.2, 3.4))
    axis.semilogx(
        frame["rank"], frame["relative_preference"], marker="o", markersize=3.5,
        linewidth=1.7, color=RRR_PCA_COLORS["rrr"],
    )
    axis.axhline(0, color="black", linewidth=.8, alpha=.6)
    axis.set_xlabel("Rank $k$ (log scale)")
    axis.set_ylabel("Relative preference: (RRR - PCA) / min(RRR, PCA)")
    axis.set_title(title)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        destination.with_suffix(".png"), dpi=300,
        bbox_inches="tight", pad_inches=.05,
    )
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight", pad_inches=.05)
    plt.close(fig)


def main() -> None:
    args = _args()
    if args.run_name and not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_name):
        raise ValueError("--run-name may contain only letters, numbers, '.', '_', and '-'")
    if args.low_performance_threshold < 0:
        raise ValueError("--low-performance-threshold must be nonnegative")

    configs = {
        dataset: f"configs/language_decoding/internal/{dataset}.yml"
        for dataset in DATASETS
    }
    grouped = _cross_dataset_groups(configs, args.run_name)
    if not grouped:
        raise RuntimeError(
            "No task/rank cells with held-out predictions were found in all datasets"
        )

    rows = []
    preference_rows = []
    for (task, rank), by_dataset in sorted(grouped.items()):
        kinds = {
            record["kind"]
            for models in by_dataset.values()
            for records in models.values()
            for record in records
        }
        if len(kinds) != 1:
            raise ValueError(
                f"Task-kind mismatch across datasets for {task}: {sorted(kinds)}"
            )
        kind = kinds.pop()
        scores, pca, rrr, delta = cross_dataset_mean_scores(by_dataset, kind)
        preferences = _relative_means(
            by_dataset, kind, args.low_performance_threshold
        )
        common_models = set.intersection(
            *(set(models) for models in by_dataset.values())
        )
        row = {
            "rank": rank, "task": task, "kind": kind,
            "pca_score": pca, "rrr_score": rrr, "delta": delta,
            "n_datasets": len(by_dataset), "n_models": len(common_models),
            # Reuse the bootstrap plot style without adding inference markers.
            "significant_rrr_gt_pca": False,
        }
        for dataset, values in scores.items():
            slug = dataset.replace("-", "_")
            row.update({f"{slug}_{key}": value for key, value in values.items()})
        rows.append(row)
        for dataset in (*DATASETS, "all"):
            preference_rows.append({
                "dataset": dataset,
                "rank": rank,
                "task": task,
                "kind": kind,
                "relative_preference": preferences[dataset],
                "n_models": len(common_models),
                "low_performance_threshold": args.low_performance_threshold,
            })

    frame = pd.DataFrame(rows)
    output = Path(args.output_dir or "results/language_decoding_mean")
    if args.run_name and args.output_dir is None:
        output /= args.run_name
    output.mkdir(parents=True, exist_ok=True)
    frame.drop(columns="significant_rrr_gt_pca").to_csv(
        output / "cross_dataset_mean_rank.csv", index=False
    )
    within_rows = []
    for row in rows:
        for dataset in DATASETS:
            slug = dataset.replace("-", "_")
            within_rows.append({
                "dataset": dataset,
                "rank": row["rank"],
                "task": row["task"],
                "kind": row["kind"],
                "pca_score": row[f"{slug}_pca"],
                "rrr_score": row[f"{slug}_rrr"],
                "delta": row[f"{slug}_delta"],
                "n_models": row["n_models"],
                "significant_rrr_gt_pca": False,
            })
    within_frame = pd.DataFrame(within_rows)
    preference_frame = pd.DataFrame(preference_rows)
    within_frame.drop(columns="significant_rrr_gt_pca").to_csv(
        output / "within_dataset_mean_rank.csv", index=False
    )
    preference_frame.to_csv(output / "relative_preference_rank.csv", index=False)
    for task, task_frame in frame.groupby("task"):
        _plot_cross_dataset_curve(
            task_frame, output / "plots" / "cross_dataset" / _slug(str(task))
        )
    for (dataset, task), task_frame in within_frame.groupby(["dataset", "task"]):
        _plot_curve(
            task_frame,
            f"{task}: equal-weighted models, {dataset}",
            output / "plots" / "within_dataset" / _slug(str(dataset)) / _slug(str(task)),
        )
    for (dataset, task), task_frame in preference_frame.groupby(["dataset", "task"]):
        scope = "equal-weighted datasets" if dataset == "all" else dataset
        _plot_preference(
            task_frame,
            f"{task}: RRR versus PCA preference, {scope}",
            output / "plots" / "relative_preference"
            / _slug(str(dataset)) / _slug(str(task)),
        )
    (output / "metadata.json").write_text(
        json.dumps({
            "datasets": list(DATASETS), "configs": configs,
            "run_name": args.run_name,
            "dataset_weighting": "equal",
            "model_weighting": "equal over models shared by every dataset",
            "layer_weighting": "equal within dataset/model",
            "within_dataset_scope": "cross-dataset-shared task/rank cells and models",
            "relative_preference": (
                "(RRR - PCA) / min(RRR, PCA), computed per model-layer "
                "before averaging"
            ),
            "low_performance_threshold": args.low_performance_threshold,
            "inference": None,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(frame)} task/rank means and plots to {output}")


if __name__ == "__main__":
    main()
