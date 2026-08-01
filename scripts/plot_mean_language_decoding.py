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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.fmri.bootstrap_language_decoding import (
    _cross_dataset_groups,
    _plot_curve,
    _plot_cross_dataset_curve,
    _slug,
    cross_dataset_mean_scores,
)


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
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.run_name and not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_name):
        raise ValueError("--run-name may contain only letters, numbers, '.', '_', and '-'")

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
    within_frame.drop(columns="significant_rrr_gt_pca").to_csv(
        output / "within_dataset_mean_rank.csv", index=False
    )
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
    (output / "metadata.json").write_text(
        json.dumps({
            "datasets": list(DATASETS), "configs": configs,
            "run_name": args.run_name,
            "dataset_weighting": "equal",
            "model_weighting": "equal over models shared by every dataset",
            "layer_weighting": "equal within dataset/model",
            "within_dataset_scope": "cross-dataset-shared task/rank cells and models",
            "inference": None,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(frame)} task/rank means and plots to {output}")


if __name__ == "__main__":
    main()
