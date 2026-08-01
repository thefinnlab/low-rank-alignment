#!/usr/bin/env python
"""Paired sentence-block inference for internal RRR versus PCA language decoding."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.fmri.run_internal_language_decoding_batch import _layer_configs
from scripts.rrr.run_internal_language_decoding_batch import build_layer_configs
from src.utils.plot_helpers import RRR_PCA_COLORS, apply_publication_style


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["podcast", "cars-2", "fmri", "all"], default="fmri",
        help="Use 'all' for the equal-weighted cross-dataset summary",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        help="Worker processes (default: SLURM_CPUS_PER_TASK, otherwise 1)",
    )
    parser.add_argument(
        "--output-dir", default=None
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def _score(kind: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if kind == "continuous":
        true_centered = y_true - np.mean(y_true)
        pred_centered = y_pred - np.mean(y_pred)
        denominator = np.sqrt(
            np.dot(true_centered, true_centered)
            * np.dot(pred_centered, pred_centered)
        )
        if denominator <= np.finfo(float).eps:
            return np.nan
        return float(np.dot(true_centered, pred_centered) / denominator)
    classes = np.unique(y_true)
    return float(np.mean([
        np.mean(y_pred[y_true == label] == label) for label in classes
    ]))


def _block_lookup(block_id: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    blocks = pd.unique(block_id.astype(str))
    return blocks, {block: np.flatnonzero(block_id.astype(str) == block) for block in blocks}


def _sample_indices(
    blocks: np.ndarray, lookup: dict[str, np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    sampled = rng.choice(blocks, size=len(blocks), replace=True)
    return np.concatenate([lookup[str(block)] for block in sampled])


def _load_pair(root: Path, task: str, rank: int) -> dict:
    arrays = {}
    for method in ("rrr", "pca"):
        path = root / "tasks" / task / method / f"k{rank}" / "heldout_predictions.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as payload:
            arrays[method] = {key: payload[key] for key in payload.files}
    for key in ("y_true", "row_index", "block_id"):
        if not np.array_equal(arrays["rrr"][key], arrays["pca"][key]):
            raise ValueError(f"RRR/PCA held-out {key} mismatch under {root}")
    return {
        "y_true": arrays["rrr"]["y_true"],
        "rrr": arrays["rrr"]["y_pred"],
        "pca": arrays["pca"]["y_pred"],
        "row_index": arrays["rrr"]["row_index"],
        "block_id": arrays["rrr"]["block_id"].astype(str),
    }


def _load_fold_pair(root: Path, task: str, rank: int) -> dict:
    """Pool paired out-of-fold predictions for Podcast or Cars 2."""
    arrays = {"rrr": [], "pca": []}
    for method in arrays:
        paths = sorted(
            (root / "tasks" / task / method / f"k{rank}").glob("fold_*/heldout_predictions.npz"),
            key=lambda path: int(path.parent.name.removeprefix("fold_")),
        )
        if not paths:
            raise FileNotFoundError(root / "tasks" / task / method / f"k{rank}")
        for path in paths:
            with np.load(path) as payload:
                arrays[method].append({key: payload[key] for key in payload.files})
    if len(arrays["rrr"]) != len(arrays["pca"]):
        raise ValueError(f"RRR/PCA fold-count mismatch under {root}, task={task}, k={rank}")
    combined = {}
    for method, folds in arrays.items():
        combined[method] = {
            key: np.concatenate([fold[key] for fold in folds]) for key in folds[0]
        }
        order = np.argsort(combined[method]["row_index"])
        combined[method] = {key: value[order] for key, value in combined[method].items()}
    for key in ("y_true", "row_index", "block_id"):
        if not np.array_equal(combined["rrr"][key], combined["pca"][key]):
            raise ValueError(f"RRR/PCA pooled held-out {key} mismatch under {root}")
    if len(np.unique(combined["rrr"]["row_index"])) != len(combined["rrr"]["row_index"]):
        raise ValueError(f"Held-out folds do not partition analysis rows under {root}")
    return {
        "y_true": combined["rrr"]["y_true"],
        "rrr": combined["rrr"]["y_pred"],
        "pca": combined["pca"]["y_pred"],
        "row_index": combined["rrr"]["row_index"],
        "block_id": combined["rrr"]["block_id"].astype(str),
    }


def paired_block_bootstrap(
    pair: dict, kind: str, repeats: int, rng: np.random.Generator
) -> tuple[float, float, float, np.ndarray]:
    """Return observed PCA/RRR scores, their difference, and paired replicates."""
    pca = _score(kind, pair["y_true"], pair["pca"])
    rrr = _score(kind, pair["y_true"], pair["rrr"])
    blocks, lookup = _block_lookup(pair["block_id"])
    boot = np.full(repeats, np.nan)
    for index in range(repeats):
        rows = _sample_indices(blocks, lookup, rng)
        boot[index] = _score(kind, pair["y_true"][rows], pair["rrr"][rows]) - _score(
            kind, pair["y_true"][rows], pair["pca"][rows]
        )
    return pca, rrr, rrr - pca, boot


def hierarchical_block_bootstrap(
    units: dict[str, list[dict]],
    kind: str,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, np.ndarray]:
    """Equal-model bootstrap with layers nested in model and shared sentence draws."""
    models = np.asarray(sorted(units))
    lookups = {
        id(pair): _block_lookup(pair["block_id"])[1]
        for layers in units.values()
        for pair in layers
    }

    def delta(pair: dict, sampled_blocks: np.ndarray | None = None) -> float:
        if sampled_blocks is None:
            rows = np.arange(len(pair["y_true"]))
        else:
            lookup = lookups[id(pair)]
            rows = np.concatenate([lookup[str(block)] for block in sampled_blocks])
        return _score(kind, pair["y_true"][rows], pair["rrr"][rows]) - _score(
            kind, pair["y_true"][rows], pair["pca"][rows]
        )

    def method_score(pair: dict, method: str) -> float:
        return _score(kind, pair["y_true"], pair[method])

    observed_pca = float(np.mean([
        np.mean([method_score(pair, "pca") for pair in units[model]])
        for model in models
    ]))
    observed_rrr = float(np.mean([
        np.mean([method_score(pair, "rrr") for pair in units[model]])
        for model in models
    ]))
    observed = observed_rrr - observed_pca
    reference_blocks, _ = _block_lookup(units[models[0]][0]["block_id"])
    boot = np.full(repeats, np.nan)
    for index in range(repeats):
        sampled_models = rng.choice(models, size=len(models), replace=True)
        sampled_blocks = rng.choice(
            reference_blocks, size=len(reference_blocks), replace=True
        )
        model_deltas = []
        for model in sampled_models:
            layers = units[str(model)]
            sampled_layers = rng.choice(len(layers), size=len(layers), replace=True)
            model_deltas.append(np.mean([
                delta(layers[layer], sampled_blocks) for layer in sampled_layers
            ]))
        boot[index] = np.mean(model_deltas)
    return observed_pca, observed_rrr, observed, boot


def cross_dataset_hierarchical_bootstrap(
    datasets: dict[str, dict[str, list[dict]]],
    kind: str,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[dict[str, dict[str, float]], float, float, float, np.ndarray]:
    """Bootstrap the equal-dataset mean with model identities shared across datasets."""
    dataset_names = sorted(datasets)
    common_models = sorted(set.intersection(*(set(datasets[name]) for name in dataset_names)))
    if not common_models:
        raise ValueError("Cross-dataset bootstrap has no model shared by every dataset")
    lookups = {
        id(record["pair"]): _block_lookup(record["pair"]["block_id"])[1]
        for name in dataset_names
        for model in common_models
        for record in datasets[name][model]
    }

    def score_pair(pair: dict, method: str, sampled_blocks=None) -> float:
        rows = np.arange(len(pair["y_true"])) if sampled_blocks is None else np.concatenate(
            [lookups[id(pair)][str(block)] for block in sampled_blocks]
        )
        return _score(kind, pair["y_true"][rows], pair[method][rows])

    scores: dict[str, dict[str, float]] = {}
    for name in dataset_names:
        scores[name] = {
            method: float(np.mean([
                np.mean([
                    score_pair(record["pair"], method)
                    for record in datasets[name][model]
                ])
                for model in common_models
            ]))
            for method in ("pca", "rrr")
        }
        scores[name]["delta"] = scores[name]["rrr"] - scores[name]["pca"]
    observed_pca = float(np.mean([scores[name]["pca"] for name in dataset_names]))
    observed_rrr = float(np.mean([scores[name]["rrr"] for name in dataset_names]))
    observed = observed_rrr - observed_pca

    boot = np.full(repeats, np.nan)
    for index in range(repeats):
        sampled_models = rng.choice(common_models, size=len(common_models), replace=True)
        dataset_deltas = []
        for name in dataset_names:
            reference = datasets[name][common_models[0]][0]["pair"]
            blocks, _ = _block_lookup(reference["block_id"])
            sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
            model_deltas = []
            for model in sampled_models:
                records = datasets[name][str(model)]
                sampled_layers = rng.choice(len(records), size=len(records), replace=True)
                model_deltas.append(float(np.mean([
                    score_pair(records[layer]["pair"], "rrr", sampled_blocks)
                    - score_pair(records[layer]["pair"], "pca", sampled_blocks)
                    for layer in sampled_layers
                ])))
            dataset_deltas.append(float(np.mean(model_deltas)))
        boot[index] = np.mean(dataset_deltas)
    return scores, observed_pca, observed_rrr, observed, boot


def _summary(delta: float, boot: np.ndarray, alpha: float) -> dict[str, float]:
    finite = boot[np.isfinite(boot)]
    if not np.isfinite(delta) or not len(finite):
        return {"ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    return {
        "ci_low": float(np.quantile(finite, alpha / 2)),
        "ci_high": float(np.quantile(finite, 1 - alpha / 2)),
        "p_value": float((1 + np.count_nonzero(finite <= 0)) / (len(finite) + 1)),
    }


def _apply_rankwise_fdr(frame: pd.DataFrame, alpha: float, groups: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    frame["q_value"] = np.nan
    frame["significant_rrr_gt_pca"] = False
    group_key = groups[0] if len(groups) == 1 else groups
    for _, indices in frame.groupby(group_key, dropna=False).groups.items():
        indices = list(indices)
        valid = frame.loc[indices, "p_value"].notna()
        selected = list(np.asarray(indices)[valid.to_numpy()])
        if not selected:
            continue
        reject, q_values, _, _ = multipletests(
            frame.loc[selected, "p_value"], alpha=alpha, method="fdr_bh"
        )
        frame.loc[selected, "q_value"] = q_values
        frame.loc[selected, "significant_rrr_gt_pca"] = reject & (
            frame.loc[selected, "delta"].to_numpy() > 0
        )
    return frame


def _discover_fmri(
    config_path: str, run_name: str | None = None
) -> tuple[list[dict], dict[tuple[str, int], dict[str, list[dict]]]]:
    manifest = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    records = []
    grouped: dict[tuple[str, int], dict[str, list[dict]]] = {}
    for model in manifest["models"]:
        _, configs = _layer_configs(config_path, model)
        for config in configs:
            identity = "heldout_wheretheressmoke"
            if run_name:
                identity += f"__{run_name}"
            root = Path(config.output_dir) / "language_decoding" / identity
            metrics_path = root / "metrics.csv"
            if not metrics_path.is_file():
                continue
            metrics = pd.read_csv(metrics_path)
            ranks = sorted(set(metrics.loc[metrics.method.isin(["rrr", "pca"]), "rank"].dropna().astype(int)))
            kinds = metrics.drop_duplicates("task").set_index("task")["kind"].to_dict()
            for rank in ranks:
                for task in manifest["tasks"]:
                    try:
                        pair = _load_pair(root, task, rank)
                    except FileNotFoundError:
                        continue
                    record = {
                        "model": model, "layer": int(config.layer), "rank": rank,
                        "task": task, "kind": kinds[task], "pair": pair,
                    }
                    records.append(record)
                    grouped.setdefault((task, rank), {}).setdefault(model, []).append(record)
    return records, grouped


def _generic_output_root(config) -> Path | None:
    candidates = Path(config.rrr_run_path).parent / "language_decoding" / "within_dataset"
    if not candidates.is_dir():
        return None
    expected_rrr = str(Path(config.rrr_run_path).expanduser().resolve())
    expected_pca = str(Path(config.pca_run_path).expanduser().resolve())
    matched = []
    for metadata_path in candidates.glob("*/run_metadata.json"):
        metadata = json.loads(metadata_path.read_text())
        saved = metadata.get("subspace_unit_dirs", {})
        saved_source_name = metadata.get("config", {}).get("source_name")
        if (
            saved.get("rrr") == expected_rrr
            and saved.get("pca") == expected_pca
            and saved_source_name == config.source_name
        ):
            matched.append(metadata_path.parent)
    if len(matched) > 1:
        raise ValueError(f"Multiple matching language-decoding outputs found: {matched}")
    return matched[0] if matched else None


def _discover_generic(
    config_path: str, run_name: str | None = None
) -> tuple[list[dict], dict[tuple[str, int], dict[str, list[dict]]]]:
    manifest = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    records = []
    grouped: dict[tuple[str, int], dict[str, list[dict]]] = {}
    for model in manifest["models"]:
        for config in build_layer_configs(config_path, model, run_name=run_name):
            root = _generic_output_root(config)
            if root is None or not (root / "grid_results.csv").is_file():
                continue
            metrics = pd.read_csv(root / "grid_results.csv")
            ranked = metrics.loc[metrics["subspace"].isin(["rrr", "pca"])]
            ranks = sorted(set(ranked["k"].dropna().astype(int)))
            kinds = metrics.drop_duplicates("task").set_index("task")["kind"].to_dict()
            for rank in ranks:
                for task in manifest["tasks"]:
                    try:
                        pair = _load_fold_pair(root, task, rank)
                    except FileNotFoundError:
                        continue
                    record = {
                        "model": model,
                        "layer": int(config.encoding.per_model_config["layer"]),
                        "rank": rank, "task": task, "kind": kinds[task], "pair": pair,
                    }
                    records.append(record)
                    grouped.setdefault((task, rank), {}).setdefault(model, []).append(record)
    return records, grouped


def _individual_job(payload: tuple[dict, int, float, int]) -> dict:
    record, repeats, alpha, seed = payload
    pca, rrr, delta, boot = paired_block_bootstrap(
        record["pair"], record["kind"], repeats, np.random.default_rng(seed)
    )
    return {key: record[key] for key in ("model", "layer", "rank", "task", "kind")} | {
        "pca_score": pca, "rrr_score": rrr, "delta": delta,
        "n_blocks": len(pd.unique(record["pair"]["block_id"])),
        "n_test": len(record["pair"]["y_true"]),
        **_summary(delta, boot, alpha),
    }


def _cross_model_job(payload: tuple[tuple[str, int], dict[str, list[dict]], int, float, int]) -> dict:
    (task, rank), by_model, repeats, alpha, seed = payload
    units = {
        model: [record["pair"] for record in model_records]
        for model, model_records in by_model.items()
    }
    kind = next(iter(by_model.values()))[0]["kind"]
    pca, rrr, delta, boot = hierarchical_block_bootstrap(
        units, kind, repeats, np.random.default_rng(seed)
    )
    return {
        "rank": rank, "task": task, "kind": kind,
        "pca_score": pca, "rrr_score": rrr, "delta": delta,
        "n_models": len(units),
        "n_layers": sum(len(layers) for layers in units.values()),
        **_summary(delta, boot, alpha),
    }


def _cross_dataset_groups(
    config_paths: dict[str, str], run_name: str | None,
) -> dict[tuple[str, int], dict[str, dict[str, list[dict]]]]:
    """Keep task/rank cells present in every dataset, organized by model."""
    discovered = {}
    for dataset, config_path in config_paths.items():
        _, grouped = (
            _discover_fmri(config_path, run_name)
            if dataset == "fmri"
            else _discover_generic(config_path, run_name)
        )
        discovered[dataset] = grouped
    shared = set.intersection(*(set(groups) for groups in discovered.values()))
    return {
        key: {dataset: discovered[dataset][key] for dataset in sorted(discovered)}
        for key in sorted(shared)
    }


def _cross_dataset_job(payload: tuple) -> dict:
    (task, rank), by_dataset, repeats, alpha, seed = payload
    kinds = {
        record["kind"]
        for models in by_dataset.values()
        for records in models.values()
        for record in records
    }
    if len(kinds) != 1:
        raise ValueError(f"Task-kind mismatch across datasets for {task}: {sorted(kinds)}")
    kind = kinds.pop()
    scores, pca, rrr, delta, boot = cross_dataset_hierarchical_bootstrap(
        by_dataset, kind, repeats, np.random.default_rng(seed)
    )
    common_models = set.intersection(*(set(models) for models in by_dataset.values()))
    result = {
        "rank": rank, "task": task, "kind": kind,
        "pca_score": pca, "rrr_score": rrr, "delta": delta,
        "n_datasets": len(by_dataset), "n_models": len(common_models),
        **_summary(delta, boot, alpha),
    }
    for dataset, values in scores.items():
        slug = dataset.replace("-", "_")
        result.update({f"{slug}_{key}": value for key, value in values.items()})
    return result


def _log_progress(stage: str, completed: int, total: int, started: float) -> None:
    elapsed = time.monotonic() - started
    rate = completed / elapsed if elapsed else 0.0
    remaining = (total - completed) / rate if rate else float("nan")
    print(
        f"[{stage}] {completed}/{total} jobs ({completed / total:.0%}); "
        f"elapsed {elapsed:.1f}s; ETA {remaining:.1f}s",
        file=sys.stderr,
        flush=True,
    )


def _run_jobs(function, payloads: list, jobs: int, stage: str) -> list:
    total = len(payloads)
    if not total:
        return []
    started = time.monotonic()
    print(
        f"[{stage}] starting {total} jobs with {jobs} worker(s)",
        file=sys.stderr,
        flush=True,
    )
    report_every = max(1, (total + 19) // 20)
    results = []

    def consume(iterator) -> None:
        for completed, result in enumerate(iterator, start=1):
            results.append(result)
            if completed % report_every == 0 or completed == total:
                _log_progress(stage, completed, total, started)

    if jobs == 1:
        consume(map(function, payloads))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            consume(executor.map(function, payloads))
    return results


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)


def _plot_curve(frame: pd.DataFrame, title: str, destination: Path) -> None:
    frame = frame.sort_values("rank")
    apply_publication_style()
    fig, axis = plt.subplots(figsize=(5.2, 3.4))
    for method in ("rrr", "pca"):
        axis.semilogx(
            frame["rank"], frame[f"{method}_score"], marker="o", markersize=3.5,
            linewidth=1.4, color=RRR_PCA_COLORS[method], label=method.upper(), zorder=4,
        )
    significant = frame["significant_rrr_gt_pca"].astype(bool).to_numpy()
    if significant.any():
        values = frame[["rrr_score", "pca_score"]].to_numpy(float)
        finite = values[np.isfinite(values)]
        span = float(np.ptp(finite)) if len(finite) else 1.0
        spacing = max(span * 0.06, 0.01)
        marker_y = np.fmax(values[:, 0], values[:, 1]) + spacing
        axis.scatter(
            frame.loc[significant, "rank"], marker_y[significant], marker="*", s=34,
            color=RRR_PCA_COLORS["rrr"], linewidths=0, zorder=10,
            label="RRR > PCA (rank-wise FDR)",
        )
        low, high = axis.get_ylim()
        axis.set_ylim(low, max(high, float(np.nanmax(marker_y[significant]) + spacing)))
    kind = str(frame["kind"].iloc[0])
    axis.set_xlabel("Rank $k$ (log scale)")
    axis.set_ylabel("Held-out correlation (r)" if kind == "continuous" else "Held-out balanced accuracy")
    axis.set_title(title)
    axis.legend(loc="best", fontsize=7)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=.05)
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight", pad_inches=.05)
    plt.close(fig)


def _write_plots(individual: pd.DataFrame, cross_model: pd.DataFrame, output: Path) -> None:
    for (model, layer, task), frame in individual.groupby(["model", "layer", "task"]):
        _plot_curve(
            frame, f"{task}: {model}, layer {int(layer)}",
            output / "plots" / "per_model_layer" / _slug(str(model)) / f"layer{int(layer)}" / _slug(str(task)),
        )
    for task, frame in cross_model.groupby("task"):
        _plot_curve(
            frame, f"{task}: equal-weighted cross-model",
            output / "plots" / "cross_model" / _slug(str(task)),
        )


def _plot_cross_dataset_curve(frame: pd.DataFrame, destination: Path) -> None:
    frame = frame.sort_values("rank")
    apply_publication_style()
    fig, axis = plt.subplots(figsize=(5.2, 3.4))
    dataset_labels = (
        ("podcast", "Podcast", "--"),
        ("cars_2", "Cars 2", ":"),
        ("fmri", "fMRI", "-."),
    )
    for method in ("rrr", "pca"):
        color = RRR_PCA_COLORS[method]
        for dataset, label, linestyle in dataset_labels:
            column = f"{dataset}_{method}"
            if column in frame:
                axis.semilogx(
                    frame["rank"], frame[column], color=color, alpha=.25,
                    linewidth=1, linestyle=linestyle,
                    label=f"{label} {method.upper()}", zorder=2,
                )
        axis.semilogx(
            frame["rank"], frame[f"{method}_score"], marker="o", markersize=3.5,
            linewidth=2, color=color, label=f"Mean {method.upper()}", zorder=5,
        )
    significant = frame["significant_rrr_gt_pca"].astype(bool).to_numpy()
    if significant.any():
        values = frame[["rrr_score", "pca_score"]].to_numpy(float)
        finite = values[np.isfinite(values)]
        spacing = max((float(np.ptp(finite)) if len(finite) else 1.0) * .06, .01)
        marker_y = np.fmax(values[:, 0], values[:, 1]) + spacing
        axis.scatter(
            frame.loc[significant, "rank"], marker_y[significant], marker="*", s=34,
            color=RRR_PCA_COLORS["rrr"], linewidths=0, zorder=10,
            label="Mean RRR > PCA (rank-wise FDR)",
        )
    axis.set_xlabel("Rank $k$ (log scale)")
    axis.set_ylabel(
        "Held-out correlation (r)" if str(frame["kind"].iloc[0]) == "continuous"
        else "Held-out balanced accuracy"
    )
    axis.set_title(f"{frame['task'].iloc[0]}: equal-weighted datasets")
    axis.legend(loc="best", fontsize=6, ncol=2)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=.05)
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight", pad_inches=.05)
    plt.close(fig)


def _run_cross_dataset(args: argparse.Namespace, output: Path, started: float) -> None:
    config_paths = {
        dataset: f"configs/language_decoding/internal/{dataset}.yml"
        for dataset in ("podcast", "cars-2", "fmri")
    }
    if args.config:
        raise ValueError("--config is not supported with --dataset all; use the three internal manifests")
    grouped = _cross_dataset_groups(config_paths, args.run_name)
    if not grouped:
        raise RuntimeError("No task/rank cells with held-out predictions were found in all datasets")
    print(
        f"[discovery] found {len(grouped)} shared cross-dataset task/rank jobs in "
        f"{time.monotonic() - started:.1f}s", file=sys.stderr, flush=True,
    )
    items = sorted(grouped.items())
    payloads = [
        (key, value, args.bootstrap_repeats, args.alpha, int(seed.generate_state(1)[0]))
        for (key, value), seed in zip(
            items, np.random.SeedSequence([args.seed, 2]).spawn(len(items))
        )
    ]
    frame = _apply_rankwise_fdr(
        pd.DataFrame(_run_jobs(_cross_dataset_job, payloads, args.jobs, "cross-dataset")),
        args.alpha, ["rank"],
    )
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "cross_dataset_rank.csv", index=False)
    if not args.no_plots:
        for task, task_frame in frame.groupby("task"):
            _plot_cross_dataset_curve(
                task_frame, output / "plots" / "cross_dataset" / _slug(str(task))
            )
    (output / "metadata.json").write_text(json.dumps({
        "dataset": "all", "configs": config_paths, "run_name": args.run_name,
        "bootstrap_repeats": args.bootstrap_repeats, "seed": args.seed,
        "alpha": args.alpha, "jobs": args.jobs, "plots": not args.no_plots,
        "datasets": ["podcast", "cars-2", "fmri"],
        "dataset_weighting": "equal; datasets treated as fixed",
        "bootstrap": "shared model draws; layers sampled within dataset/model; independent sentence-block draws by dataset",
        "alternative": "mean_rrr_gt_pca", "fdr": "BH across targets within rank",
    }, indent=2))


def main() -> None:
    args = _args()
    if args.run_name and not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_name):
        raise ValueError("--run-name may contain only letters, numbers, '.', '_', and '-'")
    config_path = args.config or f"configs/language_decoding/internal/{args.dataset}.yml"
    default_output = (
        Path("results/language_decoding_bootstrap")
        if args.dataset == "all"
        else Path(f"results/{args.dataset}/language_decoding_bootstrap")
    )
    if args.run_name:
        default_output /= args.run_name
    output_path = args.output_dir or str(default_output)
    if args.bootstrap_repeats < 1:
        raise ValueError("--bootstrap-repeats must be positive")
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between zero and one")
    analysis_started = time.monotonic()
    print(
        f"[discovery] dataset={args.dataset}; repeats={args.bootstrap_repeats}; "
        f"workers={args.jobs}; config={config_path}",
        file=sys.stderr,
        flush=True,
    )
    if args.dataset == "all":
        output = Path(output_path)
        _run_cross_dataset(args, output, analysis_started)
        print(f"Done in {time.monotonic() - analysis_started:.1f}s: {output}")
        return
    records, grouped = (
        _discover_fmri(config_path, args.run_name)
        if args.dataset == "fmri"
        else _discover_generic(config_path, args.run_name)
    )
    if not records:
        raise RuntimeError("No heldout_predictions.npz artifacts found; rerun fMRI decoding with --refresh")
    print(
        f"[discovery] found {len(records)} model/layer/task/rank jobs and "
        f"{len(grouped)} cross-model task/rank jobs in "
        f"{time.monotonic() - analysis_started:.1f}s",
        file=sys.stderr,
        flush=True,
    )
    individual_payloads = [
        (record, args.bootstrap_repeats, args.alpha, int(seed.generate_state(1)[0]))
        for record, seed in zip(
            records, np.random.SeedSequence([args.seed, 0]).spawn(len(records))
        )
    ]
    individual = _run_jobs(
        _individual_job, individual_payloads, args.jobs, "per-model/layer"
    )
    individual_frame = _apply_rankwise_fdr(
        pd.DataFrame(individual), args.alpha, ["model", "layer", "rank"]
    )
    grouped_items = sorted(grouped.items())
    cross_payloads = [
        (key, by_model, args.bootstrap_repeats, args.alpha, int(seed.generate_state(1)[0]))
        for (key, by_model), seed in zip(
            grouped_items,
            np.random.SeedSequence([args.seed, 1]).spawn(len(grouped_items)),
        )
    ]
    cross_model = _run_jobs(
        _cross_model_job, cross_payloads, args.jobs, "cross-model"
    )
    cross_frame = _apply_rankwise_fdr(
        pd.DataFrame(cross_model), args.alpha, ["rank"]
    )
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    individual_frame.to_csv(output / "per_model_layer_rank.csv", index=False)
    cross_frame.to_csv(output / "cross_model_rank.csv", index=False)
    if not args.no_plots:
        _write_plots(individual_frame, cross_frame, output)
    (output / "metadata.json").write_text(json.dumps({
        "dataset": args.dataset, "config": config_path,
        "run_name": args.run_name,
        "bootstrap_repeats": args.bootstrap_repeats,
        "seed": args.seed, "alpha": args.alpha, "jobs": args.jobs,
        "plots": not args.no_plots, "bootstrap_unit": "sentence",
        "alternative": "rrr_gt_pca", "fdr": "BH across targets within rank",
        "cross_model": "equal model weights; layers averaged within model",
    }, indent=2))
    print(f"Done in {time.monotonic() - analysis_started:.1f}s: {output}")


if __name__ == "__main__":
    main()
