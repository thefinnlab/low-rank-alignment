"""Compare every ordered pair of configured saved RRR/PCA spaces."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve()
while _ROOT != _ROOT.parent:
    if (_ROOT / "src").is_dir():
        break
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core.config import SubspaceComparisonConfig
from src.utils.config_utils import load_config, parse_args, save_effective_config
from src.utils.directed_subspace_compare import (
    ALL_METRICS,
    METRIC_IMPLEMENTATION,
    SYMMETRIC_METRICS,
    align_views,
    configure_metric_backend,
    expand_spaces,
    factor_for,
    fold_indices,
    load_evaluation_data,
    project_subspace,
    score_ordered_pair,
    shared_ranks,
    summarize_fold_metrics,
)
from src.utils.subspace_io import sanitize_name


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _space_metadata(space) -> dict:
    return {
        "label": space.label,
        "generating_dataset": space.dataset,
        "offset": space.offset,
        "model": space.model,
        "layer": space.layer,
        "kind": space.kind,
        "result_root": str(space.result_root),
        "unit_dir": str(space.unit_dir),
        "source_format": space.source_format,
        "available_ranks": sorted(space.ranks),
    }


def _cell_cache_path(output_dir, geometry, rank, source_index, target_index, fold):
    return (
        output_dir
        / "cache"
        / geometry
        / f"k{rank}"
        / f"source_{source_index:04d}"
        / f"target_{target_index:04d}_fold_{fold}.json"
    )


def _cached_metrics(path: Path, refresh: bool, calculate) -> dict:
    if path.exists() and not refresh:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("implementation") == METRIC_IMPLEMENTATION
            and set(payload.get("metrics", {})) == set(ALL_METRICS)
        ):
            return payload["metrics"]
    metrics = calculate()
    if set(metrics) != set(ALL_METRICS):
        raise ValueError(
            f"Metric implementation returned {sorted(metrics)}, expected "
            f"{sorted(ALL_METRICS)}"
        )
    _save_json(
        path,
        {"implementation": METRIC_IMPLEMENTATION, "metrics": metrics},
    )
    return metrics


def _row_prefix(source, target, geometry, rank, fold, n_train, n_test):
    return {
        "source_dataset": source.dataset,
        "source_offset": source.offset,
        "source_model": source.model,
        "source_layer": source.layer,
        "source_kind": source.kind,
        "source_label": source.label,
        "target_dataset": target.dataset,
        "target_offset": target.offset,
        "target_model": target.model,
        "target_layer": target.layer,
        "target_kind": target.kind,
        "target_label": target.label,
        "geometry": geometry,
        "rank": rank,
        "fold": fold,
        "n_train": n_train,
        "n_test": n_test,
    }


def _canonicalize_symmetric_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Make declared symmetric diagnostics exactly equal in both matrix triangles."""
    result = frame.copy()
    result["unordered_pair"] = result.apply(
        lambda row: "\0".join(sorted((row.source_label, row.target_label))), axis=1
    )
    keys = ["geometry", "rank", "fold", "unordered_pair"]
    means = result.groupby(keys, sort=False)[list(SYMMETRIC_METRICS)].transform("mean")
    result.loc[:, list(SYMMETRIC_METRICS)] = means
    return result.drop(columns="unordered_pair")


def _run_comparisons(config, spaces, ranks, evaluation, output_dir) -> pd.DataFrame:
    rows = []
    factor_cache = {}

    def cached_factor(space_index, space, rank, fold, geometry):
        key = (space_index, rank, fold, geometry)
        if key not in factor_cache:
            factor_cache[key] = factor_for(
                space, rank, fold, config.evaluation_dataset, geometry
            )
        return factor_cache[key]

    views = {
        space.key: evaluation.load_view(space.model, space.layer, space.offset)
        for space in spaces
    }
    total = sum(
        1
        for geometry in config.geometries
        for _rank in ranks
        for source_index, source in enumerate(spaces)
        for target_index, target in enumerate(spaces)
        if source_index != target_index
        and not (
            geometry == "weighted"
            and (
                source.dataset != config.evaluation_dataset
                or target.dataset != config.evaluation_dataset
                or source.source_format != "fold_factors"
                or target.source_format != "fold_factors"
            )
        )
        for _fold in range(len(evaluation.fold_test_ids))
    )
    progress = tqdm(total=total, desc="ordered comparisons", unit="fold")
    for geometry in config.geometries:
        for rank in ranks:
            for source_index, source in enumerate(spaces):
                for target_index, target in enumerate(spaces):
                    if source_index == target_index:
                        continue
                    if geometry == "weighted" and (
                        source.dataset != config.evaluation_dataset
                        or target.dataset != config.evaluation_dataset
                        or source.source_format != "fold_factors"
                        or target.source_format != "fold_factors"
                    ):
                        continue
                    common_ids, source_values, target_values = align_views(
                        views[source.key], views[target.key]
                    )
                    for fold in range(len(evaluation.fold_test_ids)):
                        train_idx, test_idx = fold_indices(
                            common_ids, source, target, evaluation, fold
                        )
                        cache_path = _cell_cache_path(
                            output_dir,
                            geometry,
                            rank,
                            source_index,
                            target_index,
                            fold,
                        )

                        def calculate_metrics():
                            source_factor = cached_factor(
                                source_index, source, rank, fold, geometry
                            )
                            target_factor = cached_factor(
                                target_index, target, rank, fold, geometry
                            )
                            if source_factor is None or target_factor is None:
                                raise RuntimeError(
                                    "Preflight accepted a comparison factor that is "
                                    "unavailable during scoring"
                                )
                            if source_values.shape[1] != source_factor.shape[0]:
                                raise ValueError(
                                    f"Ambient mismatch for {source.label}: evaluation "
                                    f"values have {source_values.shape[1]} columns but "
                                    f"factor has {source_factor.shape[0]} rows"
                                )
                            if target_values.shape[1] != target_factor.shape[0]:
                                raise ValueError(
                                    f"Ambient mismatch for {target.label}: evaluation "
                                    f"values have {target_values.shape[1]} columns but "
                                    f"factor has {target_factor.shape[0]} rows"
                                )
                            return score_ordered_pair(
                                project_subspace(
                                    source_values, source_factor, device="cuda"
                                ),
                                project_subspace(
                                    target_values, target_factor, device="cuda"
                                ),
                                train_idx,
                                test_idx,
                                ridge_alphas=config.ridge_alphas,
                                rsm_metric=config.rsm_metric,
                                device="cuda",
                            )

                        metrics = _cached_metrics(
                            cache_path,
                            config.refresh,
                            calculate_metrics,
                        )
                        rows.append(
                            {
                                **_row_prefix(
                                    source,
                                    target,
                                    geometry,
                                    rank,
                                    fold,
                                    len(train_idx),
                                    len(test_idx),
                                ),
                                **metrics,
                            }
                        )
                        progress.update(1)
    progress.close()
    if not rows:
        raise ValueError("No comparison cells were produced")
    return _canonicalize_symmetric_metrics(pd.DataFrame(rows))


def _validate_preflight(config, spaces, ranks, evaluation) -> dict:
    """Resolve every required row/factor/fold combination before scoring."""
    views = {
        space.key: evaluation.load_view(space.model, space.layer, space.offset)
        for space in spaces
    }
    for space in spaces:
        view = views[space.key]
        # Self-alignment validates row counts, uniqueness, and finite embeddings.
        align_views(view, view)
        for rank in ranks:
            for fold in range(len(evaluation.fold_test_ids)):
                factor = factor_for(
                    space, rank, fold, config.evaluation_dataset, "span"
                )
                if factor is None or factor.shape != (view.values.shape[1], rank):
                    raise ValueError(
                        f"Invalid rank-{rank} factor for {space.label}: "
                        f"expected {(view.values.shape[1], rank)}, got "
                        f"{None if factor is None else factor.shape}"
                    )
                if (
                    "weighted" in config.geometries
                    and space.dataset == config.evaluation_dataset
                    and space.source_format == "fold_factors"
                ):
                    weighted = factor_for(
                        space, rank, fold, config.evaluation_dataset, "weighted"
                    )
                    if weighted is None or weighted.shape != (view.values.shape[1], rank):
                        raise ValueError(
                            f"Invalid weighted rank-{rank} factor for {space.label}"
                        )
    minimum_train, minimum_test = None, None
    for source_index, source in enumerate(spaces):
        for target_index, target in enumerate(spaces):
            if source_index == target_index:
                continue
            common_ids, _, _ = align_views(views[source.key], views[target.key])
            for fold in range(len(evaluation.fold_test_ids)):
                train_idx, test_idx = fold_indices(
                    common_ids, source, target, evaluation, fold
                )
                minimum_train = (
                    len(train_idx)
                    if minimum_train is None
                    else min(minimum_train, len(train_idx))
                )
                minimum_test = (
                    len(test_idx)
                    if minimum_test is None
                    else min(minimum_test, len(test_idx))
                )
    return {"minimum_train_rows": minimum_train, "minimum_test_rows": minimum_test}


def _write_dataset_pair_tables(frame, summary, output_dir):
    for (source_dataset, target_dataset), group in frame.groupby(
        ["source_dataset", "target_dataset"], sort=False
    ):
        detail = (
            output_dir
            / "by_generating_data"
            / sanitize_name(source_dataset)
            / sanitize_name(target_dataset)
        )
        detail.mkdir(parents=True, exist_ok=True)
        group.to_csv(detail / "fold_metrics.csv", index=False)
        summary[
            (summary.source_dataset == source_dataset)
            & (summary.target_dataset == target_dataset)
        ].to_csv(detail / "summary.csv", index=False)


def _matrix(summary, labels, geometry, rank, metric):
    selected = summary[(summary.geometry == geometry) & (summary["rank"] == rank)]
    means = np.full((len(labels), len(labels)), np.nan)
    sems = np.full_like(means, np.nan)
    counts = np.zeros_like(means, dtype=int)
    positions = {label: i for i, label in enumerate(labels)}
    for row in selected.itertuples():
        i, j = positions[row.source_label], positions[row.target_label]
        means[i, j] = getattr(row, f"{metric}_mean")
        sems[i, j] = getattr(row, f"{metric}_sem")
        counts[i, j] = getattr(row, f"{metric}_n")
    return means, sems, counts


def _group_boundaries(spaces):
    boundaries = []
    for index in range(1, len(spaces)):
        previous, current = spaces[index - 1], spaces[index]
        if previous.dataset != current.dataset:
            boundaries.append((index - 0.5, 2.8, "black"))
        elif previous.offset != current.offset:
            boundaries.append((index - 0.5, 2.0, "#333333"))
        elif previous.model != current.model:
            boundaries.append((index - 0.5, 1.2, "white"))
    return boundaries


def _plot_matrix(matrix, spaces, metric, geometry, rank, path):
    labels = [space.label for space in spaces]
    size = max(9.0, min(28.0, 0.34 * len(labels) + 4.0))
    fig, ax = plt.subplots(figsize=(size, size))
    finite = matrix[np.isfinite(matrix)]
    if finite.size:
        lo, hi = np.nanpercentile(finite, [2, 98])
        if lo < 0 < hi:
            bound = max(abs(lo), abs(hi))
            vmin, vmax, cmap = -bound, bound, "coolwarm"
        else:
            vmin, vmax, cmap = lo, hi if hi > lo else lo + 1.0, "viridis"
    else:
        vmin, vmax, cmap = 0.0, 1.0, "viridis"
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#d9d9d9")
    image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap_obj, aspect="equal")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)), labels, fontsize=6)
    for boundary, linewidth, color in _group_boundaries(spaces):
        ax.axhline(boundary, color=color, linewidth=linewidth)
        ax.axvline(boundary, color=color, linewidth=linewidth)
    ax.set_xlabel("Target / reconstructed space")
    ax.set_ylabel("Source space")
    ax.set_title(f"{metric} — {geometry}, rank {rank}")
    fig.colorbar(image, ax=ax, shrink=0.72)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def _write_matrices_and_plots(summary, spaces, ranks, config, output_dir):
    labels = [space.label for space in spaces]
    matrix_dir = output_dir / "matrices"
    figure_dir = output_dir / "figures"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for directory in (matrix_dir, figure_dir):
        for path in directory.glob("cca_*"):
            if path.is_file() and path.suffix in {".csv", ".npz", ".png", ".svg"}:
                path.unlink()
    links = []
    for geometry in config.geometries:
        for rank in ranks:
            for metric in ALL_METRICS:
                mean, sem_values, counts = _matrix(summary, labels, geometry, rank, metric)
                stem = f"{sanitize_name(metric)}__{geometry}__k{rank}"
                np.savez_compressed(
                    matrix_dir / f"{stem}.npz",
                    labels=np.asarray(labels),
                    mean=mean,
                    sem=sem_values,
                    n_folds=counts,
                    symmetric=metric in SYMMETRIC_METRICS,
                )
                pd.DataFrame(mean, index=labels, columns=labels).to_csv(
                    matrix_dir / f"{stem}.csv"
                )
                _plot_matrix(mean, spaces, metric, geometry, rank, figure_dir / stem)
                links.append((metric, geometry, rank, f"{stem}.png"))
    html = ["<!doctype html><meta charset='utf-8'><title>Subspace heatmaps</title>"]
    html.append("<h1>Global ordered subspace heatmaps</h1>")
    for metric, geometry, rank, filename in links:
        html.append(
            f"<h2>{metric} — {geometry}, rank {rank}</h2>"
            f"<a href='{filename}'><img src='{filename}' style='max-width:95vw'></a>"
        )
    (figure_dir / "index.html").write_text("\n".join(html), encoding="utf-8")


def run_config(config: SubspaceComparisonConfig) -> Path:
    configure_metric_backend("cuda")
    output_dir = Path(config.output_root) / sanitize_name(config.analysis_name)
    save_effective_config(config, output_dir)
    spaces = expand_spaces(config)
    ranks = shared_ranks(spaces, config.max_rank)
    evaluation = load_evaluation_data(config, spaces)
    fold_counts = {
        len(space.heldout_ids)
        for space in spaces
        if space.heldout_ids is not None
    }
    if fold_counts and fold_counts != {len(evaluation.fold_test_ids)}:
        raise ValueError(
            "Evaluation-dataset spaces expose inconsistent fold counts: "
            f"{sorted(fold_counts)} versus reference {len(evaluation.fold_test_ids)}"
        )
    validation = _validate_preflight(config, spaces, ranks, evaluation)
    _save_json(
        output_dir / "manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_dataset": config.evaluation_dataset,
            "evaluation": evaluation.metadata,
            "ranks": ranks,
            "n_spaces": len(spaces),
            "n_ordered_span_pairs": len(spaces) * (len(spaces) - 1),
            "validation": validation,
            "spaces": [_space_metadata(space) for space in spaces],
            "metrics": {
                metric: {"symmetric": metric in SYMMETRIC_METRICS}
                for metric in ALL_METRICS
            },
            "metric_implementation": METRIC_IMPLEMENTATION,
            "metric_device": "cuda",
            "ridge_cv": {"folds": 5, "shared_alpha": True},
        },
    )
    frame = _run_comparisons(config, spaces, ranks, evaluation, output_dir)
    frame.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary = summarize_fold_metrics(frame)
    summary.to_csv(output_dir / "summary.csv", index=False)
    _write_dataset_pair_tables(frame, summary, output_dir)
    _write_matrices_and_plots(summary, spaces, ranks, config, output_dir)
    return output_dir


def main() -> None:
    args, overrides = parse_args(__doc__)
    config = load_config(args.config, overrides, SubspaceComparisonConfig)
    output_dir = run_config(config)
    print(f"Saved ordered subspace comparisons to {output_dir}")


if __name__ == "__main__":
    main()
