"""Reconstruct saved RRR/PCA language-model subspaces from language tasks."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.core.config import (
    LanguageDecodingMode,
    LanguageReconstructionConfig,
    get_results_dir,
)
from src.core.registry import resolve_dataset_context
from src.utils.config_utils import save_effective_config
from src.utils.constants import RIDGE_ALPHAS
from src.utils.embeddings import load_embeddings
from src.utils.language_decoding import (
    _load_within_folds_and_selection,
    _outer_folds,
    resolve_subspace_unit_dir,
    source_slug,
)
from src.utils.language_tasks import PodcastTaskContext, get_language_task
from src.utils.subspace_compare import aggregate_subspace_basis


METRICS = ("multivariate_r2", "mean_coordinate_correlation", "linear_cka")


def resolve_pca_sibling(rrr_unit_dir: str | Path) -> Path:
    """Resolve the PCA unit mirroring an RRR unit path."""
    path = Path(rrr_unit_dir)
    replacements = {
        "rrr": "pca",
        "cross_dataset_rrr": "cross_dataset_pca",
        "per_subject_rrr": "per_subject_pca",
        "per_region_rrr": "per_region_pca",
        "multi_view_rrr": "multi_view_pca",
    }
    parts = list(path.parts)
    for index, part in enumerate(parts):
        if part in replacements:
            parts[index] = replacements[part]
            return Path(*parts)
    return Path(str(path) + "_pca")


def _factor_paths(unit_dir: Path) -> list[Path]:
    paths = sorted(
        unit_dir.glob("fold_*/A.npy"),
        key=lambda path: int(path.parent.name.removeprefix("fold_")),
    )
    if not paths:
        raise FileNotFoundError(f"No fold_*/A.npy factors found under {unit_dir}")
    return paths


def _load_factors(unit_dir: Path, embedding_dim: int) -> dict[int, np.ndarray]:
    factors = {
        int(path.parent.name.removeprefix("fold_")): np.asarray(np.load(path))
        for path in _factor_paths(unit_dir)
    }
    dimensions = {factor.shape[0] for factor in factors.values()}
    if dimensions != {embedding_dim}:
        raise ValueError(
            f"Factor ambient dimensions {sorted(dimensions)} do not match "
            f"embedding dimension {embedding_dim}"
        )
    return factors


def orthonormal_factor(factor: np.ndarray, rank: int) -> np.ndarray:
    """Return a deterministic orthonormal basis for the leading factor span."""
    block = np.asarray(factor, dtype=np.float64)[:, : int(rank)]
    if block.shape[1] != int(rank):
        raise ValueError(f"Factor has insufficient width for rank {rank}")
    q, r = np.linalg.qr(block, mode="reduced")
    tolerance = np.finfo(r.dtype).eps * max(block.shape) * np.linalg.norm(r, ord=2)
    if np.count_nonzero(np.abs(np.diag(r)) > tolerance) != int(rank):
        raise ValueError(f"Factor is rank deficient at rank {rank}")
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return (q * signs).astype(np.float32)


def weighted_mean_factor(factors: list[np.ndarray], rank: int) -> np.ndarray:
    """Factorize mean(A A^T) while retaining its top-k eigenvalue weights."""
    if not factors:
        raise ValueError("At least one source factor is required")
    blocks = []
    for factor in factors:
        block = np.asarray(factor, dtype=np.float64)[:, : int(rank)]
        if block.shape[1] != int(rank):
            raise ValueError(f"Factor has insufficient width for rank {rank}")
        blocks.append(block / math.sqrt(len(factors)))
    left, singular_values, _ = np.linalg.svd(
        np.concatenate(blocks, axis=1), full_matrices=False
    )
    if len(singular_values) < int(rank) or singular_values[int(rank) - 1] <= 0:
        raise ValueError(f"Mean weighted operator is rank deficient at rank {rank}")
    return (left[:, : int(rank)] * singular_values[: int(rank)]).astype(np.float32)


def prepare_target_factors(
    factors: dict[int, np.ndarray],
    folds: list[tuple[np.ndarray, np.ndarray]],
    ranks: list[int],
    mode: LanguageDecodingMode,
    aggregation: str,
) -> dict[tuple[str, int, int], np.ndarray]:
    """Prepare weighted and span factors keyed by view, target fold, and rank."""
    source_folds = sorted(factors)
    if mode == LanguageDecodingMode.WITHIN_DATASET and source_folds != list(
        range(len(folds))
    ):
        raise ValueError(
            f"Saved factor folds {source_folds} do not match target folds "
            f"{list(range(len(folds)))}"
        )
    prepared = {}
    for rank in map(int, ranks):
        source_blocks = [factors[fold][:, :rank] for fold in source_folds]
        if any(block.shape[1] != rank for block in source_blocks):
            raise ValueError(f"A saved factor has insufficient width for rank {rank}")
        if mode == LanguageDecodingMode.CROSS_DATASET:
            weighted = weighted_mean_factor(source_blocks, rank)
            span = aggregate_subspace_basis(source_blocks, rank, aggregation)
            for fold in range(len(folds)):
                prepared[("weighted", fold, rank)] = weighted
                prepared[("span", fold, rank)] = span
        else:
            for fold in range(len(folds)):
                raw = factors[fold][:, :rank].astype(np.float32, copy=False)
                prepared[("weighted", fold, rank)] = raw
                prepared[("span", fold, rank)] = orthonormal_factor(raw, rank)
    return prepared


def _default_task_data(config: LanguageReconstructionConfig, stimulus_id: str) -> dict:
    paths = dict(config.task_data)
    paths.setdefault(
        "language_features",
        str(
            Path("processed_data")
            / "language_features"
            / config.encoding.dataset
            / stimulus_id
            / "tokens.tsv"
        ),
    )
    return paths


def load_task_matrix(
    context: PodcastTaskContext, tasks: list[str]
) -> tuple[np.ndarray, list[int], list[int], dict]:
    """Load registered task values into one mixed-type numeric design."""
    columns = []
    continuous = []
    categorical = []
    schema = {}
    n_rows = len(context.word_df)
    for index, name in enumerate(tasks):
        spec = get_language_task(name)
        values = np.asarray(spec.getter(context), dtype=np.float64)
        if values.shape != (n_rows,):
            raise ValueError(
                f"Task {name} returned shape {values.shape}; expected {(n_rows,)}"
            )
        present = np.isfinite(values)
        if not present.any():
            raise ValueError(
                f"Requested task {name!r} has no available values for this dataset"
            )
        columns.append(values)
        (continuous if spec.kind == "continuous" else categorical).append(index)
        schema[name] = {
            "kind": spec.kind,
            "n_present": int(present.sum()),
            "n_missing": int((~present).sum()),
            "n_unique_present": int(len(np.unique(values[present]))),
        }
    return np.column_stack(columns), continuous, categorical, schema


def build_preprocessor(
    continuous: list[int],
    categorical: list[int],
) -> ColumnTransformer:
    """Build leakage-safe linear preprocessing for mixed task variables."""
    transformers = []
    if continuous:
        transformers.append(
            (
                "continuous",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="median", add_indicator=True),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                continuous,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="constant", fill_value=-1.0),
                        ),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, sparse_threshold=0.0)


def reconstruction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return calibration, coordinate, and representational reconstruction metrics."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Target and prediction shapes differ: {y_true.shape} vs {y_pred.shape}"
        )
    residual = np.square(y_true - y_pred).sum()
    denominator = np.square(y_true - y_true.mean(axis=0, keepdims=True)).sum()
    multivariate_r2 = float(1 - residual / denominator) if denominator > 0 else np.nan
    correlations = []
    for index in range(y_true.shape[1]):
        left, right = y_true[:, index], y_pred[:, index]
        if np.std(left) > 0 and np.std(right) > 0:
            correlations.append(float(np.corrcoef(left, right)[0, 1]))
    mean_correlation = float(np.mean(correlations)) if correlations else np.nan
    left = y_true - y_true.mean(axis=0, keepdims=True)
    right = y_pred - y_pred.mean(axis=0, keepdims=True)
    numerator = np.square(np.linalg.norm(left.T @ right, ord="fro"))
    cka_denominator = np.linalg.norm(left.T @ left, ord="fro") * np.linalg.norm(
        right.T @ right, ord="fro"
    )
    cka = float(numerator / cka_denominator) if cka_denominator > 0 else np.nan
    return {
        "multivariate_r2": multivariate_r2,
        "mean_coordinate_correlation": mean_correlation,
        "linear_cka": cka,
    }


def _fit_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: LanguageReconstructionConfig,
) -> RidgeCV:
    n_splits = min(int(config.encoding.ridge_cv_folds), len(X_train))
    if n_splits < 2:
        raise ValueError("At least two training rows are required for RidgeCV")
    model = RidgeCV(
        alphas=np.asarray(config.ridge_alphas or RIDGE_ALPHAS, dtype=float),
        cv=KFold(n_splits=n_splits, shuffle=False),
        fit_intercept=True,
        scoring="neg_mean_squared_error",
    )
    model.fit(X_train, y_train)
    return model


def _fit_dir(
    output_dir: Path,
    subspace: str,
    target_view: str,
    rank: int,
    fold: int,
) -> Path:
    return (
        output_dir
        / "fits"
        / subspace
        / target_view
        / f"k{rank}"
        / f"fold_{fold}"
    )


def _load_cached(path: Path, refresh: bool) -> dict | None:
    metrics = path / "metrics.json"
    if refresh or not metrics.exists():
        return None
    return json.loads(metrics.read_text(encoding="utf-8"))


def _save_fit(
    path: Path,
    row: dict,
    *,
    save_models: bool,
    preprocessor=None,
    model=None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics.json").write_text(
        json.dumps(row, indent=2, allow_nan=True), encoding="utf-8"
    )
    if save_models:
        joblib.dump(preprocessor, path / "preprocessor.joblib")
        joblib.dump(model, path / "model.joblib")


def _run_fit(
    *,
    output_dir: Path,
    subspace: str,
    target_view: str,
    rank: int,
    fold: int,
    task_matrix: np.ndarray,
    continuous: list[int],
    categorical: list[int],
    target: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: LanguageReconstructionConfig,
) -> dict:
    path = _fit_dir(output_dir, subspace, target_view, rank, fold)
    cached = _load_cached(path, config.refresh)
    if cached is not None:
        return cached
    preprocessor = build_preprocessor(continuous, categorical)
    X_train = np.asarray(preprocessor.fit_transform(task_matrix[train_idx]), dtype=np.float32)
    X_test = np.asarray(preprocessor.transform(task_matrix[test_idx]), dtype=np.float32)
    y_train, y_test = target[train_idx], target[test_idx]
    fitted_model = _fit_ridge(X_train, y_train, config)
    prediction = fitted_model.predict(X_test)
    row = {
        "subspace": subspace,
        "target_view": target_view,
        "rank": int(rank),
        "fold": int(fold),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_input_features": int(X_train.shape[1]),
        **reconstruction_metrics(y_test, prediction),
        "selected_alpha": np.asarray(fitted_model.alpha_).tolist(),
    }
    _save_fit(
        path,
        row,
        save_models=config.save_models,
        preprocessor=preprocessor,
        model=fitted_model,
    )
    return row


def summarize_results(grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize linear reconstruction metrics over outer folds."""
    fold_rows = grid.copy()
    summary_keys = ["subspace", "target_view", "rank"]
    means = grid.groupby(summary_keys, sort=False)[list(METRICS)].mean()
    sems = grid.groupby(summary_keys, sort=False)[list(METRICS)].sem()
    summary = means.reset_index()
    for metric in METRICS:
        summary[f"{metric}_sem"] = sems[metric].to_numpy()
    summary["n_folds"] = (
        grid.groupby(summary_keys, sort=False).size().to_numpy(dtype=int)
    )
    return fold_rows, summary


def summarize_comparisons(
    fold_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build fold-paired RRR-minus-PCA comparisons."""
    rows = []
    index = ["target_view", "rank", "fold"]
    for metric in METRICS:
        space_table = fold_results.pivot_table(
            index=index, columns="subspace", values=metric
        )
        if {"rrr", "pca"} <= set(space_table.columns):
            values = (space_table["rrr"] - space_table["pca"]).rename("value")
            for keys, value in values.items():
                target_view, rank, fold = keys
                rows.append(
                    {
                        "target_view": target_view,
                        "rank": rank,
                        "fold": fold,
                        "metric": metric,
                        "value": value,
                    }
                )
    comparison_grid = pd.DataFrame(rows)
    group_keys = ["target_view", "rank", "metric"]
    comparisons = (
        comparison_grid.groupby(group_keys, sort=False).value
        .agg(["mean", "sem", "count"])
        .reset_index()
        .rename(columns={"count": "n_folds"})
    )
    return comparison_grid, comparisons


def plot_results(summary: pd.DataFrame, output_dir: Path) -> None:
    """Write compact linear-reconstruction rank curves."""
    for metric in METRICS:
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for axis, target_view in zip(axes, ("weighted", "span")):
            subset = summary[summary.target_view == target_view]
            for subspace, group in subset.groupby("subspace"):
                group = group.sort_values("rank")
                axis.plot(
                    group["rank"], group[metric], marker="o", label=subspace
                )
            axis.set_xscale("log", base=2)
            axis.set_title(target_view)
            axis.set_xlabel("rank")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel(metric.replace("_", " "))
        axes[-1].legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / f"{metric}_by_rank.png", dpi=160)
        plt.close(figure)


def run_language_reconstruction(config: LanguageReconstructionConfig) -> Path:
    """Run task-to-subspace reconstruction and return the artifact directory."""
    enc = config.encoding
    subject_ids = enc.subjects or list(range(1, 10))
    dataset_context = resolve_dataset_context(
        enc.dataset, subject_ids, enc.per_dataset_config, enc.data_root
    )
    embeddings, word_df = load_embeddings(
        enc.data_root,
        enc.model,
        dataset=enc.dataset,
        per_dataset_config=enc.per_dataset_config,
        dataset_context=dataset_context,
        **enc.per_model_config,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    rrr_unit = resolve_subspace_unit_dir(config.rrr_run_path, "rrr")
    pca_unit = resolve_pca_sibling(rrr_unit)
    if not _factor_paths(pca_unit):
        raise FileNotFoundError(f"Matching PCA unit was not found at {pca_unit}")
    rrr_factors = _load_factors(rrr_unit, embeddings.shape[1])
    pca_factors = _load_factors(pca_unit, embeddings.shape[1])

    if config.mode == LanguageDecodingMode.WITHIN_DATASET:
        test_idxs, selection, fold_root = _load_within_folds_and_selection(
            config, rrr_unit, word_df
        )
        folds = _outer_folds(test_idxs, len(selection))
    else:
        selection = np.arange(len(word_df), dtype=int)
        fold_root = None
        folds = list(
            KFold(n_splits=enc.num_folds, shuffle=False).split(
                np.arange(len(selection))
            )
        )
    X = embeddings[selection]
    stimulus_id = str(getattr(dataset_context, "stimulus_id", enc.dataset))
    task_paths = _default_task_data(config, stimulus_id)
    context = PodcastTaskContext(word_df, enc.data_root, task_paths)
    task_matrix, continuous, categorical, schema = load_task_matrix(context, config.tasks)
    task_matrix = task_matrix[selection]

    target_root = config.output_dir or get_results_dir(
        enc.model,
        enc.offset,
        enc.per_model_config,
        dataset_context=dataset_context,
    )
    source_identity = f"from_{source_slug(rrr_unit, config.source_name)}"
    output_dir = Path(target_root) / "language_reconstruction" / config.mode.value / source_identity
    save_effective_config(config, output_dir)
    (output_dir / "task_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {"analysis_row": np.arange(len(selection)), "transcript_word_idx": selection}
    ).to_csv(output_dir / "word_selection.csv", index=False)

    prepared = {
        "rrr": prepare_target_factors(
            rrr_factors,
            folds,
            config.rank_grid,
            config.mode,
            config.subspace_aggregation,
        ),
        "pca": prepare_target_factors(
            pca_factors,
            folds,
            config.rank_grid,
            config.mode,
            config.subspace_aggregation,
        ),
    }
    metadata = {
        "config": asdict(config),
        "rrr_unit_dir": str(rrr_unit),
        "pca_unit_dir": str(pca_unit),
        "fold_results_dir": None if fold_root is None else str(fold_root),
        "analysis_rows": int(len(selection)),
        "embedding_shape": list(embeddings.shape),
        "continuous_tasks": [config.tasks[index] for index in continuous],
        "categorical_tasks": [config.tasks[index] for index in categorical],
        "cross_weighted_strategy": (
            "top_factor_of_mean_AAt"
            if config.mode == LanguageDecodingMode.CROSS_DATASET
            else None
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    rows = []
    for subspace in ("rrr", "pca"):
        print(f"\n[{subspace.upper()} targets]")
        for target_view in config.target_views:
            print(f"  [{target_view}]")
            for fold, (train_idx, test_idx) in enumerate(folds):
                for rank in map(int, config.rank_grid):
                    factor = prepared[subspace][(target_view, fold, rank)]
                    target = np.asarray(X @ factor, dtype=np.float32)
                    rows.append(
                        _run_fit(
                            output_dir=output_dir,
                            subspace=subspace,
                            target_view=target_view,
                            rank=rank,
                            fold=fold,
                            task_matrix=task_matrix,
                            continuous=continuous,
                            categorical=categorical,
                            target=target,
                            train_idx=np.asarray(train_idx),
                            test_idx=np.asarray(test_idx),
                            config=config,
                        )
                    )
    grid = pd.DataFrame(rows)
    grid.to_csv(output_dir / "grid_results.csv", index=False)
    fold_results, summary = summarize_results(grid)
    fold_results.to_csv(output_dir / "fold_results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    comparison_grid, comparisons = summarize_comparisons(fold_results)
    comparison_grid.to_csv(output_dir / "comparison_grid_results.csv", index=False)
    comparisons.to_csv(output_dir / "rrr_pca_comparison.csv", index=False)
    plot_results(summary, output_dir)
    return output_dir
