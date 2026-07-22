"""Fold-safe linguistic decoding from saved RRR and PCA input subspaces."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.core.config import (
    LanguageDecodingConfig,
    LanguageDecodingMode,
    get_results_dir,
)
from src.core.registry import resolve_dataset_context
from src.utils.constants import RIDGE_ALPHAS
from src.utils.correlation import fisher_z_mean_sem
from src.utils.data_loaders import (
    get_precomputed_epochs_path,
    load_precomputed_epochs,
    selection_after_offset,
)
from src.utils.embeddings import load_embeddings
from src.utils.language_tasks import PodcastTaskContext, get_language_task
from src.utils.subspace_compare import (
    aggregate_subspace_basis,
    validate_aggregation_cache,
)


def resolve_subspace_unit_dir(run_path: str, kind: str = "subspace") -> Path:
    """Resolve a configured run root or direct saved-subspace unit directory."""
    root = Path(run_path).expanduser().resolve()
    candidates = [root, root / kind]
    valid = [path for path in candidates if list(path.glob("fold_*/A.npy"))]
    if len(valid) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {kind.upper()} unit at {root} or {root / kind} with "
            f"fold_*/A.npy; found {valid}"
        )
    return valid[0]


def resolve_rrr_unit_dir(run_path: str) -> Path:
    """Backward-compatible RRR-specific saved-subspace resolver."""
    return resolve_subspace_unit_dir(run_path, "rrr")


def resolve_fold_results_dir(
    unit_dir: Path, configured: str | None = None
) -> Path:
    """Find the encoding-result root whose saved folds produced a within-run A."""
    if configured:
        root = Path(configured).expanduser().resolve()
        if list(root.glob("subjects/sub_*/encoding_results.joblib")):
            return root
        raise FileNotFoundError(f"No subject encoding results found under {root}")
    for root in [unit_dir, *unit_dir.parents]:
        if list(root.glob("subjects/sub_*/encoding_results.joblib")):
            return root
    raise FileNotFoundError(
        "Could not locate subjects/sub_*/encoding_results.joblib above the RRR "
        "unit. Set fold_results_dir explicitly."
    )


def source_slug(unit_dir: Path, source_name: str | None) -> str:
    """Return a readable, collision-resistant output identity for an A source."""
    if source_name:
        label = source_name
    else:
        parts = unit_dir.parts
        start = parts.index("results") + 1 if "results" in parts else max(0, len(parts) - 4)
        label = "__".join(parts[start:])
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "rrr_source"
    digest = hashlib.sha256(str(unit_dir).encode()).hexdigest()[:8]
    return f"{safe}_{digest}"


def _load_within_folds_and_selection(
    config: LanguageDecodingConfig,
    unit_dir: Path,
    word_df: pd.DataFrame,
) -> tuple[list[np.ndarray], np.ndarray, Path]:
    enc = config.encoding
    fold_root = resolve_fold_results_dir(unit_dir, config.fold_results_dir)
    result_paths = sorted(fold_root.glob("subjects/sub_*/encoding_results.joblib"))
    configured_subjects = set(enc.subjects or [])
    if configured_subjects:
        matching = [
            path
            for path in result_paths
            if int(path.parent.name.removeprefix("sub_")) in configured_subjects
        ]
        if matching:
            result_paths = matching
    if not result_paths:
        raise FileNotFoundError(f"No compatible encoding results under {fold_root}")

    result_path = result_paths[0]
    encoding_results = joblib.load(result_path)
    test_idxs = [np.asarray(index, dtype=int) for index in encoding_results["test_idxs"]]
    metadata = encoding_results.get("word_selection")
    if metadata is not None:
        selection = np.asarray(metadata["analysis_selection"], dtype=int)
    else:
        subject = int(result_path.parent.name.removeprefix("sub_"))
        context = resolve_dataset_context(
            enc.dataset,
            [subject],
            enc.per_dataset_config,
            enc.data_root,
        )
        try:
            cache = load_precomputed_epochs(
                subject,
                enc.data_root,
                dataset=enc.dataset,
                per_dataset_config=enc.per_dataset_config,
                dataset_context=context,
            )
        except FileNotFoundError:
            h5_path = get_precomputed_epochs_path(
                subject,
                enc.data_root,
                dataset=enc.dataset,
                per_dataset_config=enc.per_dataset_config,
                dataset_context=context,
            )
            legacy_path = os.path.splitext(h5_path)[0] + ".npz"
            if not os.path.exists(legacy_path):
                raise
            with np.load(legacy_path, allow_pickle=False) as legacy:
                cache = {"selection": np.asarray(legacy["selection"], dtype=int)}
        valid_rows = None
        if enc.limit_to_glove or enc.model == "glove":
            if "in_glove" not in word_df:
                raise ValueError("GloVe word selection requires word_df['in_glove']")
            valid_rows = word_df["in_glove"].to_numpy(dtype=bool)[cache["selection"]]
        selection = selection_after_offset(
            cache["selection"], int(enc.offset), valid_rows=valid_rows
        )

    fold_union = np.concatenate(test_idxs) if test_idxs else np.array([], dtype=int)
    if len(fold_union) != len(selection) or not np.array_equal(
        np.sort(fold_union), np.arange(len(selection))
    ):
        raise ValueError(
            "Saved within-dataset folds do not partition the reconstructed RRR "
            f"selection: folds={len(fold_union)}, selection={len(selection)}"
        )
    return test_idxs, selection, fold_root


def _outer_folds(test_idxs: list[np.ndarray], n_rows: int):
    folds = []
    for test_idx in test_idxs:
        train_mask = np.ones(n_rows, dtype=bool)
        train_mask[test_idx] = False
        folds.append((np.flatnonzero(train_mask), np.asarray(test_idx, dtype=int)))
    return folds


def _inner_cv(n_rows: int, requested: int) -> KFold:
    n_splits = min(int(requested), int(n_rows))
    if n_splits < 2:
        raise ValueError(f"At least two training examples are required, got {n_rows}")
    return KFold(n_splits=n_splits, shuffle=False)


def _fit_decoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kind: str,
    config: LanguageDecodingConfig,
):
    scaler = StandardScaler()
    if kind == "continuous":
        model = RidgeCV(
            alphas=np.asarray(config.encoding.ridge_alphas or RIDGE_ALPHAS),
            cv=_inner_cv(len(y_train), config.encoding.ridge_cv_folds),
        )
    else:
        classes = np.unique(y_train)
        if len(classes) < 2:
            raise ValueError(f"Categorical training fold contains only class {classes}")
        model = LogisticRegressionCV(
            Cs=np.asarray(config.logistic_cs, dtype=float),
            cv=_inner_cv(len(y_train), config.encoding.ridge_cv_folds),
            scoring="balanced_accuracy",
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            n_jobs=1,
        )
    pipeline = make_pipeline(scaler, model)
    pipeline.fit(X_train, y_train)
    return pipeline


def _decoder_metrics(kind: str, y_true, y_pred) -> dict[str, float]:
    if kind == "continuous":
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            correlation = float("nan")
        else:
            correlation = float(np.corrcoef(y_true, y_pred)[0, 1])
        return {
            "primary_score": correlation,
            "correlation": correlation,
            "r2": float(r2_score(y_true, y_pred)),
        }
    return {
        "primary_score": float(balanced_accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _weight_payload(pipeline, A: np.ndarray | None) -> dict:
    scaler = pipeline[0]
    estimator = pipeline[-1]
    coefficient = np.asarray(estimator.coef_, dtype=float)
    if coefficient.ndim == 1:
        coefficient = coefficient[None, :]
    intercept = np.atleast_1d(np.asarray(estimator.intercept_, dtype=float))
    reduced_weights = (coefficient / scaler.scale_[None, :]).T
    corrected_intercept = intercept - scaler.mean_ @ reduced_weights
    embedding_weights = (
        reduced_weights if A is None else np.asarray(A, dtype=float) @ reduced_weights
    )
    payload = {
        "reduced_weights": reduced_weights.astype(np.float32),
        "embedding_weights": embedding_weights.astype(np.float32),
        "intercept": corrected_intercept.astype(np.float32),
        "feature_mean": np.asarray(scaler.mean_, dtype=np.float32),
        "feature_scale": np.asarray(scaler.scale_, dtype=np.float32),
    }
    if hasattr(estimator, "classes_"):
        payload["classes"] = np.asarray(estimator.classes_)
        payload["selected_regularization"] = np.asarray(estimator.C_, dtype=float)
    else:
        payload["classes"] = np.array([], dtype=float)
        payload["selected_regularization"] = np.atleast_1d(
            np.asarray(estimator.alpha_, dtype=float)
        )
    return payload


def _artifact_dir(
    output_dir: Path,
    task: str,
    fold: int,
    rank: int | None,
    subspace: str,
) -> Path:
    if rank is None:
        return output_dir / "tasks" / task / "full" / f"fold_{fold}"
    return output_dir / "tasks" / task / subspace / f"k{rank}" / f"fold_{fold}"


def _run_one_fit(
    output_dir: Path,
    task_name: str,
    task_kind: str,
    fold: int,
    rank: int | None,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: LanguageDecodingConfig,
    A: np.ndarray | None,
    subspace: str,
) -> dict:
    artifact_dir = _artifact_dir(output_dir, task_name, fold, rank, subspace)
    metrics_path = artifact_dir / "metrics.json"
    weights_path = artifact_dir / "weights.npz"
    if not config.refresh and metrics_path.exists() and weights_path.exists():
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("subspace") != subspace:
            metrics["subspace"] = subspace
            metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=True))
        return metrics

    valid = np.isfinite(y)
    task_train = train_idx[valid[train_idx]]
    task_test = test_idx[valid[test_idx]]
    if not len(task_train) or not len(task_test):
        raise ValueError(f"Task {task_name} has an empty train or test set in fold {fold}")
    pipeline = _fit_decoder(X[task_train], y[task_train], task_kind, config)
    prediction = pipeline.predict(X[task_test])
    metrics = {
        "task": task_name,
        "kind": task_kind,
        "fold": int(fold),
        "space": "full" if rank is None else "rank",
        "subspace": subspace,
        "k": None if rank is None else int(rank),
        "n_train": int(len(task_train)),
        "n_test": int(len(task_test)),
        **_decoder_metrics(task_kind, y[task_test], prediction),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez(weights_path, **_weight_payload(pipeline, A))
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=True))
    return metrics


def summarize_language_decoding(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold metrics, using Fisher-z means and SEM for correlations."""
    rows = []
    for (task, space, subspace, k), group in results.groupby(
        ["task", "space", "subspace", "k"], dropna=False, sort=False
    ):
        values = group["primary_score"].to_numpy(dtype=float)
        kind = str(group["kind"].iloc[0])
        if kind == "continuous":
            mean, lower, upper = fisher_z_mean_sem(values)
            mean, sem_lower, sem_upper = float(mean), float(lower), float(upper)
        else:
            mean = float(np.nanmean(values))
            sem = float(np.nanstd(values, ddof=1) / np.sqrt(np.isfinite(values).sum()))
            sem_lower = sem_upper = 0.0 if not np.isfinite(sem) else sem
        row = {
            "task": task,
            "kind": kind,
            "space": space,
            "subspace": subspace,
            "k": k,
            "primary_score": mean,
            "sem_lower": sem_lower,
            "sem_upper": sem_upper,
            "n_folds": int(len(group)),
        }
        for metric in ("correlation", "r2", "balanced_accuracy", "macro_f1", "accuracy"):
            if metric in group and group[metric].notna().any():
                metric_values = group[metric].to_numpy(dtype=float)
                row[metric] = (
                    float(fisher_z_mean_sem(metric_values)[0])
                    if metric == "correlation"
                    else float(np.nanmean(metric_values))
                )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_language_decoding(summary: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    task_names = list(dict.fromkeys(summary["task"].tolist()))
    n_cols = min(3, len(task_names))
    n_rows = int(np.ceil(len(task_names) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.2 * n_cols, 4.0 * n_rows), squeeze=False
    )
    for ax, task_name in zip(axes.flat, task_names):
        task_summary = summary[summary["task"] == task_name]
        spec = get_language_task(task_name)
        colors = {"rrr": "#2563eb", "pca": "#f97316"}
        for subspace in ("rrr", "pca"):
            ranked = task_summary[
                (task_summary["space"] == "rank")
                & (task_summary["subspace"] == subspace)
            ].sort_values("k")
            if ranked.empty:
                continue
            ax.errorbar(
                ranked["k"].to_numpy(dtype=float),
                ranked["primary_score"].to_numpy(dtype=float),
                yerr=np.vstack(
                    [
                        ranked["sem_lower"].to_numpy(),
                        ranked["sem_upper"].to_numpy(),
                    ]
                ),
                marker="o",
                color=colors[subspace],
                capsize=3,
                label=f"{subspace.upper()} subspace",
            )
        full = task_summary[task_summary["space"] == "full"]
        if len(full):
            ax.axhline(
                float(full.iloc[0]["primary_score"]),
                color="#6b7280",
                linestyle="--",
                label="Full embeddings",
            )
        if spec.chance is not None:
            ax.axhline(spec.chance, color="#d1d5db", linestyle=":", label="Chance")
        ax.set_xscale("log")
        ax.set_xlabel("Subspace rank")
        ax.set_ylabel("Pearson r" if spec.kind == "continuous" else "Balanced accuracy")
        ax.set_title(spec.display_name)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes.flat[len(task_names) :]:
        ax.set_visible(False)
    fig.suptitle("Language-feature decoding vs subspace rank")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _validate_factor_paths(
    factor_paths: list[Path], embedding_dim: int, kind: str
) -> None:
    if not factor_paths:
        raise FileNotFoundError(f"No {kind.upper()} fold A matrices found")
    ambient_dims = {np.load(path, mmap_mode="r").shape[0] for path in factor_paths}
    if ambient_dims != {embedding_dim}:
        raise ValueError(
            f"{kind.upper()} ambient dimensions {sorted(ambient_dims)} do not match "
            f"embeddings dimension {embedding_dim}"
        )


def _run_subspace_decoding(
    *,
    output_dir: Path,
    subspace: str,
    factor_paths: list[Path],
    folds: list[tuple[np.ndarray, np.ndarray]],
    X: np.ndarray,
    task_targets: dict[str, np.ndarray],
    config: LanguageDecodingConfig,
) -> list[dict]:
    """Load one source's A matrices and run its complete decoding sweep."""
    matrices_by_fold = {
        int(path.parent.name.removeprefix("fold_")): np.load(path)
        for path in factor_paths
    }
    if config.mode == LanguageDecodingMode.WITHIN_DATASET:
        expected_folds = set(range(len(folds)))
        if set(matrices_by_fold) != expected_folds:
            raise ValueError(
                f"{subspace.upper()} A-matrix folds {sorted(matrices_by_fold)} do not "
                f"match decoding folds {sorted(expected_folds)}"
            )
        mean_factors = None
    else:
        source_matrices = list(matrices_by_fold.values())
        mean_factors = {
            rank: aggregate_subspace_basis(
                source_matrices,
                rank=int(rank),
                method=config.subspace_aggregation,
            )
            for rank in config.rank_grid
        }

    rows = []
    print(f"\n[{subspace.upper()} subspace]")
    for fold, (train_idx, test_idx) in enumerate(folds):
        print(f"  [fold {fold}]")
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)
        for rank in config.rank_grid:
            A = (
                matrices_by_fold[fold][:, :rank]
                if mean_factors is None
                else mean_factors[rank]
            )
            if A.shape[1] < int(rank):
                raise ValueError(
                    f"Fold {fold} {subspace.upper()} A has insufficient width for "
                    f"rank {rank}"
                )
            X_rank = X @ A
            for task_name in config.tasks:
                spec = get_language_task(task_name)
                rows.append(
                    _run_one_fit(
                        output_dir,
                        task_name,
                        spec.kind,
                        fold,
                        int(rank),
                        X_rank,
                        task_targets[task_name],
                        train_idx,
                        test_idx,
                        config,
                        A,
                        subspace,
                    )
                )
    return rows


def run_language_decoding(config: LanguageDecodingConfig) -> Path:
    """Run the configured task/rank sweep and return its output directory."""
    enc = config.encoding
    if enc.dataset != "podcast":
        raise ValueError("Language decoding currently supports dataset='podcast' only")
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
    rrr_unit_dir = resolve_subspace_unit_dir(config.rrr_run_path, "rrr")
    pca_unit_dir = (
        resolve_subspace_unit_dir(config.pca_run_path, "pca")
        if config.pca_run_path
        else None
    )
    rrr_factor_paths = sorted(rrr_unit_dir.glob("fold_*/A.npy"))
    pca_factor_paths = (
        sorted(pca_unit_dir.glob("fold_*/A.npy"))
        if pca_unit_dir is not None
        else []
    )
    _validate_factor_paths(rrr_factor_paths, embeddings.shape[1], "rrr")
    if pca_factor_paths:
        _validate_factor_paths(pca_factor_paths, embeddings.shape[1], "pca")

    target_root = config.output_dir or get_results_dir(
        enc.model,
        enc.offset,
        enc.per_model_config,
        dataset_context=dataset_context,
    )
    source_identity = f"from_{source_slug(rrr_unit_dir, config.source_name)}"
    if pca_unit_dir is not None:
        source_identity += f"__with_pca_{source_slug(pca_unit_dir, None)}"
    output_dir = (
        Path(target_root) / "language_decoding" / config.mode.value / source_identity
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_aggregation_cache(
        output_dir, config.subspace_aggregation, refresh=config.refresh
    )

    context = PodcastTaskContext(word_df, enc.data_root, config.task_data)
    task_targets = {
        name: np.asarray(get_language_task(name).getter(context)) for name in config.tasks
    }
    for name, target in task_targets.items():
        if target.shape != (len(word_df),):
            raise ValueError(
                f"Task {name} returned shape {target.shape}; expected {(len(word_df),)}"
            )

    if config.mode == LanguageDecodingMode.WITHIN_DATASET:
        test_idxs, selection, fold_root = _load_within_folds_and_selection(
            config, rrr_unit_dir, word_df
        )
        X = embeddings[selection]
        task_targets = {name: target[selection] for name, target in task_targets.items()}
        folds = _outer_folds(test_idxs, len(X))
    else:
        selection = np.arange(len(word_df), dtype=int)
        X = embeddings
        fold_root = None
        folds = list(
            KFold(n_splits=enc.num_folds, shuffle=False).split(np.arange(len(X)))
        )

    if "iu_alignment" in context._cache:
        context.iu_alignment()[2].to_csv(output_dir / "iu_alignment.csv", index=False)
    pd.DataFrame(
        {
            "analysis_row": np.arange(len(selection)),
            "transcript_word_idx": selection,
        }
    ).to_csv(output_dir / "word_selection.csv", index=False)
    metadata = {
        "config": asdict(config),
        "subspace_unit_dirs": {
            "rrr": str(rrr_unit_dir),
            **({"pca": str(pca_unit_dir)} if pca_unit_dir is not None else {}),
        },
        "rrr_unit_dir": str(rrr_unit_dir),
        "pca_unit_dir": None if pca_unit_dir is None else str(pca_unit_dir),
        "fold_results_dir": None if fold_root is None else str(fold_root),
        "output_dir": str(output_dir),
        "embedding_shape": list(embeddings.shape),
        "analysis_rows": int(len(X)),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )

    rows = []
    print("\n[Full embeddings]")
    for fold, (train_idx, test_idx) in enumerate(folds):
        print(f"  [fold {fold}]")
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)
        for task_name in config.tasks:
            spec = get_language_task(task_name)
            rows.append(
                _run_one_fit(
                    output_dir,
                    task_name,
                    spec.kind,
                    fold,
                    None,
                    X,
                    task_targets[task_name],
                    train_idx,
                    test_idx,
                    config,
                    None,
                    "full",
                )
            )

    rows.extend(
        _run_subspace_decoding(
            output_dir=output_dir,
            subspace="rrr",
            factor_paths=rrr_factor_paths,
            folds=folds,
            X=X,
            task_targets=task_targets,
            config=config,
        )
    )
    if pca_factor_paths:
        rows.extend(
            _run_subspace_decoding(
                output_dir=output_dir,
                subspace="pca",
                factor_paths=pca_factor_paths,
                folds=folds,
                X=X,
                task_targets=task_targets,
                config=config,
            )
        )

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "grid_results.csv", index=False)
    summary = summarize_language_decoding(results)
    summary.to_csv(output_dir / "summary.csv", index=False)
    plot_language_decoding(summary, output_dir / "rank_analysis.png")
    return output_dir
