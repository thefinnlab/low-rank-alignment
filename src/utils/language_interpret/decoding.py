"""Fold-safe linguistic decoding from saved RRR and PCA subspaces."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.core.config import (
    LanguageDecodingConfig,
    LanguageDecodingMode,
    RRRConfig,
    ScriptMode,
    get_results_dir,
)
from src.core.registry import resolve_dataset_context
from src.utils.shared.constants import RIDGE_ALPHAS
from src.utils.shared.correlation import fisher_z_mean_sem
from src.utils.shared.ieeg import (
    get_precomputed_epochs_path,
    load_precomputed_epochs,
    selection_after_offset,
)
from src.utils.shared.embeddings import load_embeddings
from src.utils.shared.config import load_config, save_effective_config
from src.utils.language_interpret.tasks import PodcastTaskContext, get_language_task
from src.utils.rrr.multiview import (
    load_view_bases,
    load_view_embeddings,
    project_view_bands,
    resolve_rrr_views,
    target_dataset_key,
)
from src.utils.rrr.subspace_io import load_base_comparison_data
from src.utils.subspace_comparison.metrics import (
    aggregate_subspace_basis,
    validate_aggregation_cache,
)
from src.utils.shared.tensors import to_numpy


def resolve_subspace_unit_dir(run_path: str, kind: str = "subspace") -> Path:
    """Resolve a configured run root or direct saved-subspace unit directory."""
    root = Path(run_path).expanduser().resolve()
    candidates = [root, root / kind]
    valid = [
        path for path in candidates
        if list(path.glob("fold_*/A.npy")) or list(path.glob("rank_*/A.npy"))
    ]
    if len(valid) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {kind.upper()} unit at {root} or {root / kind} with "
            f"fold_*/A.npy or rank_*/A.npy; found {valid}"
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


def _configure_decoder_backend(device: str) -> str:
    """Configure Himalaya and return the effective continuous-decoder device."""
    import torch

    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be one of ['auto', 'cpu', 'cuda']")
    use_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Language decoding requested CUDA, but CUDA is unavailable")
    set_backend("torch_cuda" if use_cuda else "numpy")
    return "cuda" if use_cuda else "cpu"


def _fit_decoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kind: str,
    config: LanguageDecodingConfig,
    cv=None,
):
    scaler = StandardScaler()
    if kind == "continuous":
        _configure_decoder_backend(config.device)
        model = RidgeCV(
            alphas=np.asarray(config.encoding.ridge_alphas or RIDGE_ALPHAS),
            cv=cv or _inner_cv(len(y_train), config.encoding.ridge_cv_folds),
            fit_intercept=True,
            solver_params=config.encoding.ridge_solver_params,
            Y_in_cpu=bool(config.encoding.ridge_y_in_cpu or False),
            force_cpu=False,
        )
    else:
        classes = np.unique(y_train)
        if len(classes) < 2:
            raise ValueError(f"Categorical training fold contains only class {classes}")
        model = LogisticRegressionCV(
            Cs=np.asarray(config.logistic_cs, dtype=float),
            cv=cv or _inner_cv(len(y_train), config.encoding.ridge_cv_folds),
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
    y_true = np.asarray(to_numpy(y_true))
    y_pred = np.asarray(to_numpy(y_pred))
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


def _weight_payload(
    pipeline, A: np.ndarray | None, target_index: int | None = None
) -> dict:
    scaler = pipeline[0]
    estimator = pipeline[-1]
    coefficient = np.asarray(to_numpy(estimator.coef_), dtype=float)
    intercept = np.atleast_1d(
        np.asarray(to_numpy(estimator.intercept_), dtype=float)
    )
    if hasattr(estimator, "classes_"):
        if coefficient.ndim == 1:
            coefficient = coefficient[None, :]
        reduced_weights = (coefficient / scaler.scale_[None, :]).T
    else:
        if coefficient.ndim == 1:
            coefficient = coefficient[:, None]
        elif coefficient.shape[0] != len(scaler.scale_):
            coefficient = coefficient.T
        reduced_weights = coefficient / scaler.scale_[:, None]
        if target_index is not None:
            reduced_weights = reduced_weights[:, [target_index]]
            intercept = intercept[[target_index]]
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
        selected = np.atleast_1d(
            np.asarray(to_numpy(estimator.best_alphas_), dtype=float)
        )
        if target_index is not None:
            selected = selected[[target_index]]
        payload["selected_regularization"] = selected
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


def _load_cached_fit(
    output_dir: Path,
    task_name: str,
    fold: int,
    rank: int | None,
    subspace: str,
    refresh: bool,
) -> dict | None:
    artifact_dir = _artifact_dir(output_dir, task_name, fold, rank, subspace)
    metrics_path = artifact_dir / "metrics.json"
    weights_path = artifact_dir / "weights.npz"
    if refresh or not metrics_path.exists() or not weights_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("subspace") != subspace:
        metrics["subspace"] = subspace
        metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=True))
    return metrics


def _fit_indices(
    task_name: str,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(y)
    task_train = train_idx[valid[train_idx]]
    task_test = test_idx[valid[test_idx]]
    if not len(task_train) or not len(task_test):
        raise ValueError(
            f"Task {task_name} has an empty train or test set in fold {fold}"
        )
    return task_train, task_test


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
    block_ids: np.ndarray | None = None,
) -> dict:
    cached = _load_cached_fit(
        output_dir, task_name, fold, rank, subspace, config.refresh
    )
    if cached is not None:
        return cached

    artifact_dir = _artifact_dir(output_dir, task_name, fold, rank, subspace)
    metrics_path = artifact_dir / "metrics.json"
    weights_path = artifact_dir / "weights.npz"
    task_train, task_test = _fit_indices(
        task_name, y, train_idx, test_idx, fold
    )
    pipeline = _fit_decoder(X[task_train], y[task_train], task_kind, config)
    prediction = to_numpy(pipeline.predict(X[task_test]))
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
    np.savez_compressed(
        artifact_dir / "heldout_predictions.npz",
        y_true=np.asarray(y[task_test]),
        y_pred=np.asarray(prediction),
        row_index=np.asarray(task_test, dtype=int),
        block_id=np.asarray(
            task_test if block_ids is None else block_ids[task_test], dtype=str
        ),
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=True))
    return metrics


def _run_task_decoding(
    *,
    output_dir: Path,
    fold: int,
    rank: int | None,
    X: np.ndarray,
    task_targets: dict[str, np.ndarray],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: LanguageDecodingConfig,
    A: np.ndarray | None,
    subspace: str,
    block_ids: np.ndarray | None = None,
) -> list[dict]:
    """Fit one design, batching continuous targets with identical valid rows."""
    results: dict[str, dict] = {}
    continuous_groups: dict[tuple[tuple[int, ...], tuple[int, ...]], list[str]] = {}

    for task_name in config.tasks:
        spec = get_language_task(task_name)
        if spec.kind != "continuous":
            fit_args = (
                output_dir,
                task_name,
                spec.kind,
                fold,
                rank,
                X,
                task_targets[task_name],
                train_idx,
                test_idx,
                config,
                A,
                subspace,
            )
            results[task_name] = _run_one_fit(
                *fit_args, *(() if block_ids is None else (block_ids,))
            )
            continue
        cached = _load_cached_fit(
            output_dir, task_name, fold, rank, subspace, config.refresh
        )
        if cached is not None:
            results[task_name] = cached
            continue
        task_train, task_test = _fit_indices(
            task_name, task_targets[task_name], train_idx, test_idx, fold
        )
        key = (tuple(task_train.tolist()), tuple(task_test.tolist()))
        continuous_groups.setdefault(key, []).append(task_name)

    for (train_key, test_key), task_names in continuous_groups.items():
        task_train = np.asarray(train_key, dtype=int)
        task_test = np.asarray(test_key, dtype=int)
        if len(task_names) == 1:
            task_name = task_names[0]
            fit_args = (
                output_dir,
                task_name,
                "continuous",
                fold,
                rank,
                X,
                task_targets[task_name],
                train_idx,
                test_idx,
                config,
                A,
                subspace,
            )
            results[task_name] = _run_one_fit(
                *fit_args, *(() if block_ids is None else (block_ids,))
            )
            continue
        targets = np.column_stack(
            [task_targets[task_name][task_train] for task_name in task_names]
        )
        pipeline = _fit_decoder(X[task_train], targets, "continuous", config)
        predictions = np.asarray(to_numpy(pipeline.predict(X[task_test])))
        if predictions.ndim == 1:
            predictions = predictions[:, None]
        for target_index, task_name in enumerate(task_names):
            y = task_targets[task_name]
            metrics = {
                "task": task_name,
                "kind": "continuous",
                "fold": int(fold),
                "space": "full" if rank is None else "rank",
                "subspace": subspace,
                "k": None if rank is None else int(rank),
                "n_train": int(len(task_train)),
                "n_test": int(len(task_test)),
                **_decoder_metrics(
                    "continuous", y[task_test], predictions[:, target_index]
                ),
            }
            artifact_dir = _artifact_dir(
                output_dir, task_name, fold, rank, subspace
            )
            artifact_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                artifact_dir / "weights.npz",
                **_weight_payload(pipeline, A, target_index=target_index),
            )
            np.savez_compressed(
                artifact_dir / "heldout_predictions.npz",
                y_true=np.asarray(y[task_test]),
                y_pred=np.asarray(predictions[:, target_index]),
                row_index=np.asarray(task_test, dtype=int),
                block_id=np.asarray(
                    task_test if block_ids is None else block_ids[task_test], dtype=str
                ),
            )
            (artifact_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, allow_nan=True)
            )
            results[task_name] = metrics

    return [results[task_name] for task_name in config.tasks]


def summarize_language_decoding(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold metrics, using Fisher-z means and SEM for correlations."""
    rows = []
    for (task, space, subspace, k), group in results.groupby(
        ["task", "space", "subspace", "k"], dropna=False, sort=False
    ):
        values = group["primary_score"].to_numpy(dtype=float)
        kind = str(group["kind"].iloc[0])
        correlation_primary = (
            kind == "continuous"
            or str(group.get("primary_metric", pd.Series([""])).iloc[0])
            == "correlation"
        )
        if correlation_primary:
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
        for metric in (
            "correlation", "r2", "multivariate_r2", "mean_cosine_similarity",
            "balanced_accuracy", "macro_f1", "accuracy",
        ):
            if metric in group and group[metric].notna().any():
                metric_values = group[metric].to_numpy(dtype=float)
                row[metric] = (
                    float(fisher_z_mean_sem(metric_values)[0])
                    if metric == "correlation"
                    else float(np.nanmean(metric_values))
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    """Return Pearson correlation, or NaN for invalid/constant paired values."""
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def _saved_decision_scores(
    output_dir: Path,
    task: str,
    subspace: str,
    rank: int,
    fold: int,
    design: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct held-out continuous/logit scores from a saved decoder artifact."""
    path = _artifact_dir(output_dir, task, fold, rank, subspace) / "weights.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing saved decoder weights: {path}")
    with np.load(path, allow_pickle=False) as payload:
        weights = np.asarray(payload["reduced_weights"], dtype=float)
        intercept = np.asarray(payload["intercept"], dtype=float).reshape(-1)
        classes = np.asarray(payload["classes"])
    if weights.ndim == 1:
        weights = weights[:, None]
    scores = np.asarray(design[test_idx], dtype=float) @ weights + intercept
    if scores.ndim == 1:
        scores = scores[:, None]
    return scores, classes


def _matched_score_correlation(
    left: np.ndarray, right: np.ndarray
) -> tuple[float, int]:
    """Correlate matched score columns and Fisher-average multiclass values."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim == 1:
        left = left[:, None]
    if right.ndim == 1:
        right = right[:, None]
    if left.shape != right.shape:
        raise ValueError(
            f"Paired decoder score shapes differ: {left.shape} vs {right.shape}"
        )
    correlations = np.asarray(
        [_safe_pearson(left[:, index], right[:, index]) for index in range(left.shape[1])]
    )
    finite = correlations[np.isfinite(correlations)]
    if not len(finite):
        return float("nan"), 0
    return float(fisher_z_mean_sem(finite)[0]), int(len(finite))


def compute_prediction_agreement(
    *,
    output_dir: Path,
    rrr_prepared: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    pca_prepared: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    folds: list[tuple[np.ndarray, np.ndarray]],
    task_targets: dict[str, np.ndarray],
    config: LanguageDecodingConfig,
) -> pd.DataFrame:
    """Compare fold-held-out RRR/PCA decoder outputs at each task and rank."""
    rows = []
    for fold, (train_idx, test_idx) in enumerate(folds):
        for rank in map(int, config.rank_grid):
            rrr_design = rrr_prepared[(fold, rank)][0]
            pca_design = pca_prepared[(fold, rank)][0]
            for task in config.tasks:
                spec = get_language_task(task)
                _, task_test = _fit_indices(
                    task,
                    task_targets[task],
                    np.asarray(train_idx),
                    np.asarray(test_idx),
                    fold,
                )
                rrr_scores, rrr_classes = _saved_decision_scores(
                    output_dir, task, "rrr", rank, fold, rrr_design, task_test
                )
                pca_scores, pca_classes = _saved_decision_scores(
                    output_dir, task, "pca", rank, fold, pca_design, task_test
                )
                if not np.array_equal(rrr_classes, pca_classes):
                    raise ValueError(
                        f"RRR/PCA decoder classes differ for {task}, fold={fold}, "
                        f"rank={rank}: {rrr_classes} vs {pca_classes}"
                    )
                prediction_correlation, n_score_columns = _matched_score_correlation(
                    rrr_scores, pca_scores
                )
                error_correlation = float("nan")
                if spec.kind == "continuous":
                    target = np.asarray(task_targets[task][task_test], dtype=float)
                    error_correlation = _safe_pearson(
                        target - rrr_scores[:, 0], target - pca_scores[:, 0]
                    )
                rows.append(
                    {
                        "task": task,
                        "kind": spec.kind,
                        "k": rank,
                        "fold": fold,
                        "n_test": int(len(task_test)),
                        "n_score_columns": n_score_columns,
                        "prediction_correlation": prediction_correlation,
                        "error_correlation": error_correlation,
                    }
                )
    return pd.DataFrame(rows)


def summarize_prediction_agreement(results: pd.DataFrame) -> pd.DataFrame:
    """Fisher-average paired decoder correlations across outer folds."""
    rows = []
    for (task, kind, rank), group in results.groupby(
        ["task", "kind", "k"], sort=False
    ):
        row = {
            "task": task,
            "kind": kind,
            "k": int(rank),
            "n_folds": int(len(group)),
        }
        for metric in ("prediction_correlation", "error_correlation"):
            values = group[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if len(finite):
                mean, lower, upper = fisher_z_mean_sem(finite)
                row[metric] = float(mean)
                row[f"{metric}_sem_lower"] = float(lower)
                row[f"{metric}_sem_upper"] = float(upper)
            else:
                row[metric] = float("nan")
                row[f"{metric}_sem_lower"] = float("nan")
                row[f"{metric}_sem_upper"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def plot_language_decoding(
    summary: pd.DataFrame,
    output_path: Path,
    agreement: pd.DataFrame | None = None,
) -> None:
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
        handles, labels = ax.get_legend_handles_labels()
        if agreement is not None:
            task_agreement = agreement[agreement["task"] == task_name].sort_values("k")
            if not task_agreement.empty:
                correlation_axis = ax.twinx()
                correlation_axis.errorbar(
                    task_agreement["k"],
                    task_agreement["prediction_correlation"],
                    yerr=np.vstack(
                        [
                            task_agreement["prediction_correlation_sem_lower"],
                            task_agreement["prediction_correlation_sem_upper"],
                        ]
                    ),
                    marker="s",
                    linestyle="--",
                    color="#059669",
                    alpha=0.85,
                    capsize=2,
                    label="RRR–PCA score correlation",
                )
                if spec.kind == "continuous" and task_agreement[
                    "error_correlation"
                ].notna().any():
                    correlation_axis.errorbar(
                        task_agreement["k"],
                        task_agreement["error_correlation"],
                        yerr=np.vstack(
                            [
                                task_agreement["error_correlation_sem_lower"],
                                task_agreement["error_correlation_sem_upper"],
                            ]
                        ),
                        marker="^",
                        linestyle=":",
                        color="#7c3aed",
                        alpha=0.85,
                        capsize=2,
                        label="RRR–PCA error correlation",
                    )
                correlation_axis.set_ylim(-1.05, 1.05)
                correlation_axis.set_ylabel("RRR–PCA correlation")
                correlation_handles, correlation_labels = (
                    correlation_axis.get_legend_handles_labels()
                )
                handles += correlation_handles
                labels += correlation_labels
        ax.legend(handles, labels, fontsize=7, loc="best")
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


def _block_diagonal(matrices: list[np.ndarray]) -> np.ndarray:
    """Construct a block-diagonal map without conflating distinct model spaces."""
    rows = sum(int(matrix.shape[0]) for matrix in matrices)
    cols = sum(int(matrix.shape[1]) for matrix in matrices)
    result = np.zeros((rows, cols), dtype=np.float32)
    row = col = 0
    for matrix in matrices:
        matrix = np.asarray(matrix, dtype=np.float32)
        result[row : row + matrix.shape[0], col : col + matrix.shape[1]] = matrix
        row += matrix.shape[0]
        col += matrix.shape[1]
    return result


def compose_multiview_factor(
    component_factors: list[np.ndarray], final_factor: np.ndarray
) -> np.ndarray:
    """Compose separate raw view factors with one final multiview factor."""
    block = _block_diagonal(component_factors)
    final_factor = np.asarray(final_factor, dtype=np.float32)
    if block.shape[1] != final_factor.shape[0]:
        raise ValueError(
            "Multiview component widths do not match the final A ambient dimension: "
            f"{block.shape[1]} vs {final_factor.shape[0]}"
        )
    return np.asarray(block @ final_factor, dtype=np.float32)


def _load_saved_multiview_config(unit_dir: Path) -> RRRConfig | None:
    path = unit_dir / "config.yml"
    if not path.exists():
        if "multi_view_rrr" in unit_dir.parts:
            raise FileNotFoundError(
                f"Multiview language decoding requires the effective RRR config at {path}. "
                "Re-run multiview RRR with config provenance enabled."
            )
        return None
    saved = load_config(str(path), {}, RRRConfig)
    return saved if saved.mode == ScriptMode.MULTI_VIEW else None


def _raw_embedding_slices(
    embeddings: dict[str, np.ndarray], view_names: list[str]
) -> dict[str, slice]:
    result = {}
    start = 0
    for name in view_names:
        stop = start + int(embeddings[name].shape[1])
        result[name] = slice(start, stop)
        start = stop
    return result


def _apply_raw_factor(
    embeddings: dict[str, np.ndarray],
    view_names: list[str],
    factor: np.ndarray,
    selection: np.ndarray,
) -> np.ndarray:
    """Apply a factor over concatenated raw spaces without materializing the concat."""
    slices = _raw_embedding_slices(embeddings, view_names)
    projected = None
    for name in view_names:
        value = np.asarray(embeddings[name])[selection] @ factor[slices[name]]
        projected = value if projected is None else projected + value
    return np.asarray(projected, dtype=np.float32)


def _prepare_multiview_subspaces(
    *,
    subspace: str,
    final_factor_paths: list[Path],
    component_bases: dict[str, dict[int, np.ndarray]],
    embeddings: dict[str, np.ndarray],
    view_names: list[str],
    selection: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: LanguageDecodingConfig,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Prepare projected embeddings and ambient factors for a multiview source."""
    final_by_fold = {
        int(path.parent.name.removeprefix("fold_")): np.load(path)
        for path in final_factor_paths
    }
    source_folds = sorted(final_by_fold)
    if not source_folds:
        raise FileNotFoundError(f"No {subspace.upper()} fold A matrices found")
    for name in view_names:
        if set(component_bases[name]) != set(source_folds):
            raise ValueError(
                f"View {name} component folds do not match final {subspace} folds"
            )

    within = config.mode == LanguageDecodingMode.WITHIN_DATASET
    if within and set(source_folds) != set(range(len(folds))):
        raise ValueError(
            f"{subspace.upper()} A-matrix folds {source_folds} do not match "
            f"decoding folds {list(range(len(folds)))}"
        )

    prepared: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for rank in map(int, config.rank_grid):
        composed = []
        for source_fold in source_folds:
            final = np.asarray(final_by_fold[source_fold])
            if final.shape[1] < rank:
                raise ValueError(
                    f"Fold {source_fold} {subspace.upper()} A has insufficient "
                    f"width for rank {rank}"
                )
            components = [component_bases[name][source_fold] for name in view_names]
            composed.append(compose_multiview_factor(components, final[:, :rank]))
        if within:
            for fold, factor in zip(source_folds, composed):
                prepared[(fold, rank)] = (
                    _apply_raw_factor(embeddings, view_names, factor, selection),
                    factor,
                )
        else:
            factor = aggregate_subspace_basis(
                composed, rank=rank, method=config.subspace_aggregation
            )
            design = _apply_raw_factor(embeddings, view_names, factor, selection)
            for fold in range(len(folds)):
                prepared[(fold, rank)] = (design, factor)
    return prepared


def _prepare_single_view_subspaces(
    *,
    subspace: str,
    factor_paths: list[Path],
    folds: list[tuple[np.ndarray, np.ndarray]],
    X: np.ndarray,
    config: LanguageDecodingConfig,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Prepare projected embeddings and ambient factors for one embedding space."""
    ranked_paths = [path for path in factor_paths if path.parent.name.startswith("rank_")]
    if ranked_paths:
        if len(ranked_paths) != len(factor_paths):
            raise ValueError("Cannot mix fold-specific and rank-specific factors")
        if config.mode == LanguageDecodingMode.WITHIN_DATASET:
            raise ValueError("Within-dataset decoding requires fold-specific factors")
        matrices_by_rank = {
            int(path.parent.name.removeprefix("rank_")): np.load(path)
            for path in ranked_paths
        }
        missing = sorted(set(map(int, config.rank_grid)) - set(matrices_by_rank))
        if missing:
            raise ValueError(f"{subspace.upper()} factors are missing ranks {missing}")
        prepared = {}
        for fold, (train_idx, _) in enumerate(folds):
            mean = X[np.asarray(train_idx)].mean(axis=0, dtype=np.float64)
            scale = X[np.asarray(train_idx)].std(axis=0, dtype=np.float64)
            scale[scale == 0] = 1.0
            X_fold = ((X - mean) / scale).astype(np.float32)
            for rank in map(int, config.rank_grid):
                factor = np.asarray(matrices_by_rank[rank])
                if factor.shape != (X.shape[1], rank):
                    raise ValueError(
                        f"Rank {rank} {subspace.upper()} factor has shape "
                        f"{factor.shape}; expected {(X.shape[1], rank)}"
                    )
                prepared[(fold, rank)] = (X_fold @ factor, factor)
        return prepared

    matrices_by_fold = {
        int(path.parent.name.removeprefix("fold_")): np.load(path)
        for path in factor_paths
    }
    scalers_by_fold = {}
    for path in factor_paths:
        fold = int(path.parent.name.removeprefix("fold_"))
        scaler_path = path.parent / "input_scaler.npz"
        if scaler_path.exists():
            with np.load(scaler_path, allow_pickle=False) as payload:
                scalers_by_fold[fold] = {
                    "mean": np.asarray(payload["mean"]),
                    "scale": np.asarray(payload["scale"]),
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
        mean_factors = {}
        for rank in config.rank_grid:
            if len(source_matrices) == 1:
                factor = np.asarray(source_matrices[0])[:, : int(rank)]
                if factor.shape[1] != int(rank):
                    raise ValueError(
                        f"{subspace.upper()} factor has insufficient width for "
                        f"rank {rank}"
                    )
                mean_factors[int(rank)] = factor
            else:
                mean_factors[int(rank)] = aggregate_subspace_basis(
                    source_matrices,
                    rank=int(rank),
                    method=config.subspace_aggregation,
                )

    prepared = {}
    for fold, (train_idx, test_idx) in enumerate(folds):
        if mean_factors is None:
            if fold in scalers_by_fold:
                scaler = scalers_by_fold[fold]
                X_fold = ((X - scaler["mean"]) / scaler["scale"]).astype(
                    np.float32
                )
            else:
                # Compatibility for explicitly supplied legacy path-based factors.
                X_fold = X
        else:
            mean = X[np.asarray(train_idx)].mean(axis=0, dtype=np.float64)
            scale = X[np.asarray(train_idx)].std(axis=0, dtype=np.float64)
            scale[scale == 0] = 1.0
            X_fold = ((X - mean) / scale).astype(np.float32)
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
            prepared[(fold, int(rank))] = (
                np.asarray(X_fold @ A), np.asarray(A)
            )
    return prepared


def _run_subspace_decoding(
    *,
    output_dir: Path,
    subspace: str,
    prepared: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    folds: list[tuple[np.ndarray, np.ndarray]],
    task_targets: dict[str, np.ndarray],
    config: LanguageDecodingConfig,
    block_ids: np.ndarray | None = None,
) -> list[dict]:
    """Run one shared task/rank/fold sweep over prepared subspace inputs."""
    rows = []
    print(f"\n[{subspace.upper()} subspace]")
    for fold, (train_idx, test_idx) in enumerate(folds):
        print(f"  [fold {fold}]")
        for rank in map(int, config.rank_grid):
            try:
                X_rank, ambient_factor = prepared[(fold, rank)]
            except KeyError as exc:
                raise ValueError(
                    f"Missing prepared {subspace} input for fold={fold}, rank={rank}"
                ) from exc
            rows.extend(
                _run_task_decoding(
                    output_dir=output_dir,
                    fold=fold,
                    rank=rank,
                    X=X_rank,
                    task_targets=task_targets,
                    train_idx=np.asarray(train_idx),
                    test_idx=np.asarray(test_idx),
                    config=config,
                    A=ambient_factor,
                    subspace=subspace,
                    block_ids=block_ids,
                )
            )
    return rows


def run_language_decoding(config: LanguageDecodingConfig) -> Path:
    """Run the configured task/rank sweep and return its output directory."""
    effective_device = _configure_decoder_backend(config.device)
    print(
        "Decoder backends: "
        f"continuous=himalaya/{effective_device}, categorical=scikit-learn/cpu"
    )
    enc = config.encoding
    subject_ids = enc.subjects or list(range(1, 10))
    dataset_context = resolve_dataset_context(
        enc.dataset, subject_ids, enc.per_dataset_config, enc.data_root
    )
    rrr_unit_dir = resolve_subspace_unit_dir(config.rrr_run_path, "rrr")
    multiview_config = _load_saved_multiview_config(rrr_unit_dir)
    multiview_embeddings = None
    component_bases = None
    source_test_idxs = None
    source_selection = None
    view_names: list[str] = []
    if multiview_config is None:
        embeddings, word_df = load_embeddings(
            enc.data_root,
            enc.model,
            dataset=enc.dataset,
            per_dataset_config=enc.per_dataset_config,
            dataset_context=dataset_context,
            limit_to_glove=enc.limit_to_glove or enc.model == "glove",
            glove_dim=enc.per_model_config.get("dim", 300),
            **enc.per_model_config,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
    else:
        source_subjects = multiview_config.encoding.subjects or list(range(1, 10))
        resolved_views = resolve_rrr_views(multiview_config)
        source_key = target_dataset_key(multiview_config, source_subjects)
        matching_runs = [
            run for run in resolved_views.values() if run.dataset_key == source_key
        ]
        if not matching_runs:
            raise ValueError(
                "Saved multiview config has no component from its target dataset"
            )
        source_data = load_base_comparison_data(matching_runs[0])
        source_selection = np.asarray(source_data.selection, dtype=int)
        source_test_idxs = [np.asarray(idx, dtype=int) for idx in source_data.test_idxs]
        component_bases, _ = load_view_bases(
            multiview_config,
            resolved_views,
            source_key,
            source_selection,
            source_test_idxs,
        )
        target_view_config = deepcopy(multiview_config)
        target_view_config.encoding = deepcopy(enc)
        multiview_embeddings, word_df = load_view_embeddings(
            target_view_config, resolved_views, subject_ids
        )
        multiview_embeddings = {
            name: np.asarray(values, dtype=np.float32)
            for name, values in multiview_embeddings.items()
        }
        view_names = [view.name for view in multiview_config.views]
        embeddings = None
    pca_unit_dir = (
        resolve_subspace_unit_dir(config.pca_run_path, "pca")
        if config.pca_run_path
        else None
    )
    rrr_factor_paths = sorted(rrr_unit_dir.glob("fold_*/A.npy")) or sorted(
        rrr_unit_dir.glob("rank_*/A.npy")
    )
    pca_factor_paths = (
        (sorted(pca_unit_dir.glob("fold_*/A.npy")) or
         sorted(pca_unit_dir.glob("rank_*/A.npy")))
        if pca_unit_dir is not None
        else []
    )
    if multiview_config is None:
        _validate_factor_paths(rrr_factor_paths, embeddings.shape[1], "rrr")
        if pca_factor_paths:
            _validate_factor_paths(pca_factor_paths, embeddings.shape[1], "pca")
    else:
        stacked_dim = sum(int(view.rank) for view in multiview_config.views)
        _validate_factor_paths(rrr_factor_paths, stacked_dim, "rrr")
        if pca_factor_paths:
            _validate_factor_paths(pca_factor_paths, stacked_dim, "pca")

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
    save_effective_config(config, output_dir)
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
        if multiview_config is not None:
            if not np.array_equal(selection, source_selection):
                raise ValueError(
                    "Language-decoding selection differs from the saved multiview run"
                )
            if len(test_idxs) != len(source_test_idxs) or any(
                not np.array_equal(left, right)
                for left, right in zip(test_idxs, source_test_idxs)
            ):
                raise ValueError(
                    "Language-decoding folds differ from the saved multiview run"
                )
        X = None if embeddings is None else embeddings[selection]
        task_targets = {name: target[selection] for name, target in task_targets.items()}
        block_ids = context.language_features["sentence_id"].astype(str).to_numpy()[selection]
        folds = _outer_folds(test_idxs, len(selection))
    else:
        selection = np.arange(len(word_df), dtype=int)
        X = embeddings
        fold_root = None
        block_ids = context.language_features["sentence_id"].astype(str).to_numpy()
        folds = list(
            KFold(n_splits=enc.num_folds, shuffle=False).split(
                np.arange(len(selection))
            )
        )

    if multiview_config is not None:
        if config.mode == LanguageDecodingMode.WITHIN_DATASET:
            full_designs = []
            full_projections = []
            for fold in range(len(folds)):
                bands, _ = project_view_bands(
                    multiview_embeddings, component_bases, view_names, fold
                )
                full_designs.append(
                    np.concatenate([band[selection] for band in bands], axis=1)
                )
                full_projections.append(
                    _block_diagonal(
                        [component_bases[name][fold] for name in view_names]
                    )
                )
        else:
            aggregated_components = {
                name: aggregate_subspace_basis(
                    [component_bases[name][fold] for fold in sorted(component_bases[name])],
                    rank=int(next(v.rank for v in multiview_config.views if v.name == name)),
                    method=config.subspace_aggregation,
                )
                for name in view_names
            }
            full_design = np.concatenate(
                [
                    multiview_embeddings[name][selection]
                    @ aggregated_components[name]
                    for name in view_names
                ],
                axis=1,
            )
            full_designs = [full_design] * len(folds)
            full_projection = _block_diagonal(
                [aggregated_components[name] for name in view_names]
            )
            full_projections = [full_projection] * len(folds)
        X = full_designs

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
        "embedding_shape": (
            list(embeddings.shape)
            if multiview_embeddings is None
            else {name: list(values.shape) for name, values in multiview_embeddings.items()}
        ),
        "multiview": multiview_config is not None,
        "view_names": view_names,
        "multiview_source_config": (
            None if multiview_config is None else str(rrr_unit_dir / "config.yml")
        ),
        "view_ranks": (
            {}
            if multiview_config is None
            else {view.name: int(view.rank) for view in multiview_config.views}
        ),
        "raw_embedding_boundaries": (
            {}
            if multiview_embeddings is None
            else {
                name: [bounds.start, bounds.stop]
                for name, bounds in _raw_embedding_slices(
                    multiview_embeddings, view_names
                ).items()
            }
        ),
        "cross_dataset_factor_strategy": (
            "compose_component_and_final_factors_then_aggregate"
            if multiview_config is not None
            and config.mode == LanguageDecodingMode.CROSS_DATASET
            else None
        ),
        "analysis_rows": int(len(selection)),
        "decoder_backends": {
            "continuous": "himalaya",
            "categorical": "scikit-learn",
        },
        "requested_device": config.device,
        "effective_continuous_device": effective_device,
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
        X_full = X[fold] if isinstance(X, list) else X
        full_projection = (
            full_projections[fold] if multiview_config is not None else None
        )
        rows.extend(
            _run_task_decoding(
                output_dir=output_dir,
                fold=fold,
                rank=None,
                X=X_full,
                task_targets=task_targets,
                train_idx=train_idx,
                test_idx=test_idx,
                config=config,
                A=full_projection,
                subspace="full",
                block_ids=block_ids,
            )
        )

    if multiview_config is None:
        rrr_prepared = _prepare_single_view_subspaces(
            subspace="rrr",
            factor_paths=rrr_factor_paths,
            folds=folds,
            X=X,
            config=config,
        )
    else:
        rrr_prepared = _prepare_multiview_subspaces(
            subspace="rrr",
            final_factor_paths=rrr_factor_paths,
            component_bases=component_bases,
            embeddings=multiview_embeddings,
            view_names=view_names,
            selection=selection,
            folds=folds,
            config=config,
        )
    rows.extend(_run_subspace_decoding(
        output_dir=output_dir,
        subspace="rrr",
        prepared=rrr_prepared,
        folds=folds,
        task_targets=task_targets,
        config=config,
        block_ids=block_ids,
    ))
    pca_prepared = None
    if pca_factor_paths:
        if multiview_config is None:
            pca_prepared = _prepare_single_view_subspaces(
                subspace="pca",
                factor_paths=pca_factor_paths,
                folds=folds,
                X=X,
                config=config,
            )
        else:
            pca_prepared = _prepare_multiview_subspaces(
                subspace="pca",
                final_factor_paths=pca_factor_paths,
                component_bases=component_bases,
                embeddings=multiview_embeddings,
                view_names=view_names,
                selection=selection,
                folds=folds,
                config=config,
            )
        rows.extend(_run_subspace_decoding(
            output_dir=output_dir,
            subspace="pca",
            prepared=pca_prepared,
            folds=folds,
            task_targets=task_targets,
            config=config,
            block_ids=block_ids,
        ))

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "grid_results.csv", index=False)
    summary = summarize_language_decoding(results)
    summary.to_csv(output_dir / "summary.csv", index=False)
    agreement_summary = None
    if pca_prepared is not None:
        agreement = compute_prediction_agreement(
            output_dir=output_dir,
            rrr_prepared=rrr_prepared,
            pca_prepared=pca_prepared,
            folds=folds,
            task_targets=task_targets,
            config=config,
        )
        agreement.to_csv(output_dir / "prediction_agreement.csv", index=False)
        agreement_summary = summarize_prediction_agreement(agreement)
        agreement_summary.to_csv(
            output_dir / "prediction_agreement_summary.csv", index=False
        )
    plot_language_decoding(
        summary,
        output_dir / "rank_analysis.png",
        agreement=agreement_summary,
    )
    return output_dir
