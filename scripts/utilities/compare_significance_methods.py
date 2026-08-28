#!/usr/bin/env python
"""Compare three iEEG lag-significance methods on matched encoding results."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from himalaya.backend import set_backend
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_significance import Config as SignificanceClientConfig
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.shared.alpha_grids import resolve_ridge_alphas
from src.utils.shared.config import load_config
from src.utils.shared.embeddings import get_glove_embeddings, load_embeddings
from src.utils.shared.encoding import get_ridge_model_constructor, train_encoding
from src.utils.shared.ieeg import (
    compute_shared_word_selection,
    load_data_from_cache,
)
from src.utils.significance.ieeg import (
    derive_subject_null_seed,
    fisher_mean_correlations,
    fold_local_comparison_null_scores,
    compute_per_fold_correlations,
    reshape_fold_predictions_and_targets,
    score_fold_lag_clusters,
)


METHODS = ("lag_corr", "lag_sem", "cluster")
METHOD_LABELS = {
    "lag_corr": "Lag correlation",
    "lag_sem": "Lag correlation / SEM",
    "cluster": "Cluster",
}


def canonical_config_path(dataset: str, model: str) -> Path:
    """Return the repository's canonical significance config for a comparison."""
    if dataset == "podcast":
        path = PROJECT_ROOT / "configs" / "significance" / model / "significance.yml"
    elif dataset == "cars-2":
        path = (
            PROJECT_ROOT
            / "configs"
            / "brain_treebank"
            / "cars_2"
            / "significance"
            / model
            / "significance.yml"
        )
    else:
        raise ValueError(
            f"Unsupported iEEG dataset {dataset!r}; expected 'podcast' or 'cars-2'"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"No canonical significance config for dataset={dataset!r}, "
            f"model={model!r}: {path}"
        )
    return path


def _scientific_source_signature(client_config: SignificanceClientConfig) -> tuple:
    selected = client_config.selected_config()
    enc = selected.encoding
    return (
        enc.dataset,
        enc.model,
        enc.per_model_config.get("model_dir"),
        enc.per_model_config.get("dim"),
        enc.per_dataset_config.get("movie"),
    )


def load_comparison_config(
    dataset: str, model: str, layer: int, config_path: str | Path | None = None
):
    """Load a checked canonical/explicit config and select exactly one layer."""
    canonical = canonical_config_path(dataset, model)
    source = Path(config_path).resolve() if config_path is not None else canonical
    if not source.is_file():
        raise FileNotFoundError(f"Significance config does not exist: {source}")

    canonical_client = load_config(str(canonical), {}, SignificanceClientConfig)
    source_client = load_config(str(source), {}, SignificanceClientConfig)
    if _scientific_source_signature(source_client) != _scientific_source_signature(
        canonical_client
    ):
        raise ValueError(
            f"Explicit config {source} does not match dataset={dataset!r}, "
            f"model={model!r}"
        )

    configured = load_config(
        str(source),
        {"ieeg.encoding.per_model_config.layer": int(layer)},
        SignificanceClientConfig,
    )
    if configured.modality != "ieeg":
        raise ValueError("Significance-method comparison only supports iEEG configs")
    return configured.selected_config(), source


def empirical_max_pvalues(observed: np.ndarray, null_maxima: np.ndarray) -> np.ndarray:
    """Return plus-one-corrected p-values against electrode-specific maxima."""
    observed = np.asarray(observed, dtype=np.float64)
    null_maxima = np.asarray(null_maxima, dtype=np.float64)
    if null_maxima.ndim != 2 or null_maxima.shape[1] != len(observed):
        raise ValueError(
            "null_maxima must have shape (n_null, n_electrodes), got "
            f"{null_maxima.shape} for {len(observed)} observed scores"
        )
    pvalues = np.ones(len(observed), dtype=float)
    valid = np.isfinite(observed)
    pvalues[valid] = (
        np.sum(null_maxima[:, valid] >= observed[None, valid], axis=0) + 1.0
    ) / (len(null_maxima) + 1.0)
    return pvalues


def _finite_max_per_electrode(scores: np.ndarray) -> np.ndarray:
    finite = np.where(np.isfinite(scores), scores, -np.inf)
    maxima = np.max(finite, axis=1)
    maxima[~np.isfinite(maxima)] = np.nan
    return maxima


def _observed_method_scores(cluster_result: dict, fold_corrs: np.ndarray) -> dict:
    lag_corr = fisher_mean_correlations(fold_corrs, axis=0)
    lag_sem = np.asarray(cluster_result["target_scores"], dtype=np.float64)
    n_electrodes = lag_sem.shape[0]
    cluster = np.full(n_electrodes, np.nan, dtype=float)
    for item in cluster_result["scored_clusters"]:
        electrode = int(item["electrode_idx"])
        score = float(item["cluster_score"])
        if not np.isfinite(cluster[electrode]) or score > cluster[electrode]:
            cluster[electrode] = score
    return {
        "lag_corr": _finite_max_per_electrode(lag_corr),
        "lag_sem": _finite_max_per_electrode(lag_sem),
        "cluster": cluster,
        "mean_all_lags_corr": fisher_mean_correlations(lag_corr, axis=1),
    }


def _fit_subject(
    config,
    subject,
    context,
    aligned_embeddings,
    word_in_glove,
    shared,
    selected_electrodes=None,
    return_predictions=True,
):
    enc = config.encoding
    X_arr, Y, _, elec_names = load_data_from_cache(
        [subject],
        enc.data_root,
        enc.offset,
        aligned_embeddings,
        sig_electrodes=(
            {subject: list(selected_electrodes)}
            if selected_electrodes is not None
            else None
        ),
        valid_rows=word_in_glove,
        dataset=enc.dataset,
        per_dataset_config=enc.per_dataset_config,
        dataset_context=context,
        shared_selection=shared,
    )
    n_electrodes = len(elec_names[0])
    n_lags = Y.shape[1] // n_electrodes
    explicit_alphas = (
        np.asarray(enc.ridge_alphas, dtype=float)
        if enc.ridge_alphas is not None
        else None
    )
    alpha_context = {
        "script": "compare_significance_methods",
        "role": "subject",
        "subject": int(subject),
        "model": enc.model,
        "per_model_config": enc.per_model_config,
        "offset": enc.offset,
        "n_features": int(X_arr.shape[1]),
        "standardize": True,
    }
    alphas = resolve_ridge_alphas(
        explicit_alphas,
        X_arr,
        standardize=True,
        context=alpha_context,
        cache_dir=None,
        policy=enc.alpha_grid,
    )
    constructor = get_ridge_model_constructor(
        alphas=alphas,
        ridge_cv_folds=enc.ridge_cv_folds,
        solver_params=enc.ridge_solver_params,
        y_in_cpu=enc.ridge_y_in_cpu,
        force_cpu=enc.ridge_force_cpu,
    )
    trained = train_encoding(
        [X_arr] * enc.num_folds,
        Y,
        constructor,
        epochs_shape=(n_electrodes, n_lags),
        num_folds=enc.num_folds,
        return_preds=return_predictions,
    )
    corrs, models, _, test_idxs, scalers = trained[:5]
    predictions = trained[5] if return_predictions else None
    if return_predictions:
        fold_predictions, fold_targets = reshape_fold_predictions_and_targets(
            predictions, test_idxs, Y, n_electrodes, n_lags
        )
    else:
        fold_predictions = fold_targets = None
    return {
        "X": X_arr,
        "Y": Y,
        "models": models,
        "scalers": scalers,
        "predictions": predictions,
        "correlations": corrs,
        "fold_predictions": fold_predictions,
        "fold_targets": fold_targets,
        "electrodes": list(map(str, elec_names[0])),
    }


def run_comparison(config) -> pd.DataFrame:
    """Fit and score all configured subjects, returning one row per electrode."""
    enc = config.encoding
    subjects = enc.subjects or list(range(1, 10))
    context = resolve_dataset_context(
        dataset=enc.dataset,
        subject_ids=subjects,
        per_dataset_config=enc.per_dataset_config,
        data_root=enc.data_root,
    )
    get_dataset_adapter(enc.dataset).validate_subjects_share_stimulus(
        subjects, enc.per_dataset_config, enc.data_root
    )

    if torch.cuda.is_available():
        set_backend("torch_cuda")
        print("Using torch_cuda backend.")
    else:
        print("CUDA not available, using CPU.")

    aligned_embeddings, transcript = load_embeddings(
        enc.data_root,
        enc.model,
        dataset=enc.dataset,
        per_dataset_config=enc.per_dataset_config,
        dataset_context=context,
        **enc.per_model_config,
    )
    word_in_glove = None
    if enc.limit_to_glove or enc.model == "glove":
        word_in_glove = get_glove_embeddings(
            enc.data_root, transcript, **enc.per_model_config
        )[1]["in_glove"]
    shared = None
    if enc.shared_word_selection:
        shared = compute_shared_word_selection(
            subjects,
            enc.data_root,
            dataset=enc.dataset,
            per_dataset_config=enc.per_dataset_config,
            dataset_context=context,
        )

    mean_corr_threshold = (
        float(np.tanh(config.mean_z_threshold))
        if config.mean_z_threshold is not None
        else float(config.mean_corr_threshold)
    )
    rows = []
    for subject in subjects:
        print(f"\nProcessing subject {subject}: fitting encoding model...")
        fitted = _fit_subject(
            config, subject, context, aligned_embeddings, word_in_glove, shared
        )
        fold_predictions = fitted["fold_predictions"]
        fold_targets = fitted["fold_targets"]
        cluster_result = score_fold_lag_clusters(
            fold_predictions,
            fold_targets,
            config.blocks_per_fold,
            max_matching_distance=config.max_matching_distance,
            positive_fraction=config.positive_fraction,
            mean_corr_threshold=mean_corr_threshold,
            device=config.device,
        )
        fold_corrs = compute_per_fold_correlations(
            fold_predictions, fold_targets, device=config.device
        )
        observed = _observed_method_scores(cluster_result, fold_corrs)
        subject_seed = derive_subject_null_seed(config.null_seed, subject)
        nulls, _ = fold_local_comparison_null_scores(
            fold_predictions,
            fold_targets,
            blocks_per_fold=config.blocks_per_fold,
            max_matching_distance=config.max_matching_distance,
            positive_fraction=config.positive_fraction,
            mean_corr_threshold=mean_corr_threshold,
            minimal_shift=config.minimal_shift,
            shift_step=config.shift_step,
            num_null_shifts=config.num_null_shifts,
            seed=subject_seed,
            device=config.device,
            null_batch_size=config.null_batch_size,
        )
        pvalues = {
            method: empirical_max_pvalues(observed[method], nulls[method])
            for method in METHODS
        }
        for electrode_idx, electrode in enumerate(fitted["electrodes"]):
            row = {
                "subject": int(subject),
                "electrode": electrode,
                "mean_all_lags_corr": float(
                    observed["mean_all_lags_corr"][electrode_idx]
                ),
            }
            for method in METHODS:
                row[f"{method}_score"] = float(observed[method][electrode_idx])
                row[f"{method}_p"] = float(pvalues[method][electrode_idx])
            rows.append(row)

        del fitted, fold_predictions, fold_targets, cluster_result, fold_corrs, nulls
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return apply_global_fdr(pd.DataFrame(rows), config.fdr_q)


def apply_global_fdr(frame: pd.DataFrame, q: float) -> pd.DataFrame:
    """Apply each method's BH correction across all subject-electrode rows."""
    frame = frame.copy()
    for method in METHODS:
        rejected, adjusted, _, _ = multipletests(
            frame[f"{method}_p"].to_numpy(), alpha=q, method="fdr_bh"
        )
        frame[f"{method}_q"] = adjusted
        frame[f"{method}_significant"] = rejected.astype(bool)
    return frame


def significant_sets(frame: pd.DataFrame) -> dict[str, set[str]]:
    identifiers = frame.apply(
        lambda row: f"{int(row['subject'])}:{row['electrode']}", axis=1
    )
    return {
        method: set(identifiers[frame[f"{method}_significant"]])
        for method in METHODS
    }


def membership_groups(sets: dict[str, set[str]]) -> dict[str, set[str]]:
    """Return the seven non-empty three-method membership regions."""
    corr, sem, cluster = (sets[method] for method in METHODS)
    return {
        "all three": corr & sem & cluster,
        "lag correlation + lag SEM only": (corr & sem) - cluster,
        "lag correlation + cluster only": (corr & cluster) - sem,
        "lag SEM + cluster only": (sem & cluster) - corr,
        "lag correlation only": corr - sem - cluster,
        "lag SEM only": sem - corr - cluster,
        "cluster only": cluster - corr - sem,
    }


def mean_significant_correlations(frame: pd.DataFrame) -> dict[str, float]:
    """Fisher-average all-lag electrode correlations over each selected set."""
    means = {}
    for method in METHODS:
        values = frame.loc[
            frame[f"{method}_significant"], "mean_all_lags_corr"
        ].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        means[method] = (
            float(fisher_mean_correlations(finite, axis=0))
            if len(finite)
            else np.nan
        )
    return means


def print_report(frame: pd.DataFrame, config, config_path: Path, dataset, model, layer):
    sets = significant_sets(frame)
    mean_correlations = mean_significant_correlations(frame)
    print("\n" + "=" * 72)
    print("SIGNIFICANCE METHOD COMPARISON")
    print("=" * 72)
    print(f"Config: {config_path}")
    print(f"Dataset: {dataset} | Model: {model} | Layer: {layer}")
    print(
        f"Subjects: {sorted(frame.subject.unique())} | Electrodes: {len(frame)} | "
        f"Null shifts: {config.num_null_shifts} | Global BH q: {config.fdr_q}"
    )
    print(
        f"Folds: {config.encoding.num_folds} | Blocks/fold: {config.blocks_per_fold} | "
        f"Cluster distance: {config.max_matching_distance} | "
        f"Cluster positive fraction: {config.positive_fraction} | "
        f"Cluster mean-r threshold: {config.mean_corr_threshold}"
    )

    print("\nSignificant electrode counts")
    count_rows = []
    for method in METHODS:
        row = {
            "method": METHOD_LABELS[method],
            "all": len(sets[method]),
            "mean all-lag r (Fisher z→r)": mean_correlations[method],
        }
        for subject in sorted(frame.subject.unique()):
            mask = frame.subject.eq(subject) & frame[f"{method}_significant"]
            row[f"subject {subject}"] = int(mask.sum())
        count_rows.append(row)
    print(
        pd.DataFrame(count_rows).to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nPairwise overlap")
    pair_rows = []
    for left_idx, left in enumerate(METHODS):
        for right in METHODS[left_idx + 1 :]:
            intersection = sets[left] & sets[right]
            union = sets[left] | sets[right]
            pair_rows.append(
                {
                    "first": METHOD_LABELS[left],
                    "second": METHOD_LABELS[right],
                    "shared": len(intersection),
                    "first only": len(sets[left] - sets[right]),
                    "second only": len(sets[right] - sets[left]),
                    "union": len(union),
                    "Jaccard": len(intersection) / len(union) if union else 1.0,
                }
            )
    print(pd.DataFrame(pair_rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nThree-way membership")
    for label, members in membership_groups(sets).items():
        print(f"\n{label} ({len(members)}):")
        print("  " + (", ".join(sorted(members)) if members else "—"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["podcast", "cars-2"])
    parser.add_argument("--model", required=True, help="Canonical config model name")
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional matching significance YAML instead of the canonical path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config, path = load_comparison_config(
        args.dataset, args.model, args.layer, args.config
    )
    frame = run_comparison(config)
    print_report(frame, config, path, args.dataset, args.model, args.layer)


if __name__ == "__main__":
    main()
