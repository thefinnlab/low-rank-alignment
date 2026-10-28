"""Single-lag correlation significance for iEEG encoding models."""

import gc
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from himalaya.backend import set_backend
from statsmodels.stats.multitest import multipletests

from src.core.config import SignificanceConfig
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.shared.alpha_grids import resolve_ridge_alphas
from src.utils.shared.embeddings import load_embeddings
from src.utils.shared.encoding import extract_best_alphas, get_ridge_model_constructor, train_encoding
from src.utils.shared.ieeg import build_word_selection_metadata, compute_shared_word_selection, load_data_from_cache
from src.utils.significance.ieeg import (
    compute_per_fold_correlations, derive_subject_null_seed, fisher_mean_correlations,
    fold_local_max_lag_correlation_null_scores, reshape_fold_predictions_and_targets,
)


def _empirical_pvalues(observed, null_maxima):
    """Return per-lag p-values against each electrode's max-lag null."""
    observed = np.asarray(observed, dtype=np.float64)
    null_maxima = np.asarray(null_maxima, dtype=np.float64)
    if null_maxima.ndim != 2 or null_maxima.shape[1] != observed.shape[0]:
        raise ValueError("null_maxima must have shape (n_null, n_electrodes)")
    pvalues = np.ones(observed.shape, dtype=float)
    valid = np.isfinite(observed)
    exceedances = np.sum(null_maxima[:, :, None] >= observed[None, :, :], axis=0)
    pvalues[valid] = (exceedances[valid] + 1.0) / (len(null_maxima) + 1.0)
    return pvalues


def _best_lag_indices(lag_correlations):
    """Choose the first maximum finite lag per electrode, or -1."""
    values = np.asarray(lag_correlations, dtype=np.float64)
    finite = np.where(np.isfinite(values), values, -np.inf)
    indices = np.argmax(finite, axis=1).astype(int)
    indices[np.all(~np.isfinite(values), axis=1)] = -1
    return indices


def run_ieeg(config: SignificanceConfig):
    from scripts.run_significance import Artifact

    config.validate()
    enc = config.encoding
    subjects = enc.subjects or list(range(1, 10))
    context = resolve_dataset_context(
        dataset=enc.dataset, subject_ids=subjects,
        per_dataset_config=enc.per_dataset_config, data_root=enc.data_root,
    )
    get_dataset_adapter(enc.dataset).validate_subjects_share_stimulus(
        subjects, enc.per_dataset_config, enc.data_root
    )
    artifact = Artifact.from_ieeg_config(config, context).begin(
        config=config, refresh=config.refresh
    )
    if torch.cuda.is_available():
        set_backend("torch_cuda")
        print("Using torch_cuda backend.")
    else:
        print("CUDA not available, using CPU.")
    print(f"Artifact identity: {artifact.identity_record()}")

    embeddings, transcript = load_embeddings(
        enc.data_root, enc.model, dataset=enc.dataset,
        per_dataset_config=enc.per_dataset_config, dataset_context=context,
        limit_to_glove=enc.limit_to_glove or enc.model == "glove",
        glove_dim=enc.per_model_config.get("dim", 300),
        **enc.per_model_config,
    )
    word_in_glove = None
    if enc.limit_to_glove or enc.model == "glove":
        word_in_glove = transcript["in_glove"]
    shared = None
    if enc.shared_word_selection:
        shared = compute_shared_word_selection(
            subjects, enc.data_root, dataset=enc.dataset,
            per_dataset_config=enc.per_dataset_config, dataset_context=context,
        )
        print(f"Using shared word-selection intersection: {len(shared):,} cached words")
    explicit_alphas = (
        np.asarray(enc.ridge_alphas, dtype=float) if enc.ridge_alphas is not None else None
    )

    results = []
    for subject in subjects:
        print(f"\n{'=' * 60}\nProcessing subject {subject}\n{'=' * 60}")
        selector = {"subject": str(subject)}
        if not config.refresh and artifact.has_field("checkpoints", selector=selector):
            checkpoint = artifact.load_field("checkpoints", selector=selector)
            if checkpoint.get("method") != "single_lag_correlation":
                raise ValueError(
                    "The existing significance checkpoint uses the retired cluster "
                    "method; rerun with refresh=true."
                )
            results.append(checkpoint)
            print("  Reusing complete subject checkpoint")
            continue
        X_arr, Y, selection, elec_names = load_data_from_cache(
            [subject], enc.data_root, enc.offset, embeddings,
            valid_rows=word_in_glove, dataset=enc.dataset,
            per_dataset_config=enc.per_dataset_config, dataset_context=context,
            shared_selection=shared,
        )
        electrodes = list(map(str, elec_names[0]))
        n_electrodes = len(electrodes)
        n_lags = Y.shape[1] // n_electrodes
        lags = np.linspace(-2.0, 2.0, n_lags, endpoint=False)
        alpha_context = {
            "script": "run_significance", "role": "subject", "subject": int(subject),
            "model": enc.model, "per_model_config": enc.per_model_config,
            "offset": enc.offset, "n_features": int(X_arr.shape[1]), "standardize": True,
        }
        if not config.refresh and artifact.has_field("diagnostics", selector=selector):
            alphas = np.asarray(
                artifact.load_field("diagnostics", selector=selector)["alpha_grid"], dtype=float
            )
        else:
            alphas = resolve_ridge_alphas(
                explicit_alphas, X_arr, standardize=True, context=alpha_context,
                cache_dir=None, policy=enc.alpha_grid,
            )
        constructor = get_ridge_model_constructor(
            alphas=alphas, ridge_cv_folds=enc.ridge_cv_folds,
            solver_params=enc.ridge_solver_params, y_in_cpu=enc.ridge_y_in_cpu,
            force_cpu=enc.ridge_force_cpu,
        )
        print("  Training encoding model...")
        corrs, models, train_idxs, test_idxs, scalers, predictions = train_encoding(
            [X_arr] * enc.num_folds, Y, constructor,
            epochs_shape=(n_electrodes, n_lags), num_folds=enc.num_folds,
            return_preds=True,
        )
        encoding_result = {
            "selection": selection, "test_idxs": test_idxs, "train_idxs": train_idxs,
            "lags": lags, "scalers": scalers, "elec_names": list(elec_names[0]),
            "subject_id": subject, "offset": enc.offset,
            "per_model_config": dict(enc.per_model_config),
        }
        if enc.shared_word_selection:
            encoding_result["word_selection"] = build_word_selection_metadata(
                "shared_intersection", subjects, selection, enc.offset,
                valid_rows=word_in_glove[selection] if word_in_glove is not None else None,
            )
        artifact.save_subject_field("encoding_results", subject, encoding_result, overwrite=config.refresh)
        artifact.save_subject_field("correlations", subject, corrs, overwrite=config.refresh)
        if enc.log_alphas:
            artifact.save_subject_field("diagnostics", subject, {
                "alpha_grid": np.asarray(alphas),
                "best_alphas": [extract_best_alphas(model) for model in models],
                "context": alpha_context, "policy": enc.alpha_grid,
            }, overwrite=config.refresh)

        fold_predictions, fold_targets = reshape_fold_predictions_and_targets(
            predictions, test_idxs, Y, n_electrodes, n_lags
        )
        lag_correlations = fisher_mean_correlations(
            compute_per_fold_correlations(fold_predictions, fold_targets, device=config.device),
            axis=0,
        )
        print("  Scoring max-lag fold-local circular-shift null...")
        null_maxima, shift_metadata = fold_local_max_lag_correlation_null_scores(
            fold_predictions, fold_targets, minimal_shift=config.minimal_shift,
            shift_step=config.shift_step, num_null_shifts=config.num_null_shifts,
            seed=derive_subject_null_seed(config.null_seed, subject), device=config.device,
            null_batch_size=config.null_batch_size,
        )
        lag_pvalues = _empirical_pvalues(lag_correlations, null_maxima)
        best_lags = _best_lag_indices(lag_correlations)
        best_pvalues = np.ones(n_electrodes, dtype=float)
        valid = best_lags >= 0
        best_pvalues[valid] = lag_pvalues[np.arange(n_electrodes)[valid], best_lags[valid]]
        result = {
            "method": "single_lag_correlation",
            "subject": int(subject), "electrodes": electrodes, "lags": lags,
            "lag_correlations": lag_correlations, "lag_pvalues": lag_pvalues,
            "best_lags": best_lags, "best_pvalues": best_pvalues,
            "null_distribution": null_maxima, "shift_metadata": shift_metadata,
        }
        results.append(result)
        artifact.save_subject_field("checkpoints", subject, result, overwrite=config.refresh)
        del X_arr, Y, models, scalers, predictions, fold_predictions, fold_targets
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_best_pvalues = np.concatenate([item["best_pvalues"] for item in results])
    rejected, adjusted, _, _ = multipletests(all_best_pvalues, alpha=config.fdr_q, method="fdr_bh")
    cursor = 0
    for item in results:
        subject, electrodes = item["subject"], item["electrodes"]
        count = len(electrodes)
        electrode_sig = rejected[cursor:cursor + count]
        electrode_q = adjusted[cursor:cursor + count]
        cursor += count
        rows = []
        for electrode_idx, electrode in enumerate(electrodes):
            best_lag = int(item["best_lags"][electrode_idx])
            for lag_idx, lag_sec in enumerate(item["lags"]):
                is_best = lag_idx == best_lag
                rows.append({
                    "subject": subject, "electrode": electrode,
                    "subject_electrode_index": electrode_idx, "lag_idx": lag_idx,
                    "lag_sec": float(lag_sec),
                    "obs_corr": float(item["lag_correlations"][electrode_idx, lag_idx]),
                    "p_value": float(item["lag_pvalues"][electrode_idx, lag_idx]),
                    "best_lag_p_value": float(item["best_pvalues"][electrode_idx]),
                    "best_lag_q_value": float(electrode_q[electrode_idx]),
                    "is_best_lag": is_best,
                    "is_significant_best_lag": bool(is_best and electrode_sig[electrode_idx]),
                    "sig": bool(electrode_sig[electrode_idx]),
                })
        targets = pd.DataFrame({
            "subject": str(subject), "target_id": electrodes,
            "target_kind": "electrode", "significant": electrode_sig.astype(bool),
        })
        artifact.save_subject_field("target_significance", subject, targets, overwrite=True)
        artifact.save_subject_field("significance_details", subject, pd.DataFrame(rows), overwrite=True)

    artifact.save_field("effective_config", asdict(config), overwrite=True)
    artifact.save_field("run_metadata", {
        "method": "single_lag_correlation", "subjects": list(map(str, subjects)),
        "multiple_comparison_unit": "electrode_best_lag",
    }, overwrite=True)
    artifact.save_subject_field("null_distribution", "all", {
        "max_lag_correlation_per_electrode": [item["null_distribution"] for item in results],
        "subjects": np.array([item["subject"] for item in results], dtype=int),
        "electrodes": [item["electrodes"] for item in results],
        "minimal_shift": int(config.minimal_shift), "shift_step": int(config.shift_step),
        "num_null_shifts": int(config.num_null_shifts), "null_seed": int(config.null_seed),
        "null_sampling_mode": "iid_with_replacement_per_fold",
        "subject_shift_metadata": {item["subject"]: item["shift_metadata"] for item in results},
    }, overwrite=True)
    required = {
        "target_significance": [{"subject": str(s)} for s in subjects],
        "significance_details": [{"subject": str(s)} for s in subjects],
        "encoding_results": [{"subject": str(s)} for s in subjects],
        "correlations": [{"subject": str(s)} for s in subjects],
        "checkpoints": [{"subject": str(s)} for s in subjects],
    }
    artifact.mark_complete(required)
    print(f"Completed Artifact {artifact.identity_record()}")
    return artifact
