"""Multiview iEEG RRR mode."""

import numpy as np

from scripts.run_rrr import Artifact, UnitSelector
from src.core.config import RRRConfig
from src.utils.rrr.multiview import (
    fit_banded_ridge_baseline,
    load_view_bases,
    load_view_embeddings,
    project_view_bands,
    resolve_rrr_views,
    target_dataset_key,
)
from src.utils.rrr.significance import PreparedSignificance
from src.utils.shared.alpha_grids import resolve_ridge_alphas

from .common import (_build_rrr_decomposition_inputs,
                     _load_shared_word_selection_metadata,
                     _reference_encoding_results, _run_rrr_for_unit)


def run_multi_view(
    config: RRRConfig,
    artifact: Artifact,
    significance: PreparedSignificance,
    subject_ids: list[int],
) -> list[tuple[UnitSelector, int]]:
    """Fit banded ridge over saved RRR views, then run the standard RRR sweep."""
    from sklearn.preprocessing import StandardScaler

    resolved_views = resolve_rrr_views(config)

    sig_frame = significance.significant_electrodes
    union = {
        int(subject): group["elec"].astype(str).tolist()
        for subject, group in sig_frame.groupby("subject")
    }
    significance_sources = [significance.reference.identity_record()]
    for identity in significance.source_identities:
        if identity not in significance_sources:
            significance_sources.append(identity)
    for view in config.views:
        view_enc = resolved_views[view.name].config.encoding
        if int(view_enc.offset) != int(config.encoding.offset):
            raise ValueError(
                f"View {view.name} uses offset {view_enc.offset}; "
                f"target uses {config.encoding.offset}"
            )
    embeddings, word_df = load_view_embeddings(config, resolved_views, subject_ids)

    enc = config.encoding
    word_in_glove = None
    if enc.limit_to_glove or enc.model == "glove":
        word_in_glove = word_df["in_glove"]
    shared_selection_metadata = None
    if enc.shared_word_selection:
        shared_selection_metadata = _load_shared_word_selection_metadata(
            significance, subject_ids
        )

    reference = _reference_encoding_results(significance, subject_ids)
    prepared_values = _build_rrr_decomposition_inputs(
        subject_ids,
        union,
        reference,
        significance.detailed_significance,
        enc,
        embeddings,
        word_df,
        compute_sig_nonsig=config.compute_sig_nonsig,
        word_in_glove=word_in_glove,
        shared_selection_metadata=shared_selection_metadata,
        return_aligned_embeddings=True,
    )
    if prepared_values is None:
        return []
    (
        X_aligned, prepared_baselines, test_idxs, sig_mask_flat,
        target_manifest, aligned_embeddings,
    ) = prepared_values
    selection = np.asarray(reference["selection"])
    bases, basis_modes, view_scalers = load_view_bases(
        config,
        resolved_views,
        target_dataset_key(config, subject_ids),
        selection,
        test_idxs,
        return_scalers=True,
    )
    view_names = [view.name for view in config.views]
    n_words = X_aligned.shape[0]
    all_rows = np.arange(n_words, dtype=int)
    widths = [int(view.rank) for view in config.views]
    stops = np.cumsum(widths)
    band_boundaries = list(zip([0, *stops[:-1].tolist()], stops.tolist()))
    empty_design = np.empty((n_words, int(stops[-1])), dtype=np.float32)
    fold_designs: list[np.ndarray] = [empty_design] * len(test_idxs)
    baselines: list[dict | None] = [None] * len(test_idxs)
    unit = UnitSelector(method="rrr")
    artifact.save_target_manifest(unit=unit, target_manifest=target_manifest,
                                  overwrite=config.refresh)
    rrr_plan = artifact.resume_plan(
        unit=unit, ranks=config.rank_grid, folds=range(len(test_idxs)),
        refresh=config.refresh, require_target_factors=config.save_b_matrices,
        required_metric_keys={
            "nr_orth_corr", "nr_orth_r2_per_target",
        } | ({"nr_orth_best_alphas"} if config.log_alphas else set()) | (
            {"null_aug_corr_per_target", "null_aug_r2_per_target"}
            if config.run_null_augmented else set()
        ),
    )
    pca_plan = (
        artifact.resume_plan(
            unit=UnitSelector(method="pca"), ranks=config.rank_grid,
            folds=range(len(test_idxs)), refresh=config.refresh,
        )
        if config.run_pca_baseline else None
    )
    folds_to_build = set(rrr_plan["folds_to_build"])
    if pca_plan is not None:
        folds_to_build |= pca_plan["folds_to_build"]
    explicit_banded_alphas = (
        np.asarray(config.banded_ridge_alphas, dtype=float)
        if config.banded_ridge_alphas is not None
        else None
    )

    for fold_f, test_idx in enumerate(test_idxs):
        if fold_f not in folds_to_build:
            continue
        test_idx = np.asarray(test_idx, dtype=int)
        train_idx = np.setdiff1d(all_rows, test_idx, assume_unique=False)
        bands, boundaries = project_view_bands(
            aligned_embeddings, bases, view_names, fold_f, view_scalers
        )
        if band_boundaries != boundaries:
            raise ValueError("Multiview band boundaries differ across folds")
        X_full = np.concatenate(bands, axis=1).astype(np.float32, copy=False)
        fold_designs[fold_f] = X_full

        raw_targets = prepared_baselines[fold_f]["Y_train_raw"] if "Y_train_raw" in prepared_baselines[fold_f] else None
        baseline = prepared_baselines[fold_f]
        Y_train = baseline["Y_train"]
        Y_test = baseline["Y_test"]
        if raw_targets is not None:
            target_scaler = StandardScaler().fit(raw_targets)
            Y_train = target_scaler.transform(raw_targets).astype(
            np.float32
            )
        alpha_context = {
            "script": "run_rrr",
            "role": "multi_view_banded_full",
            "fold": int(fold_f),
            "n_features": int(X_full.shape[1]),
            "bands": view_names,
            "standardize": False,
        }
        banded_alphas = resolve_ridge_alphas(
            explicit_banded_alphas,
            X_full[train_idx],
            standardize=False,
            context=alpha_context,
            cache_dir=None,
            policy=enc.alpha_grid,
        )
        fitted = fit_banded_ridge_baseline(
            [band[train_idx] for band in bands],
            [band[test_idx] for band in bands],
            Y_train,
            alphas=banded_alphas,
            ridge_cv_folds=enc.ridge_cv_folds,
            solver_params=config.banded_ridge_solver_params,
            y_in_cpu=enc.ridge_y_in_cpu,
            force_cpu=enc.ridge_force_cpu,
        )
        if config.log_alphas:
            artifact.save_diagnostics(unit=unit, fold=fold_f, diagnostics={
                "view_names": view_names, "band_boundaries": boundaries,
                "alphas": banded_alphas, "deltas": fitted.deltas,
                "kernel_weights": np.exp(fitted.deltas),
                "best_alphas": fitted.best_alphas, "intercept": fitted.intercept,
            }, overwrite=config.refresh)
        baselines[fold_f] = {
            "train_idx": train_idx,
            "test_idx": test_idx,
            "Y_train": Y_train,
            "Y_test": Y_test,
            "Y_hat_full_train": fitted.Y_hat_train,
            "Y_hat_full_test": fitted.Y_hat_test,
            # The banded model and final RRR both operate directly on the
            # concatenated projected bands. Persist an explicit identity scaler
            # so these factors satisfy the standard Artifact coordinate contract.
            "input_coordinates": "standardized",
            "input_mean": np.zeros((1, X_full.shape[1]), dtype=np.float32),
            "input_scale": np.ones((1, X_full.shape[1]), dtype=np.float32),
            "beta_full_input_coordinates": fitted.beta_full_raw,
        }

    metadata = {
        "mode": "multi_view",
        "target_dataset": artifact.dataset,
        "views": view_names,
        "view_ranks": {view.name: int(view.rank) for view in config.views},
        "basis_modes": basis_modes,
        "subspace_aggregation": config.subspace_aggregation,
        "band_boundaries": band_boundaries,
        "significance_sources": significance_sources,
    }
    artifact.save_field("mode_metadata", metadata, selector={"kind": "multi_view"},
                        overwrite=config.refresh)

    print(
        f"  Multiview X: {(n_words, int(stops[-1]))}, "
        f"bands: {dict(zip(view_names, [b - a for a, b in band_boundaries]))}"
    )
    _run_rrr_for_unit(
        artifact=artifact,
        unit=unit,
        X_raw=fold_designs,
        test_idxs=test_idxs,
        per_fold_baselines=baselines,
        config=config,
        sig_mask_flat=sig_mask_flat,
        ranks_by_fold=rrr_plan["ranks_by_fold"],
        completed_pairs=rrr_plan["completed_pairs"],
        pca_ranks_by_fold=(pca_plan["ranks_by_fold"] if pca_plan else None),
        pca_completed_pairs=(pca_plan["completed_pairs"] if pca_plan else None),
    )
    return [(unit, len(test_idxs))]


__all__ = ["run_multi_view"]
