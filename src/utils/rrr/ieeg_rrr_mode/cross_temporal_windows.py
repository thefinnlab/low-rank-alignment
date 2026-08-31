"""Cross-temporal-window iEEG RRR mode."""

import numpy as np
import pandas as pd

from scripts.run_rrr import Artifact, UnitSelector
from src.utils.rrr.inputs import _get_decomposition_inputs
from src.utils.shared.encoding import build_per_fold_baselines

from .common import _reference_encoding_results, _temporal_window_target_mask
from .cross_dataset_helpers import (_load_cross_temporal_A_by_fold,
                                     _run_cross_dataset_subspace_for_unit)


def run_cross_temporal_windows(config, enc, artifact, significance, subject_ids,
                               sig_electrodes, aligned_embeddings, word_df,
                               word_in_glove=None, shared_selection_metadata=None):
    """Transfer fold-matched source factors between configured target windows."""
    del word_df
    source_kwargs = dict(config.ieeg_rrr_source or {})
    source_kwargs.setdefault("results_root", config.results_root)
    source = Artifact(**source_kwargs)
    if source.variant is None:
        source._resolve_variant()
    print(f"Loading temporal-window subspaces from {source.identity_record()}")

    reference = _reference_encoding_results(significance, subject_ids)
    prepared = _get_decomposition_inputs(
        subject_ids, sig_electrodes, enc, aligned_embeddings,
        compute_sig_nonsig=config.compute_sig_nonsig,
        word_in_glove=word_in_glove,
        shared_selection_metadata=shared_selection_metadata,
        encoding_results=reference,
        significance=significance.detailed_significance,
    )
    if prepared is None:
        return []
    X_raw = next(iter(prepared.embeddings.values()))
    n_folds = len(reference["test_idxs"])
    windows = {window.name: window for window in config.temporal_windows}
    pairs = ([(pair.source, pair.target) for pair in config.temporal_window_pairs]
             if config.temporal_window_pairs else [
                 (source_name, target_name) for source_name in windows
                 for target_name in windows if source_name != target_name
             ])
    if not pairs:
        raise ValueError("cross_temporal_windows requires at least two windows or an explicit pair")

    returned_units, combined = [], []
    for source_window, target_window in pairs:
        target_mask = _temporal_window_target_mask(
            prepared.target_manifest, windows[target_window])
        target_indices = np.flatnonzero(target_mask)
        manifest = prepared.target_manifest.loc[target_mask].copy().reset_index(drop=True)
        manifest["original_target_index"] = target_indices
        Y_target = prepared.targets[:, target_mask]
        sig_mask = (prepared.sig_mask_flat[target_mask]
                    if prepared.sig_mask_flat is not None else None)
        print(f"\nTemporal transfer: {source_window} -> {target_window}")

        for method in (["rrr", "pca"] if config.run_pca_baseline else ["rrr"]):
            unit = UnitSelector(method=method, source_temporal_window=source_window,
                                target_temporal_window=target_window)
            plan = artifact.resume_plan(unit=unit, ranks=config.rank_grid,
                                        folds=range(n_folds), refresh=config.refresh,
                                        require_input_factors=False,
                                        required_metric_keys=(
                                            {"nr_orth_corr", "nr_orth_r2_per_target",
                                             "nr_orth_best_alphas"}
                                            if method == "rrr" else None
                                        ))
            artifact.save_target_manifest(unit=unit, target_manifest=manifest,
                                          overwrite=config.refresh)
            per_fold_baselines = build_per_fold_baselines(
                X_raw, Y_target, prepared.test_idxs,
                alphas=(np.array(enc.ridge_alphas, dtype=float)
                        if enc.ridge_alphas is not None else None),
                ridge_cv_folds=enc.ridge_cv_folds,
                ridge_solver_params=enc.ridge_solver_params,
                ridge_y_in_cpu=enc.ridge_y_in_cpu, ridge_force_cpu=enc.ridge_force_cpu,
                alpha_policy=enc.alpha_grid, alpha_cache_dir=None,
                alpha_context={"script": "run_rrr", "mode": "cross_temporal_windows",
                               "source_window": source_window, "target_window": target_window,
                               "method": method, "model": enc.model,
                               "per_model_config": enc.per_model_config, "offset": enc.offset},
                folds_to_build=plan["folds_to_build"])
            A_by_rank = _load_cross_temporal_A_by_fold(
                source, method, source_window, config.rank_grid, n_folds)
            _run_cross_dataset_subspace_for_unit(
                artifact=artifact, unit=unit, A_by_rank=A_by_rank, X_raw=X_raw,
                test_idxs=prepared.test_idxs, per_fold_baselines=per_fold_baselines,
                config=config, primary_label="pca" if method == "pca" else "nr",
                sig_mask_flat=sig_mask, ranks_by_fold=plan["ranks_by_fold"],
                completed_pairs=plan["completed_pairs"],
                evaluate_orthonormalized=method == "rrr")
            summary = artifact.load_summary(unit=unit).copy()
            summary["source_window"], summary["target_window"] = source_window, target_window
            summary["method"] = method
            combined.append(summary)
            returned_units.append((unit, n_folds, False, True))

    artifact.save_field("mode_metadata", pd.concat(combined, ignore_index=True),
                        selector={"kind": "cross_temporal_window_summary"},
                        overwrite=config.refresh)
    return returned_units


__all__ = ["run_cross_temporal_windows"]
