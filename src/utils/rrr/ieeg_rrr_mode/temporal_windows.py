"""Temporal-window iEEG RRR mode."""

import numpy as np
import pandas as pd

from scripts.run_rrr import Artifact, UnitSelector
from src.core.config import RRRConfig
from src.utils.rrr.inputs import _get_decomposition_inputs
from src.utils.shared.encoding import build_per_fold_baselines

from .common import (_reference_encoding_results, _run_rrr_for_unit,
                     _temporal_window_target_mask)


def run_temporal_windows(
    config: RRRConfig,
    enc,
    artifact: Artifact,
    significance,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    word_df,
    word_in_glove=None,
    shared_selection_metadata=None,
):
    """Run independent cross-subject RRR analyses for configured lag windows."""
    del word_df  # Retained for the common branch-call interface.
    reference = _reference_encoding_results(significance, subject_ids)
    n_folds = len(reference["test_idxs"])
    resume_plans = {}
    for window in config.temporal_windows:
        unit = UnitSelector(method="rrr", temporal_window=window.name)
        rrr_plan = artifact.resume_plan(
            unit=unit, ranks=config.rank_grid, folds=range(n_folds),
            refresh=config.refresh, require_target_factors=config.save_b_matrices,
            required_metric_keys={
                "nr_orth_corr", "nr_orth_r2_per_target", "nr_orth_best_alphas",
            } | (
                {"null_aug_corr_per_target", "null_aug_r2_per_target"}
                if config.run_null_augmented else set()
            ),
        )
        pca_plan = (
            artifact.resume_plan(
                unit=UnitSelector(method="pca", temporal_window=window.name),
                ranks=config.rank_grid, folds=range(n_folds), refresh=config.refresh,
            )
            if config.run_pca_baseline else None
        )
        folds_to_build = set(rrr_plan["folds_to_build"])
        if pca_plan is not None:
            folds_to_build |= pca_plan["folds_to_build"]
        resume_plans[window.name] = {
            "rrr": rrr_plan, "pca": pca_plan, "folds_to_build": folds_to_build,
        }

    prepared = _get_decomposition_inputs(
        subject_ids,
        sig_electrodes,
        reference,
        significance.detailed_significance,
        enc,
        aligned_embeddings,
        compute_sig_nonsig=config.compute_sig_nonsig,
        word_in_glove=word_in_glove,
        shared_selection_metadata=shared_selection_metadata,
    )
    if prepared is None:
        return

    X_raw = next(iter(prepared.embeddings.values()))
    window_records = []
    for window in config.temporal_windows:
        print(
            f"\n{'=' * 60}\nTemporal window: {window.name} "
            f"[{window.start_sec}, {window.end_sec}) seconds"
        )
        target_mask = _temporal_window_target_mask(prepared.target_manifest, window)
        target_indices = np.flatnonzero(target_mask)
        manifest = (
            prepared.target_manifest.loc[target_mask].copy().reset_index(drop=True)
        )
        manifest["original_target_index"] = target_indices
        unique_lags = np.sort(manifest["lag"].unique())
        unit = UnitSelector(method="rrr", temporal_window=window.name)
        artifact.save_target_manifest(unit=unit, target_manifest=manifest,
                                      overwrite=config.refresh)

        plan = resume_plans[window.name]
        Y_window = prepared.targets[:, target_mask]
        sig_mask = (
            prepared.sig_mask_flat[target_mask]
            if prepared.sig_mask_flat is not None
            else None
        )
        per_fold_baselines = build_per_fold_baselines(
            X_raw,
            Y_window,
            prepared.test_idxs,
            alphas=(
                np.array(enc.ridge_alphas, dtype=float)
                if enc.ridge_alphas is not None
                else None
            ),
            ridge_cv_folds=enc.ridge_cv_folds,
            ridge_solver_params=enc.ridge_solver_params,
            ridge_y_in_cpu=enc.ridge_y_in_cpu,
            ridge_force_cpu=enc.ridge_force_cpu,
            alpha_policy=enc.alpha_grid,
            alpha_cache_dir=None,
            alpha_context={
                "script": "run_rrr",
                "mode": "temporal_windows",
                "window": window.name,
                "model": enc.model,
                "per_model_config": enc.per_model_config,
                "offset": enc.offset,
            },
            folds_to_build=plan["folds_to_build"],
        )
        _run_rrr_for_unit(
            artifact=artifact,
            unit=unit,
            X_raw=X_raw,
            test_idxs=prepared.test_idxs,
            per_fold_baselines=per_fold_baselines,
            config=config,
            sig_mask_flat=sig_mask,
            ranks_by_fold=plan["rrr"]["ranks_by_fold"],
            completed_pairs=plan["rrr"]["completed_pairs"],
            pca_ranks_by_fold=(
                plan["pca"]["ranks_by_fold"] if plan["pca"] else None
            ),
            pca_completed_pairs=(
                plan["pca"]["completed_pairs"] if plan["pca"] else None
            ),
        )
        units = {"rrr": unit}
        if config.run_pca_baseline:
            units["pca"] = UnitSelector(method="pca", temporal_window=window.name)
        window_records.append(
            {
                "metadata": {
                    "window": window.name,
                    "start_sec": float(window.start_sec),
                    "end_sec": float(window.end_sec),
                    "actual_lag_min_sec": float(unique_lags.min()),
                    "actual_lag_max_sec": float(unique_lags.max()),
                    "n_lags": int(len(unique_lags)),
                    "n_targets": int(target_mask.sum()),
                },
                "units": units,
            }
        )

    combined = []
    returned_units = []
    for record in window_records:
        for method, unit in record["units"].items():
            frame = artifact.load_summary(unit=unit).copy()
            for key, value in record["metadata"].items():
                frame[key] = value
            frame["method"] = method
            combined.append(frame)
            returned_units.append((unit, n_folds))
    artifact.save_field("mode_metadata", pd.concat(combined, ignore_index=True),
                        selector={"kind": "temporal_window_summary"},
                        overwrite=config.refresh)
    return returned_units


__all__ = ["run_temporal_windows"]
