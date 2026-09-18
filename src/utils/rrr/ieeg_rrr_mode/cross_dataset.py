"""Cross-dataset iEEG RRR mode."""

from scripts.run_rrr import Artifact, UnitSelector
from src.core.config import RRRConfig
from src.utils.shared.fmri import apply_embedding_scaler
from src.utils.rrr.significance import validate_cross_dataset_significance_alignment

from .common import _build_rrr_decomposition_inputs, _reference_encoding_results
from .cross_dataset_helpers import (
    load_cross_dataset_A_by_rank,
    load_cross_dataset_scaler,
    _run_cross_dataset_subspace_for_unit,
)


def run_cross_dataset(
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
    """Evaluate target models using whole-stimulus source input bases."""
    if config.ieeg_rrr_source:
        source_kwargs = dict(config.ieeg_rrr_source)
        source_kwargs.setdefault("results_root", config.results_root)
        source = Artifact(**source_kwargs)
    elif config.fmri_rrr_source:
        source_kwargs = dict(config.fmri_rrr_source)
        source_kwargs.setdefault("results_root", config.results_root)
        source = Artifact(**source_kwargs)
    else:
        raise ValueError("cross_dataset requires an explicit RRR source Artifact")
    if source.variant is None:
        source._resolve_variant()
    if config.ieeg_rrr_source and source.identity["mode"] != "full_fit":
        raise ValueError(
            "Corrected cross-dataset transfer requires an iEEG full_fit source "
            "with one saved whole-dataset scaler"
        )
    validate_cross_dataset_significance_alignment(
        config.significance_selection,
        artifact.identity,
        source.requested_significance_selection,
        source.identity,
    )
    unit = UnitSelector(method="rrr")
    pca_unit = None
    pca_resume_plan = None
    if config.run_pca_baseline:
        pca_unit = UnitSelector(method="pca")

    reference = _reference_encoding_results(significance, subject_ids)
    n_folds = len(reference["test_idxs"])
    raw_plan = artifact.resume_plan(
        unit=unit,
        ranks=config.rank_grid,
        folds=range(n_folds),
        refresh=config.refresh,
        require_input_factors=False,
    )
    rrr_resume_plan = {
        "folds_to_build": raw_plan["folds_to_build"],
        "ranks_by_fold": raw_plan["ranks_by_fold"],
        "completed_pairs": raw_plan["completed_pairs"],
    }
    folds_to_build = set(rrr_resume_plan["folds_to_build"])
    if config.run_pca_baseline and pca_unit is not None:
        raw_pca_plan = artifact.resume_plan(
            unit=pca_unit,
            ranks=config.rank_grid,
            folds=range(n_folds),
            refresh=config.refresh,
            require_input_factors=False,
        )
        pca_resume_plan = {
            "folds_to_build": raw_pca_plan["folds_to_build"],
            "ranks_by_fold": raw_pca_plan["ranks_by_fold"],
            "completed_pairs": raw_pca_plan["completed_pairs"],
        }
        folds_to_build |= pca_resume_plan["folds_to_build"]

    if not folds_to_build:
        print("  Cross-dataset RRR/PCA outputs complete; skipping data/regression setup.")
        return

    print(f"Loading source RRR subspaces from {source.identity_record()}")
    A_rrr_by_rank = load_cross_dataset_A_by_rank(
        source, kind="rrr", ranks=config.rank_grid, source_format="rank_bases",
        orthonormalize=True,
    )
    rrr_scaler = load_cross_dataset_scaler(source, "rrr", config.rank_grid[0])
    A_pca_by_rank = None
    if config.run_pca_baseline:
        print(f"Loading source PCA subspaces from {source.identity_record()}")
        A_pca_by_rank = load_cross_dataset_A_by_rank(
            source, kind="pca", ranks=config.rank_grid,
            source_format="rank_bases",
        )

    results = _build_rrr_decomposition_inputs(
        subject_ids,
        sig_electrodes,
        reference,
        significance.detailed_significance,
        enc,
        aligned_embeddings,
        word_df,
        compute_sig_nonsig=config.compute_sig_nonsig,
        word_in_glove=word_in_glove,
        folds_to_build=folds_to_build,
        shared_selection_metadata=shared_selection_metadata,
    )
    if results is None:
        return
    X_raw, per_fold_baselines, test_idxs, sig_mask_flat, target_manifest = results
    X_rrr_transfer = apply_embedding_scaler(X_raw, rrr_scaler)
    artifact.save_target_manifest(
        unit=unit, target_manifest=target_manifest, overwrite=config.refresh
    )

    print("  Running cross-dataset RRR-source evaluation")
    _run_cross_dataset_subspace_for_unit(
        artifact=artifact,
        unit=unit,
        A_by_rank=A_rrr_by_rank,
        X_raw=X_rrr_transfer,
        test_idxs=test_idxs,
        per_fold_baselines=per_fold_baselines,
        config=config,
        primary_label="nr",
        sig_mask_flat=sig_mask_flat,
        ranks_by_fold=rrr_resume_plan["ranks_by_fold"],
        completed_pairs=rrr_resume_plan["completed_pairs"],
        standardize_subspace_inputs=True,
    )

    units = [(unit, n_folds, False, True)]
    if config.run_pca_baseline and A_pca_by_rank is not None and pca_unit is not None:
        artifact.save_target_manifest(
            unit=pca_unit, target_manifest=target_manifest, overwrite=config.refresh
        )
        print("  Running cross-dataset PCA-source evaluation")
        _run_cross_dataset_subspace_for_unit(
            artifact=artifact,
            unit=pca_unit,
            A_by_rank=A_pca_by_rank,
            # PCA and RRR share the already-built source-standardized design.
            X_raw=X_rrr_transfer,
            test_idxs=test_idxs,
            per_fold_baselines=per_fold_baselines,
            config=config,
            primary_label="pca",
            sig_mask_flat=sig_mask_flat,
            ranks_by_fold=pca_resume_plan["ranks_by_fold"],
            completed_pairs=pca_resume_plan["completed_pairs"],
            standardize_subspace_inputs=True,
        )
        units.append((pca_unit, n_folds, False, True))
    return units


__all__ = [
    "run_cross_dataset",
    "load_cross_dataset_A_by_rank",
    "_run_cross_dataset_subspace_for_unit",
]
