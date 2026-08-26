"""Cross-subject/default iEEG RRR mode."""

from scripts.run_rrr import Artifact, UnitSelector
from src.core.config import RRRConfig

from .common import (_build_rrr_decomposition_inputs, _reference_encoding_results,
                     _rrr_output_resume_plan, _run_rrr_for_unit)


def run_cross_subject(
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
    """RRR on raw embeddings using all subjects' significant electrodes."""
    unit = UnitSelector(method="rrr")
    reference = _reference_encoding_results(significance, subject_ids)
    n_folds = len(reference["test_idxs"])
    resume_plan = _rrr_output_resume_plan(artifact, unit, config, n_folds)
    folds_to_build = resume_plan["folds_to_build"]
    if not folds_to_build:
        print(
            "  Cross-subject RRR/PCA outputs complete; skipping data/regression setup."
        )
        return [(unit, n_folds)]

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
        return []
    X_raw, per_fold_baselines, test_idxs, sig_mask_flat, target_manifest = results
    artifact.save_target_manifest(unit=unit, target_manifest=target_manifest,
                                  overwrite=config.refresh)
    _run_rrr_for_unit(
        artifact=artifact,
        unit=unit,
        X_raw=X_raw,
        test_idxs=test_idxs,
        per_fold_baselines=per_fold_baselines,
        config=config,
        sig_mask_flat=sig_mask_flat,
        ranks_by_fold=resume_plan["rrr_ranks_by_fold"],
        completed_pairs=resume_plan["rrr_completed_pairs"],
        pca_ranks_by_fold=resume_plan["pca_ranks_by_fold"],
        pca_completed_pairs=resume_plan["pca_completed_pairs"],
    )
    units = [(unit, n_folds)]
    if config.run_pca_baseline:
        units.append((UnitSelector(method="pca"), n_folds))
    return units


__all__ = ["run_cross_subject"]
