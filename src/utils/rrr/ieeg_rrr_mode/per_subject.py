"""Per-subject iEEG RRR mode."""

from scripts.run_rrr import Artifact, UnitSelector
from src.core.config import RRRConfig

from .common import (_build_rrr_decomposition_inputs, _reference_encoding_results,
                     _rrr_output_resume_plan, _run_rrr_for_unit)


def run_per_subject(
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
    """RRR on raw embeddings independently per subject."""
    units = []
    for sub_id in subject_ids:
        print(f"\n{'='*60}\nSubject {sub_id}")

        unit = UnitSelector(method="rrr", subject=int(sub_id))
        sub_sig_elecs = sig_electrodes.get(sub_id, [])
        if not sub_sig_elecs:
            print("  No significant electrodes; skipping.")
            continue

        reference = _reference_encoding_results(significance, [sub_id])
        n_folds = len(reference["test_idxs"])
        resume_plan = _rrr_output_resume_plan(artifact, unit, config, n_folds)
        folds_to_build = resume_plan["folds_to_build"]
        if not folds_to_build:
            print("  Subject RRR/PCA outputs complete; skipping data/regression setup.")
            continue

        results = _build_rrr_decomposition_inputs(
            [sub_id],
            {sub_id: sub_sig_elecs},
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
            continue
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
        units.append((unit, n_folds))
        if config.run_pca_baseline:
            units.append((UnitSelector(method="pca", subject=int(sub_id)), n_folds))
    return units


__all__ = ["run_per_subject"]
