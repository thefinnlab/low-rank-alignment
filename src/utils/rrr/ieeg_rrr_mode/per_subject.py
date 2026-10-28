"""Per-subject iEEG RRR mode."""

from scripts.run_rrr import Artifact, UnitSelector
from src.core.config import RRRConfig

from .common import (_build_rrr_decomposition_inputs,
                     _reference_encoding_results, _run_rrr_for_unit)


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
        rrr_plan = artifact.resume_plan(
            unit=unit, ranks=config.rank_grid, folds=range(n_folds),
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
                unit=UnitSelector(method="pca", subject=int(sub_id)),
                ranks=config.rank_grid, folds=range(n_folds), refresh=config.refresh,
            )
            if config.run_pca_baseline else None
        )
        folds_to_build = set(rrr_plan["folds_to_build"])
        if pca_plan is not None:
            folds_to_build |= pca_plan["folds_to_build"]
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
            ranks_by_fold=rrr_plan["ranks_by_fold"],
            completed_pairs=rrr_plan["completed_pairs"],
            pca_ranks_by_fold=(pca_plan["ranks_by_fold"] if pca_plan else None),
            pca_completed_pairs=(pca_plan["completed_pairs"] if pca_plan else None),
        )
        units.append((unit, n_folds))
        if config.run_pca_baseline:
            units.append((UnitSelector(method="pca", subject=int(sub_id)), n_folds))
    return units


__all__ = ["run_per_subject"]
