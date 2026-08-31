"""Shared leave-one-out orchestration for iEEG RRR modes."""

import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.run_rrr import UnitSelector
from scripts.run_rrr import Artifact
from src.core.config import RRRConfig

from .common import (_build_results, _build_rrr_decomposition_inputs,
                     _load_existing_grid, _merge_grid_rows,
                     _reference_encoding_results, _run_common_subspace_fold_k,
                     _run_rrr_for_unit)


def _run_leave_one_out_transfer_for_unit(
    artifact: Artifact,
    unit: UnitSelector,
    X_raw: np.ndarray,
    test_idxs: list,
    per_fold_baselines: list,
    train_unit: UnitSelector,
    config: "RRRConfig",
    sig_mask_flat: "np.ndarray | None" = None,
    ranks_by_fold: "dict[int, list[int]] | None" = None,
    completed_pairs: "set[tuple[int, int]] | None" = None,
) -> pd.DataFrame:
    """Evaluate a learned input factor on one held-out leave-one-out unit.

    For each fold and rank k, loads the A matrix saved by _run_rrr_for_unit on the
    training subjects, projects X through A_k, trains a fresh ridge on the held-out
    subject's training data, and evaluates on the held-out fold.

    Args:
        transfer_dir: Output directory for transfer metrics.
        X_raw: (n_words, n_embed) embedding matrix (shared across all subjects).
        test_idxs: List of test index arrays (one per fold); used only for n_folds.
        per_fold_baselines: Fold baselines built from the held-out subject's Y matrix.
        train_unit: Logical selector for the trained RRR factors.
        config: RRRConfig with rank_grid and encoding settings.
        sig_mask_flat: Optional boolean mask for significant targets in held-out subject.

    Returns:
        DataFrame with one row per (fold, k) containing transfer metrics.
    """
    import torch

    enc = config.encoding
    alphas = np.array(enc.ridge_alphas) if enc.ridge_alphas is not None else None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_folds = len(test_idxs)

    existing_grid = _load_existing_grid(artifact, unit)
    if ranks_by_fold is None or completed_pairs is None:
        plan = artifact.resume_plan(unit=unit, ranks=config.rank_grid,
                                    folds=range(n_folds), refresh=config.refresh,
                                    require_input_factors=False,
                                    required_metric_keys={
                                        "transfer_orth_corr",
                                        "transfer_orth_r2_per_target",
                                        "transfer_orth_best_alphas",
                                    })
        ranks_by_fold, completed_pairs = plan["ranks_by_fold"], plan["completed_pairs"]
    for fold_f in range(n_folds):
        if not artifact.has_field(
            "input_factors", selector=artifact._selector(train_unit, fold=fold_f)
        ):
            raise ValueError(f"Missing leave-one-out training factors for fold {fold_f}")
    total_to_run = sum(len(ranks) for ranks in ranks_by_fold.values())
    if completed_pairs:
        print(f"  Reusing {len(completed_pairs)} completed transfer fold/rank pair(s)")
    if total_to_run == 0:
        print(
            "  All requested transfer fold/rank pairs complete; rebuilding summary/plots from existing artifacts."
        )
        df = _merge_grid_rows(
            existing_grid,
            [],
            completed_pairs,
            config.rank_grid,
            artifact=artifact,
            unit=unit,
            primary_label="transfer",
            sig_mask_flat=sig_mask_flat,
        )
        _build_results(df, sig_mask_flat, artifact, unit, primary_label="transfer",
                       plot=config.plot, overwrite=config.refresh)
        return df

    all_rows = []
    for fold_f in range(n_folds):
        ranks_to_run = ranks_by_fold[fold_f]
        if not ranks_to_run:
            print(f"\n[Transfer fold {fold_f + 1}/{n_folds}] complete; skipping")
            continue

        fold_start = time.time()
        print(f"\n[Transfer fold {fold_f + 1}/{n_folds}]")

        # Load A matrix from training subjects (saved by _run_rrr_for_unit)
        A_full = artifact.load_factors(unit=train_unit, fold=fold_f)["input_factors"]

        baseline = per_fold_baselines[fold_f]
        train_idx = baseline["train_idx"]
        test_idx = baseline["test_idx"]
        X_train = X_raw[train_idx]
        X_test = X_raw[test_idx]
        Y_train = baseline["Y_train"]
        Y_test = baseline["Y_test"]

        for k in tqdm(
            ranks_to_run, desc=f"  transfer fold {fold_f}", leave=True, unit="rank"
        ):
            common = _run_common_subspace_fold_k(
                A_full,
                k,
                X_train,
                X_test,
                Y_train,
                Y_test,
                sig_mask_flat,
                fold_f,
                alphas=alphas,
                alpha_policy=config.encoding.alpha_grid,
                standardize_subspace_inputs=config.standardize_subspace_inputs,
                ridge_cv_folds=config.encoding.ridge_cv_folds,
                ridge_solver_params=config.encoding.ridge_solver_params,
                ridge_y_in_cpu=config.encoding.ridge_y_in_cpu,
                ridge_force_cpu=config.encoding.ridge_force_cpu,
                device=device,
                artifact=artifact,
                unit=unit,
                primary_label="transfer",
                evaluate_orthonormalized=True,
            )

            row = {
                "fold": fold_f,
                "k": k,
                **common,
            }
            all_rows.append(row)

        print(f"  Transfer fold {fold_f} done in {time.time() - fold_start:.1f}s")

    df = _merge_grid_rows(
        existing_grid,
        all_rows,
        completed_pairs,
        config.rank_grid,
        artifact=artifact,
        unit=unit,
        primary_label="transfer",
        sig_mask_flat=sig_mask_flat,
    )
    _build_results(df, sig_mask_flat, artifact, unit, primary_label="transfer",
                   plot=config.plot, overwrite=config.refresh)
    return df


@dataclass(frozen=True)
class LeaveOneOutPartition:
    """Neural-target partition and Artifact selectors for one held-out unit."""

    label: str
    train_subjects: list[int]
    train_electrodes: dict[int, list[str]]
    held_subjects: list[int]
    held_electrodes: dict[int, list[str]]
    train_unit: UnitSelector
    transfer_unit: UnitSelector
    baseline_unit: UnitSelector
    allow_baseline_without_training: bool = False


def _electrode_identities(
    subjects: list[int], electrodes: dict[int, list[str]]
) -> set[tuple[int, str]]:
    return {
        (int(subject), str(electrode))
        for subject in subjects
        for electrode in electrodes.get(subject, [])
    }


def validate_disjoint_electrodes(partition: LeaveOneOutPartition) -> None:
    """Fail before fitting if a neural target occurs on both sides of a split."""
    overlap = _electrode_identities(
        partition.train_subjects, partition.train_electrodes
    ) & _electrode_identities(partition.held_subjects, partition.held_electrodes)
    if overlap:
        formatted = ", ".join(
            f"(subject={subject}, electrode={electrode!r})"
            for subject, electrode in sorted(overlap)
        )
        raise ValueError(
            f"Leave-one-out train and held-out electrodes overlap for "
            f"{partition.label}: {formatted}"
        )


def run_leave_one_out(
    config: "RRRConfig",
    enc,
    artifact: Artifact,
    significance,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    word_df,
    word_in_glove=None,
    shared_selection_metadata=None,
    *,
    partitions: list[LeaveOneOutPartition],
):
    """Run shared train, transfer, and held-out baseline leave-one-out phases."""
    units = []
    for partition in partitions:
        print(f"\n{'='*60}\nHeld out: {partition.label}")
        validate_disjoint_electrodes(partition)

        train_available = bool(
            _electrode_identities(
                partition.train_subjects, partition.train_electrodes
            )
        )
        reference_subjects = (
            partition.train_subjects if train_available else partition.held_subjects
        )
        reference = _reference_encoding_results(significance, reference_subjects)
        test_idxs = reference["test_idxs"]
        n_folds = len(test_idxs)
        train_rrr_plan = train_pca_plan = None
        if train_available:
            print(f"  Training subjects: {partition.train_subjects}")
            train_rrr_plan = artifact.resume_plan(
                unit=partition.train_unit, ranks=config.rank_grid, folds=range(n_folds),
                refresh=config.refresh,
                require_target_factors=config.save_b_matrices,
                required_metric_keys={
                    "nr_orth_corr", "nr_orth_r2_per_target", "nr_orth_best_alphas",
                } | (
                    {"null_aug_corr_per_target", "null_aug_r2_per_target"}
                    if config.run_null_augmented else set()
                ),
            )
            train_pca_plan = (
                artifact.resume_plan(
                    unit=replace(partition.train_unit, method="pca"),
                    ranks=config.rank_grid, folds=range(n_folds),
                    refresh=config.refresh,
                )
                if config.run_pca_baseline else None
            )
            train_folds_to_build = set(train_rrr_plan["folds_to_build"])
            if train_pca_plan is not None:
                train_folds_to_build |= train_pca_plan["folds_to_build"]
            if train_folds_to_build:
                results_train = _build_rrr_decomposition_inputs(
                    partition.train_subjects,
                    partition.train_electrodes,
                    reference,
                    significance.detailed_significance,
                    enc,
                    aligned_embeddings,
                    word_df,
                    compute_sig_nonsig=config.compute_sig_nonsig,
                    word_in_glove=word_in_glove,
                    folds_to_build=train_folds_to_build,
                    shared_selection_metadata=shared_selection_metadata,
                )
                if results_train is None:
                    train_available = False
                else:
                    (X_raw, per_fold_baselines_train, test_idxs, sig_mask_train,
                     train_manifest) = results_train
                    artifact.save_target_manifest(
                        unit=partition.train_unit,
                        target_manifest=train_manifest,
                        overwrite=config.refresh,
                    )
                    print("  Running leave-one-out training RRR")
                    _run_rrr_for_unit(
                        artifact=artifact, unit=partition.train_unit, X_raw=X_raw,
                        test_idxs=test_idxs,
                        per_fold_baselines=per_fold_baselines_train, config=config,
                        sig_mask_flat=sig_mask_train,
                        ranks_by_fold=train_rrr_plan["ranks_by_fold"],
                        completed_pairs=train_rrr_plan["completed_pairs"],
                        pca_ranks_by_fold=(train_pca_plan["ranks_by_fold"]
                                           if train_pca_plan else None),
                        pca_completed_pairs=(train_pca_plan["completed_pairs"]
                                              if train_pca_plan else None),
                    )
            else:
                print("  Leave-one-out training RRR/PCA outputs complete; skipping setup.")
        else:
            print("  No leave-one-out training electrodes.")
        if not train_available and not partition.allow_baseline_without_training:
            print("  Skipping held-out phases because training data are unavailable.")
            continue
        if not _electrode_identities(
            partition.held_subjects, partition.held_electrodes
        ):
            print("  No held-out electrodes; skipping transfer/baseline.")
            continue

        transfer_plan = None
        if train_available:
            transfer_plan = artifact.resume_plan(
                unit=partition.transfer_unit, ranks=config.rank_grid,
                folds=range(n_folds), refresh=config.refresh,
                require_input_factors=False,
                required_metric_keys={
                    "transfer_orth_corr", "transfer_orth_r2_per_target",
                    "transfer_orth_best_alphas",
                },
            )
        baseline_rrr_plan = artifact.resume_plan(
            unit=partition.baseline_unit, ranks=config.rank_grid, folds=range(n_folds),
            refresh=config.refresh, require_target_factors=config.save_b_matrices,
            required_metric_keys={
                "nr_orth_corr", "nr_orth_r2_per_target", "nr_orth_best_alphas",
            } | (
                {"null_aug_corr_per_target", "null_aug_r2_per_target"}
                if config.run_null_augmented else set()
            ),
        )
        baseline_pca_plan = (
            artifact.resume_plan(
                unit=replace(partition.baseline_unit, method="pca"),
                ranks=config.rank_grid, folds=range(n_folds), refresh=config.refresh,
            )
            if config.run_pca_baseline else None
        )
        baseline_folds_to_build = set(baseline_rrr_plan["folds_to_build"])
        if baseline_pca_plan is not None:
            baseline_folds_to_build |= baseline_pca_plan["folds_to_build"]
        held_folds_to_build = set(baseline_folds_to_build)
        if transfer_plan is not None:
            held_folds_to_build |= transfer_plan["folds_to_build"]
        if not held_folds_to_build:
            print("  Transfer/baseline outputs complete; skipping held-out setup.")
            continue

        print("  Building held-out data...")
        results_held = _build_rrr_decomposition_inputs(
            partition.held_subjects,
            partition.held_electrodes,
            _reference_encoding_results(significance, partition.held_subjects),
            significance.detailed_significance,
            enc,
            aligned_embeddings,
            word_df,
            compute_sig_nonsig=config.compute_sig_nonsig,
            word_in_glove=word_in_glove,
            folds_to_build=held_folds_to_build,
            shared_selection_metadata=shared_selection_metadata,
        )
        if results_held is None:
            print("  No held-out data; skipping transfer/baseline.")
            continue
        (X_raw_held, per_fold_baselines_held, _, sig_mask_held,
         held_manifest) = results_held
        if transfer_plan is not None:
            artifact.save_target_manifest(
                unit=partition.transfer_unit, target_manifest=held_manifest,
                overwrite=config.refresh,
            )
        artifact.save_target_manifest(unit=partition.baseline_unit, target_manifest=held_manifest,
                                      overwrite=config.refresh)

        if transfer_plan is not None:
            print("  Running transfer evaluation")
            _run_leave_one_out_transfer_for_unit(
                artifact=artifact, unit=partition.transfer_unit,
                X_raw=X_raw_held, test_idxs=test_idxs,
                per_fold_baselines=per_fold_baselines_held,
                train_unit=partition.train_unit, config=config,
                sig_mask_flat=sig_mask_held,
                ranks_by_fold=transfer_plan["ranks_by_fold"],
                completed_pairs=transfer_plan["completed_pairs"],
            )

        # --- Step D: within-subject baseline RRR on held-out subject alone ---
        print("  Running within-held-out baseline RRR")
        _run_rrr_for_unit(
            artifact=artifact,
            unit=partition.baseline_unit,
            X_raw=X_raw_held,
            test_idxs=test_idxs,
            per_fold_baselines=per_fold_baselines_held,
            config=config,
            sig_mask_flat=sig_mask_held,
            ranks_by_fold=baseline_rrr_plan["ranks_by_fold"],
            completed_pairs=baseline_rrr_plan["completed_pairs"],
            pca_ranks_by_fold=(
                baseline_pca_plan["ranks_by_fold"] if baseline_pca_plan else None
            ),
            pca_completed_pairs=(
                baseline_pca_plan["completed_pairs"] if baseline_pca_plan else None
            ),
        )
        if train_available:
            units.extend([
                (partition.train_unit, n_folds),
                (partition.transfer_unit, n_folds, False, False),
            ])
        units.append((partition.baseline_unit, n_folds))
        if config.run_pca_baseline:
            if train_available:
                units.append((replace(partition.train_unit, method="pca"), n_folds))
            units.append((replace(partition.baseline_unit, method="pca"), n_folds))
    return units


__all__ = [
    "LeaveOneOutPartition",
    "run_leave_one_out",
    "validate_disjoint_electrodes",
]
