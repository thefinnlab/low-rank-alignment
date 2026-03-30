"""
run_pca.py — PCA-based subspace analysis baseline.

Computes PCA on word embeddings to find a low-dimensional input subspace,
then trains ridge regression from the projected embeddings to neural targets.
Sweeps over a configured rank grid analogous to RRR, producing the same
metrics (minimal_r2, recovery_r2, correlations, etc.) so results are
directly comparable.

Three modes controlled by PCAConfig.mode (mirrors RRRConfig.mode):

  cross_subject (default):
    Builds global Y across all subjects (significant electrodes).
    Writes to {output_dir}/pca/.

  per_subject:
    Runs PCA independently per subject.
    Writes to {output_dir}/per_subject_pca/subjects/sub_{id}/.

  per_region:
    Runs PCA independently per atlas region.
    Writes to {output_dir}/per_region_pca/regions/{region}/.

Output layout per unit:
    {pca_unit_dir}/
        grid_results.csv
        summary.csv
"""

import gc
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import (
    build_global_sig_mask,
    load_embeddings,
    apply_offset,
    build_XY_from_subjects,
    load_precomputed_epochs,
    epochs_from_cache,
)
from utils.data_loaders import load_sig_electrodes
from utils.atlas import build_electrode_region_map
from utils.config import PCAConfig, SVDMode, get_results_dir
from utils.config_utils import parse_args, load_config
from utils.encoding import (
    load_encoding_results,
    build_enc_subjects_info,
    build_per_fold_baselines,
    get_and_offset_arbitrary_embeddings,
)
from utils.analysis import build_sig_mask_flat_for_elecs
from utils.rrr_utils import add_sig_nonsig_corrs, apply_sig_nonsig_splits_and_strip, make_sig_split_df
from utils.subspace_utils import (
    evaluate_subspace_ols,
    evaluate_complement_and_arbitrary,
    add_primary_cross_model_metrics,
    _add_complement_sig_nonsig,
    _add_arbitrary_sig_nonsig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_rrr_overlay_dir(mode: SVDMode, output_dir: str) -> str | None:
    """Return the corresponding RRR output directory for overlay, or None."""
    if mode == SVDMode.PER_REGION:
        candidate = os.path.join(output_dir, "per_region_rrr")
    elif mode == SVDMode.PER_SUBJECT:
        candidate = os.path.join(output_dir, "per_subject_rrr")
    else:
        candidate = os.path.join(output_dir, "rrr")
    return candidate if os.path.isdir(candidate) else None


def _load_rrr_overlay(rrr_dir: str, unit_suffix: str = "") -> tuple:
    """Try to load RRR summary.csv for overlay.

    Returns (rrr_summary, rrr_nr_summary) DataFrames or (None, None).
    ``rrr_summary`` has reduced-rank columns in standard names.
    ``rrr_nr_summary`` has NR columns renamed to standard names.
    """
    summary_path = os.path.join(rrr_dir, unit_suffix, "summary.csv")
    if not os.path.exists(summary_path):
        return None, None
    try:
        rrr_df = pd.read_csv(summary_path)
    except Exception:
        return None, None

    # Reduced-rank overlay: already uses standard column names
    rrr_summary = rrr_df

    # NR overlay: rename nr_ columns to standard names
    nr_col_map = {
        "nr_r2": "minimal_r2",
        "nr_corr": "minimal_corr",
        "nr_recovery_r2": "recovery_r2",
        "nr_recovery_corr": "recovery_corr",
        "nr_pred_r2": "pred_r2",
        "nr_cca_similarity": "cca_similarity",
        "nr_procrustes_similarity": "procrustes_similarity",
        "nr_corr_sig": "minimal_corr_sig",
        "nr_corr_nonsig": "minimal_corr_nonsig",
        "nr_recovery_corr_sig": "recovery_corr_sig",
        "nr_recovery_corr_nonsig": "recovery_corr_nonsig",
    }
    drop_cols = [v for v in nr_col_map.values() if v in rrr_df.columns]
    rrr_nr_summary = rrr_df.drop(columns=drop_cols).rename(columns=nr_col_map)

    return rrr_summary, rrr_nr_summary


# ---------------------------------------------------------------------------
# Core per-unit loop
# ---------------------------------------------------------------------------


def _run_pca_for_unit(
    pca_unit_dir: str,
    X_raw: np.ndarray,
    test_idxs: list,
    per_fold_baselines: list,
    config: PCAConfig,
    sig_mask_flat: np.ndarray | None = None,
    rrr_overlay_dir: str | None = None,
    X_arbitrary: np.ndarray | None = None,
) -> pd.DataFrame:
    """Core PCA loop for one analysis unit (cross-subject, per-subject, or per-region).

    Args:
        pca_unit_dir: Root output directory for this unit.
        X_raw: (n_words, n_embed) raw embedding matrix.
        test_idxs: List of test index arrays, one per fold.
        per_fold_baselines: Output of build_per_fold_baselines.
        config: PCAConfig.
        rrr_overlay_dir: Path to RRR unit directory for overlay (or None).
        X_arbitrary: (n_words, n_embed) arbitrary embeddings for null baseline.

    Returns:
        DataFrame of per-(fold, k) metrics for grid_results.csv.
    """
    os.makedirs(pca_unit_dir, exist_ok=True)

    enc = config.encoding
    alphas = np.array(enc.ridge_alphas) if enc.ridge_alphas is not None else None

    n_folds = len(test_idxs)
    all_rows = []

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold_f in range(n_folds):
        fold_start = time.time()
        print(f"\n[Fold {fold_f + 1}/{n_folds}]")

        baseline = per_fold_baselines[fold_f]
        train_idx = baseline["train_idx"]
        test_idx = baseline["test_idx"]
        X_train = X_raw[train_idx]
        X_test = X_raw[test_idx]
        Y_train = baseline["Y_train"]
        Y_test = baseline["Y_test"]
        Y_hat_full = baseline["Y_hat_full_test"]

        # Compute PCA on training data only (proper CV)
        X_mean = X_train.mean(axis=0)
        X_centered = X_train - X_mean
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        V_full = Vt.T  # (n_embed, min(n_train, n_embed))
        max_components = V_full.shape[1]

        # Clamp rank grid to available components
        valid_ranks = [k for k in config.rank_grid if k <= max_components]
        if len(valid_ranks) < len(config.rank_grid):
            skipped = [k for k in config.rank_grid if k > max_components]
            print(f"  Skipping ranks {skipped} (> {max_components} PCA components)")

        print(f"  Running {len(valid_ranks)} rank(s): {valid_ranks}")
        for k in tqdm(valid_ranks, desc=f"  fold {fold_f}", leave=True, unit="rank"):
            A = V_full[:, :k]  # (n_embed, k) — top-k PCA components

            nr_result = evaluate_subspace_ols(
                A,
                X_train,
                X_test,
                Y_train,
                Y_test,
                Y_hat_full,
                compute_cca=config.compute_cca,
                compute_procrustes=config.compute_procrustes,
                device=device,
            )

            # Complement and arbitrary evaluations
            complement_result, arbitrary_result = evaluate_complement_and_arbitrary(
                A,
                X_train,
                X_test,
                X_arbitrary,
                train_idx,
                test_idx,
                Y_train,
                Y_test,
                nr_result["Y_hat_train"],
                nr_result["Y_hat"],
                alphas=alphas,
                device=device,
            )

            suf = nr_result["suf"]
            metrics = {
                "full_r2": suf["full_r2"],
                "minimal_r2": suf["minimal_r2"],
                "recovery_r2": suf["recovery_r2"],
                "pred_r2": suf["pred_r2"],
                "full_corr": suf["full_corr"],
                "minimal_corr": suf["minimal_corr"],
                "recovery_corr": suf["recovery_corr"],
                "complement_corr": complement_result["complement_corr"],
                "complement_recovery_corr": complement_result[
                    "complement_recovery_corr"
                ],
                "fold": fold_f,
                "k": k,
            }
            if arbitrary_result is not None:
                metrics["arbitrary_corr"] = arbitrary_result["arbitrary_corr"]
                metrics["arbitrary_recovery_corr"] = arbitrary_result[
                    "arbitrary_recovery_corr"
                ]
            if nr_result["cca_sim"] is not None:
                metrics["cca_similarity"] = nr_result["cca_sim"]
            if nr_result["proc_sim"] is not None:
                metrics["procrustes_similarity"] = nr_result["proc_sim"]

            if sig_mask_flat is not None:
                add_sig_nonsig_corrs(
                    metrics,
                    suf,
                    sig_mask_flat,
                    [
                        ("full_corr", "full_corr_per_target"),
                        ("minimal_corr", "minimal_corr_per_target"),
                        ("recovery_corr", "recovery_corr_per_target"),
                    ],
                )
                _add_complement_sig_nonsig(metrics, complement_result, sig_mask_flat)
                if arbitrary_result is not None:
                    _add_arbitrary_sig_nonsig(metrics, arbitrary_result, sig_mask_flat)

            # Cross-model metrics
            Y_hat_pca = nr_result["Y_hat"]
            Y_hat_complement = complement_result["complement_Y_hat"]
            Y_hat_arbitrary = (
                arbitrary_result["arbitrary_Y_hat"]
                if arbitrary_result is not None
                else None
            )
            add_primary_cross_model_metrics(metrics, Y_hat_pca, Y_hat_full, Y_hat_complement, Y_hat_arbitrary, Y_test, "pca")

            # Split any *_per_target arrays into *_sig/*_nonsig scalars, then strip them
            apply_sig_nonsig_splits_and_strip(metrics, sig_mask_flat)

            all_rows.append(metrics)

        print(f"  Fold {fold_f} done in {time.time() - fold_start:.1f}s")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(pca_unit_dir, "grid_results.csv"), index=False)

    # Summary: average over folds
    agg_cols = [
        "full_r2",
        "minimal_r2",
        "recovery_r2",
        "pred_r2",
        "full_corr",
        "minimal_corr",
        "recovery_corr",
        "complement_corr",
        "complement_recovery_corr",
        # Cross-model metrics for PCA model
        "pca_full_pred_corr",
        "pca_full_joint_corr",
        "pca_complement_pred_corr",
        "pca_complement_joint_corr",
        # Complement vs other models
        "complement_full_pred_corr",
        "complement_full_joint_corr",
    ]
    if X_arbitrary is not None:
        agg_cols.extend(
            [
                "arbitrary_corr",
                "arbitrary_recovery_corr",
                "pca_arbitrary_pred_corr",
                "pca_arbitrary_joint_corr",
                "complement_arbitrary_pred_corr",
                "complement_arbitrary_joint_corr",
            ]
        )
    has_sig_split = sig_mask_flat is not None
    if has_sig_split:
        agg_cols.extend(
            [
                "full_corr_sig",
                "full_corr_nonsig",
                "minimal_corr_sig",
                "minimal_corr_nonsig",
                "recovery_corr_sig",
                "recovery_corr_nonsig",
                "complement_corr_sig",
                "complement_corr_nonsig",
                "complement_recovery_corr_sig",
                "complement_recovery_corr_nonsig",
                # Cross-model sig/nonsig for PCA
                "pca_full_pred_corr_sig",
                "pca_full_pred_corr_nonsig",
                "pca_full_joint_corr_sig",
                "pca_full_joint_corr_nonsig",
                "pca_complement_pred_corr_sig",
                "pca_complement_pred_corr_nonsig",
                "pca_complement_joint_corr_sig",
                "pca_complement_joint_corr_nonsig",
                # Complement vs other
                "complement_full_pred_corr_sig",
                "complement_full_pred_corr_nonsig",
                "complement_full_joint_corr_sig",
                "complement_full_joint_corr_nonsig",
            ]
        )
        if X_arbitrary is not None:
            agg_cols.extend(
                [
                    "arbitrary_corr_sig",
                    "arbitrary_corr_nonsig",
                    "arbitrary_recovery_corr_sig",
                    "arbitrary_recovery_corr_nonsig",
                    "pca_arbitrary_pred_corr_sig",
                    "pca_arbitrary_pred_corr_nonsig",
                    "pca_arbitrary_joint_corr_sig",
                    "pca_arbitrary_joint_corr_nonsig",
                    "complement_arbitrary_pred_corr_sig",
                    "complement_arbitrary_pred_corr_nonsig",
                    "complement_arbitrary_joint_corr_sig",
                    "complement_arbitrary_joint_corr_nonsig",
                ]
            )
    if config.compute_cca:
        agg_cols.append("cca_similarity")
    if config.compute_procrustes:
        agg_cols.append("procrustes_similarity")
    summary = df.groupby("k")[agg_cols].mean().reset_index()
    summary.to_csv(os.path.join(pca_unit_dir, "summary.csv"), index=False)

    # --- Plotting ---
    from utils.plotting import plot_rrr_rank_results

    # Load RRR overlay if available
    rrr_summary, rrr_nr_summary = None, None
    if rrr_overlay_dir:
        rrr_summary, rrr_nr_summary = _load_rrr_overlay(rrr_overlay_dir)

    # Main plot: PCA subspace ridge results (with RRR NR overlay if available)
    plot_rrr_rank_results(
        summary,
        grid_df=df,
        title=f"PCA: perf vs rank ({os.path.basename(pca_unit_dir)})",
        save_path=os.path.join(pca_unit_dir, "rank_analysis.png"),
        overlay_summary_df=rrr_nr_summary,
        overlay_label="RRR (NR)",
        primary_label="pca",
    )
    if has_sig_split:
        plot_rrr_rank_results(
            make_sig_split_df(summary, "sig"),
            grid_df=make_sig_split_df(df, "sig"),
            title=f"PCA (sig targets): perf vs rank ({os.path.basename(pca_unit_dir)})",
            save_path=os.path.join(pca_unit_dir, "rank_analysis_sig.png"),
            overlay_summary_df=rrr_nr_summary,
            overlay_label="RRR (NR)",
            primary_label="pca",
        )
        plot_rrr_rank_results(
            make_sig_split_df(summary, "nonsig"),
            grid_df=make_sig_split_df(df, "nonsig"),
            title=f"PCA (nonsig targets): perf vs rank ({os.path.basename(pca_unit_dir)})",
            save_path=os.path.join(pca_unit_dir, "rank_analysis_nonsig.png"),
            overlay_summary_df=rrr_nr_summary,
            overlay_label="RRR (NR)",
            primary_label="pca",
        )

    # Second plot: PCA with RRR reduced-rank overlay (only when RRR data exists)
    if rrr_summary is not None:
        plot_rrr_rank_results(
            summary,
            grid_df=df,
            title=f"PCA: perf vs rank ({os.path.basename(pca_unit_dir)})",
            save_path=os.path.join(pca_unit_dir, "rank_analysis_rrr_raw.png"),
            overlay_summary_df=rrr_summary,
            overlay_label="RRR",
            primary_label="pca",
        )
    if rrr_summary is not None and has_sig_split:
        plot_rrr_rank_results(
            make_sig_split_df(summary, "sig"),
            grid_df=make_sig_split_df(df, "sig"),
            title=f"PCA (sig targets): perf vs rank ({os.path.basename(pca_unit_dir)})",
            save_path=os.path.join(pca_unit_dir, "rank_analysis_rrr_raw_sig.png"),
            overlay_summary_df=rrr_summary,
            overlay_label="RRR",
            primary_label="pca",
        )
        plot_rrr_rank_results(
            make_sig_split_df(summary, "nonsig"),
            grid_df=make_sig_split_df(df, "nonsig"),
            title=f"PCA (nonsig targets): perf vs rank ({os.path.basename(pca_unit_dir)})",
            save_path=os.path.join(pca_unit_dir, "rank_analysis_rrr_raw_nonsig.png"),
            overlay_summary_df=rrr_summary,
            overlay_label="RRR",
            primary_label="pca",
        )

    return df


# ---------------------------------------------------------------------------
# Branch: cross-subject PCA
# ---------------------------------------------------------------------------


def _run_cross_subject_pca(
    config,
    enc,
    output_dir,
    sig_dir,
    pca_dir,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    df_word,
):
    """PCA on raw embeddings using all subjects' significant electrodes."""
    enc_results = load_encoding_results(
        os.path.join(
            output_dir, "subjects", f"sub_{subject_ids[0]}", "encoding_results.joblib"
        )
    )
    ref_test_idxs = enc_results["test_idxs"]

    subs_with_sig = [
        s for s in subject_ids if s in sig_electrodes and sig_electrodes[s]
    ]
    if not subs_with_sig:
        print("  No significant electrodes found; skipping cross-subject PCA.")
        return

    X_ref, Y_full, selection = build_XY_from_subjects(
        subject_ids,
        enc,
        sig_electrodes,
        aligned_embeddings,
        df_word,
    )
    print(f"  X: {X_ref.shape}, Y: {Y_full.shape}")

    enc_subjects_info = build_enc_subjects_info(
        output_dir, subs_with_sig, sig_electrodes
    )
    per_fold_baselines = build_per_fold_baselines(
        X_ref, Y_full, ref_test_idxs, enc_subjects_info
    )

    # Generate arbitrary embeddings for null baseline
    X_arbitrary = get_and_offset_arbitrary_embeddings(
        aligned_embeddings, selection, df_word, enc.offset, config.arbitrary_seed
    )

    rrr_overlay_dir = _resolve_rrr_overlay_dir(config.mode, output_dir)

    sig_mask_flat = None
    if config.compute_sig_nonsig:
        sig_mask_flat = build_sig_mask_flat_for_elecs(
            output_dir,
            {s: sig_electrodes[s] for s in subs_with_sig},
            enc_results,
            sig_dir=sig_dir,
        )

    _run_pca_for_unit(
        pca_unit_dir=pca_dir,
        X_raw=X_ref,
        test_idxs=ref_test_idxs,
        per_fold_baselines=per_fold_baselines,
        config=config,
        sig_mask_flat=sig_mask_flat,
        rrr_overlay_dir=rrr_overlay_dir,
        X_arbitrary=X_arbitrary,
    )


# ---------------------------------------------------------------------------
# Branch: per-subject PCA
# ---------------------------------------------------------------------------


def _run_per_subject_pca(
    config,
    enc,
    output_dir,
    sig_dir,
    pca_dir,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    df_word,
):
    """PCA on raw embeddings independently per subject."""
    rrr_overlay_dir = _resolve_rrr_overlay_dir(config.mode, output_dir)

    for sub_id in subject_ids:
        print(f"\n{'='*60}\nSubject {sub_id}")

        pca_sub_dir = os.path.join(pca_dir, "subjects", f"sub_{sub_id}")
        sub_sig_elecs = sig_electrodes.get(sub_id, [])
        if not sub_sig_elecs:
            print(f"  No significant electrodes; skipping.")
            continue

        enc_results = load_encoding_results(
            os.path.join(
                output_dir, "subjects", f"sub_{sub_id}", "encoding_results.joblib"
            )
        )
        test_idxs = enc_results["test_idxs"]

        cache = load_precomputed_epochs(sub_id, enc.data_root)
        selection = cache["selection"]
        data = epochs_from_cache(cache, elec_names=sub_sig_elecs)
        X_arr, Y_sub = apply_offset(aligned_embeddings[selection], data, enc.offset)
        del data
        gc.collect()

        Y_sub = Y_sub.astype(np.float32)
        print(f"  X: {X_arr.shape}, Y: {Y_sub.shape}")

        # Generate arbitrary embeddings for null baseline
        X_arbitrary = get_and_offset_arbitrary_embeddings(
            aligned_embeddings, selection, df_word, enc.offset, config.arbitrary_seed
        )

        del cache
        gc.collect()

        enc_subjects_info = build_enc_subjects_info(
            output_dir, [sub_id], {sub_id: sub_sig_elecs}
        )
        per_fold_baselines = build_per_fold_baselines(
            X_arr, Y_sub, test_idxs, enc_subjects_info
        )

        rrr_sub_overlay = None
        if rrr_overlay_dir:
            candidate = os.path.join(rrr_overlay_dir, "subjects", f"sub_{sub_id}")
            if os.path.isdir(candidate):
                rrr_sub_overlay = candidate

        sig_mask_flat = None
        if config.compute_sig_nonsig:
            sig_mask_flat = build_sig_mask_flat_for_elecs(
                output_dir,
                {sub_id: sub_sig_elecs},
                enc_results,
                sig_dir=sig_dir,
            )

        _run_pca_for_unit(
            pca_unit_dir=pca_sub_dir,
            X_raw=X_arr,
            test_idxs=test_idxs,
            per_fold_baselines=per_fold_baselines,
            config=config,
            sig_mask_flat=sig_mask_flat,
            rrr_overlay_dir=rrr_sub_overlay,
            X_arbitrary=X_arbitrary,
        )


# ---------------------------------------------------------------------------
# Branch: per-region PCA
# ---------------------------------------------------------------------------


def _run_per_region_pca(
    config,
    enc,
    output_dir,
    sig_dir,
    pca_dir,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    df_word,
):
    """PCA on raw embeddings independently per atlas region."""
    enc_results = load_encoding_results(
        os.path.join(
            output_dir, "subjects", f"sub_{subject_ids[0]}", "encoding_results.joblib"
        )
    )
    ref_test_idxs = enc_results["test_idxs"]

    print("Building electrode region map...")
    electrode_region_map = build_electrode_region_map(
        subject_ids,
        enc.data_root,
        getattr(config, "atlas_region_groups", None),
        getattr(config, "atlas_path", None),
    )
    print(f"  Regions found: {list(electrode_region_map.keys())}")

    rrr_overlay_dir = _resolve_rrr_overlay_dir(config.mode, output_dir)

    for region_name, sub_elec_map in electrode_region_map.items():
        print(f"\n{'='*60}\nRegion: {region_name}")

        pca_region_dir = os.path.join(pca_dir, "regions", region_name)

        # Intersect region electrodes with significant electrodes
        per_subject_region_elecs = {
            sub_id: [e for e in elecs if e in set(sig_electrodes.get(sub_id, []))]
            for sub_id, elecs in sub_elec_map.items()
        }
        per_subject_region_elecs = {
            s: e for s, e in per_subject_region_elecs.items() if e
        }

        if not per_subject_region_elecs:
            print(f"  No significant electrodes in region; skipping.")
            continue

        subs_in_region = list(per_subject_region_elecs.keys())
        X_ref, Y_full, selection = build_XY_from_subjects(
            subs_in_region,
            enc,
            per_subject_region_elecs,
            aligned_embeddings,
            df_word,
        )
        print(f"  X: {X_ref.shape}, Y: {Y_full.shape}")

        enc_subjects_info = build_enc_subjects_info(
            output_dir, subs_in_region, per_subject_region_elecs
        )
        per_fold_baselines = build_per_fold_baselines(
            X_ref, Y_full, ref_test_idxs, enc_subjects_info
        )

        # Generate arbitrary embeddings for null baseline
        X_arbitrary = get_and_offset_arbitrary_embeddings(
            aligned_embeddings, selection, df_word, enc.offset, config.arbitrary_seed
        )

        rrr_region_overlay = None
        if rrr_overlay_dir:
            candidate = os.path.join(rrr_overlay_dir, "regions", region_name)
            if os.path.isdir(candidate):
                rrr_region_overlay = candidate

        sig_mask_flat = None
        if config.compute_sig_nonsig:
            sig_mask_flat = build_sig_mask_flat_for_elecs(
                output_dir,
                per_subject_region_elecs,
                enc_results,
                sig_dir=sig_dir,
            )

        _run_pca_for_unit(
            pca_unit_dir=pca_region_dir,
            X_raw=X_ref,
            test_idxs=ref_test_idxs,
            per_fold_baselines=per_fold_baselines,
            config=config,
            sig_mask_flat=sig_mask_flat,
            rrr_overlay_dir=rrr_region_overlay,
            X_arbitrary=X_arbitrary,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args, overrides = parse_args("PCA: embedding PCA subspace baseline")
    config: PCAConfig = load_config(args.config, overrides, PCAConfig)
    enc = config.encoding

    import torch
    from himalaya.backend import set_backend

    if torch.cuda.is_available():
        set_backend("torch_cuda")

    subject_ids = enc.subjects or list(range(1, 10))
    output_dir = config.output_dir or get_results_dir(enc.model, enc.layer, enc.offset)

    # Resolve PCA output directory
    if config.pca_output_dir:
        pca_dir = config.pca_output_dir
    elif config.mode == SVDMode.PER_REGION:
        pca_dir = os.path.join(output_dir, "per_region_pca")
    elif config.mode == SVDMode.PER_SUBJECT:
        pca_dir = os.path.join(output_dir, "per_subject_pca")
    else:
        pca_dir = os.path.join(output_dir, "pca")

    os.makedirs(pca_dir, exist_ok=True)
    print(f"Output dir:  {output_dir}")
    print(f"PCA dir:     {pca_dir}")
    print(f"Mode:        {config.mode.value}")

    # Load significance CSV and build subject → electrode mapping
    sig_dir = config.sig_dir or output_dir
    sig_electrodes, _ = load_sig_electrodes(output_dir, subject_ids, sig_dir=sig_dir)

    # Load embeddings
    print("Loading embeddings...")
    aligned_embeddings, df_word = load_embeddings(enc.data_root, enc.layer)

    branch_kwargs = dict(
        config=config,
        enc=enc,
        output_dir=output_dir,
        sig_dir=sig_dir,
        pca_dir=pca_dir,
        subject_ids=subject_ids,
        sig_electrodes=sig_electrodes,
        aligned_embeddings=aligned_embeddings,
        df_word=df_word,
    )

    if config.mode == SVDMode.PER_REGION:
        _run_per_region_pca(**branch_kwargs)
    elif config.mode == SVDMode.PER_SUBJECT:
        _run_per_subject_pca(**branch_kwargs)
    else:
        _run_cross_subject_pca(**branch_kwargs)

    print("\nDone.")


if __name__ == "__main__":
    main()
