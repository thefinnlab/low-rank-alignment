"""
run_rrr.py — Raw-Input Reduced Rank Regression baseline.

Performs RRR directly from raw embeddings to neural targets — no SVD projection.
Sweeps over a configured rank grid to find the minimal rank that captures all
variance explained by an unconstrained ridge baseline.

Three modes controlled by RRRConfig.mode (mirrors SVDConfig.mode):

  cross_subject (default):
    Builds global Y across all subjects (significant electrodes).
    Writes to {output_dir}/rrr/.

  per_subject:
    Runs RRR independently per subject.
    Writes to {output_dir}/per_subject_rrr/subjects/sub_{id}/.

  per_region:
    Runs RRR independently per atlas region.
    Writes to {output_dir}/per_region_rrr/regions/{region}/.

Output layout per unit (subject or region):
    {rrr_unit_dir}/
        grid_results.csv
        summary.csv
        fold_{f}/
            A.npy, B.npy          (full-rank factor matrices, shared across k)
        k{k}/fold_{f}/
            metrics.npz
"""

import gc
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import (
    load_embeddings,
    build_XY_from_subjects,
)
from src.utils.data_loaders import load_sig_electrodes
from src.utils.atlas import build_electrode_region_map
from src.core.config import RRRConfig, ScriptMode, get_results_dir
from src.utils.config_utils import parse_args, load_config
from src.utils.encoding import (
    load_encoding_results,
    build_enc_subjects_info,
    build_per_fold_baselines,
    get_and_offset_arbitrary_embeddings,
)
from src.utils.sae import solve_regression
from src.utils.srrr import (
    fit_dense_rrr,
    save_rrr_factors,
    save_rrr_result,
    compute_correlation,
)
from src.utils.analysis import build_sig_mask_flat_for_elecs
from src.utils.rrr_utils import (
    add_sig_nonsig_corrs,
    apply_sig_nonsig_splits_and_strip,
    make_sig_split_df,
    run_prediction_metrics,
)
from src.utils.subspace_utils import (
    evaluate_subspace_ols,
    evaluate_complement_and_arbitrary,
    add_primary_cross_model_metrics,
    build_nr_metrics_dict,
    compute_stacked_joint_corr,
)


# ---------------------------------------------------------------------------
# Core per-unit loop
# ---------------------------------------------------------------------------


def _run_k_for_rrr(
    A_full,
    B_full,
    k,
    X_train,
    X_test,
    X_arbitrary,
    Y_train,
    Y_test,
    Y_hat_full,
    Y_hat_ols_train,
    train_idx,
    test_idx,
    sig_mask_flat,
    fold_f,
    alphas,
    config,
    rrr_unit_dir,
    oof_spill_dir,
    device,
):
    A = A_full[:, :k]
    B = B_full[:, :k]

    # Effective rank of A (may be < k if A is rank-deficient)
    sv = np.linalg.svd(A, compute_uv=False)
    effective_rank = int(np.sum(sv > 1e-6 * sv[0])) if len(sv) > 0 and sv[0] > 0 else 0

    Y_hat_rrr = Y_hat_full @ B @ B.T
    Y_hat_rrr_train = Y_hat_ols_train @ B @ B.T

    suf, cca_sim, proc_sim = run_prediction_metrics(
        Y_hat_rrr,
        Y_hat_rrr_train,
        Y_train,
        X_train,
        X_test,
        Y_test,
        Y_hat_full,
        alphas=alphas,
        compute_cca=config.compute_cca,
        compute_procrustes=config.compute_procrustes,
    )

    # Non-reduced-rank OLS from k-dim subspace
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
        A_full,
        k,
        X_train,
        X_test,
        X_arbitrary,
        train_idx,
        test_idx,
        Y_train,
        Y_test,
        nr_result["Y_hat_train"],
        nr_result["Y_hat"],
        alphas=np.logspace(1, 10, 13) if k < 50 else np.logspace(-2, 10, 13),
        device=device,
    )

    metrics = {
        "full_r2": suf["full_r2"],
        "minimal_r2": suf["minimal_r2"],
        "recovery_r2": suf["recovery_r2"],
        "pred_r2": suf["pred_r2"],
        "full_corr": suf["full_corr"],
        "minimal_corr": suf["minimal_corr"],
        "recovery_corr": suf["recovery_corr"],
        "effective_rank": effective_rank,
        "fold": fold_f,
        # "ridge_corr": ridge_result["correlations"].mean(),
        "k": k,
    }
    if cca_sim is not None:
        metrics["cca_similarity"] = cca_sim
    if proc_sim is not None:
        metrics["procrustes_similarity"] = proc_sim

    if sig_mask_flat is not None:
        add_sig_nonsig_corrs(
            metrics,
            suf,
            sig_mask_flat,
            [
                ("full_corr", "full_corr_per_target"),
                ("minimal_corr", "minimal_corr_per_target"),
                ("recovery_corr", "recovery_corr_per_target"),
                # ("ridge_corr", "ridge_corr_per_target"),
            ],
        )

    metrics.update(
        build_nr_metrics_dict(
            nr_result,
            sig_mask_flat=sig_mask_flat,
            complement_result=complement_result,
            arbitrary_result=arbitrary_result,
        )
    )

    # Cross-model metrics
    Y_hat_nr = nr_result["Y_hat"]
    Y_hat_complement = complement_result["complement_Y_hat"]
    Y_hat_arbitrary = (
        arbitrary_result["arbitrary_Y_hat"] if arbitrary_result is not None else None
    )
    add_primary_cross_model_metrics(
        metrics,
        Y_hat_rrr,
        Y_hat_full,
        Y_hat_complement,
        Y_hat_arbitrary,
        Y_test,
        "rrr",
    )
    add_primary_cross_model_metrics(
        metrics,
        Y_hat_nr,
        Y_hat_full,
        Y_hat_complement,
        Y_hat_arbitrary,
        Y_test,
        "nr",
    )

    # Spill OOF predictions to disk to avoid OOM (loaded one k at a time post-loop)
    for _field, _arr in [
        ("Y_hat_rrr", Y_hat_rrr),
        ("Y_hat_nr", Y_hat_nr),
        ("Y_hat_complement", Y_hat_complement),
        ("Y_hat_full", Y_hat_full),
        ("Y_hat_arbitrary", Y_hat_arbitrary),
        ("Y_test", Y_test),
    ]:
        _spill_path = os.path.join(oof_spill_dir, f"oof_{k}_{fold_f}_{_field}.npy")
        if _arr is not None:
            np.save(_spill_path, _arr)
        else:
            np.save(_spill_path, np.array(None, dtype=object))

    # Split any *_per_target arrays into *_sig/*_nonsig scalars, then strip them
    apply_sig_nonsig_splits_and_strip(metrics, sig_mask_flat)

    save_rrr_result(rrr_unit_dir, k=k, fold=fold_f, metrics=metrics)
    # Now drop nr corr per target.
    del metrics["nr_corr_per_target"]
    return metrics


def _calculate_joint_correlations(
    oof_spill_dir: str, n_folds: int, config: RRRConfig, sig_mask_flat, all_rows
):
    _oof_fields = [
        "Y_hat_rrr",
        "Y_hat_nr",
        "Y_hat_complement",
        "Y_hat_full",
        "Y_hat_arbitrary",
        "Y_test",
    ]
    _stacked_pairs = [("nr", "full"), ("nr", "complement"), ("nr", "arbitrary")]
    for k in config.rank_grid:
        # Load OOF arrays for this k from disk (one k at a time to stay within RAM)
        oof: dict[str, list] = {field: [] for field in _oof_fields}
        for fold_f in range(n_folds):
            for field in _oof_fields:
                _spill_path = os.path.join(
                    oof_spill_dir, f"oof_{k}_{fold_f}_{field}.npy"
                )
                _loaded = np.load(_spill_path, allow_pickle=True)
                oof[field].append(
                    None if _loaded.dtype == object and _loaded.ndim == 0 else _loaded
                )

        stacked_joint: dict[str, float] = {}
        for label_a, label_b in _stacked_pairs:
            corr = compute_stacked_joint_corr(
                oof[f"Y_hat_{label_a}"],
                oof[f"Y_hat_{label_b}"],
                oof["Y_test"],
                n_splits=n_folds,
            )
            stacked_joint[f"{label_a}_{label_b}_joint_corr"] = float(corr.mean())
            stacked_joint[f"{label_a}_{label_b}_joint_corr_per_target"] = corr
        for row in all_rows:
            if row["k"] == k:
                row.update(stacked_joint)

        # TODO: Remove this when cleaning up the code.
        # Save the nr_complement joint corr per target so we can use it for significance testing later.
        # Read metrics.npz and write to sub-fold folder
        for fold in range(n_folds):
            metrics_path = os.path.join(
                rrr_unit_dir, f"k{k}", f"fold_{fold}", "metrics.npz"
            )
            if os.path.exists(metrics_path):
                metrics = dict(np.load(metrics_path))
                curr_complement_joint_corr_per_target = None
                for row in all_rows:
                    if row["k"] == k and row["fold"] == fold:
                        curr_complement_joint_corr_per_target = row.get(
                            "nr_complement_joint_corr_per_target"
                        )
                if curr_complement_joint_corr_per_target is not None:
                    metrics.update(
                        {
                            "nr_complement_joint_corr_per_target": curr_complement_joint_corr_per_target[
                                fold
                            ],
                        }
                    )
                    np.savez(metrics_path, **metrics)
                else:
                    print(
                        f"Unable to find matching row for fold {fold} and k {k} to save complement joint corr per target."
                    )

    for metrics in all_rows:
        apply_sig_nonsig_splits_and_strip(metrics, sig_mask_flat)

    # Clean up OOF spill directory now that stacked joint corrs are computed
    import shutil

    shutil.rmtree(oof_spill_dir, ignore_errors=True)

    return all_rows


def _plot_results(df, X_arbitrary, sig_mask_flat, rrr_unit_dir):
    # Summary: average over folds (only include columns that were computed)
    agg_cols = [
        "full_r2",
        "minimal_r2",
        "recovery_r2",
        "pred_r2",
        "full_corr",
        "minimal_corr",
        "recovery_corr",
        "effective_rank",
        "nr_r2",
        "nr_corr",
        "nr_recovery_r2",
        "nr_recovery_corr",
        "nr_pred_r2",
        "nr_complement_corr",
        "nr_complement_recovery_corr",
        # "ridge_corr",
        # Cross-model metrics for RRR reduced-rank model
        "rrr_full_pred_corr",
        "rrr_full_joint_corr",
        "rrr_complement_pred_corr",
        "rrr_complement_joint_corr",
        # Cross-model metrics for NR model
        "nr_full_pred_corr",
        "nr_full_joint_corr",
        "nr_complement_pred_corr",
        "nr_complement_joint_corr",
        # Complement vs other models
        "complement_full_pred_corr",
        "complement_full_joint_corr",
    ]
    if X_arbitrary is not None:
        agg_cols.extend(
            [
                "nr_arbitrary_corr",
                "nr_arbitrary_recovery_corr",
                "rrr_arbitrary_pred_corr",
                "rrr_arbitrary_joint_corr",
                "nr_arbitrary_pred_corr",
                "nr_arbitrary_joint_corr",
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
                "nr_corr_sig",
                "nr_corr_nonsig",
                "nr_recovery_corr_sig",
                "nr_recovery_corr_nonsig",
                "nr_complement_corr_sig",
                "nr_complement_corr_nonsig",
                "nr_complement_recovery_corr_sig",
                "nr_complement_recovery_corr_nonsig",
                # Cross-model sig/nonsig for RRR reduced-rank
                "rrr_full_pred_corr_sig",
                "rrr_full_pred_corr_nonsig",
                "rrr_full_joint_corr_sig",
                "rrr_full_joint_corr_nonsig",
                "rrr_complement_pred_corr_sig",
                "rrr_complement_pred_corr_nonsig",
                "rrr_complement_joint_corr_sig",
                "rrr_complement_joint_corr_nonsig",
                # Cross-model sig/nonsig for NR
                "nr_full_pred_corr_sig",
                "nr_full_pred_corr_nonsig",
                "nr_full_joint_corr_sig",
                "nr_full_joint_corr_nonsig",
                "nr_complement_pred_corr_sig",
                "nr_complement_pred_corr_nonsig",
                "nr_complement_joint_corr_sig",
                "nr_complement_joint_corr_nonsig",
                # Cross-model sig/nonsig for complement
                "complement_full_pred_corr_sig",
                "complement_full_pred_corr_nonsig",
                "complement_full_joint_corr_sig",
                "complement_full_joint_corr_nonsig",
            ]
        )
        if X_arbitrary is not None:
            agg_cols.extend(
                [
                    "nr_arbitrary_corr_sig",
                    "nr_arbitrary_corr_nonsig",
                    "nr_arbitrary_recovery_corr_sig",
                    "nr_arbitrary_recovery_corr_nonsig",
                    "rrr_arbitrary_pred_corr_sig",
                    "rrr_arbitrary_pred_corr_nonsig",
                    "rrr_arbitrary_joint_corr_sig",
                    "rrr_arbitrary_joint_corr_nonsig",
                    "nr_arbitrary_pred_corr_sig",
                    "nr_arbitrary_pred_corr_nonsig",
                    "nr_arbitrary_joint_corr_sig",
                    "nr_arbitrary_joint_corr_nonsig",
                    "complement_arbitrary_pred_corr_sig",
                    "complement_arbitrary_pred_corr_nonsig",
                    "complement_arbitrary_joint_corr_sig",
                    "complement_arbitrary_joint_corr_nonsig",
                ]
            )
    summary = df.groupby("k")[agg_cols].mean().reset_index()
    summary.to_csv(os.path.join(rrr_unit_dir, "summary.csv"), index=False)

    from src.utils.plotting import plot_rrr_rank_results

    plot_rrr_rank_results(
        summary,
        grid_df=df,
        title=f"RRR: perf vs rank ({os.path.basename(rrr_unit_dir)})",
        save_path=os.path.join(rrr_unit_dir, "rank_analysis.png"),
        primary_label="rrr",
    )
    if has_sig_split:
        plot_rrr_rank_results(
            make_sig_split_df(summary, "sig"),
            grid_df=make_sig_split_df(df, "sig"),
            title=f"RRR (sig targets): perf vs rank ({os.path.basename(rrr_unit_dir)})",
            save_path=os.path.join(rrr_unit_dir, "rank_analysis_sig.png"),
            primary_label="rrr",
        )
        plot_rrr_rank_results(
            make_sig_split_df(summary, "nonsig"),
            grid_df=make_sig_split_df(df, "nonsig"),
            title=f"RRR (nonsig targets): perf vs rank ({os.path.basename(rrr_unit_dir)})",
            save_path=os.path.join(rrr_unit_dir, "rank_analysis_nonsig.png"),
            primary_label="rrr",
        )

    # Non-reduced-rank plot: rename nr_ metric columns to standard names and reuse plotter
    nr_col_map = {
        "nr_r2": "minimal_r2",
        "nr_corr": "minimal_corr",
        "nr_recovery_r2": "recovery_r2",
        "nr_recovery_corr": "recovery_corr",
        "nr_pred_r2": "pred_r2",
        "nr_cca_similarity": "cca_similarity",
        "nr_procrustes_similarity": "procrustes_similarity",
        "nr_complement_corr": "complement_corr",
        "nr_complement_recovery_corr": "complement_recovery_corr",
        "nr_arbitrary_corr": "arbitrary_corr",
        "nr_arbitrary_recovery_corr": "arbitrary_recovery_corr",
        "nr_corr_sig": "minimal_corr_sig",
        "nr_corr_nonsig": "minimal_corr_nonsig",
        "nr_recovery_corr_sig": "recovery_corr_sig",
        "nr_recovery_corr_nonsig": "recovery_corr_nonsig",
        "nr_complement_corr_sig": "complement_corr_sig",
        "nr_complement_corr_nonsig": "complement_corr_nonsig",
        "nr_complement_recovery_corr_sig": "complement_recovery_corr_sig",
        "nr_complement_recovery_corr_nonsig": "complement_recovery_corr_nonsig",
        "nr_arbitrary_corr_sig": "arbitrary_corr_sig",
        "nr_arbitrary_corr_nonsig": "arbitrary_corr_nonsig",
        "nr_arbitrary_recovery_corr_sig": "arbitrary_recovery_corr_sig",
        "nr_arbitrary_recovery_corr_nonsig": "arbitrary_recovery_corr_nonsig",
    }
    # Drop original reduced-rank columns, then rename nr_ columns into their place.
    # Also drop RRR-specific cross-model cols — the NR plot uses nr_*_pred_corr directly.
    drop_cols = [v for v in nr_col_map.values() if v in df.columns]
    rrr_cross_cols = [
        c
        for c in df.columns
        if c.startswith("rrr_") and ("_pred_corr" in c or "_joint_corr" in c)
    ]
    nr_grid = df.drop(columns=drop_cols + rrr_cross_cols, errors="ignore").rename(
        columns=nr_col_map
    )
    nr_summary = summary.drop(
        columns=[c for c in drop_cols + rrr_cross_cols if c in summary.columns]
    ).rename(columns=nr_col_map)
    plot_rrr_rank_results(
        nr_summary,
        grid_df=nr_grid,
        title=f"RRR (non-reduced ridge): perf vs rank ({os.path.basename(rrr_unit_dir)})",
        save_path=os.path.join(rrr_unit_dir, "rank_analysis_nr.png"),
        primary_label="nr",
    )
    if has_sig_split:
        plot_rrr_rank_results(
            make_sig_split_df(nr_summary, "sig"),
            grid_df=make_sig_split_df(nr_grid, "sig"),
            title=f"RRR (non-reduced ridge, sig targets): perf vs rank ({os.path.basename(rrr_unit_dir)})",
            save_path=os.path.join(rrr_unit_dir, "rank_analysis_nr_sig.png"),
            primary_label="nr",
        )
        plot_rrr_rank_results(
            make_sig_split_df(nr_summary, "nonsig"),
            grid_df=make_sig_split_df(nr_grid, "nonsig"),
            title=f"RRR (non-reduced ridge, nonsig targets): perf vs rank ({os.path.basename(rrr_unit_dir)})",
            save_path=os.path.join(rrr_unit_dir, "rank_analysis_nr_nonsig.png"),
            primary_label="nr",
        )


def _run_rrr_for_unit(
    rrr_unit_dir: str,
    X_raw: np.ndarray,
    test_idxs: list,
    per_fold_baselines: list,
    config: RRRConfig,
    sig_mask_flat: np.ndarray | None = None,
    X_arbitrary: np.ndarray | None = None,
) -> pd.DataFrame:
    """Core RRR loop for one analysis unit (cross-subject, per-subject, or per-region).

    Args:
        rrr_unit_dir: Root output directory for this unit.
        X_raw: (n_words, n_embed) raw embedding matrix.
        test_idxs: List of test index arrays, one per fold.
        per_fold_baselines: Output of build_per_fold_baselines — pre-computed scaled Y
            arrays and baseline predictions from the saved encoding model.
        config: RRRConfig.
        sig_mask_flat: Boolean mask over targets (True = significant). If provided,
            sig/nonsig split metrics are computed.
        X_arbitrary: (n_words, n_embed) arbitrary embeddings for null baseline.

    Returns:
        DataFrame of per-(fold, k) metrics for grid_results.csv.
    """
    os.makedirs(rrr_unit_dir, exist_ok=True)

    enc = config.encoding
    alphas = np.array(enc.ridge_alphas) if enc.ridge_alphas is not None else None

    n_folds = len(test_idxs)
    n_embed = X_raw.shape[1]
    selected_idx = np.arange(n_embed)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_rows = []
    # Per-fold alpha tracking (populated when config.log_alphas is True)
    per_fold_rank_alphas: list[dict[int, np.ndarray]] = []

    # OOF combined OLS predictions spilled to disk to avoid OOM; loaded one k at a time post-loop
    # Deleted at end of run.
    oof_spill_dir = os.path.join(rrr_unit_dir, "_oof_spill")
    os.makedirs(oof_spill_dir, exist_ok=True)

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
        Y_hat_ols_train = baseline["Y_hat_full_train"]

        # Fit RRR once at max rank — the SVD is identical for all k values;
        # only the truncation point differs.
        max_k = max(config.rank_grid)
        rrr_result = fit_dense_rrr(
            X_train,
            Y_train,
            k=max_k,
            selected_idx=selected_idx,
            precomputed_Y_hat_ols=Y_hat_ols_train,
            precomputed_beta_full=baseline["beta_full_raw"],
        )
        A_full = rrr_result["A_dense"]
        B_full = rrr_result["B_dense"]
        save_rrr_factors(rrr_unit_dir, fold=fold_f, A=A_full, B=B_full)

        rank_alpha_dict: dict[int, np.ndarray] = {}

        print(f"  Running {len(config.rank_grid)} rank(s): {config.rank_grid}")
        for k in tqdm(
            config.rank_grid, desc=f"  fold {fold_f}", leave=True, unit="rank"
        ):
            metrics = _run_k_for_rrr(
                A_full,
                B_full,
                k,
                X_train,
                X_test,
                X_arbitrary,
                Y_train,
                Y_test,
                Y_hat_full,
                Y_hat_ols_train,
                train_idx,
                test_idx,
                sig_mask_flat,
                fold_f,
                alphas,
                config,
                rrr_unit_dir,
                oof_spill_dir,
                device,
            )
            all_rows.append(metrics)

        print(f"  Fold {fold_f} done in {time.time() - fold_start:.1f}s")

    # Compute stacked OLS joint corr using properly held-out OOF predictions.
    # The joint_corr in compute_cross_model_metrics was removed because it fit and
    # evaluated the stacking OLS on the same test fold.  Here we pool OOF predictions
    # across all outer folds and run a second K-fold CV so the stacking weights are
    # never trained on the data they are evaluated on.
    print("\nComputing stacked OLS joint correlations...")
    all_rows = _calculate_joint_correlations(
        oof_spill_dir, n_folds, config, sig_mask_flat, all_rows
    )

    # Alpha-vs-rank diagnostic plot (aggregate across folds)
    if config.log_alphas and per_fold_rank_alphas:
        from src.utils.plotting import plot_alpha_vs_rank
        from src.utils.constants import RIDGE_ALPHAS as _default_alphas

        _alpha_grid = alphas if alphas is not None else _default_alphas
        # Merge across folds: concatenate per-rank arrays
        merged: dict[int, np.ndarray] = {}
        for frd in per_fold_rank_alphas:
            for k_val, arr in frd.items():
                merged[k_val] = np.concatenate([merged.get(k_val, np.array([])), arr])
        plot_alpha_vs_rank(
            merged,
            _alpha_grid,
            title=f"RRR alpha vs rank ({os.path.basename(rrr_unit_dir)})",
            save_path=os.path.join(rrr_unit_dir, "alpha_vs_rank.png"),
        )

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(rrr_unit_dir, "grid_results.csv"), index=False)
    _plot_results(df, X_arbitrary, sig_mask_flat, rrr_unit_dir)

    return df


# ---------------------------------------------------------------------------
# Significance mask helper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Branch: cross-subject RRR
# ---------------------------------------------------------------------------


def _get_decomposition_inputs(
    subject_ids,
    output_dir,
    sig_electrodes,
    encoding_config,
    embeddings,
    word_df,
    compute_sig_nonsig=False,
    sig_dir=None,
):
    enc_results = load_encoding_results(
        os.path.join(
            output_dir, "subjects", f"sub_{subject_ids[0]}", "encoding_results.joblib"
        )
    )
    test_idxs = enc_results["test_idxs"]

    subs_with_sig = [
        s for s in subject_ids if s in sig_electrodes and sig_electrodes[s]
    ]
    if not subs_with_sig:
        print("  No significant electrodes found; skipping cross-subject RRR.")
        return

    X_raw, Y_full, selection = build_XY_from_subjects(
        subject_ids,
        encoding_config,
        sig_electrodes,
        embeddings,
        word_df,
    )
    print(f"  X: {X_raw.shape}, Y: {Y_full.shape}")

    enc_subjects_info = build_enc_subjects_info(
        output_dir, subs_with_sig, sig_electrodes
    )
    per_fold_baselines = build_per_fold_baselines(
        X_raw, Y_full, test_idxs, enc_subjects_info
    )

    # Generate arbitrary embeddings for null baseline
    X_arbitrary = get_and_offset_arbitrary_embeddings(
        embeddings,
        selection,
        word_df,
        encoding_config.offset,
        encoding_config.arbitrary_seed,
    )

    sig_mask_flat = None
    if compute_sig_nonsig:
        sig_mask_flat = build_sig_mask_flat_for_elecs(
            output_dir,
            {s: sig_electrodes[s] for s in subs_with_sig},
            enc_results,
            sig_dir=sig_dir,
        )

    return X_raw, X_arbitrary, per_fold_baselines, test_idxs, sig_mask_flat


def _run_cross_subject_rrr(
    config: RRRConfig,
    enc,
    output_dir,
    sig_dir,
    rrr_dir,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    word_df,
):
    """RRR on raw embeddings using all subjects' significant electrodes."""
    X_raw, X_arbitrary, per_fold_baselines, test_idxs, sig_mask_flat = (
        _get_decomposition_inputs(
            subject_ids,
            output_dir,
            sig_electrodes,
            enc,
            aligned_embeddings,
            word_df,
            compute_sig_nonsig=config.compute_sig_nonsig,
            sig_dir=sig_dir,
        )
    )

    _run_rrr_for_unit(
        rrr_unit_dir=rrr_dir,
        X_raw=X_raw,
        test_idxs=test_idxs,
        per_fold_baselines=per_fold_baselines,
        config=config,
        sig_mask_flat=sig_mask_flat,
        X_arbitrary=X_arbitrary,
    )


# ---------------------------------------------------------------------------
# Branch: per-subject RRR
# ---------------------------------------------------------------------------


def _run_per_subject_rrr(
    config: RRRConfig,
    enc,
    output_dir,
    sig_dir,
    rrr_dir,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    word_df,
):
    """RRR on raw embeddings independently per subject."""
    for sub_id in subject_ids:
        print(f"\n{'='*60}\nSubject {sub_id}")

        rrr_sub_dir = os.path.join(rrr_dir, "subjects", f"sub_{sub_id}")
        sub_sig_elecs = sig_electrodes.get(sub_id, [])
        if not sub_sig_elecs:
            print(f"  No significant electrodes; skipping.")
            continue

        X_raw, X_arbitrary, per_fold_baselines, test_idxs, sig_mask_flat = (
            _get_decomposition_inputs(
                [sub_id],
                output_dir,
                sub_sig_elecs,
                enc,
                aligned_embeddings,
                word_df,
                compute_sig_nonsig=config.compute_sig_nonsig,
                sig_dir=sig_dir,
            )
        )

        _run_rrr_for_unit(
            rrr_unit_dir=rrr_sub_dir,
            X_raw=X_raw,
            test_idxs=test_idxs,
            per_fold_baselines=per_fold_baselines,
            config=config,
            sig_mask_flat=sig_mask_flat,
            X_arbitrary=X_arbitrary,
        )


# ---------------------------------------------------------------------------
# Branch: per-region RRR
# ---------------------------------------------------------------------------


def _run_per_region_rrr(
    config: RRRConfig,
    enc,
    output_dir,
    sig_dir,
    rrr_dir,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    word_df,
):
    """RRR on raw embeddings independently per atlas region."""

    print("Building electrode region map...")
    electrode_region_map = build_electrode_region_map(
        subject_ids,
        enc.data_root,
        getattr(config, "atlas_region_groups", None),
        getattr(config, "atlas_path", None),
    )
    print(f"  Regions found: {list(electrode_region_map.keys())}")

    for region_name, sub_elec_map in electrode_region_map.items():
        print(f"\n{'='*60}\nRegion: {region_name}")

        rrr_region_dir = os.path.join(rrr_dir, "regions", region_name)

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

        X_raw, X_arbitrary, per_fold_baselines, test_idxs, sig_mask_flat = (
            _get_decomposition_inputs(
                subs_in_region,
                output_dir,
                per_subject_region_elecs,
                enc,
                aligned_embeddings,
                word_df,
                compute_sig_nonsig=config.compute_sig_nonsig,
                sig_dir=sig_dir,
            )
        )

        _run_rrr_for_unit(
            rrr_unit_dir=rrr_region_dir,
            X_raw=X_raw,
            test_idxs=test_idxs,
            per_fold_baselines=per_fold_baselines,
            config=config,
            sig_mask_flat=sig_mask_flat,
            X_arbitrary=X_arbitrary,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args, overrides = parse_args("RRR: raw-input reduced rank regression baseline")
    config: RRRConfig = load_config(args.config, overrides, RRRConfig)
    enc = config.encoding

    import torch
    from himalaya.backend import set_backend

    if torch.cuda.is_available():
        set_backend("torch_cuda")

    subject_ids = enc.subjects or list(range(1, 10))
    output_dir = config.output_dir or get_results_dir(
        enc.model, enc.offset, enc.per_model_config
    )

    # Resolve RRR output directory
    if config.rrr_output_dir:
        rrr_dir = config.rrr_output_dir
    elif config.mode == ScriptMode.PER_REGION:
        rrr_dir = os.path.join(output_dir, "per_region_rrr")
    elif config.mode == ScriptMode.PER_SUBJECT:
        rrr_dir = os.path.join(output_dir, "per_subject_rrr")
    else:
        rrr_dir = os.path.join(output_dir, "rrr")

    os.makedirs(rrr_dir, exist_ok=True)
    print(f"Output dir:  {output_dir}")
    print(f"RRR dir:     {rrr_dir}")
    print(f"Mode:        {config.mode.value}")

    # Load significance CSV and build subject → electrode mapping
    sig_dir = config.sig_dir or output_dir
    sig_electrodes, _ = load_sig_electrodes(output_dir, subject_ids, sig_dir=sig_dir)

    # Load embeddings
    print("Loading embeddings...")
    aligned_embeddings, df_word = load_embeddings(
        enc.data_root, enc.model, **enc.per_model_config
    )

    branch_kwargs = dict(
        config=config,
        enc=enc,
        output_dir=output_dir,
        sig_dir=sig_dir,
        rrr_dir=rrr_dir,
        subject_ids=subject_ids,
        sig_electrodes=sig_electrodes,
        aligned_embeddings=aligned_embeddings,
        df_word=df_word,
    )

    if config.mode == ScriptMode.PER_REGION:
        _run_per_region_rrr(**branch_kwargs)
    elif config.mode == ScriptMode.PER_SUBJECT:
        _run_per_subject_rrr(**branch_kwargs)
    else:
        _run_cross_subject_rrr(**branch_kwargs)

    print("\nDone.")


if __name__ == "__main__":
    main()
