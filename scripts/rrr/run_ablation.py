"""
run_ablation.py -- RRR latent space ablation study.

Loads pre-computed RRR A and B matrices, projects embeddings into the
latent space, applies quantile-based masks (tail-only, center-only, random),
trains a fresh ridge regression from the masked projections to neural targets,
and evaluates reconstruction performance.  This determines whether encoding
information is concentrated in distributional tails or centers.

Three modes controlled by AblationConfig.mode:

  cross_subject (default):
    Builds global Y across all subjects (significant electrodes).
    Writes to {output_dir}/ablation/.

  per_subject:
    Runs independently per subject.
    Writes to {output_dir}/per_subject_ablation/subjects/sub_{id}/.

  per_region:
    Runs independently per atlas region.
    Writes to {output_dir}/per_region_ablation/regions/{region}/.

Output layout per unit:
    {ablation_unit_dir}/
        grid_results.csv          (fold x k x center_pct x mask_type)
        summary.csv               (averaged over folds)
        ablation_vs_rank.png      (performance vs k, one panel per condition)
        k{k}/
            ablation_pct_zeroed.png   (performance vs pct_zeroed for this k)
"""

import gc
import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.utils import (
    build_XY_from_subjects,
    load_embeddings,
    apply_offset,
    load_precomputed_epochs,
    epochs_from_cache,
)
from src.utils.analysis import build_sig_mask_flat_for_elecs
from src.utils.atlas import build_electrode_region_map
from src.core.config import AblationConfig, ScriptMode, get_results_dir
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.config_utils import parse_args, load_config
from src.utils.data_loaders import load_sig_electrodes
from src.utils.encoding import (
    load_encoding_results,
    build_per_fold_baselines,
)
from src.utils.rrr_utils import (
    add_sig_nonsig_corrs,
    resolve_rrr_source_dir,
)
from src.utils.sae import solve_regression
from src.utils.tensor_utils import to_numpy
from src.utils.correlation import fisher_z_mean
from himalaya.scoring import correlation_score, r2_score

# ---------------------------------------------------------------------------
# Masking functions
# ---------------------------------------------------------------------------


def compute_quantile_thresholds(
    X_proj_train: np.ndarray, center_pct: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-column quantile thresholds from training data.

    The center zone is symmetric around the **median** of each dimension.

    Args:
        X_proj_train: (n_train, k) latent activations on training set.
        center_pct: Percentage of distribution to consider "center"
            (e.g., 50 means middle 50%).

    Returns:
        lo: (k,) lower quantile boundary per dimension.
        hi: (k,) upper quantile boundary per dimension.
    """
    lo = np.percentile(X_proj_train, 50 - center_pct / 2, axis=0)
    hi = np.percentile(X_proj_train, 50 + center_pct / 2, axis=0)
    return lo, hi


def apply_quantile_mask(
    X_proj: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    mask_type: str,
) -> np.ndarray:
    """Apply a quantile-based mask to latent activations.

    Args:
        X_proj: (n, k) latent activations.
        lo: (k,) lower quantile boundary per dimension (from training set).
        hi: (k,) upper quantile boundary per dimension (from training set).
        mask_type: ``"tail"`` keeps tails (zeros center) or
            ``"center"`` keeps center (zeros tails).

    Returns:
        (n, k) masked activations (zeros where masked out).
    """
    in_center = (X_proj >= lo[None, :]) & (X_proj <= hi[None, :])
    if mask_type == "tail":
        keep = ~in_center
    else:
        keep = in_center
    return X_proj * keep.astype(X_proj.dtype)


def apply_random_mask(
    X_proj: np.ndarray,
    frac_zero: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Zero out a random fraction of entries per column.

    Args:
        X_proj: (n, k) latent activations.
        frac_zero: Fraction of entries to zero per column (0-1).
        rng: Numpy random generator for reproducibility.

    Returns:
        (n, k) masked activations.
    """
    mask = np.ones_like(X_proj, dtype=bool)
    n = X_proj.shape[0]
    n_zero = int(n * frac_zero)
    for col in range(X_proj.shape[1]):
        idx = rng.choice(n, size=n_zero, replace=False)
        mask[idx, col] = False
    return X_proj * mask.astype(X_proj.dtype)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def _eval_ablated(
    X_proj_train: np.ndarray,
    X_proj_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    Y_hat_full: np.ndarray,
    alphas: np.ndarray | None,
    sig_mask_flat: np.ndarray | None,
    device: str,
) -> dict:
    """Train a fresh ridge from (masked) projections to Y and evaluate.

    Args:
        X_proj_train: (n_train, k) masked latent activations (train).
        X_proj_test: (n_test, k) masked latent activations (test).
        Y_train: (n_train, n_targets) neural data (train).
        Y_test: (n_test, n_targets) neural data (test).
        Y_hat_full: (n_test, n_targets) full baseline predictions.
        alphas: Ridge alpha grid.
        sig_mask_flat: Boolean mask for sig/nonsig split.
        device: torch device string.

    Returns:
        Dict of scalar metrics.
    """
    W, b = solve_regression(
        torch.as_tensor(X_proj_train, dtype=torch.float32, device=device),
        torch.as_tensor(Y_train, dtype=torch.float32, device=device),
        "ridge",
        device,
        standardize=False,
        alphas=alphas,
    )
    X_test_t = torch.as_tensor(X_proj_test, dtype=torch.float32, device=device)
    Y_hat_test = to_numpy(X_test_t @ W + b)

    nr_corr_arr = to_numpy(correlation_score(Y_test, Y_hat_test))
    nr_r2_arr = to_numpy(r2_score(Y_test, Y_hat_test))
    full_corr_arr = to_numpy(correlation_score(Y_test, Y_hat_full))
    full_r2_arr = to_numpy(r2_score(Y_test, Y_hat_full))

    metrics = {
        "full_corr": float(fisher_z_mean(full_corr_arr)),
        "full_r2": float(full_r2_arr.mean()),
        "nr_corr": float(fisher_z_mean(nr_corr_arr)),
        "nr_r2": float(nr_r2_arr.mean()),
    }
    if sig_mask_flat is not None:
        for col, arr in [("full_corr", full_corr_arr), ("nr_corr", nr_corr_arr)]:
            metrics[f"{col}_sig"] = (
                float(fisher_z_mean(arr[sig_mask_flat]))
                if sig_mask_flat.any()
                else float("nan")
            )
            metrics[f"{col}_nonsig"] = (
                float(fisher_z_mean(arr[~sig_mask_flat]))
                if (~sig_mask_flat).any()
                else float("nan")
            )
    return metrics


# ---------------------------------------------------------------------------
# Core per-unit loop
# ---------------------------------------------------------------------------


def _run_ablation_for_unit(
    ablation_unit_dir: str,
    rrr_unit_dir: str,
    X_raw: np.ndarray,
    test_idxs: list,
    per_fold_baselines: list,
    config: AblationConfig,
    sig_mask_flat: np.ndarray | None = None,
) -> pd.DataFrame:
    """Core ablation loop for one analysis unit.

    For each fold and rank k, loads the pre-computed RRR A matrix, projects
    embeddings into the k-dim latent space, applies quantile-based masks,
    trains a fresh ridge regression from the masked projections to neural
    targets, and evaluates reconstruction performance.

    Args:
        ablation_unit_dir: Root output directory for this unit.
        rrr_unit_dir: Directory containing pre-computed RRR A/B matrices.
        X_raw: (n_words, n_embed) raw embedding matrix.
        test_idxs: List of test index arrays, one per fold.
        per_fold_baselines: Output of build_per_fold_baselines.
        config: AblationConfig.
        sig_mask_flat: Optional boolean mask for sig/nonsig split.

    Returns:
        DataFrame of per-(fold, k, center_pct, mask_type) metrics.
    """
    os.makedirs(ablation_unit_dir, exist_ok=True)

    enc = config.encoding
    alphas = np.array(enc.ridge_alphas) if enc.ridge_alphas is not None else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_folds = len(test_idxs)
    rank_grid = config.rank_grid

    # Load RRR baseline metrics to include as mask_type="none" rows
    rrr_grid_path = os.path.join(rrr_unit_dir, "grid_results.csv")
    rrr_baseline_df = (
        pd.read_csv(rrr_grid_path) if os.path.exists(rrr_grid_path) else None
    )

    all_rows = []

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

        # Load full-rank A from RRR output
        A_full = np.load(os.path.join(rrr_unit_dir, f"fold_{fold_f}", "A.npy"))

        for k in tqdm(rank_grid, desc=f"  fold {fold_f}", leave=True, unit="rank"):
            A = A_full[:, :k]
            X_proj_train = X_train @ A
            X_proj_test = X_test @ A

            # Include unmasked RRR baseline from pre-computed results
            if rrr_baseline_df is not None:
                rrr_row = rrr_baseline_df[
                    (rrr_baseline_df["fold"] == fold_f) & (rrr_baseline_df["k"] == k)
                ]
                if len(rrr_row) == 1:
                    row = rrr_row.iloc[0]
                    baseline_metrics = {
                        "fold": fold_f,
                        "k": k,
                        "center_pct": 0,
                        "mask_type": "none",
                        "pct_zeroed": 0.0,
                    }
                    for col in ["full_r2", "full_corr", "nr_r2", "nr_corr"]:
                        if col in row.index:
                            baseline_metrics[col] = float(row[col])
                    all_rows.append(baseline_metrics)

            rng = np.random.default_rng(42)

            for center_pct in config.center_pcts:
                # Compute quantile thresholds from training data
                lo, hi = compute_quantile_thresholds(X_proj_train, center_pct)

                for mask_type in config.mask_types:
                    if mask_type == "random":
                        frac_zero = center_pct / 100.0
                        seed_metrics = []
                        for seed_i in range(config.n_random_seeds):
                            seed_rng = np.random.default_rng(42 + seed_i)
                            X_masked_train = apply_random_mask(
                                X_proj_train, frac_zero, seed_rng
                            )
                            X_masked_test = apply_random_mask(
                                X_proj_test, frac_zero, seed_rng
                            )
                            m = _eval_ablated(
                                X_masked_train,
                                X_masked_test,
                                Y_train,
                                Y_test,
                                Y_hat_full,
                                alphas,
                                sig_mask_flat,
                                device,
                            )
                            seed_metrics.append(m)

                        # Average metrics across random seeds
                        avg_metrics = {}
                        for key in seed_metrics[0]:
                            vals = [m[key] for m in seed_metrics]
                            avg_metrics[key] = float(np.mean(vals))

                        all_rows.append(
                            {
                                "fold": fold_f,
                                "k": k,
                                "center_pct": center_pct,
                                "mask_type": "random",
                                "pct_zeroed": float(center_pct),
                                **avg_metrics,
                            }
                        )
                    else:
                        X_masked_train = apply_quantile_mask(
                            X_proj_train, lo, hi, mask_type
                        )
                        X_masked_test = apply_quantile_mask(
                            X_proj_test, lo, hi, mask_type
                        )

                        m = _eval_ablated(
                            X_masked_train,
                            X_masked_test,
                            Y_train,
                            Y_test,
                            Y_hat_full,
                            alphas,
                            sig_mask_flat,
                            device,
                        )

                        pct_zeroed = (
                            float(center_pct)
                            if mask_type == "tail"
                            else float(100 - center_pct)
                        )

                        all_rows.append(
                            {
                                "fold": fold_f,
                                "k": k,
                                "center_pct": center_pct,
                                "mask_type": mask_type,
                                "pct_zeroed": pct_zeroed,
                                **m,
                            }
                        )

        print(f"  Fold {fold_f} done in {time.time() - fold_start:.1f}s")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(ablation_unit_dir, "grid_results.csv"), index=False)

    # Summary: average over folds
    group_cols = ["k", "center_pct", "mask_type", "pct_zeroed"]
    agg_cols = [
        c
        for c in df.columns
        if c not in group_cols + ["fold"]
        and df[c].dtype in (np.float64, np.float32, float)
    ]
    summary = df.groupby(group_cols, dropna=False)[agg_cols].mean().reset_index()
    summary.to_csv(os.path.join(ablation_unit_dir, "summary.csv"), index=False)

    # Generate plots
    from src.utils.plotting import plot_ablation_pct_zeroed, plot_ablation_vs_rank

    for k in rank_grid:
        k_dir = os.path.join(ablation_unit_dir, f"k{k}")
        os.makedirs(k_dir, exist_ok=True)
        k_summary = summary[summary["k"] == k]
        plot_ablation_pct_zeroed(
            k_summary,
            title=f"Ablation at k={k} ({os.path.basename(ablation_unit_dir)})",
            save_path=os.path.join(k_dir, "ablation_pct_zeroed.png"),
        )

    plot_ablation_vs_rank(
        summary,
        title=f"Ablation vs rank ({os.path.basename(ablation_unit_dir)})",
        save_path=os.path.join(ablation_unit_dir, "ablation_vs_rank.png"),
    )

    return df


# ---------------------------------------------------------------------------
# Branch: cross-subject ablation
# ---------------------------------------------------------------------------


def _run_cross_subject_ablation(
    config,
    enc,
    output_dir,
    sig_dir,
    ablation_dir,
    rrr_source_dir,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    df_word,
):
    """Ablation on RRR-projected embeddings using all subjects' significant electrodes."""
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
        print("  No significant electrodes found; skipping cross-subject ablation.")
        return

    X_ref, Y_full, _ = build_XY_from_subjects(
        subject_ids,
        enc,
        sig_electrodes,
        aligned_embeddings,
        df_word,
    )
    print(f"  X: {X_ref.shape}, Y: {Y_full.shape}")

    per_fold_baselines = build_per_fold_baselines(X_ref, Y_full, ref_test_idxs)

    sig_mask_flat = build_sig_mask_flat_for_elecs(
        output_dir,
        {s: sig_electrodes[s] for s in subs_with_sig},
        enc_results,
        sig_dir=sig_dir,
    )

    _run_ablation_for_unit(
        ablation_unit_dir=ablation_dir,
        rrr_unit_dir=rrr_source_dir,
        X_raw=X_ref,
        test_idxs=ref_test_idxs,
        per_fold_baselines=per_fold_baselines,
        config=config,
        sig_mask_flat=sig_mask_flat,
    )


# ---------------------------------------------------------------------------
# Branch: per-subject ablation
# ---------------------------------------------------------------------------


def _run_per_subject_ablation(
    config,
    enc,
    output_dir,
    sig_dir,
    ablation_dir,
    rrr_source_dir,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    df_word,
):
    """Ablation on RRR-projected embeddings independently per subject."""
    for sub_id in subject_ids:
        print(f"\n{'='*60}\nSubject {sub_id}")

        ablation_sub_dir = os.path.join(ablation_dir, "subjects", f"sub_{sub_id}")
        rrr_sub_dir = os.path.join(rrr_source_dir, "subjects", f"sub_{sub_id}")
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

        cache = load_precomputed_epochs(
            sub_id,
            enc.data_root,
            dataset=enc.dataset,
            per_dataset_config=enc.per_dataset_config,
        )
        data = epochs_from_cache(cache, elec_names=sub_sig_elecs)
        X_arr, Y_sub = apply_offset(
            aligned_embeddings[cache["selection"]], data, enc.offset
        )
        del cache, data
        gc.collect()

        Y_sub = Y_sub.astype(np.float32)
        print(f"  X: {X_arr.shape}, Y: {Y_sub.shape}")

        per_fold_baselines = build_per_fold_baselines(X_arr, Y_sub, test_idxs)

        sig_mask_flat = build_sig_mask_flat_for_elecs(
            output_dir,
            {sub_id: sub_sig_elecs},
            enc_results,
            sig_dir=sig_dir,
        )

        _run_ablation_for_unit(
            ablation_unit_dir=ablation_sub_dir,
            rrr_unit_dir=rrr_sub_dir,
            X_raw=X_arr,
            test_idxs=test_idxs,
            per_fold_baselines=per_fold_baselines,
            config=config,
            sig_mask_flat=sig_mask_flat,
        )


# ---------------------------------------------------------------------------
# Branch: per-region ablation
# ---------------------------------------------------------------------------


def _run_per_region_ablation(
    config,
    enc,
    output_dir,
    sig_dir,
    ablation_dir,
    rrr_source_dir,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    df_word,
):
    """Ablation on RRR-projected embeddings independently per atlas region."""
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

    for region_name, sub_elec_map in electrode_region_map.items():
        print(f"\n{'='*60}\nRegion: {region_name}")

        ablation_region_dir = os.path.join(ablation_dir, "regions", region_name)
        rrr_region_dir = os.path.join(rrr_source_dir, "regions", region_name)

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
        X_ref, Y_full, _ = build_XY_from_subjects(
            subs_in_region,
            enc,
            per_subject_region_elecs,
            aligned_embeddings,
            df_word,
        )
        print(f"  X: {X_ref.shape}, Y: {Y_full.shape}")

        per_fold_baselines = build_per_fold_baselines(X_ref, Y_full, ref_test_idxs)

        sig_mask_flat = build_sig_mask_flat_for_elecs(
            output_dir,
            per_subject_region_elecs,
            enc_results,
            sig_dir=sig_dir,
        )

        _run_ablation_for_unit(
            ablation_unit_dir=ablation_region_dir,
            rrr_unit_dir=rrr_region_dir,
            X_raw=X_ref,
            test_idxs=ref_test_idxs,
            per_fold_baselines=per_fold_baselines,
            config=config,
            sig_mask_flat=sig_mask_flat,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args, overrides = parse_args("RRR latent space ablation study")
    config: AblationConfig = load_config(args.config, overrides, AblationConfig)
    enc = config.encoding

    if torch.cuda.is_available():
        from himalaya.backend import set_backend

        set_backend("torch_cuda")

    subject_ids = enc.subjects or list(range(1, 10))
    dataset_context = resolve_dataset_context(
        dataset=enc.dataset,
        subject_ids=subject_ids,
        per_dataset_config=enc.per_dataset_config,
        data_root=enc.data_root,
    )
    get_dataset_adapter(enc.dataset).validate_subjects_share_stimulus(
        subject_ids, enc.per_dataset_config, enc.data_root
    )
    output_dir = config.output_dir or get_results_dir(
        enc.model,
        enc.offset,
        enc.per_model_config,
        dataset_context=dataset_context,
    )

    # Resolve RRR source directory
    rrr_source_dir = resolve_rrr_source_dir(
        config.mode, output_dir, config.rrr_source_dir
    )

    # Resolve ablation output directory
    if config.ablation_output_dir:
        ablation_dir = config.ablation_output_dir
    elif config.mode == ScriptMode.PER_REGION:
        ablation_dir = os.path.join(output_dir, "per_region_ablation")
    elif config.mode == ScriptMode.PER_SUBJECT:
        ablation_dir = os.path.join(output_dir, "per_subject_ablation")
    else:
        ablation_dir = os.path.join(output_dir, "ablation")

    os.makedirs(ablation_dir, exist_ok=True)
    print(f"Output dir:      {output_dir}")
    print(f"RRR source dir:  {rrr_source_dir}")
    print(f"Ablation dir:    {ablation_dir}")
    print(f"Mode:            {config.mode.value}")

    sig_dir = config.sig_dir or output_dir
    sig_electrodes, _ = load_sig_electrodes(
        output_dir,
        subject_ids,
        sig_dir=sig_dir,
        dataset=enc.dataset,
        per_dataset_config=enc.per_dataset_config,
        dataset_context=dataset_context,
        data_root=enc.data_root,
    )

    print("Loading embeddings...")
    aligned_embeddings, df_word = load_embeddings(
        enc.data_root,
        enc.model,
        dataset=enc.dataset,
        per_dataset_config=enc.per_dataset_config,
        dataset_context=dataset_context,
        **enc.per_model_config,
    )

    branch_kwargs = dict(
        config=config,
        enc=enc,
        output_dir=output_dir,
        sig_dir=sig_dir,
        ablation_dir=ablation_dir,
        rrr_source_dir=rrr_source_dir,
        subject_ids=subject_ids,
        sig_electrodes=sig_electrodes,
        aligned_embeddings=aligned_embeddings,
        df_word=df_word,
    )

    if config.mode == ScriptMode.PER_REGION:
        _run_per_region_ablation(**branch_kwargs)
    elif config.mode == ScriptMode.PER_SUBJECT:
        _run_per_subject_ablation(**branch_kwargs)
    else:
        _run_cross_subject_ablation(**branch_kwargs)

    print("\nDone.")


if __name__ == "__main__":
    main()
