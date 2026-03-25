"""
run_partial_svd.py — Partial SVD analysis of encoding model directions.

Removes contiguous windows of SVD directions from the word embeddings,
retrains ridge regression, and measures the correlation drop to quantify
how much each group of SVD directions contributes to encoding performance.

Config parameters (parallel lists, one entry per group-size sweep):
    partial_group_sizes : how many consecutive SVD directions to partial out
    max_partial_direction : max rank to sweep to
    start_direction : where to start the sweep
    step_size : how much to slide the window per step

Output:
    {output_dir}/partial_svd/
        partial_svd_results.npz   — full per-electrode/per-lag correlation matrices
        summary.csv               — tabular summary of mean correlations per window
        cross_subject_sweep.png   — cross-subject correlation vs window start
        subjects/sub_{id}/sweep.png      — per-subject plots
        regions/{region}/sweep.png       — per-region plots (if atlas provided)
        lag_heatmap_gs{N}.png     — per-electrode/per-lag drop heatmap
"""

import gc
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from utils import (
    build_global_sig_mask,
    load_embeddings,
    build_XY_from_subjects,
    load_all_subject_correlations,
    train_encoding,
)
from utils.analysis import partial_out_svd_directions
from utils.atlas import build_electrode_region_map
from utils.config import PartialSVDConfig, get_results_dir
from utils.config_utils import parse_args, load_config
from utils.encoding import get_ridge_model_constructor
from utils.data_loaders import load_sig_electrodes, load_fold_indices
from utils.plotting import plot_partial_svd_sweep, plot_partial_svd_lag_heatmap
from utils.srrr import load_svd_decomposition


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

def _run_partial_svd_sweep(
    partial_svd_dir,
    X,
    Y,
    Vt_per_fold,
    provided_idxs,
    n_elecs,
    n_lags,
    config,
):
    """Core loop: partial out SVD directions and retrain ridge regression.

    Args:
        partial_svd_dir: Output directory.
        X: (n_words, n_features) embedding matrix.
        Y: (n_words, n_elecs * n_lags) neural data.
        Vt_per_fold: (n_folds, n_dirs, n_features) SVD directions per fold.
        provided_idxs: list of (train_idx, test_idx) tuples.
        n_elecs: number of electrodes.
        n_lags: number of time lags.
        config: PartialSVDConfig.

    Returns:
        dict mapping (group_size, window_start) → (n_folds, n_elecs, n_lags) corrs.
    """
    n_folds = len(provided_idxs)
    n_dirs_available = Vt_per_fold.shape[1]
    alphas = np.array(config.encoding.ridge_alphas) if config.encoding.ridge_alphas is not None else None
    model_constructor = get_ridge_model_constructor(alphas=alphas)

    all_results = {}
    n_sweeps = len(config.partial_group_sizes)

    for i in range(n_sweeps):
        group_size = config.partial_group_sizes[i]
        start = config.start_direction[i]
        max_dir = config.max_partial_direction[i]
        step = config.step_size[i]

        print(f"\nSweep {i + 1}/{n_sweeps}: group_size={group_size}, "
              f"start={start}, max_dir={max_dir}, step={step}")

        window_start = start
        while window_start + group_size <= max_dir and window_start + group_size <= n_dirs_available:
            print(f"  Partialling directions [{window_start}, {window_start + group_size})...")

            # Build per-fold X_partial
            X_partials = [
                partial_out_svd_directions(X, Vt_per_fold[f], window_start, group_size)
                for f in range(n_folds)
            ]

            corrs, _, _, _, _ = train_encoding(
                X=X_partials,
                Y=Y,
                model_constructor_fn=model_constructor,
                epochs_shape=(n_elecs, n_lags),
                provided_idxs=provided_idxs,
            )
            # corrs: (n_folds, n_elecs, n_lags)

            all_results[(group_size, window_start)] = corrs
            print(f"    mean corr = {corrs.mean():.4f}")

            # Free per-fold partials
            del X_partials
            gc.collect()

            window_start += step

    return all_results


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def _save_partial_svd_results(
    partial_svd_dir,
    all_results,
    baseline_corrs,
    sig_mask,
    config,
):
    """Save all partial SVD results to .npz and summary.csv."""

    # Build npz contents
    npz_data = {
        "baseline_corrs": baseline_corrs,
        "sig_mask": sig_mask,
    }

    group_sizes = sorted(set(g for g, _ in all_results.keys()))
    npz_data["group_sizes"] = np.array(group_sizes)

    for gs in group_sizes:
        window_starts = sorted(ws for g, ws in all_results.keys() if g == gs)
        npz_data[f"window_starts_gs{gs}"] = np.array(window_starts)
        for ws in window_starts:
            npz_data[f"corrs_gs{gs}_ws{ws}"] = all_results[(gs, ws)]

    np.savez(os.path.join(partial_svd_dir, "partial_svd_results.npz"), **npz_data)

    # Build summary CSV
    baseline_mean_all = baseline_corrs.mean()
    baseline_mean_sig = baseline_corrs.mean(axis=0)[sig_mask].mean()
    baseline_mean_nonsig = baseline_corrs.mean(axis=0)[~sig_mask].mean()

    rows = []
    for (gs, ws), corrs in sorted(all_results.items()):
        mean_all = corrs.mean()
        mean_sig = corrs.mean(axis=0)[sig_mask].mean()
        mean_nonsig = corrs.mean(axis=0)[~sig_mask].mean()
        rows.append({
            "group_size": gs,
            "window_start": ws,
            "mean_corr_all": mean_all,
            "mean_corr_sig": mean_sig,
            "mean_corr_nonsig": mean_nonsig,
            "drop_sig": baseline_mean_sig - mean_sig,
            "drop_nonsig": baseline_mean_nonsig - mean_nonsig,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(partial_svd_dir, "summary.csv"), index=False)
    print(f"\nSaved summary.csv ({len(rows)} rows)")
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _generate_plots(
    partial_svd_dir,
    all_results,
    baseline_corrs,
    sig_mask,
    metadata_df,
    lags,
    config,
    subject_ids,
):
    """Generate cross-subject, per-subject, per-region, and heatmap plots."""

    group_sizes = sorted(set(g for g, _ in all_results.keys()))

    # -- Cross-subject sweep plot --
    print("\nGenerating cross-subject sweep plot...")
    plot_partial_svd_sweep(
        all_results, baseline_corrs, sig_mask, group_sizes,
        title="Cross-subject: partial SVD sweep",
        save_path=os.path.join(partial_svd_dir, "cross_subject_sweep.png"),
    )

    # -- Per-subject sweep plots --
    for sub_id in subject_ids:
        sub_rows = metadata_df.index[metadata_df["subject"] == sub_id].tolist()
        if not sub_rows:
            continue

        sub_dir = os.path.join(partial_svd_dir, "subjects", f"sub_{sub_id}")
        os.makedirs(sub_dir, exist_ok=True)

        sig_mask_sub = sig_mask[sub_rows]
        baseline_sub = baseline_corrs[:, sub_rows, :]

        sub_results = {
            (gs, ws): corrs[:, sub_rows, :]
            for (gs, ws), corrs in all_results.items()
        }

        print(f"  Generating subject {sub_id} sweep plot...")
        plot_partial_svd_sweep(
            sub_results, baseline_sub, sig_mask_sub, group_sizes,
            title=f"Subject {sub_id}: partial SVD sweep",
            save_path=os.path.join(sub_dir, "sweep.png"),
        )

    # -- Per-region sweep plots (if atlas provided) --
    if config.atlas_region_groups is not None:
        print("\nBuilding electrode region map for plotting...")
        enc = config.encoding
        electrode_region_map = build_electrode_region_map(
            subject_ids, enc.data_root, config.atlas_region_groups, config.atlas_path,
        )

        for region_name, sub_elec_map in electrode_region_map.items():
            # Find metadata rows for this region
            region_rows = []
            for sub_id, elec_list in sub_elec_map.items():
                elec_set = set(elec_list)
                for row_i in metadata_df.index[metadata_df["subject"] == sub_id].tolist():
                    if metadata_df.loc[row_i, "electrode"] in elec_set:
                        region_rows.append(row_i)

            if not region_rows:
                continue

            region_dir = os.path.join(partial_svd_dir, "regions", region_name)
            os.makedirs(region_dir, exist_ok=True)

            sig_mask_region = sig_mask[region_rows]
            baseline_region = baseline_corrs[:, region_rows, :]

            region_results = {
                (gs, ws): corrs[:, region_rows, :]
                for (gs, ws), corrs in all_results.items()
            }

            print(f"  Generating region {region_name} sweep plot...")
            plot_partial_svd_sweep(
                region_results, baseline_region, sig_mask_region, group_sizes,
                title=f"Region {region_name}: partial SVD sweep",
                save_path=os.path.join(region_dir, "sweep.png"),
            )

    # -- Per-lag heatmaps (one per group size, using the first window) --
    if lags is not None:
        for gs in group_sizes:
            windows = sorted(ws for g, ws in all_results.keys() if g == gs)
            if not windows:
                continue
            # Use the first window for the heatmap
            first_ws = windows[0]
            print(f"  Generating lag heatmap for group_size={gs}, window_start={first_ws}...")
            plot_partial_svd_lag_heatmap(
                all_results[(gs, first_ws)], baseline_corrs, sig_mask, lags,
                title=f"Correlation drop: group_size={gs}, directions [{first_ws}, {first_ws + gs})",
                save_path=os.path.join(partial_svd_dir, f"lag_heatmap_gs{gs}.png"),
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args, overrides = parse_args("Partial SVD: direction importance via partialling")
    config: PartialSVDConfig = load_config(args.config, overrides, PartialSVDConfig)
    enc = config.encoding

    # Validate parallel list lengths
    n_sweeps = len(config.partial_group_sizes)
    for name in ("max_partial_direction", "start_direction", "step_size"):
        actual = len(getattr(config, name))
        if actual != n_sweeps:
            raise ValueError(
                f"Length mismatch: partial_group_sizes has {n_sweeps} entries but "
                f"{name} has {actual}. All four lists must be the same length."
            )

    subject_ids = enc.subjects or list(range(1, 10))
    output_dir = config.output_dir or get_results_dir(enc.model, enc.layer, enc.offset)
    partial_svd_dir = config.partial_svd_output_dir or os.path.join(output_dir, "partial_svd")

    os.makedirs(partial_svd_dir, exist_ok=True)
    print(f"Output dir:       {output_dir}")
    print(f"Partial SVD dir:  {partial_svd_dir}")

    # -- Load significance --
    sig_electrodes, sig_df = load_sig_electrodes(output_dir, subject_ids)

    sig_detailed_csv = os.path.join(output_dir, "significance_detailed.csv")
    if not os.path.exists(sig_detailed_csv):
        raise FileNotFoundError(
            f"significance_detailed.csv not found at {sig_detailed_csv}. "
            "Run run_significance.py first."
        )
    sig_detailed_df = pd.read_csv(sig_detailed_csv)

    subs_with_sig = [s for s in subject_ids if s in sig_electrodes]

    # -- Load embeddings --
    print("Loading embeddings...")
    aligned_embeddings, df_word = load_embeddings(enc.data_root, enc.layer)

    # -- Build combined X, Y --
    print("Building combined X, Y from all subjects...")
    X, Y = build_XY_from_subjects(
        subject_ids, enc, sig_electrodes, aligned_embeddings, df_word,
    )
    print(f"  X: {X.shape}, Y: {Y.shape}")

    # -- Load fold indices from first subject's encoding results --
    provided_idxs, lags = load_fold_indices(output_dir, subject_ids, sig_electrodes)
    n_folds = len(provided_idxs)

    # -- Load SVD --
    svd_dir = os.path.join(output_dir, "svd")
    print(f"Loading SVD from {svd_dir}...")
    Vt_per_fold = load_svd_decomposition(svd_dir, n_folds)
    print(f"  Vt shape: {Vt_per_fold.shape}")

    # -- Load baseline correlations --
    print("Loading baseline correlations...")
    baseline_corrs, metadata_df = load_all_subject_correlations(
        output_dir, subjects=subs_with_sig,
    )
    n_elecs = baseline_corrs.shape[1]
    n_lags = baseline_corrs.shape[2]
    print(f"  Baseline corrs: {baseline_corrs.shape}")

    # -- Build significance mask --
    sig_mask = build_global_sig_mask(sig_detailed_df, metadata_df, n_lags)
    print(f"  Significant pairs: {sig_mask.sum()} / {sig_mask.size}")

    # -- Run sweep --
    print("\nStarting partial SVD sweep...")
    all_results = _run_partial_svd_sweep(
        partial_svd_dir, X, Y, Vt_per_fold, provided_idxs,
        n_elecs, n_lags, config,
    )

    # -- Save results --
    _save_partial_svd_results(
        partial_svd_dir, all_results, baseline_corrs, sig_mask, config,
    )

    # -- Generate plots --
    _generate_plots(
        partial_svd_dir, all_results, baseline_corrs, sig_mask,
        metadata_df, lags, config, subs_with_sig,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
