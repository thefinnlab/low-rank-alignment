"""
run_partial_rrr.py — Partial RRR subspace importance analysis.

Loads RRR-learned subspaces (A matrices) for each rank k, partials them out
of word embeddings, and retrains ridge regression to measure how much each
rank-k subspace contributes to encoding performance.

Three modes controlled by PartialRRRConfig.mode:

  cross_subject (default):
    Builds global Y across all subjects (significant electrodes).
    Reads RRR from {output_dir}/rrr/.
    Writes to {output_dir}/partial_rrr/.

  per_subject:
    Runs independently per subject.
    Reads RRR from {output_dir}/per_subject_rrr/subjects/sub_{id}/.
    Writes to {output_dir}/per_subject_partial_rrr/subjects/sub_{id}/.

  per_region:
    Runs independently per atlas region.
    Reads RRR from {output_dir}/per_region_rrr/regions/{region}/.
    Writes to {output_dir}/per_region_partial_rrr/regions/{region}/.

Output per unit:
    {partial_rrr_dir}/
        partial_rrr_results.npz   — full per-electrode/per-lag correlation matrices
        summary.csv               — tabular summary of mean correlations per rank k
        sweep.png                 — correlation vs rank k plot
"""

import gc
import os
import time

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import (
    build_global_sig_mask,
    load_embeddings,
    build_XY_from_subjects,
    load_all_subject_correlations,
    train_encoding,
    load_raws,
    build_word_epochs,
    apply_offset,
)
from utils.analysis import partial_out_subspace
from utils.atlas import build_electrode_region_map
from utils.config import PartialRRRConfig, SVDMode, get_results_dir
from utils.config_utils import parse_args, load_config
from utils.encoding import get_ridge_model_constructor
from utils.data_loaders import load_sig_electrodes, load_fold_indices
from utils.encoding import load_encoding_results
from utils.plotting import plot_partial_rrr_sweep
from utils.srrr import load_rrr_result


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------


def _discover_ranks(rrr_source_dir, rank_grid=None):
    """Discover available ranks from an RRR output directory.

    Scans for subdirectories matching the ``k{rank}`` naming convention and
    returns them sorted.  When *rank_grid* is provided, the discovered ranks
    are filtered to only those present in *rank_grid*.

    Args:
        rrr_source_dir: Root directory containing ``k{rank}/`` subdirs.
        rank_grid: Optional list of ranks to keep.  ``None`` keeps all.

    Returns:
        Sorted list of integer rank values.

    Raises:
        FileNotFoundError: If *rrr_source_dir* does not exist or contains
            no ``k*`` subdirectories.
    """
    if not os.path.isdir(rrr_source_dir):
        raise FileNotFoundError(f"RRR source directory not found: {rrr_source_dir}")

    discovered = []
    for name in os.listdir(rrr_source_dir):
        if name.startswith("k") and os.path.isdir(os.path.join(rrr_source_dir, name)):
            try:
                discovered.append(int(name[1:]))
            except ValueError:
                continue

    if not discovered:
        raise FileNotFoundError(f"No k* rank directories found in {rrr_source_dir}")

    discovered.sort()

    if rank_grid is not None:
        discovered = [k for k in discovered if k in rank_grid]
        if not discovered:
            raise ValueError(
                f"None of the requested ranks {rank_grid} were found in {rrr_source_dir}"
            )

    return discovered


def _run_partial_rrr_sweep(
    partial_rrr_dir,
    X,
    Y,
    rrr_source_dir,
    provided_idxs,
    n_elecs,
    n_lags,
    rank_grid=None,
    alphas=None,
):
    """Core loop: for each rank k, load A, partial out subspace, retrain ridge.

    Args:
        partial_rrr_dir: Output directory.
        X: (n_words, n_features) embedding matrix.
        Y: (n_words, n_elecs * n_lags) neural data.
        rrr_source_dir: Directory containing RRR results (k{k}/fold_{f}/).
        provided_idxs: list of (train_idx, test_idx) tuples.
        n_elecs: number of electrodes.
        n_lags: number of time lags.
        rank_grid: Optional list of rank values to filter to. None discovers all.

    Returns:
        dict mapping k (int) → (n_folds, n_elecs, n_lags) correlations.
    """
    rank_grid = _discover_ranks(rrr_source_dir, rank_grid)
    n_folds = len(provided_idxs)
    model_constructor = get_ridge_model_constructor(alphas=alphas)

    all_results = {}
    sweep_start = time.time()

    print(f"  RRR source:  {rrr_source_dir}")
    print(f"  Rank grid:   {rank_grid} ({len(rank_grid)} ranks)")
    print(f"  Folds:       {n_folds}")
    print(f"  Electrodes:  {n_elecs}, Lags: {n_lags}")

    for k in tqdm(rank_grid, desc="  Partial RRR sweep", unit="rank"):
        k_start = time.time()

        # Build per-fold X_partial
        X_partials = []
        for f in range(n_folds):
            rrr_data = load_rrr_result(rrr_source_dir, k, f)
            A = rrr_data["A"]  # (n_embed, k)
            X_partials.append(partial_out_subspace(X, A.T))

        corrs, _, _, _, _ = train_encoding(
            X=X_partials,
            Y=Y,
            model_constructor_fn=model_constructor,
            epochs_shape=(n_elecs, n_lags),
            provided_idxs=provided_idxs,
        )
        # corrs: (n_folds, n_elecs, n_lags)

        all_results[k] = corrs
        k_elapsed = time.time() - k_start
        tqdm.write(f"    k={k:>3d}  mean_corr={corrs.mean():.4f}  ({k_elapsed:.1f}s)")

        del X_partials
        gc.collect()

    total_elapsed = time.time() - sweep_start
    print(f"\n  Sweep complete: {len(rank_grid)} ranks in {total_elapsed:.1f}s")

    return all_results


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def _save_partial_rrr_results(
    partial_rrr_dir,
    all_results,
    baseline_corrs,
    sig_mask,
):
    """Save all partial RRR results to .npz and summary.csv."""

    npz_data = {
        "baseline_corrs": baseline_corrs,
        "sig_mask": sig_mask,
        "rank_grid": np.array(sorted(all_results.keys())),
    }
    for k, corrs in all_results.items():
        npz_data[f"corrs_k{k}"] = corrs

    np.savez(os.path.join(partial_rrr_dir, "partial_rrr_results.npz"), **npz_data)

    # Build summary CSV
    baseline_mean_all = baseline_corrs.mean()
    baseline_mean_sig = baseline_corrs.mean(axis=0)[sig_mask].mean()
    baseline_mean_nonsig = baseline_corrs.mean(axis=0)[~sig_mask].mean()

    rows = []
    for k in sorted(all_results.keys()):
        corrs = all_results[k]
        mean_all = corrs.mean()
        mean_sig = corrs.mean(axis=0)[sig_mask].mean()
        mean_nonsig = corrs.mean(axis=0)[~sig_mask].mean()
        rows.append(
            {
                "rank_k": k,
                "mean_corr_all": mean_all,
                "mean_corr_sig": mean_sig,
                "mean_corr_nonsig": mean_nonsig,
                "drop_sig": baseline_mean_sig - mean_sig,
                "drop_nonsig": baseline_mean_nonsig - mean_nonsig,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(partial_rrr_dir, "summary.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nSaved partial_rrr_results.npz and summary.csv ({len(rows)} rows)")
    print(f"  Baseline mean corr (sig):     {baseline_mean_sig:.4f}")
    print(f"  Baseline mean corr (non-sig): {baseline_mean_nonsig:.4f}")
    print(
        f"\n  {'k':>5s}  {'corr_sig':>10s}  {'corr_nonsig':>12s}  {'drop_sig':>10s}  {'drop_nonsig':>12s}"
    )
    print(f"  {'-'*55}")
    for _, row in df.iterrows():
        print(
            f"  {int(row['rank_k']):>5d}  {row['mean_corr_sig']:>10.4f}  {row['mean_corr_nonsig']:>12.4f}"
            f"  {row['drop_sig']:>+10.4f}  {row['drop_nonsig']:>+12.4f}"
        )

    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _generate_plots(
    partial_rrr_dir,
    all_results,
    baseline_corrs,
    sig_mask,
    metadata_df,
    config,
    subject_ids,
):
    """Generate sweep plots (cross-subject, per-subject, per-region)."""

    # -- Main sweep plot --
    print("\nGenerating sweep plot...")
    plot_partial_rrr_sweep(
        all_results,
        baseline_corrs,
        sig_mask,
        title="Partial RRR sweep",
        save_path=os.path.join(partial_rrr_dir, "sweep.png"),
    )

    # -- Per-subject sweep plots --
    for sub_id in subject_ids:
        sub_rows = metadata_df.index[metadata_df["subject"] == sub_id].tolist()
        if not sub_rows:
            continue

        sub_dir = os.path.join(partial_rrr_dir, "subjects", f"sub_{sub_id}")
        os.makedirs(sub_dir, exist_ok=True)

        sig_mask_sub = sig_mask[sub_rows]
        baseline_sub = baseline_corrs[:, sub_rows, :]

        sub_results = {k: corrs[:, sub_rows, :] for k, corrs in all_results.items()}

        print(f"  Generating subject {sub_id} sweep plot...")
        plot_partial_rrr_sweep(
            sub_results,
            baseline_sub,
            sig_mask_sub,
            title=f"Subject {sub_id}: partial RRR sweep",
            save_path=os.path.join(sub_dir, "sweep.png"),
        )

    # -- Per-region sweep plots (if atlas provided) --
    if config.atlas_region_groups is not None:
        print("\nBuilding electrode region map for plotting...")
        enc = config.encoding
        electrode_region_map = build_electrode_region_map(
            subject_ids,
            enc.data_root,
            config.atlas_region_groups,
            config.atlas_path,
        )

        for region_name, sub_elec_map in electrode_region_map.items():
            region_rows = []
            for sub_id, elec_list in sub_elec_map.items():
                elec_set = set(elec_list)
                for row_i in metadata_df.index[
                    metadata_df["subject"] == sub_id
                ].tolist():
                    if metadata_df.loc[row_i, "electrode"] in elec_set:
                        region_rows.append(row_i)

            if not region_rows:
                continue

            region_dir = os.path.join(partial_rrr_dir, "regions", region_name)
            os.makedirs(region_dir, exist_ok=True)

            sig_mask_region = sig_mask[region_rows]
            baseline_region = baseline_corrs[:, region_rows, :]

            region_results = {
                k: corrs[:, region_rows, :] for k, corrs in all_results.items()
            }

            print(f"  Generating region {region_name} sweep plot...")
            plot_partial_rrr_sweep(
                region_results,
                baseline_region,
                sig_mask_region,
                title=f"Region {region_name}: partial RRR sweep",
                save_path=os.path.join(region_dir, "sweep.png"),
            )


# ---------------------------------------------------------------------------
# Branch: cross-subject
# ---------------------------------------------------------------------------


def _run_cross_subject(
    config,
    enc,
    output_dir,
    partial_rrr_dir,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    df_word,
):
    """Partial RRR using all subjects' significant electrodes."""
    subs_with_sig = [
        s for s in subject_ids if s in sig_electrodes and sig_electrodes[s]
    ]
    if not subs_with_sig:
        print("  No significant electrodes found; skipping.")
        return

    print(f"\nSubjects with significant electrodes: {subs_with_sig}")
    print("Building combined X, Y from all subjects...")
    X, Y = build_XY_from_subjects(
        subject_ids,
        enc,
        sig_electrodes,
        aligned_embeddings,
        df_word,
    )
    print(f"  X: {X.shape}, Y: {Y.shape}")

    print("Loading fold indices...")
    provided_idxs, lags = load_fold_indices(output_dir, subject_ids, sig_electrodes)

    # Load baseline correlations
    print("Loading baseline correlations...")
    baseline_corrs, metadata_df = load_all_subject_correlations(
        output_dir,
        subjects=subs_with_sig,
    )
    n_elecs = baseline_corrs.shape[1]
    n_lags = baseline_corrs.shape[2]
    print(f"  Baseline corrs: {baseline_corrs.shape}")

    # Build significance mask
    sig_detailed_csv = os.path.join(output_dir, "significance_detailed.csv")
    sig_detailed_df = pd.read_csv(sig_detailed_csv)
    sig_mask = build_global_sig_mask(sig_detailed_df, metadata_df, n_lags)
    print(f"  Significant pairs: {sig_mask.sum()} / {sig_mask.size}")

    # Resolve RRR source dir
    rrr_source_dir = config.rrr_source_dir or os.path.join(output_dir, "rrr")

    # Run sweep
    print("\nStarting partial RRR sweep...")
    alphas = np.array(config.encoding.ridge_alphas) if config.encoding.ridge_alphas is not None else None
    all_results = _run_partial_rrr_sweep(
        partial_rrr_dir,
        X,
        Y,
        rrr_source_dir,
        provided_idxs,
        n_elecs,
        n_lags,
        config.rank_grid,
        alphas=alphas,
    )

    # Save results
    print("\nSaving results...")
    _save_partial_rrr_results(
        partial_rrr_dir,
        all_results,
        baseline_corrs,
        sig_mask,
    )

    # Generate plots
    print("\nGenerating plots...")
    _generate_plots(
        partial_rrr_dir,
        all_results,
        baseline_corrs,
        sig_mask,
        metadata_df,
        config,
        subs_with_sig,
    )


# ---------------------------------------------------------------------------
# Branch: per-subject
# ---------------------------------------------------------------------------


def _run_per_subject(
    config,
    enc,
    output_dir,
    partial_rrr_dir,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    df_word,
):
    """Partial RRR independently per subject."""
    n_subjects = len(subject_ids)
    for si, sub_id in enumerate(subject_ids):
        print(f"\n{'='*60}")
        print(f"Subject {sub_id} ({si + 1}/{n_subjects})")
        print(f"{'='*60}")

        sub_sig_elecs = sig_electrodes.get(sub_id, [])
        if not sub_sig_elecs:
            print(f"  No significant electrodes; skipping.")
            continue

        print(f"  Significant electrodes: {len(sub_sig_elecs)}")
        sub_out_dir = os.path.join(partial_rrr_dir, "subjects", f"sub_{sub_id}")
        os.makedirs(sub_out_dir, exist_ok=True)

        # Load data for this subject
        print("  Loading neural data...")
        enc_results = load_encoding_results(
            os.path.join(
                output_dir, "subjects", f"sub_{sub_id}", "encoding_results.joblib"
            )
        )
        test_idxs = enc_results["test_idxs"]

        raws = load_raws(
            [sub_id],
            enc.data_root,
            use_high_gamma=True,
            per_subject_electrodes={sub_id: sub_sig_elecs},
        )
        raw = raws[0]
        epochs, data = build_word_epochs(raw, df_word)
        X_arr, Y_sub = apply_offset(
            aligned_embeddings[epochs.selection], data, enc.offset
        )
        del raws, raw, epochs, data
        gc.collect()

        Y_sub = Y_sub.astype(np.float32)
        n_elecs_sub = len(sub_sig_elecs)
        n_lags = Y_sub.shape[1] // n_elecs_sub
        print(f"  X: {X_arr.shape}, Y: {Y_sub.shape}")

        # Build provided_idxs
        n_words = X_arr.shape[0]
        provided_idxs = []
        for test_idx in test_idxs:
            train_mask = np.ones(n_words, dtype=bool)
            train_mask[test_idx] = False
            train_idx = np.where(train_mask)[0]
            provided_idxs.append((train_idx, test_idx))

        # Load baseline correlations for this subject
        print("  Loading baseline correlations...")
        baseline_corrs, metadata_df = load_all_subject_correlations(
            output_dir,
            subjects=[sub_id],
        )
        print(f"  Baseline corrs: {baseline_corrs.shape}")

        # Build significance mask
        sig_detailed_csv = os.path.join(output_dir, "significance_detailed.csv")
        sig_detailed_df = pd.read_csv(sig_detailed_csv)
        sig_mask = build_global_sig_mask(sig_detailed_df, metadata_df, n_lags)
        print(f"  Significant pairs: {sig_mask.sum()} / {sig_mask.size}")

        # Resolve RRR source dir
        rrr_source_dir = config.rrr_source_dir
        if rrr_source_dir:
            rrr_source_dir = os.path.join(rrr_source_dir, "subjects", f"sub_{sub_id}")
        else:
            rrr_source_dir = os.path.join(
                output_dir, "per_subject_rrr", "subjects", f"sub_{sub_id}"
            )

        # Run sweep
        print("\n  Starting partial RRR sweep...")
        sub_start = time.time()
        alphas = np.array(config.encoding.ridge_alphas) if config.encoding.ridge_alphas is not None else None
        all_results = _run_partial_rrr_sweep(
            sub_out_dir,
            X_arr,
            Y_sub,
            rrr_source_dir,
            provided_idxs,
            n_elecs_sub,
            n_lags,
            config.rank_grid,
            alphas=alphas,
        )

        _save_partial_rrr_results(
            sub_out_dir,
            all_results,
            baseline_corrs,
            sig_mask,
        )

        # Simple sweep plot (no sub-breakdowns for per-subject mode)
        plot_partial_rrr_sweep(
            all_results,
            baseline_corrs,
            sig_mask,
            title=f"Subject {sub_id}: partial RRR sweep",
            save_path=os.path.join(sub_out_dir, "sweep.png"),
        )
        print(f"  Subject {sub_id} complete in {time.time() - sub_start:.1f}s")


# ---------------------------------------------------------------------------
# Branch: per-region
# ---------------------------------------------------------------------------


def _run_per_region(
    config,
    enc,
    output_dir,
    partial_rrr_dir,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    df_word,
):
    """Partial RRR independently per atlas region."""
    print("Loading fold indices...")
    provided_idxs, lags = load_fold_indices(output_dir, subject_ids, sig_electrodes)

    print("Building electrode region map...")
    electrode_region_map = build_electrode_region_map(
        subject_ids,
        enc.data_root,
        getattr(config, "atlas_region_groups", None),
        getattr(config, "atlas_path", None),
    )
    region_names = list(electrode_region_map.keys())
    n_regions = len(region_names)
    print(f"  Regions found ({n_regions}): {region_names}")

    for ri, (region_name, sub_elec_map) in enumerate(electrode_region_map.items()):
        print(f"\n{'='*60}")
        print(f"Region: {region_name} ({ri + 1}/{n_regions})")
        print(f"{'='*60}")

        region_out_dir = os.path.join(partial_rrr_dir, "regions", region_name)

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

        total_elecs = sum(len(e) for e in per_subject_region_elecs.values())
        print(
            f"  Subjects: {list(per_subject_region_elecs.keys())}, "
            f"Electrodes: {total_elecs}"
        )

        os.makedirs(region_out_dir, exist_ok=True)

        print("  Building X, Y...")
        subs_in_region = list(per_subject_region_elecs.keys())
        X, Y = build_XY_from_subjects(
            subs_in_region,
            enc,
            per_subject_region_elecs,
            aligned_embeddings,
            df_word,
        )
        print(f"  X: {X.shape}, Y: {Y.shape}")

        # Load baseline correlations for this region's subjects/electrodes
        print("  Loading baseline correlations...")
        baseline_corrs, metadata_df = load_all_subject_correlations(
            output_dir,
            subjects=subs_in_region,
        )

        # Filter to only this region's electrodes
        region_rows = []
        for sub_id, elec_list in per_subject_region_elecs.items():
            elec_set = set(elec_list)
            for row_i in metadata_df.index[metadata_df["subject"] == sub_id].tolist():
                if metadata_df.loc[row_i, "electrode"] in elec_set:
                    region_rows.append(row_i)

        n_region_elecs = len(region_rows)
        n_lags = baseline_corrs.shape[2]

        # Build sig mask for region electrodes
        sig_detailed_csv = os.path.join(output_dir, "significance_detailed.csv")
        sig_detailed_df = pd.read_csv(sig_detailed_csv)

        n_elecs_region = Y.shape[1] // n_lags

        sig_mask = build_global_sig_mask(sig_detailed_df, metadata_df, n_lags)
        sig_mask_region = sig_mask[region_rows]
        print(f"  Significant pairs: {sig_mask_region.sum()} / {sig_mask_region.size}")

        # Baseline corrs filtered to region
        baseline_region = baseline_corrs[:, region_rows, :]

        # Resolve RRR source dir
        rrr_source_dir = config.rrr_source_dir
        if rrr_source_dir:
            rrr_source_dir = os.path.join(rrr_source_dir, "regions", region_name)
        else:
            rrr_source_dir = os.path.join(
                output_dir, "per_region_rrr", "regions", region_name
            )

        # Run sweep
        print("\n  Starting partial RRR sweep...")
        region_start = time.time()
        alphas = np.array(config.encoding.ridge_alphas) if config.encoding.ridge_alphas is not None else None
        all_results = _run_partial_rrr_sweep(
            region_out_dir,
            X,
            Y,
            rrr_source_dir,
            provided_idxs,
            n_elecs_region,
            n_lags,
            config.rank_grid,
            alphas=alphas,
        )

        _save_partial_rrr_results(
            region_out_dir,
            all_results,
            baseline_region,
            sig_mask_region,
        )

        plot_partial_rrr_sweep(
            all_results,
            baseline_region,
            sig_mask_region,
            title=f"Region {region_name}: partial RRR sweep",
            save_path=os.path.join(region_out_dir, "sweep.png"),
        )
        print(f"  Region {region_name} complete in {time.time() - region_start:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args, overrides = parse_args("Partial RRR: subspace importance via partialling")
    config: PartialRRRConfig = load_config(args.config, overrides, PartialRRRConfig)
    enc = config.encoding

    subject_ids = enc.subjects or list(range(1, 10))
    output_dir = config.output_dir or get_results_dir(enc.model, enc.layer, enc.offset)

    # Resolve output directory based on mode
    if config.partial_rrr_output_dir:
        partial_rrr_dir = config.partial_rrr_output_dir
    elif config.mode == SVDMode.PER_REGION:
        partial_rrr_dir = os.path.join(output_dir, "per_region_partial_rrr")
    elif config.mode == SVDMode.PER_SUBJECT:
        partial_rrr_dir = os.path.join(output_dir, "per_subject_partial_rrr")
    else:
        partial_rrr_dir = os.path.join(output_dir, "partial_rrr")

    os.makedirs(partial_rrr_dir, exist_ok=True)
    print(f"Output dir:         {output_dir}")
    print(f"Partial RRR dir:    {partial_rrr_dir}")
    print(f"Mode:               {config.mode}")
    print(f"Rank grid:          {config.rank_grid or 'auto-discover'}")
    print(f"Subjects:           {subject_ids}")

    # Load significance
    print("\nLoading significance data...")
    sig_electrodes, _ = load_sig_electrodes(output_dir, subject_ids)
    total_sig = sum(len(e) for e in sig_electrodes.values())
    print(f"  {total_sig} significant electrodes across {len(sig_electrodes)} subjects")

    # Load embeddings
    print("\nLoading embeddings...")
    aligned_embeddings, df_word = load_embeddings(enc.data_root, enc.layer)
    print(f"  Embeddings: {aligned_embeddings.shape}")

    branch_kwargs = dict(
        config=config,
        enc=enc,
        output_dir=output_dir,
        partial_rrr_dir=partial_rrr_dir,
        subject_ids=subject_ids,
        sig_electrodes=sig_electrodes,
        aligned_embeddings=aligned_embeddings,
        df_word=df_word,
    )

    if config.mode == SVDMode.PER_REGION:
        _run_per_region(**branch_kwargs)
    elif config.mode == SVDMode.PER_SUBJECT:
        _run_per_subject(**branch_kwargs)
    else:
        _run_cross_subject(**branch_kwargs)

    print("\nDone.")


if __name__ == "__main__":
    main()
