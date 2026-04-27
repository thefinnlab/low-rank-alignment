"""
compute_significance.py — Determine significant electrodes from a phase-randomization null distribution.

Reads null distribution shards produced by generate_null_distribution.py,
re-runs encoding on real (non-permuted) data to obtain observed per-electrode
max-across-lags correlations, and applies the max-statistic significance
procedure with FDR correction.

Procedure (per subject):
    1. Re-run encoding on real data to get observed correlations.
    2. For each electrode, compute max correlation across all lags.
    3. Load the null distribution (max across electrodes per permutation).
    4. P-value = fraction of null values >= electrode's observed max.
    5. FDR correction (Benjamini-Hochberg) across all electrodes (all subjects).

Writes:
  {output_dir}/all_subject_sig.csv         — electrodes with q < fdr_q
  {output_dir}/significance_detailed.csv   — one row per (subject, electrode, lag)
"""

import gc
import glob
import os

import numpy as np
import pandas as pd
import torch
from statsmodels.stats.multitest import multipletests

from src.utils import (
    load_embeddings,
    apply_offset,
    train_encoding,
    get_all_predictions_and_targets_test_set,
    load_precomputed_epochs,
    epochs_from_cache,
)
from src.core.config import NewSignificanceConfig, get_results_dir
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.config_utils import parse_args, load_config
from src.utils.encoding import get_ridge_model_constructor
from src.utils.significance import compute_observed_max_per_electrode


def main():
    args, overrides = parse_args("Compute significance from null distribution")
    config: NewSignificanceConfig = load_config(
        args.config, overrides, NewSignificanceConfig
    )
    enc = config.encoding

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
    null_dir = config.null_dir or os.path.join(output_dir, "null_dist")

    # Use GPU backend if available
    if torch.cuda.is_available():
        from himalaya.backend import set_backend

        set_backend("torch_cuda")
        print("Using torch_cuda backend.")
    else:
        print("CUDA not available, using CPU.")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")
    print(f"Null dir:   {null_dir}")

    # Load embeddings once
    aligned_embeddings, df_word = load_embeddings(
        enc.data_root,
        enc.model,
        dataset=enc.dataset,
        per_dataset_config=enc.per_dataset_config,
        dataset_context=dataset_context,
        **enc.per_model_config,
    )

    alphas = np.array(enc.ridge_alphas) if enc.ridge_alphas is not None else None
    model_constructor = get_ridge_model_constructor(alphas=alphas)

    # Collect per-electrode results across all subjects
    all_electrode_records = (
        []
    )  # (subject, electrode, obs_max, p_value, obs_corrs_per_lag)

    for bids_sub_id in subject_ids:
        print(f"\n{'='*60}")
        print(f"Subject {bids_sub_id}")
        print(f"{'='*60}")

        # -------------------------------------------------------------------
        # 1. Load null distribution for this subject
        # -------------------------------------------------------------------
        sub_null_dir = os.path.join(null_dir, f"sub_{bids_sub_id}")
        shard_files = sorted(glob.glob(os.path.join(sub_null_dir, "*.npy")))
        if not shard_files:
            print(
                f"  WARNING: No null distribution shards found in {sub_null_dir}. Skipping."
            )
            continue

        null_values = np.concatenate([np.load(f) for f in shard_files])
        n_permutations = len(null_values)
        print(
            f"  Loaded {n_permutations} null permutations from {len(shard_files)} shard(s)"
        )

        if n_permutations < config.min_permutations:
            print(
                f"  WARNING: Only {n_permutations} permutations found, "
                f"recommended minimum is {config.min_permutations}."
            )

        # -------------------------------------------------------------------
        # 2. Re-run encoding on real (non-permuted) data
        # -------------------------------------------------------------------
        cache = load_precomputed_epochs(
            bids_sub_id,
            enc.data_root,
            dataset=enc.dataset,
            per_dataset_config=enc.per_dataset_config,
            dataset_context=dataset_context,
        )
        electrode_names = cache["ch_names"]
        n_elec = len(electrode_names)

        data = epochs_from_cache(cache)
        n_words, _, n_lags = data.shape

        X_arr, Y = apply_offset(
            aligned_embeddings[cache["selection"]], data, enc.offset
        )
        X = [X_arr] * enc.num_folds

        actual_lag_sec = np.linspace(-2.0, 2.0, n_lags, endpoint=False)

        print("  Training encoding model on real data...")
        corrs, models, tr_idxs, test_idxs, scalers = train_encoding(
            X,
            Y,
            model_constructor,
            epochs_shape=(n_elec, n_lags),
        )

        print("  Computing predictions...")
        all_predictions, all_targets = get_all_predictions_and_targets_test_set(
            models, test_idxs, X, Y, n_elec, n_lags
        )

        # -------------------------------------------------------------------
        # 3. Compute observed max-across-lags per electrode
        # -------------------------------------------------------------------
        max_per_electrode, obs_corrs = compute_observed_max_per_electrode(
            all_predictions, all_targets
        )
        # obs_corrs: (n_elec, n_lags)
        # max_per_electrode: (n_elec,)

        # -------------------------------------------------------------------
        # 4. Compute p-values using the null distribution
        # -------------------------------------------------------------------
        for elec_idx, elec_name in enumerate(electrode_names):
            obs_max = max_per_electrode[elec_idx]
            p_value = (null_values >= obs_max).mean()
            all_electrode_records.append(
                {
                    "subject": bids_sub_id,
                    "electrode": elec_name,
                    "obs_max": float(obs_max),
                    "p_value": float(p_value),
                    "obs_corrs_per_lag": obs_corrs[elec_idx],  # (n_lags,)
                    "n_lags": n_lags,
                    "lag_sec": actual_lag_sec,
                }
            )

        # Free memory
        del cache, data, X_arr, Y, X
        del corrs, models, tr_idxs, test_idxs, scalers
        del all_predictions, all_targets, max_per_electrode, obs_corrs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 5. FDR correction across all electrodes (all subjects)
    # -----------------------------------------------------------------------
    if not all_electrode_records:
        print(
            "\nERROR: No subjects had null distribution data. Cannot compute significance."
        )
        return

    p_values = np.array([r["p_value"] for r in all_electrode_records])
    reject, q_values, _, _ = multipletests(
        p_values, alpha=config.fdr_q, method="fdr_bh"
    )

    for i, rec in enumerate(all_electrode_records):
        rec["sig"] = bool(reject[i])
        rec["q_value"] = float(q_values[i])

    n_sig = sum(r["sig"] for r in all_electrode_records)
    print(f"\n{'='*60}")
    print(
        f"Significant electrodes: {n_sig} / {len(all_electrode_records)} (FDR q < {config.fdr_q})"
    )
    print(f"{'='*60}")

    # -----------------------------------------------------------------------
    # 6. Write output files
    # -----------------------------------------------------------------------

    # significance_detailed.csv — one row per (subject, electrode, lag_idx)
    # For significant electrodes: all lags marked sig=True
    detailed_rows = []
    for rec in all_electrode_records:
        for lag_idx in range(rec["n_lags"]):
            detailed_rows.append(
                {
                    "subject": rec["subject"],
                    "electrode": rec["electrode"],
                    "lag_idx": lag_idx,
                    "lag_sec": float(rec["lag_sec"][lag_idx]),
                    "sig": rec["sig"],
                    "p_value": rec["p_value"],
                    "q_value": rec["q_value"],
                    "obs_corr": float(rec["obs_corrs_per_lag"][lag_idx]),
                }
            )

    df_detailed = pd.DataFrame(detailed_rows)
    detailed_path = os.path.join(output_dir, "significance_detailed.csv")
    df_detailed.to_csv(detailed_path, index=False)
    print(f"Wrote {detailed_path}  ({len(df_detailed):,} rows)")

    # all_subject_sig.csv — one row per significant (subject, electrode)
    sig_rows = [
        {"subject": r["subject"], "elec": r["electrode"]}
        for r in all_electrode_records
        if r["sig"]
    ]
    df_sig = pd.DataFrame(sig_rows, columns=["subject", "elec"])
    sig_path = os.path.join(output_dir, "all_subject_sig.csv")
    df_sig.to_csv(sig_path, index=False)
    print(f"Wrote {sig_path}  ({len(df_sig):,} rows)")


if __name__ == "__main__":
    main()
