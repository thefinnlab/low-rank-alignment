"""
run_significance.py — Batched per-subject encoding + permutation significance test.

Processes one subject at a time to avoid OOM from concatenating all subjects.
Writes:
  {output_dir}/significance_detailed.csv   — one row per (subject, electrode, lag)
  {output_dir}/all_subject_sig.csv         — electrodes with ≥1 significant lag
"""

import gc
import os

import numpy as np
import pandas as pd
from himalaya.backend import set_backend
import torch

from utils import (
    load_raws,
    train_encoding,
    get_all_predictions_and_targets_test_set,
    significant_lags_all_electrodes_gpu,
    load_embeddings,
    get_ridge_model_constructor,
    build_word_epochs,
    apply_offset,
    get_models_coefficients,
    save_subject_results,
)
from utils.config import SignificanceConfig, get_results_dir
from utils.config_utils import parse_args, load_config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args, overrides = parse_args("Per-subject encoding + significance testing")
    config: SignificanceConfig = load_config(args.config, overrides, SignificanceConfig)
    enc = config.encoding

    subject_ids = enc.subjects or list(range(1, 10))
    output_dir = config.output_dir or get_results_dir(enc.model, enc.layer, enc.offset)

    # Use GPU backend if available
    if torch.cuda.is_available():
        set_backend("torch_cuda")
        print("Using torch_cuda backend.")
    else:
        print("CUDA not available, using CPU.")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Results directory: {output_dir}")

    # Load embeddings once — shared across all subjects
    aligned_embeddings, df_word = load_embeddings(enc.data_root, enc.layer)

    # Lag axis: epochs are built with tmin=-2, tmax=2 then resampled to 32 Hz
    # That gives 4 s * 32 Hz = 128 samples but MNE epoch layout is:
    #   original sfreq=512 → np.arange(-2*512, 2*512, 1) / 512, then every 16th sample
    lag_sec = np.arange(-2 * 512, 2 * 512, 16) / 512  # length 64

    model_constructor = get_ridge_model_constructor()

    all_rows = []

    for bids_sub_id in subject_ids:
        print(f"\n{'='*60}")
        print(f"Processing subject {bids_sub_id}")
        print(f"{'='*60}")

        # ------------------------------------------------------------------
        # a. Load raw data
        # ------------------------------------------------------------------
        raws = load_raws([bids_sub_id], enc.data_root, use_high_gamma=True)
        raw = raws[0]
        electrode_names = raw.ch_names
        print(f"  Electrodes: {len(electrode_names)}")

        # ------------------------------------------------------------------
        # b. Build epochs
        # ------------------------------------------------------------------
        epochs, data = build_word_epochs(raw, df_word)
        print(f"  Epoch data shape: {data.shape}")

        # ------------------------------------------------------------------
        # c. Align X and Y by epoch selection and word offset
        # ------------------------------------------------------------------
        n_words, n_elec, n_lags = data.shape
        X_arr, Y = apply_offset(aligned_embeddings[epochs.selection], data, enc.offset)
        X = [X_arr, X_arr]  # one per fold

        # Update lag_sec based on actual n_lags (in case resampling gives different count)
        actual_lag_sec = np.linspace(-2.0, 2.0, n_lags, endpoint=False)

        # ------------------------------------------------------------------
        # e. Train encoding model
        # ------------------------------------------------------------------
        print("  Training encoding model...")
        corrs, models, tr_idxs, test_idxs, scalers = train_encoding(
            X,
            Y,
            model_constructor,
            epochs_shape=(n_elec, n_lags),
        )

        # ------------------------------------------------------------------
        # f. Get predictions
        # ------------------------------------------------------------------
        print("  Computing predictions...")
        all_predictions, all_targets = get_all_predictions_and_targets_test_set(
            models, test_idxs, X, Y, n_elec, n_lags
        )

        # ------------------------------------------------------------------
        # g. Significance test
        # ------------------------------------------------------------------
        print(f"  Running significance test ({config.n_permutations} permutations)...")
        sig, p_values, q_values, obs_corrs = significant_lags_all_electrodes_gpu(
            all_predictions.transpose(1, 2, 0),
            all_targets.transpose(1, 2, 0),
            n_permutations=config.n_permutations,
            fdr_q=config.fdr_q,
        )
        # shapes: (n_elec, n_lags)

        # ------------------------------------------------------------------
        # h. Optionally save weights for significant electrodes
        # ------------------------------------------------------------------
        if config.save_weights:
            sig_elec_indices = np.where(sig.any(axis=1))[0]
            if len(sig_elec_indices) > 0:
                coefficients, intercepts = get_models_coefficients(
                    models, n_elec, n_lags, scaled_model=False
                )
                sig_coefs = [c[sig_elec_indices] for c in coefficients]
                sig_ints = [i.reshape(n_elec, n_lags)[sig_elec_indices].ravel() for i in intercepts]
                sig_corrs = corrs[:, sig_elec_indices, :]
                sig_names = [electrode_names[i] for i in sig_elec_indices]
                save_subject_results(
                    output_dir, bids_sub_id, sig_corrs,
                    sig_coefs, sig_ints, models, tr_idxs, test_idxs,
                    scalers, sig_names, enc.offset, enc.layer,
                    actual_lag_sec,
                )
                del coefficients, intercepts

        # ------------------------------------------------------------------
        # i. Collect rows
        # ------------------------------------------------------------------
        for elec_idx, elec_name in enumerate(electrode_names):
            for lag_idx in range(n_lags):
                all_rows.append(
                    {
                        "subject": bids_sub_id,
                        "electrode": elec_name,
                        "lag_idx": lag_idx,
                        "lag_sec": float(actual_lag_sec[lag_idx]),
                        "sig": bool(sig[elec_idx, lag_idx]),
                        "p_value": float(p_values[elec_idx, lag_idx]),
                        "q_value": float(q_values[elec_idx, lag_idx]),
                        "obs_corr": float(obs_corrs[elec_idx, lag_idx]),
                    }
                )

        # ------------------------------------------------------------------
        # j. Free memory
        # ------------------------------------------------------------------
        del raws, raw, epochs, data, X, Y, models, scalers
        del all_predictions, all_targets, sig, p_values, q_values, obs_corrs, corrs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4. Build DataFrames
    # ------------------------------------------------------------------
    df_detailed = pd.DataFrame(all_rows)

    # ------------------------------------------------------------------
    # 5. Write output files
    # ------------------------------------------------------------------
    detailed_path = os.path.join(output_dir, "significance_detailed.csv")
    df_detailed.to_csv(detailed_path, index=False)
    print(f"\nWrote {detailed_path}  ({len(df_detailed):,} rows)")

    # all_subject_sig.csv — one row per (subject, electrode) that has ≥1 sig lag
    df_any_sig = (
        df_detailed.groupby(["subject", "electrode"])["sig"]
        .any()
        .reset_index()
        .query("sig == True")[["subject", "electrode"]]
        .rename(columns={"electrode": "elec"})
    )
    sig_path = os.path.join(output_dir, "all_subject_sig.csv")
    df_any_sig.to_csv(sig_path, index=False)
    print(f"Wrote {sig_path}  ({len(df_any_sig):,} rows)")


if __name__ == "__main__":
    main()
