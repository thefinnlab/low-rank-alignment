"""
run_significance.py — Batched per-subject encoding + permutation significance test.

Processes one subject at a time to avoid OOM from concatenating all subjects.
Writes:
  {output_dir}/significance_detailed.csv   — one row per (subject, electrode, lag)
  {output_dir}/all_subject_sig.csv         — electrodes with ≥1 significant lag
"""

import argparse
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
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Per-subject encoding + significance testing")
    parser.add_argument("--data-root", default="data/", help="Root of BIDS dataset")
    parser.add_argument("--output-dir", default="results/", help="Directory for output CSVs")
    parser.add_argument("--layer", type=int, default=24, help="GPT-2-XL layer index")
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Subject BIDS IDs to process (default: 1–9). E.g. --subjects 1 3 5",
    )
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--fdr-q", type=float, default=0.01)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.subjects is None:
        subject_ids = list(range(1, 10))
    else:
        subject_ids = [int(s) for s in args.subjects]

    # Use GPU backend if available
    if torch.cuda.is_available():
        set_backend("torch_cuda")
        print("Using torch_cuda backend.")
    else:
        print("CUDA not available, using CPU.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load embeddings once — shared across all subjects
    aligned_embeddings, df_word = load_embeddings(args.data_root, args.layer)

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
        raws = load_raws([bids_sub_id], args.data_root, use_high_gamma=True)
        raw = raws[0]
        electrode_names = raw.ch_names
        print(f"  Electrodes: {len(electrode_names)}")

        # ------------------------------------------------------------------
        # b. Build epochs
        # ------------------------------------------------------------------
        epochs, data = build_word_epochs(raw, df_word)
        print(f"  Epoch data shape: {data.shape}")

        # ------------------------------------------------------------------
        # c. Align X to epoch selection
        # ------------------------------------------------------------------
        X = aligned_embeddings[epochs.selection].astype(np.float32)
        X = [X, X]  # one per fold

        # ------------------------------------------------------------------
        # d. Build Y
        # ------------------------------------------------------------------
        n_words, n_elec, n_lags = data.shape
        Y = data.reshape(n_words, -1).astype(np.float32)

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
        print(f"  Running significance test ({args.n_permutations} permutations)...")
        sig, p_values, q_values, obs_corrs = significant_lags_all_electrodes_gpu(
            all_predictions.transpose(1, 2, 0),
            all_targets.transpose(1, 2, 0),
            n_permutations=args.n_permutations,
            fdr_q=args.fdr_q,
        )
        # shapes: (n_elec, n_lags)

        # ------------------------------------------------------------------
        # h. Collect rows
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
        # i. Free memory
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
    detailed_path = os.path.join(args.output_dir, "significance_detailed.csv")
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
    sig_path = os.path.join(args.output_dir, "all_subject_sig.csv")
    df_any_sig.to_csv(sig_path, index=False)
    print(f"Wrote {sig_path}  ({len(df_any_sig):,} rows)")


if __name__ == "__main__":
    main()
