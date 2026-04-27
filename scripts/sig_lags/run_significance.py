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
from src.utils.embeddings import get_glove_embeddings
import torch

from src.utils import (
    train_encoding,
    get_all_predictions_and_targets_test_set,
    significant_lags_all_electrodes_gpu,
    load_embeddings,
    get_models_coefficients,
    save_subject_results,
)
from src.core.config import SignificanceConfig, get_results_dir
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.config_utils import parse_args, load_config
from src.utils.encoding import get_ridge_model_constructor
from src.utils.data_loaders import load_data_from_cache


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args, overrides = parse_args("Per-subject encoding + significance testing")
    config: SignificanceConfig = load_config(args.config, overrides, SignificanceConfig)
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

    # Use GPU backend if available
    if torch.cuda.is_available():
        set_backend("torch_cuda")
        print("Using torch_cuda backend.")
    else:
        print("CUDA not available, using CPU.")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Results directory: {output_dir}")

    # Load embeddings once — shared across all subjects
    aligned_embeddings, transcript_df = load_embeddings(
        enc.data_root,
        enc.model,
        dataset=enc.dataset,
        per_dataset_config=enc.per_dataset_config,
        dataset_context=dataset_context,
        **enc.per_model_config,
    )

    word_in_glove = None
    if enc.limit_to_glove or enc.model == "glove":
        word_in_glove = get_glove_embeddings(
            enc.data_root, transcript_df, **enc.per_model_config
        )[1]["in_glove"]

    alphas = np.array(enc.ridge_alphas) if enc.ridge_alphas is not None else None
    model_constructor = get_ridge_model_constructor(alphas=alphas)

    all_rows = []

    for bids_sub_id in subject_ids:
        print(f"\n{'='*60}")
        print(f"Processing subject {bids_sub_id}")
        print(f"{'='*60}")

        X_arr, Y, selection, elec_names = load_data_from_cache(
            [bids_sub_id],
            enc.data_root,
            enc.offset,
            aligned_embeddings,
            valid_rows=word_in_glove,
            dataset=enc.dataset,
            per_dataset_config=enc.per_dataset_config,
            dataset_context=dataset_context,
        )
        _, n_targets = Y.shape
        elec_counts = [len(x) for x in elec_names]
        n_lags = n_targets // sum(elec_counts)
        n_elec = sum(elec_counts)
        X = [X_arr] * enc.num_folds

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
            num_folds=enc.num_folds,
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
        print(f"  Running significance test...")
        sig, p_values, q_values, obs_corrs = significant_lags_all_electrodes_gpu(
            all_predictions.transpose(1, 2, 0),
            all_targets.transpose(1, 2, 0),
            minimal_shift=config.minimal_shift,
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
                sig_ints = [
                    i.reshape(n_elec, n_lags)[sig_elec_indices].ravel()
                    for i in intercepts
                ]
                sig_corrs = corrs[:, sig_elec_indices, :]
                sig_names = [elec_names[0][i] for i in sig_elec_indices]
                save_subject_results(
                    output_dir,
                    bids_sub_id,
                    sig_corrs,
                    sig_coefs,
                    sig_ints,
                    models,
                    tr_idxs,
                    test_idxs,
                    scalers,
                    sig_names,
                    enc.offset,
                    actual_lag_sec,
                    per_model_config=enc.per_model_config,
                )
                del coefficients, intercepts

        # ------------------------------------------------------------------
        # i. Collect rows
        # ------------------------------------------------------------------
        for elec_idx, elec_name in enumerate(elec_names[0]):
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
        del X_arr, X, Y, models, scalers
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
