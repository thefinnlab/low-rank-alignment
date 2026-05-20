"""
run_heuristic_significance.py - Heuristic lag-cluster significance.

Processes one subject at a time, trains encoding models, builds reliable
electrode-lag clusters from out-of-fold predictions, and marks every lag in a
cluster significant when that cluster has at least ``min_cluster_size`` lag
indices. This entrypoint does not score circular-shift nulls or apply FDR.
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
    load_embeddings,
)
from src.core.config import SignificanceConfig, get_results_dir
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.config_utils import (
    parse_args,
    load_config,
    expand_per_model_layer_configs,
)
from src.utils.encoding import extract_best_alphas, get_ridge_model_constructor
from src.utils.alpha_grids import resolve_ridge_alphas, save_alpha_diagnostics
from src.utils.data_loaders import load_data_from_cache
from src.utils.significance import (
    mark_clusters_by_min_size,
    reshape_fold_predictions_and_targets,
    score_fold_lag_clusters,
    validate_min_cluster_size,
)


def run_config(config: SignificanceConfig):
    enc = config.encoding
    min_cluster_size = validate_min_cluster_size(config.min_cluster_size)
    mean_corr_threshold = (
        float(np.tanh(config.mean_z_threshold))
        if config.mean_z_threshold is not None
        else float(config.mean_corr_threshold)
    )

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

    if torch.cuda.is_available():
        set_backend("torch_cuda")
        print("Using torch_cuda backend.")
    else:
        print("CUDA not available, using CPU.")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Results directory: {output_dir}")
    print(f"Heuristic min_cluster_size: {min_cluster_size}")

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

    explicit_alphas = (
        np.array(enc.ridge_alphas, dtype=float)
        if enc.ridge_alphas is not None
        else None
    )

    all_lag_records = []

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
        del selection
        _, n_targets = Y.shape
        elec_counts = [len(x) for x in elec_names]
        n_lags = n_targets // sum(elec_counts)
        n_elec = sum(elec_counts)
        X = [X_arr] * enc.num_folds
        alpha_context = {
            "script": "run_heuristic_significance",
            "role": "subject",
            "subject": int(bids_sub_id),
            "model": enc.model,
            "per_model_config": enc.per_model_config,
            "offset": enc.offset,
            "n_features": int(X_arr.shape[1]),
            "standardize": True,
        }
        alpha_dir = os.path.join(output_dir, "subjects", f"sub_{bids_sub_id}")
        alphas = resolve_ridge_alphas(
            explicit_alphas,
            X_arr,
            standardize=True,
            context=alpha_context,
            cache_dir=alpha_dir,
            policy=enc.alpha_grid,
        )
        model_constructor = get_ridge_model_constructor(
            alphas=alphas,
            ridge_cv_folds=enc.ridge_cv_folds,
            solver_params=enc.ridge_solver_params,
            y_in_cpu=enc.ridge_y_in_cpu,
            force_cpu=enc.ridge_force_cpu,
        )

        actual_lag_sec = np.linspace(-2.0, 2.0, n_lags, endpoint=False)

        print("  Training encoding model...")
        corrs, models, tr_idxs, test_idxs, scalers, fold_predictions = train_encoding(
            X,
            Y,
            model_constructor,
            epochs_shape=(n_elec, n_lags),
            num_folds=enc.num_folds,
            return_preds=True,
        )
        del tr_idxs
        if enc.log_alphas:
            save_alpha_diagnostics(
                alpha_dir,
                alphas,
                best_alphas=[extract_best_alphas(m) for m in models],
                context=alpha_context,
                policy=enc.alpha_grid,
            )

        fold_predictions_3d, fold_targets_3d = reshape_fold_predictions_and_targets(
            fold_predictions, test_idxs, Y, n_elec, n_lags
        )
        print("  Computing observed block correlations and lag clusters...")
        cluster_result = score_fold_lag_clusters(
            fold_predictions_3d,
            fold_targets_3d,
            config.blocks_per_fold,
            max_matching_distance=config.max_matching_distance,
            positive_fraction=config.positive_fraction,
            mean_corr_threshold=mean_corr_threshold,
            device=config.device,
        )
        obs_corrs = cluster_result["corr"]
        positive_frac = cluster_result["positive_fraction"]
        mean_corr = cluster_result["mean_corr"]
        cluster_labels = cluster_result["cluster_labels"]
        scored_clusters = mark_clusters_by_min_size(
            cluster_result["scored_clusters"], min_cluster_size=min_cluster_size
        )

        cluster_lookup = {}
        for cluster in scored_clusters:
            elec_idx = int(cluster["electrode_idx"])
            electrode = elec_names[0][elec_idx]
            record = {
                "subject": bids_sub_id,
                "electrode": electrode,
                "subject_electrode_index": elec_idx,
                "cluster_id": int(cluster["cluster_id"]),
                "lag_indices": cluster["lag_indices"],
                "cluster_score": float(cluster["cluster_score"]),
                "cluster_size": int(cluster["cluster_size"]),
                "peak_corr": float(cluster["peak_corr"]),
                "peak_score": float(cluster["peak_target_score"]),
                "cluster_p_value": np.nan,
                "cluster_q_value": np.nan,
                "sig": bool(cluster["sig"]),
            }
            cluster_lookup[(electrode, int(cluster["cluster_id"]))] = record

        for elec_idx, elec_name in enumerate(elec_names[0]):
            for lag_idx in range(n_lags):
                label = int(cluster_labels[elec_idx, lag_idx])
                rec = cluster_lookup.get((elec_name, label))
                all_lag_records.append(
                    {
                        "subject": bids_sub_id,
                        "electrode": elec_name,
                        "lag_idx": lag_idx,
                        "lag_sec": float(actual_lag_sec[lag_idx]),
                        "sig": bool(rec["sig"]) if rec is not None else False,
                        "p_value": np.nan,
                        "q_value": np.nan,
                        "obs_corr": float(obs_corrs[elec_idx, lag_idx]),
                        "cluster_id": label,
                        "cluster_score": (
                            float(rec["cluster_score"]) if rec is not None else np.nan
                        ),
                        "cluster_p_value": np.nan,
                        "cluster_q_value": np.nan,
                        "positive_fraction": float(positive_frac[elec_idx, lag_idx]),
                        "mean_corr": float(mean_corr[elec_idx, lag_idx]),
                        "cluster_size": (
                            int(rec["cluster_size"]) if rec is not None else np.nan
                        ),
                    }
                )
        n_sig_clusters = sum(1 for cluster in scored_clusters if cluster["sig"])
        print(
            f"  Found {len(scored_clusters):,} candidate clusters across "
            f"{n_elec:,} electrodes; {n_sig_clusters:,} pass size threshold"
        )

        del X_arr, X, Y, models, scalers
        del corrs, fold_predictions, fold_predictions_3d, fold_targets_3d
        del cluster_result, obs_corrs
        del positive_frac, mean_corr, cluster_labels
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    detailed_path = os.path.join(output_dir, "significance_detailed.csv")
    df_detailed = pd.DataFrame(all_lag_records)
    df_detailed.to_csv(detailed_path, index=False)

    sig_pairs = (
        df_detailed[df_detailed["sig"]][["subject", "electrode"]]
        .drop_duplicates()
        .rename(columns={"electrode": "elec"})
    )
    sig_path = os.path.join(output_dir, "all_subject_sig.csv")
    sig_pairs.to_csv(sig_path, index=False, columns=["subject", "elec"])

    print(f"\nWrote {detailed_path}  ({len(df_detailed):,} rows)")
    print(f"Wrote {sig_path}  ({len(sig_pairs):,} electrodes)")
    print(
        "Heuristic run complete; cluster_null_distribution.npz was not written."
    )


def main():
    args, overrides = parse_args("Heuristic lag-cluster significance")
    config: SignificanceConfig = load_config(args.config, overrides, SignificanceConfig)
    configs = expand_per_model_layer_configs(config)
    for layer_config in configs:
        if len(configs) > 1:
            print(
                f"\nComputing heuristic cluster significance for layer "
                f"{layer_config.encoding.per_model_config['layer']}"
            )
        run_config(layer_config)


if __name__ == "__main__":
    main()
