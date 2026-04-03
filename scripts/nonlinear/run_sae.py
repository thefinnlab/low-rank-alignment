"""
run_sae.py — SAE-based encoding analysis.

Loads an RRR-learned subspace (A matrix) at a given rank k, projects word
embeddings into that k-dimensional space, applies a sparse autoencoder
(overcomplete hidden layer with ReLU + L1 penalty on activations), and
regresses onto neural targets.  Supports two training modes:

  end_to_end:   SAE + linear regression head trained jointly via backprop.
  alternating:  Alternating optimisation — train SAE encoder for N steps,
                then re-solve regression optimally (OLS or Ridge).

Three modes controlled by SAEConfig.mode:

  cross_subject (default):
    Builds global Y across all subjects (significant electrodes).
    Writes to {output_dir}/sae/.

  per_subject:
    Runs independently per subject.
    Writes to {output_dir}/per_subject_sae/subjects/sub_{id}/.

  per_region:
    Runs independently per atlas region.
    Writes to {output_dir}/per_region_sae/regions/{region}/.

Output layout per unit:
    {sae_unit_dir}/
        grid_results.csv
        summary.csv
        k{k}/fold_{f}/
            sae_weights.pt, regression_weights.npy, metrics.npz, training_log.csv
"""

import copy
import gc
import itertools
import os
import time
from dataclasses import asdict, fields
from typing import get_type_hints, get_origin, get_args, Union

import numpy as np
import pandas as pd
import torch
import yaml

from src.utils import (
    load_embeddings,
    apply_offset,
    build_XY_from_subjects,
    load_precomputed_epochs,
    epochs_from_cache,
)
from src.utils.data_loaders import load_sig_electrodes
from src.utils.atlas import build_electrode_region_map
from src.core.config import SAEConfig, SVDMode, get_results_dir
from src.utils.config_utils import parse_args, load_config
from src.utils.encoding import (
    load_encoding_results,
    build_enc_subjects_info,
    build_per_fold_baselines,
)
from src.utils.srrr import (
    compute_r2,
    compute_correlation,
    run_sufficiency_test,
    load_rrr_result,
)
from src.utils.sae import (
    train_sae_end_to_end,
    train_sae_alternating,
)


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------


def _resolve_type(hint):
    """Unwrap Optional[X] → X; return hint unchanged otherwise."""
    origin = get_origin(hint)
    if origin is Union:
        args = [a for a in get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return hint


_SCALAR_TYPES = (int, float, str, bool)


def _detect_sweep_params(config: SAEConfig):
    """Detect which SAEConfig fields are set to lists (indicating a sweep).

    Returns:
        k_values: list of k values to sweep (single-element if scalar).
        sweep_params: OrderedDict mapping param name → list of values (excludes k).
    """
    hints = get_type_hints(SAEConfig)
    k_values = [config.k] if not isinstance(config.k, list) else config.k
    sweep_params = {}
    for f in fields(SAEConfig):
        if f.name == "k":
            continue
        resolved = _resolve_type(hints[f.name])
        if not isinstance(resolved, type) or not issubclass(resolved, _SCALAR_TYPES):
            continue
        value = getattr(config, f.name)
        if isinstance(value, list):
            sweep_params[f.name] = value
    return k_values, sweep_params


def _build_experiment_subdir(param_dict: dict) -> str | None:
    """Build hierarchical subdir path from sweep param values.

    Keys are sorted alphabetically for consistent ordering across runs.
    Returns None if param_dict is empty.
    """
    if not param_dict:
        return None
    return os.path.join(*(f"{k}={v}" for k, v in sorted(param_dict.items())))


def _make_scalar_config(config: SAEConfig, k_value, overrides: dict) -> SAEConfig:
    """Return a copy of config with k and sweep params set to scalar values."""
    cfg = copy.deepcopy(config)
    cfg.k = k_value
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


# ---------------------------------------------------------------------------
# Core per-unit loop
# ---------------------------------------------------------------------------


def _run_sae_for_unit(
    sae_unit_dir: str,
    rrr_unit_dir: str,
    X_raw: np.ndarray,
    test_idxs: list,
    per_fold_baselines: list,
    config: SAEConfig,
    n_electrodes: int | None = None,
    experiment_subdir: str | None = None,
) -> pd.DataFrame:
    """Core SAE loop for one analysis unit (cross-subject, per-subject, or per-region).

    Args:
        sae_unit_dir: Root output directory for this unit.
        rrr_unit_dir: RRR results directory to load A matrices from.
        X_raw: (n_words, n_embed) raw embedding matrix.
        test_idxs: List of test index arrays, one per fold.
        per_fold_baselines: Output of build_per_fold_baselines.
        config: SAEConfig.
        n_electrodes: Number of electrodes (for reshaping per-target correlations
            into (n_electrodes, n_lags) for the timecourse plot). None to skip plot.
        experiment_subdir: Optional subdirectory under k_dir for sweep experiments
            (e.g. "hidden_dim=5/l1_penalty=0.005"). None = write directly to k_dir.

    Returns:
        DataFrame of per-fold metrics for grid_results.csv.
    """
    os.makedirs(sae_unit_dir, exist_ok=True)

    enc = config.encoding
    alphas = np.array(enc.ridge_alphas) if enc.ridge_alphas is not None else None

    n_folds = len(test_idxs)
    k = config.k
    all_rows = []
    fold_corrs_sae = []
    fold_corrs_full = []
    fold_corrs_rrr = []

    for fold_f in range(n_folds):
        fold_start = time.time()
        print(f"\n[Fold {fold_f + 1}/{n_folds}]")

        # Load RRR A matrix for this fold
        rrr_result = load_rrr_result(rrr_unit_dir, k=k, fold=fold_f)
        A = rrr_result["A"]  # (n_embed, k)
        print(f"  Loaded RRR A matrix: {A.shape}")

        # Project embeddings into k-dim RRR subspace
        baseline = per_fold_baselines[fold_f]
        train_idx = baseline["train_idx"]
        test_idx = baseline["test_idx"]

        X_proj_train = X_raw[train_idx] @ A  # (n_train, k)
        X_proj_test = X_raw[test_idx] @ A  # (n_test, k)
        Y_train = baseline["Y_train"]
        Y_test = baseline["Y_test"]
        Y_hat_full = baseline["Y_hat_full_test"]

        print(f"  X_proj_train: {X_proj_train.shape}, Y_train: {Y_train.shape}")

        # Build optional log callback for recovery tracking during training.
        log_callback = None
        if config.track_recovery_during_training:

            def _log_callback(Y_hat_train, Y_hat_test):
                extra = {}
                suf = run_sufficiency_test(
                    X_raw[train_idx],
                    Y_train - Y_hat_train,
                    X_raw[test_idx],
                    Y_test,
                    Y_hat_full=Y_hat_full,
                    Y_hat_min=Y_hat_test,
                    null_r2=None,
                    regression_type="ridge",
                    alphas=alphas,
                )
                extra["recovery_corr"] = suf["recovery_corr"]
                extra["recovery_r2"] = suf["recovery_r2"]
                return extra

            log_callback = _log_callback

        # Train SAE + regression
        if config.training_mode == "alternating":
            result = train_sae_alternating(
                X_proj_train,
                Y_train,
                X_proj_test,
                Y_test,
                hidden_dim=config.hidden_dim,
                l1_penalty=config.l1_penalty,
                regression_type=config.regression_type,
                sae_lr=config.sae_learning_rate,
                sae_steps_per_cycle=config.sae_steps_per_cycle,
                num_cycles=config.num_cycles,
                batch_size=config.batch_size,
                log_every=config.log_every,
                resample_interval=config.resample_interval,
                weight_l1_by_decoder_norm=config.weight_l1_by_decoder_norm,
                l1_warmup_start=config.l1_warmup_start,
                l1_warmup_epochs=config.l1_warmup_epochs,
                regression_momentum=config.regression_momentum,
                input_jitter_std=config.input_jitter_std,
                log_callback=log_callback,
                ridge_alphas=alphas,
            )
        else:
            result = train_sae_end_to_end(
                X_proj_train,
                Y_train,
                X_proj_test,
                Y_test,
                hidden_dim=config.hidden_dim,
                l1_penalty=config.l1_penalty,
                lr=config.learning_rate,
                num_epochs=config.num_epochs,
                batch_size=config.batch_size,
                log_every=config.log_every,
                resample_interval=config.resample_interval,
                weight_l1_by_decoder_norm=config.weight_l1_by_decoder_norm,
                l1_warmup_start=config.l1_warmup_start,
                l1_warmup_epochs=config.l1_warmup_epochs,
                log_callback=log_callback,
            )

        Y_hat_sae = result["Y_hat_test"]
        Y_hat_sae_train = result["Y_hat_train"]

        # Per-target correlations for timecourse plot
        if n_electrodes is not None:
            n_targets = Y_test.shape[1]
            n_lags = n_targets // n_electrodes
            per_target_corr_sae = compute_correlation(Y_test, Y_hat_sae)
            per_target_corr_full = compute_correlation(Y_test, Y_hat_full)
            fold_corrs_sae.append(per_target_corr_sae.reshape(n_electrodes, n_lags))
            fold_corrs_full.append(per_target_corr_full.reshape(n_electrodes, n_lags))

            # NR ridge baseline: load per-target correlations from saved RRR results
            rrr_metrics = rrr_result.get("metrics", {})
            nr_corr_per_target = rrr_metrics.get("nr_corr_per_target")
            if nr_corr_per_target is not None:
                fold_corrs_rrr.append(
                    np.asarray(nr_corr_per_target).reshape(n_electrodes, n_lags)
                )

        # Recovery test: partial out SAE predictions and retrain ridge on residuals
        suf = run_sufficiency_test(
            X_raw[train_idx],
            Y_train - Y_hat_sae_train,
            X_raw[test_idx],
            Y_test,
            Y_hat_full=Y_hat_full,
            Y_hat_min=Y_hat_sae,
            null_r2=None,
            regression_type="ridge",
            alphas=alphas,
        )
        print(
            f"  Recovery test R²: {suf['recovery_r2']:.4f}, corr: {suf['recovery_corr']:.4f}"
        )

        # Load NR ridge baseline metrics at same k for comparison
        rrr_baseline_metrics = rrr_result.get("metrics", {})
        rrr_baseline_r2 = float(rrr_baseline_metrics.get("nr_r2", np.nan))
        rrr_baseline_corr = float(rrr_baseline_metrics.get("nr_corr", np.nan))
        rrr_recovery_corr = float(rrr_baseline_metrics.get("nr_recovery_corr", np.nan))

        # Assemble metrics row
        m = result["metrics"]
        metrics = {
            "fold": fold_f,
            "correlation": m["correlation"],
            "r2": m["r2"],
            "l0_sparsity": m["l0_sparsity"],
            "dead_neurons": m["dead_neurons"],
            "mean_activation": m["mean_activation"],
            "l1_loss": m["l1_loss"],
            "recovery_r2": suf["recovery_r2"],
            "recovery_corr": suf["recovery_corr"],
            "rrr_baseline_r2": rrr_baseline_r2,
            "rrr_baseline_corr": rrr_baseline_corr,
            "rrr_recovery_corr": rrr_recovery_corr,
        }
        # Save per-fold outputs
        k_dir = os.path.join(sae_unit_dir, f"k{k}")
        if experiment_subdir:
            k_dir = os.path.join(k_dir, experiment_subdir)
        fold_dir = os.path.join(k_dir, f"fold_{fold_f}")
        os.makedirs(fold_dir, exist_ok=True)

        torch.save(
            result["encoder_state_dict"], os.path.join(fold_dir, "sae_weights.pt")
        )
        np.save(
            os.path.join(fold_dir, "regression_weights.npy"), result["regressor_weight"]
        )

        # Save metrics (scalars + activation_frequency array)
        npz_metrics = {mk: mv for mk, mv in metrics.items()}
        npz_metrics["activation_frequency"] = m["activation_frequency"]
        np.savez(os.path.join(fold_dir, "metrics.npz"), **npz_metrics)

        # Save training log and extract start/end values for sweep summary
        log_df = pd.DataFrame(result["training_log"])
        log_df.to_csv(os.path.join(fold_dir, "training_log.csv"), index=False)

        for tl_metric in _TRAINING_LOG_METRICS:
            if tl_metric in log_df.columns:
                col = log_df[tl_metric].dropna()
                if len(col) > 0:
                    metrics[f"{tl_metric}_start"] = float(col.iloc[0])
                    metrics[f"{tl_metric}_end"] = float(col.iloc[-1])

        from src.utils.plotting import plot_sae_training_curves

        plot_sae_training_curves(
            result["training_log"],
            title=f"SAE training — k={k}, fold {fold_f}",
            save_path=os.path.join(fold_dir, "training_curves.png"),
            rrr_recovery_corr=rrr_recovery_corr,
        )

        all_rows.append(metrics)
        print(f"  Fold {fold_f} done in {time.time() - fold_start:.1f}s")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(k_dir, "grid_results.csv"), index=False)

    # Summary: average over folds
    agg_cols = [c for c in df.columns if c != "fold"]
    summary = df[agg_cols].mean().to_frame().T
    summary.to_csv(os.path.join(k_dir, "summary.csv"), index=False)

    # Recovery summary plot: SAE vs RRR across folds
    if "recovery_corr" in df.columns and "rrr_recovery_corr" in df.columns:
        from src.utils.plotting import plot_sae_recovery_summary

        plot_sae_recovery_summary(
            df,
            title=f"SAE vs RRR recovery — k={k}",
            save_path=os.path.join(k_dir, "recovery_summary.png"),
        )

    # Correlation timecourse plot: SAE vs RRR-direct vs full model
    if n_electrodes is not None and fold_corrs_sae:
        from src.utils.plotting import plot_sae_corrs_vs_full

        corrs_sae = np.stack(fold_corrs_sae, axis=0)  # (n_folds, n_electrodes, n_lags)
        corrs_full = np.stack(
            fold_corrs_full, axis=0
        )  # (n_folds, n_electrodes, n_lags)
        corrs_rrr = np.stack(fold_corrs_rrr, axis=0)  # (n_folds, n_electrodes, n_lags)
        n_lags = corrs_sae.shape[2]
        lags = np.linspace(-2.0, 2.0, n_lags, endpoint=False)
        plot_sae_corrs_vs_full(
            corrs_sae,
            corrs_full,
            lags,
            corrs_rrr=corrs_rrr,
            title=f"SAE vs RRR vs full model — k={k}",
            save_path=os.path.join(k_dir, "corrs_sae_vs_full.png"),
        )
        # Save correlation arrays for downstream use
        np.save(os.path.join(k_dir, "corrs_sae.npy"), corrs_sae)
        np.save(os.path.join(k_dir, "corrs_full.npy"), corrs_full)
        np.save(os.path.join(k_dir, "corrs_rrr.npy"), corrs_rrr)

    return df


# ---------------------------------------------------------------------------
# Branch: cross-subject SAE
# ---------------------------------------------------------------------------


def _run_cross_subject_sae(
    config,
    enc,
    output_dir,
    sig_dir,
    sae_dir,
    rrr_source_dir,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    df_word,
    experiment_subdir=None,
):
    """SAE on RRR-projected embeddings using all subjects' significant electrodes."""
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
        print("  No significant electrodes found; skipping cross-subject SAE.")
        return

    X_ref, Y_full, _ = build_XY_from_subjects(
        subject_ids,
        enc,
        sig_electrodes,
        aligned_embeddings,
        df_word,
    )
    total_electrodes = sum(len(sig_electrodes[s]) for s in subs_with_sig)
    print(f"  X: {X_ref.shape}, Y: {Y_full.shape}")

    enc_subjects_info = build_enc_subjects_info(
        output_dir, subs_with_sig, sig_electrodes
    )
    per_fold_baselines = build_per_fold_baselines(
        X_ref, Y_full, ref_test_idxs, enc_subjects_info
    )

    return _run_sae_for_unit(
        sae_unit_dir=sae_dir,
        rrr_unit_dir=rrr_source_dir,
        X_raw=X_ref,
        test_idxs=ref_test_idxs,
        per_fold_baselines=per_fold_baselines,
        config=config,
        n_electrodes=total_electrodes,
        experiment_subdir=experiment_subdir,
    )


# ---------------------------------------------------------------------------
# Branch: per-subject SAE
# ---------------------------------------------------------------------------


def _run_per_subject_sae(
    config,
    enc,
    output_dir,
    sig_dir,
    sae_dir,
    rrr_source_dir,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    df_word,
    experiment_subdir=None,
):
    """SAE on RRR-projected embeddings independently per subject."""
    unit_dfs = []
    subjects_to_run = subject_ids
    if config.max_subjects is not None:
        subjects_to_run = subjects_to_run[: config.max_subjects]

    for sub_id in subjects_to_run:
        print(f"\n{'='*60}\nSubject {sub_id}")

        sae_sub_dir = os.path.join(sae_dir, "subjects", f"sub_{sub_id}")
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

        cache = load_precomputed_epochs(sub_id, enc.data_root)
        data = epochs_from_cache(cache, elec_names=sub_sig_elecs)
        n_elecs_sub = data.shape[1]
        X_arr, Y_sub = apply_offset(
            aligned_embeddings[cache["selection"]], data, enc.offset
        )
        del cache, data
        gc.collect()

        Y_sub = Y_sub.astype(np.float32)
        print(f"  X: {X_arr.shape}, Y: {Y_sub.shape}")

        enc_subjects_info = build_enc_subjects_info(
            output_dir, [sub_id], {sub_id: sub_sig_elecs}
        )
        per_fold_baselines = build_per_fold_baselines(
            X_arr, Y_sub, test_idxs, enc_subjects_info
        )

        unit_dfs.append(
            _run_sae_for_unit(
                sae_unit_dir=sae_sub_dir,
                rrr_unit_dir=rrr_sub_dir,
                X_raw=X_arr,
                test_idxs=test_idxs,
                per_fold_baselines=per_fold_baselines,
                config=config,
                n_electrodes=n_elecs_sub,
                experiment_subdir=experiment_subdir,
            )
        )

    return unit_dfs


# ---------------------------------------------------------------------------
# Branch: per-region SAE
# ---------------------------------------------------------------------------


def _run_per_region_sae(
    config,
    enc,
    output_dir,
    sig_dir,
    sae_dir,
    rrr_source_dir,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    df_word,
    experiment_subdir=None,
):
    """SAE on RRR-projected embeddings independently per atlas region."""
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

    unit_dfs = []
    regions_to_run = list(electrode_region_map.items())
    if config.max_regions is not None:
        regions_to_run = regions_to_run[: config.max_regions]

    for region_name, sub_elec_map in regions_to_run:
        print(f"\n{'='*60}\nRegion: {region_name}")

        sae_region_dir = os.path.join(sae_dir, "regions", region_name)
        rrr_region_dir = os.path.join(rrr_source_dir, "regions", region_name)

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
        region_n_electrodes = sum(len(e) for e in per_subject_region_elecs.values())
        X_ref, Y_full, _ = build_XY_from_subjects(
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

        unit_dfs.append(
            _run_sae_for_unit(
                sae_unit_dir=sae_region_dir,
                rrr_unit_dir=rrr_region_dir,
                X_raw=X_ref,
                test_idxs=ref_test_idxs,
                per_fold_baselines=per_fold_baselines,
                config=config,
                n_electrodes=region_n_electrodes,
                experiment_subdir=experiment_subdir,
            )
        )

    return unit_dfs


# Metrics to extract start/end values from training logs
_TRAINING_LOG_METRICS = [
    "test_corr",
    "test_r2",
    "l0_sparsity",
    "dead_frac",
    "recovery_corr",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _dispatch_branch(config, branch_kwargs):
    """Call the appropriate branch function based on config.mode."""
    if config.mode == SVDMode.PER_REGION:
        return _run_per_region_sae(**branch_kwargs)
    elif config.mode == SVDMode.PER_SUBJECT:
        return _run_per_subject_sae(**branch_kwargs)
    else:
        return _run_cross_subject_sae(**branch_kwargs)


def main():
    args, overrides = parse_args("SAE: sparse autoencoder encoding analysis")
    config: SAEConfig = load_config(args.config, overrides, SAEConfig)
    enc = config.encoding

    import torch
    from himalaya.backend import set_backend

    if torch.cuda.is_available():
        set_backend("torch_cuda")

    subject_ids = enc.subjects or list(range(1, 10))
    output_dir = config.output_dir or get_results_dir(enc.model, enc.layer, enc.offset)

    # Resolve SAE output directory
    if config.sae_output_dir:
        sae_dir = config.sae_output_dir
    elif config.mode == SVDMode.PER_REGION:
        sae_dir = os.path.join(output_dir, "per_region_sae")
    elif config.mode == SVDMode.PER_SUBJECT:
        sae_dir = os.path.join(output_dir, "per_subject_sae")
    else:
        sae_dir = os.path.join(output_dir, "sae")

    # Resolve RRR source directory
    if config.rrr_source_dir:
        rrr_source_dir = config.rrr_source_dir
    elif config.mode == SVDMode.PER_REGION:
        rrr_source_dir = os.path.join(output_dir, "per_region_rrr")
    elif config.mode == SVDMode.PER_SUBJECT:
        rrr_source_dir = os.path.join(output_dir, "per_subject_rrr")
    else:
        rrr_source_dir = os.path.join(output_dir, "rrr")

    os.makedirs(sae_dir, exist_ok=True)
    print(f"Output dir:      {output_dir}")
    print(f"SAE dir:         {sae_dir}")
    print(f"RRR source dir:  {rrr_source_dir}")
    print(f"Mode:            {config.mode.value}")
    print(f"Training mode:   {config.training_mode}")

    # Load significance CSV and build subject → electrode mapping
    sig_dir = config.sig_dir or output_dir
    sig_electrodes, _ = load_sig_electrodes(output_dir, subject_ids, sig_dir=sig_dir)

    # Load embeddings
    print("Loading embeddings...")
    aligned_embeddings, df_word = load_embeddings(enc.data_root, enc.layer)

    # Detect sweep parameters
    k_values, sweep_params = _detect_sweep_params(config)
    is_sweep = bool(sweep_params) or len(k_values) > 1

    if not is_sweep:
        # --- No sweep: existing behaviour ---
        print(f"k={config.k}, hidden_dim={config.hidden_dim}, l1={config.l1_penalty}")

        # Save config for reproducibility
        k_dir = os.path.join(sae_dir, f"k{config.k}")
        os.makedirs(k_dir, exist_ok=True)
        with open(os.path.join(k_dir, "config.yml"), "w") as f:
            yaml.dump(asdict(config), f, default_flow_style=False)

        branch_kwargs = dict(
            config=config,
            enc=enc,
            output_dir=output_dir,
            sig_dir=sig_dir,
            sae_dir=sae_dir,
            rrr_source_dir=rrr_source_dir,
            subject_ids=subject_ids,
            sig_electrodes=sig_electrodes,
            aligned_embeddings=aligned_embeddings,
            df_word=df_word,
        )
        _dispatch_branch(config, branch_kwargs)
    else:
        # --- Sweep mode ---
        print(f"k values: {k_values}")
        if sweep_params:
            for name, vals in sweep_params.items():
                print(f"  sweep {name}: {vals}")

        combos = (
            list(itertools.product(*sweep_params.values())) if sweep_params else [()]
        )
        combo_keys = list(sweep_params.keys())
        total_experiments = len(k_values) * len(combos)
        print(f"Total experiments: {total_experiments}")

        # Save the full sweep config (with list values) at the sae_dir level
        with open(os.path.join(sae_dir, "sweep_config.yml"), "w") as f:
            yaml.dump(asdict(config), f, default_flow_style=False)

        all_k_results = []

        for k_val in k_values:
            print(f"\n{'='*60}\nk = {k_val}\n{'='*60}")
            k_dir = os.path.join(sae_dir, f"k{k_val}")
            os.makedirs(k_dir, exist_ok=True)
            k_experiment_results = []

            for combo_i, combo in enumerate(combos):
                param_dict = dict(zip(combo_keys, combo))
                scalar_config = _make_scalar_config(config, k_val, param_dict)
                experiment_subdir = _build_experiment_subdir(param_dict)

                exp_label = ", ".join(f"{k}={v}" for k, v in sorted(param_dict.items()))
                print(
                    f"\n--- Experiment {combo_i + 1}/{len(combos)}: k={k_val}, {exp_label} ---"
                )

                # Save per-experiment config
                exp_dir = (
                    os.path.join(k_dir, experiment_subdir)
                    if experiment_subdir
                    else k_dir
                )
                os.makedirs(exp_dir, exist_ok=True)
                with open(os.path.join(exp_dir, "config.yml"), "w") as f:
                    yaml.dump(asdict(scalar_config), f, default_flow_style=False)

                branch_kwargs = dict(
                    config=scalar_config,
                    enc=enc,
                    output_dir=output_dir,
                    sig_dir=sig_dir,
                    sae_dir=sae_dir,
                    rrr_source_dir=rrr_source_dir,
                    subject_ids=subject_ids,
                    sig_electrodes=sig_electrodes,
                    aligned_embeddings=aligned_embeddings,
                    df_word=df_word,
                    experiment_subdir=experiment_subdir,
                )
                df_result = _dispatch_branch(scalar_config, branch_kwargs)

                # Normalise result to a single DataFrame
                if isinstance(df_result, list):
                    df_result = (
                        pd.concat(df_result, ignore_index=True)
                        if df_result
                        else pd.DataFrame()
                    )
                if df_result is not None and not df_result.empty:
                    for pname, pval in param_dict.items():
                        df_result[pname] = pval
                    df_result["k"] = k_val
                    k_experiment_results.append(df_result)

                gc.collect()

            if k_experiment_results:
                combined_k = pd.concat(k_experiment_results, ignore_index=True)
                combined_k.to_csv(os.path.join(k_dir, "sweep_results.csv"), index=False)
                all_k_results.append(combined_k)

                # Generate per-k summary plots (one per swept param)
                if sweep_params:
                    from src.utils.plotting import plot_sae_sweep_summary

                    for param_name in sweep_params:
                        plot_sae_sweep_summary(
                            combined_k,
                            sweep_param=param_name,
                            save_path=os.path.join(k_dir, f"sweep_{param_name}.png"),
                            title_prefix=f"k={k_val}",
                        )

        # Root-level summary if multiple k values
        if len(k_values) > 1 and all_k_results:
            full_combined = pd.concat(all_k_results, ignore_index=True)
            full_combined.to_csv(
                os.path.join(sae_dir, "sweep_results.csv"), index=False
            )

            from src.utils.plotting import plot_sae_sweep_summary

            # Summary per swept param across k values
            sweep_params_with_k = {"k": k_values, **sweep_params}
            for param_name in sweep_params_with_k:
                plot_sae_sweep_summary(
                    full_combined,
                    sweep_param=param_name,
                    save_path=os.path.join(sae_dir, f"sweep_{param_name}.png"),
                    title_prefix="all k",
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
