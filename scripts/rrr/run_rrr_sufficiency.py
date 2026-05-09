"""
run_rrr_sufficiency.py — RRR/PCA rank sufficiency bootstrap.

Loads saved real RRR fold bases (A.npy), or recomputes fold PCA bases from the
training embeddings, trains full and rank-k ridge models once on the reference
folds, builds out-of-fold predictions, then uses a paired block bootstrap over
words to estimate uncertainty in:

    T_k = max_target fisher_z(r_full_target) - fisher_z(r_rank_k_target)

Rank k is sufficient when the upper confidence bound for T_k is below an
epsilon margin.

Here "full" is the full transformed basis space for the current fold
(`X @ A_full`), so RRR sufficiency compares the full RRR-space embedding
representation against its first-k subspace.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core.config import RRRSufficiencyConfig, ScriptMode, get_results_dir
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils import build_XY_from_subjects, load_embeddings
from src.utils.config_utils import (
    expand_per_model_layer_configs,
    load_config,
    parse_args,
)
from src.utils.data_loaders import load_sig_electrodes
from src.utils.embeddings import get_glove_embeddings
from src.utils.encoding import build_per_fold_baselines, load_encoding_results
from src.utils.rrr_sufficiency import (
    bootstrap_full_model_ci_widths,
    ci_bounds,
    is_sufficient,
    paired_corr_delta_bootstrap,
)
from src.utils.subspace_utils import evaluate_input_space

OOF_PREDICTION_VERSION = 2


def _reference_test_idxs(output_dir: str, subject_ids: list[int]) -> list[np.ndarray]:
    enc_results = load_encoding_results(
        os.path.join(
            output_dir, "subjects", f"sub_{subject_ids[0]}", "encoding_results.joblib"
        )
    )
    return enc_results["test_idxs"]


def _resolve_rrr_source_dir(config: RRRSufficiencyConfig, output_dir: str) -> str:
    if config.rrr_source_dir:
        return config.rrr_source_dir
    if config.mode == ScriptMode.LEAVE_ONE_SUBJECT_OUT:
        return os.path.join(output_dir, "loso_rrr")
    return os.path.join(output_dir, "rrr")


def _resolve_pca_sibling_dir(rrr_dir: str) -> str:
    for rrr_part, pca_part in [
        ("per_subject_rrr", "per_subject_pca"),
        ("per_region_rrr", "per_region_pca"),
        ("loso_rrr", "loso_pca"),
        ("/rrr", "/pca"),
    ]:
        if rrr_part in rrr_dir:
            return rrr_dir.replace(rrr_part, pca_part, 1)
    return rrr_dir + "_pca"


def _validate_subspace_types(subspace_types: list[str] | str) -> list[str]:
    if isinstance(subspace_types, str):
        subspace_types = [subspace_types]
    normalized = []
    for subspace_type in subspace_types:
        value = str(subspace_type).lower()
        if value not in {"rrr", "pca"}:
            raise ValueError(
                f"Unsupported subspace type '{subspace_type}'. Expected 'rrr' or 'pca'."
            )
        if value not in normalized:
            normalized.append(value)
    return normalized


def _resolve_held_out_subjects(
    config: RRRSufficiencyConfig, subject_ids: list[int]
) -> list[int]:
    if config.held_out_subjects is None:
        return list(subject_ids)

    held_out_subjects = [int(subject) for subject in config.held_out_subjects]
    unknown = [subject for subject in held_out_subjects if subject not in subject_ids]
    if unknown:
        raise ValueError(
            "held_out_subjects must be a subset of encoding.subjects/default "
            f"subjects. Unknown held-out subject(s): {unknown}; "
            f"available subjects: {subject_ids}"
        )
    return held_out_subjects


def _check_fold_bases(rrr_source_dir: str, n_folds: int) -> None:
    missing = [
        os.path.join(rrr_source_dir, f"fold_{fold_f}", "A.npy")
        for fold_f in range(n_folds)
        if not os.path.exists(os.path.join(rrr_source_dir, f"fold_{fold_f}", "A.npy"))
    ]
    if missing:
        raise FileNotFoundError(
            "Missing saved RRR fold basis file(s): " + ", ".join(missing)
        )


def _load_or_compute_basis(
    subspace_type: str,
    basis_source_dir: str | None,
    fold_f: int,
    X_train: np.ndarray,
) -> np.ndarray:
    if subspace_type == "rrr":
        if basis_source_dir is None:
            raise ValueError("basis_source_dir is required for RRR sufficiency")
        return np.load(os.path.join(basis_source_dir, f"fold_{fold_f}", "A.npy"))
    if subspace_type == "pca":
        X_mean = X_train.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_train - X_mean, full_matrices=False)
        return Vt.T
    raise ValueError(f"Unsupported subspace type: {subspace_type}")


def _empty_oof_like(Y: np.ndarray) -> np.ndarray:
    return np.full(Y.shape, np.nan, dtype=np.float32)


def _oof_predictions_path(suff_dir: str) -> str:
    return os.path.join(suff_dir, "oof_predictions.npz")


def _oof_cache_version_ok(suff_dir: str) -> bool:
    path = _oof_predictions_path(suff_dir)
    if not os.path.exists(path):
        return False
    try:
        with np.load(path) as payload:
            version = int(payload["oof_prediction_version"])
    except (KeyError, OSError, ValueError):
        return False
    return version == OOF_PREDICTION_VERSION


def _load_oof_predictions(
    suff_dir: str,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    path = _oof_predictions_path(suff_dir)
    with np.load(path) as payload:
        version = int(payload["oof_prediction_version"])
        if version != OOF_PREDICTION_VERSION:
            raise ValueError(
                f"Cached OOF predictions at {path} are version {version}; "
                f"expected {OOF_PREDICTION_VERSION}. Rebuild with "
                "refresh_oof_predictions=true."
            )
        Y_oof = payload["Y_oof"]
        Yhat_full_oof = payload["Yhat_full_oof"]
        Yhat_rank_oof = {}
        for key in payload.files:
            if key.startswith("Yhat_rank_k"):
                k = int(key.replace("Yhat_rank_k", ""))
                Yhat_rank_oof[k] = payload[key]
    return Y_oof, Yhat_full_oof, Yhat_rank_oof


def _write_oof_predictions(
    suff_dir: str,
    Y_oof: np.ndarray,
    Yhat_full_oof: np.ndarray,
    Yhat_rank_oof: dict[int, np.ndarray],
) -> str:
    os.makedirs(suff_dir, exist_ok=True)
    path = _oof_predictions_path(suff_dir)
    payload = {
        "oof_prediction_version": np.array(OOF_PREDICTION_VERSION, dtype=np.int64),
        "Y_oof": Y_oof,
        "Yhat_full_oof": Yhat_full_oof,
        **{f"Yhat_rank_k{int(k)}": value for k, value in Yhat_rank_oof.items()},
    }
    np.savez(path, **payload)
    return path


def _should_load_cached_oof(output_unit_dir: str, config: RRRSufficiencyConfig) -> bool:
    return (
        config.cache_oof_predictions
        and not config.refresh_oof_predictions
        and _oof_cache_version_ok(
            os.path.join(output_unit_dir, "sufficiency_bootstrap")
        )
    )


def build_oof_predictions(
    X_raw: np.ndarray,
    Y_full: np.ndarray,
    test_idxs: list[np.ndarray],
    basis_source_dir: str | None,
    rank_grid: list[int],
    config: RRRSufficiencyConfig,
    out_dir: str | None = None,
    subspace_type: str = "rrr",
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Train full/rank ridge models once per fold and return OOF predictions."""
    subspace_type = subspace_type.lower()
    if subspace_type == "rrr" and basis_source_dir is not None:
        _check_fold_bases(basis_source_dir, len(test_idxs))

    enc = config.encoding
    alphas = np.array(enc.ridge_alphas, dtype=float) if enc.ridge_alphas else None

    per_fold_baselines = build_per_fold_baselines(
        X_raw,
        Y_full,
        test_idxs,
        alphas=alphas,
        ridge_cv_folds=enc.ridge_cv_folds,
        ridge_solver_params=enc.ridge_solver_params,
        ridge_y_in_cpu=enc.ridge_y_in_cpu,
        ridge_force_cpu=enc.ridge_force_cpu,
        alpha_policy=enc.alpha_grid,
        alpha_cache_dir=out_dir,
        alpha_context={
            "script": "run_rrr_sufficiency",
            "model": enc.model,
            "per_model_config": enc.per_model_config,
            "offset": enc.offset,
        },
    )

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Y_oof = _empty_oof_like(Y_full)
    Yhat_full_oof = _empty_oof_like(Y_full)
    Yhat_rank_oof = {int(k): _empty_oof_like(Y_full) for k in rank_grid}

    for fold_f, baseline in enumerate(per_fold_baselines):
        print(f"\n[Fold {fold_f + 1}/{len(test_idxs)}]")
        test_idx = baseline["test_idx"]
        train_idx = baseline["train_idx"]
        X_train = X_raw[train_idx]
        X_test = X_raw[test_idx]
        Y_train = baseline["Y_train"]
        Y_test = baseline["Y_test"]
        A_full = _load_or_compute_basis(
            subspace_type, basis_source_dir, fold_f, X_train
        )
        max_rank = A_full.shape[1]
        X_full_train = X_train @ A_full
        X_full_test = X_test @ A_full
        full_result = evaluate_input_space(
            X_full_train,
            X_full_test,
            Y_train,
            Y_test,
            device=device,
            alphas=alphas,
            ridge_cv_folds=enc.ridge_cv_folds,
            standardize=config.standardize_subspace_inputs,
            ridge_solver_params=enc.ridge_solver_params,
            ridge_y_in_cpu=enc.ridge_y_in_cpu,
            ridge_force_cpu=enc.ridge_force_cpu,
            subspace_name="full_basis",
        )
        Y_oof[test_idx] = Y_test
        Yhat_full_oof[test_idx] = full_result["full_basis_Y_hat"].astype(np.float32)

        valid_ranks = [int(k) for k in rank_grid if int(k) <= max_rank]
        skipped = [int(k) for k in rank_grid if int(k) > max_rank]
        if skipped:
            print(
                f"  Skipping ranks {skipped}; {subspace_type.upper()} basis has "
                f"only {max_rank} columns"
            )

        for k in tqdm(valid_ranks, desc=f"  fold {fold_f}", unit="rank"):
            X_proj_train = X_train @ A_full[:, :k]
            X_proj_test = X_test @ A_full[:, :k]
            result = evaluate_input_space(
                X_proj_train,
                X_proj_test,
                Y_train,
                Y_test,
                device=device,
                alphas=alphas,
                ridge_cv_folds=enc.ridge_cv_folds,
                standardize=config.standardize_subspace_inputs,
                ridge_solver_params=enc.ridge_solver_params,
                ridge_y_in_cpu=enc.ridge_y_in_cpu,
                ridge_force_cpu=enc.ridge_force_cpu,
                subspace_name="rank",
            )
            Yhat_rank_oof[k][test_idx] = result["rank_Y_hat"].astype(np.float32)

    return Y_oof, Yhat_full_oof, Yhat_rank_oof


def run_bootstrap(
    Y_oof: np.ndarray,
    Yhat_full_oof: np.ndarray,
    Yhat_rank_oof: dict[int, np.ndarray],
    config: RRRSufficiencyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict, dict]:
    rng = np.random.default_rng(config.random_seed)
    full_uncertainty = bootstrap_full_model_ci_widths(
        Y_oof,
        Yhat_full_oof,
        n_bootstrap=config.n_bootstrap,
        block_size=config.block_size,
        ci=config.ci,
        rng=rng,
    )
    median_width = full_uncertainty["median_ci_width"]
    mean_width = full_uncertainty["mean_ci_width"]
    suggested_eps = {
        "suggested_epsilon_25pct_median_full_ci": 0.25 * median_width,
        "suggested_epsilon_50pct_median_full_ci": 0.50 * median_width,
    }

    rows = []
    per_target_rows = []
    target_payload = {}
    boot_payload = {}
    for k in sorted(Yhat_rank_oof):
        if not np.isfinite(Yhat_rank_oof[k]).all():
            print(f"Rank {k}: incomplete OOF predictions; skipping bootstrap")
            continue

        print(f"\nBootstrapping rank {k} ({config.n_bootstrap} samples)")
        boot = paired_corr_delta_bootstrap(
            Y_oof,
            Yhat_full_oof,
            Yhat_rank_oof[k],
            n_bootstrap=config.n_bootstrap,
            block_size=config.block_size,
            rng=rng,
            save_per_target=True,
        )
        lo, hi = ci_bounds(boot["max_delta"], config.ci)
        obs = boot["observed"]
        z_full = obs["z_full"]
        denom = np.maximum(np.abs(z_full), 1e-6)
        normalized_delta = obs["delta"] / denom
        observed_max_fraction = (
            float(np.nanmax(normalized_delta))
            if np.isfinite(normalized_delta).any()
            else float("nan")
        )
        if boot["target_delta"] is not None:
            boot_fraction = boot["target_delta"] / denom[np.newaxis, :]
            boot_max_fraction = np.nanmax(boot_fraction, axis=1)
            _, fraction_hi = ci_bounds(boot_max_fraction, config.ci)
            alpha = (1.0 - config.ci) / 2.0
            target_delta_lo = np.nanquantile(boot["target_delta"], alpha, axis=0)
            target_delta_hi = np.nanquantile(boot["target_delta"], 1.0 - alpha, axis=0)
            target_fraction_lo = np.nanquantile(boot_fraction, alpha, axis=0)
            target_fraction_hi = np.nanquantile(boot_fraction, 1.0 - alpha, axis=0)
        else:
            boot_max_fraction = np.full(config.n_bootstrap, np.nan)
            fraction_hi = float("nan")
            target_delta_lo = np.full_like(obs["delta"], np.nan, dtype=float)
            target_delta_hi = np.full_like(obs["delta"], np.nan, dtype=float)
            target_fraction_lo = np.full_like(obs["delta"], np.nan, dtype=float)
            target_fraction_hi = np.full_like(obs["delta"], np.nan, dtype=float)

        row = {
            "k": int(k),
            "observed_max_delta": obs["max_delta"],
            "ci_lower": lo,
            "ci_upper": hi,
            "bootstrap_p_gt_0": float(np.nanmean(boot["max_delta"] > 0.0)),
            "bootstrap_percentile_observed": float(
                np.nanmean(boot["max_delta"] <= obs["max_delta"])
            ),
            "full_model_median_ci_width": median_width,
            "full_model_mean_ci_width": mean_width,
            **suggested_eps,
            "epsilon_fraction_of_full": config.epsilon_fraction_of_full,
            "observed_max_fraction_of_full": observed_max_fraction,
            "ci_upper_fraction_of_full": fraction_hi,
            "sufficient_fraction_of_full": is_sufficient(
                fraction_hi, config.epsilon_fraction_of_full
            ),
        }
        if config.track_per_target_sufficiency:
            row["median_target_ci_upper_delta"] = float(np.nanmedian(target_delta_hi))
            row["mean_target_ci_upper_delta"] = float(np.nanmean(target_delta_hi))
            row["median_target_ci_upper_fraction_of_full"] = float(
                np.nanmedian(target_fraction_hi)
            )
            row["mean_target_ci_upper_fraction_of_full"] = float(
                np.nanmean(target_fraction_hi)
            )
            row["fraction_targets_sufficient_fraction_of_full"] = float(
                np.nanmean(target_fraction_hi < config.epsilon_fraction_of_full)
            )
        for eps in config.epsilon_values:
            suffix = str(eps).replace(".", "p")
            row[f"p_max_delta_ge_epsilon_{suffix}"] = float(
                np.nanmean(boot["max_delta"] >= eps)
            )
            row[f"sufficient_epsilon_{suffix}"] = is_sufficient(hi, eps)
            if config.track_per_target_sufficiency:
                row[f"fraction_targets_sufficient_epsilon_{suffix}"] = float(
                    np.nanmean(target_delta_hi < eps)
                )
        rows.append(row)

        if config.track_per_target_sufficiency:
            for target_idx in range(obs["delta"].shape[0]):
                target_row = {
                    "k": int(k),
                    "target_idx": int(target_idx),
                    "r_full": obs["r_full"][target_idx],
                    "r_rank": obs["r_rank"][target_idx],
                    "z_full": obs["z_full"][target_idx],
                    "z_rank": obs["z_rank"][target_idx],
                    "delta": obs["delta"][target_idx],
                    "delta_ci_lower": target_delta_lo[target_idx],
                    "delta_ci_upper": target_delta_hi[target_idx],
                    "delta_fraction_of_full": normalized_delta[target_idx],
                    "delta_fraction_ci_lower": target_fraction_lo[target_idx],
                    "delta_fraction_ci_upper": target_fraction_hi[target_idx],
                    "sufficient_fraction_of_full": is_sufficient(
                        target_fraction_hi[target_idx],
                        config.epsilon_fraction_of_full,
                    ),
                }
                for eps in config.epsilon_values:
                    suffix = str(eps).replace(".", "p")
                    target_row[f"sufficient_epsilon_{suffix}"] = is_sufficient(
                        target_delta_hi[target_idx], eps
                    )
                per_target_rows.append(target_row)

        target_payload[f"r_full_k{k}"] = obs["r_full"]
        target_payload[f"r_rank_k{k}"] = obs["r_rank"]
        target_payload[f"z_full_k{k}"] = obs["z_full"]
        target_payload[f"z_rank_k{k}"] = obs["z_rank"]
        target_payload[f"delta_k{k}"] = obs["delta"]
        target_payload[f"delta_fraction_of_full_k{k}"] = normalized_delta
        boot_payload[f"max_delta_k{k}"] = boot["max_delta"]
        if boot["target_delta"] is not None:
            boot_payload[f"target_delta_k{k}"] = boot["target_delta"]
            boot_payload[f"max_delta_fraction_of_full_k{k}"] = boot_max_fraction

    summary = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    per_target_summary = (
        pd.DataFrame(per_target_rows)
        .sort_values(["k", "target_idx"])
        .reset_index(drop=True)
        if config.track_per_target_sufficiency
        else None
    )
    boot_payload = {
        **boot_payload,
        "full_model_z_boot": full_uncertainty["z_boot"],
        "full_model_ci_width": full_uncertainty["ci_width"],
    }
    return summary, per_target_summary, target_payload, boot_payload


def _write_bootstrap_outputs(
    suff_dir: str,
    summary: pd.DataFrame,
    per_target_summary: pd.DataFrame | None,
    target_payload: dict,
    boot_payload: dict,
    metadata: dict,
) -> None:
    os.makedirs(suff_dir, exist_ok=True)
    summary.to_csv(os.path.join(suff_dir, "sufficiency_summary.csv"), index=False)
    if per_target_summary is not None:
        per_target_summary.to_csv(
            os.path.join(suff_dir, "per_target_sufficiency.csv"), index=False
        )
    np.savez(os.path.join(suff_dir, "target_deltas.npz"), **target_payload)
    np.savez(os.path.join(suff_dir, "bootstrap_distribution.npz"), **boot_payload)
    with open(os.path.join(suff_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\nWrote sufficiency outputs to {suff_dir}")


def _run_sufficiency_unit(
    unit_label: str,
    X_raw: np.ndarray | None,
    Y_full: np.ndarray | None,
    test_idxs: list[np.ndarray],
    subspace_type: str,
    basis_source_dir: str | None,
    output_unit_dir: str,
    config: RRRSufficiencyConfig,
    metadata_base: dict,
) -> None:
    print(f"\n{'=' * 60}\nSufficiency unit: {unit_label} [{subspace_type.upper()}]")
    suff_dir = os.path.join(output_unit_dir, "sufficiency_bootstrap")
    oof_path = _oof_predictions_path(suff_dir)
    loaded_from_cache = (
        config.cache_oof_predictions
        and not config.refresh_oof_predictions
        and _oof_cache_version_ok(suff_dir)
    )
    if loaded_from_cache:
        print(f"Loading cached OOF predictions from {oof_path}")
        Y_oof, Yhat_full_oof, Yhat_rank_oof = _load_oof_predictions(suff_dir)
    else:
        if X_raw is None or Y_full is None:
            raise ValueError(
                f"OOF prediction cache is unavailable for {unit_label}; "
                "X_raw and Y_full are required to build predictions."
            )
        Y_oof, Yhat_full_oof, Yhat_rank_oof = build_oof_predictions(
            X_raw,
            Y_full,
            test_idxs,
            basis_source_dir,
            config.rank_grid,
            config,
            out_dir=suff_dir,
            subspace_type=subspace_type,
        )
        if config.cache_oof_predictions:
            oof_path = _write_oof_predictions(
                suff_dir, Y_oof, Yhat_full_oof, Yhat_rank_oof
            )
            print(f"Cached OOF predictions to {oof_path}")

    summary, per_target_summary, target_payload, boot_payload = run_bootstrap(
        Y_oof, Yhat_full_oof, Yhat_rank_oof, config
    )
    _write_bootstrap_outputs(
        suff_dir,
        summary,
        per_target_summary,
        target_payload,
        boot_payload,
        {
            **metadata_base,
            "unit_label": unit_label,
            "subspace_type": subspace_type,
            "basis_source_dir": basis_source_dir,
            "output_unit_dir": output_unit_dir,
            "sufficiency_dir": suff_dir,
            "oof_predictions_path": oof_path if config.cache_oof_predictions else None,
            "oof_prediction_version": OOF_PREDICTION_VERSION,
            "cache_oof_predictions": bool(config.cache_oof_predictions),
            "refresh_oof_predictions": bool(config.refresh_oof_predictions),
            "loaded_oof_predictions_from_cache": bool(loaded_from_cache),
            "n_words": int(Y_oof.shape[0]),
            "n_targets": int(Y_oof.shape[1]),
            "n_folds": int(len(test_idxs)),
        },
    )


def _build_xy(subject_ids, enc, sig_electrodes, aligned_embeddings, word_in_glove):
    unit_sig_electrodes = {
        sub_id: sig_electrodes.get(sub_id, [])
        for sub_id in subject_ids
        if sig_electrodes.get(sub_id, [])
    }
    if not unit_sig_electrodes:
        return None
    return build_XY_from_subjects(
        subject_ids,
        enc,
        unit_sig_electrodes,
        aligned_embeddings,
        valid_rows=word_in_glove,
    )[:2]


def _run_cross_subject_sufficiency(
    config: RRRSufficiencyConfig,
    enc,
    output_dir: str,
    rrr_source_dir: str,
    subspace_types: list[str],
    subject_ids: list[int],
    sig_electrodes: dict,
    aligned_embeddings: np.ndarray,
    word_in_glove,
    metadata_base: dict,
) -> None:
    test_idxs = _reference_test_idxs(output_dir, subject_ids)
    xy = None

    for subspace_type in subspace_types:
        if subspace_type == "rrr":
            basis_source_dir = rrr_source_dir
            output_unit_dir = rrr_source_dir
        else:
            basis_source_dir = None
            output_unit_dir = _resolve_pca_sibling_dir(rrr_source_dir)
        if xy is None and not _should_load_cached_oof(output_unit_dir, config):
            xy = _build_xy(
                subject_ids, enc, sig_electrodes, aligned_embeddings, word_in_glove
            )
            if xy is None:
                print("No significant electrodes found for cross-subject sufficiency.")
                return
            print(f"Cross-subject X: {xy[0].shape}, Y: {xy[1].shape}")
        _run_sufficiency_unit(
            "cross_subject",
            xy[0] if xy is not None else None,
            xy[1] if xy is not None else None,
            test_idxs,
            subspace_type,
            basis_source_dir,
            output_unit_dir,
            config,
            metadata_base,
        )


def _run_loso_sufficiency(
    config: RRRSufficiencyConfig,
    enc,
    output_dir: str,
    rrr_source_dir: str,
    subspace_types: list[str],
    subject_ids: list[int],
    sig_electrodes: dict,
    aligned_embeddings: np.ndarray,
    word_in_glove,
    metadata_base: dict,
) -> None:
    pca_root_dir = _resolve_pca_sibling_dir(rrr_source_dir)
    held_out_subjects = _resolve_held_out_subjects(config, subject_ids)
    for held_out_sub in held_out_subjects:
        print(f"\n{'#' * 60}\nHeld-out subject: {held_out_sub}")
        train_subs = [s for s in subject_ids if s != held_out_sub]
        test_idxs = _reference_test_idxs(output_dir, train_subs)

        loso_rrr_sub_dir = os.path.join(rrr_source_dir, f"held_out_sub_{held_out_sub}")
        loso_pca_sub_dir = os.path.join(pca_root_dir, f"held_out_sub_{held_out_sub}")

        for subspace_type in subspace_types:
            if subspace_type == "rrr":
                train_basis_dir = os.path.join(loso_rrr_sub_dir, "train_subjects")
                train_output_dir = train_basis_dir
                transfer_basis_dir = train_basis_dir
                transfer_output_dir = os.path.join(loso_rrr_sub_dir, "transfer")
                baseline_basis_dir = os.path.join(loso_rrr_sub_dir, "baseline")
                baseline_output_dir = baseline_basis_dir
            else:
                train_basis_dir = None
                train_output_dir = os.path.join(loso_pca_sub_dir, "train_subjects")
                transfer_basis_dir = None
                transfer_output_dir = os.path.join(loso_pca_sub_dir, "transfer")
                baseline_basis_dir = None
                baseline_output_dir = os.path.join(loso_pca_sub_dir, "baseline")

            train_xy = None
            if not _should_load_cached_oof(train_output_dir, config):
                train_xy = _build_xy(
                    train_subs, enc, sig_electrodes, aligned_embeddings, word_in_glove
                )
            if train_xy is not None:
                _run_sufficiency_unit(
                    f"held_out_sub_{held_out_sub}/train_subjects",
                    train_xy[0],
                    train_xy[1],
                    test_idxs,
                    subspace_type,
                    train_basis_dir,
                    train_output_dir,
                    config,
                    {
                        **metadata_base,
                        "held_out_subject": int(held_out_sub),
                        "subjects_in_unit": train_subs,
                    },
                )
                del train_xy
            elif _should_load_cached_oof(train_output_dir, config):
                _run_sufficiency_unit(
                    f"held_out_sub_{held_out_sub}/train_subjects",
                    None,
                    None,
                    test_idxs,
                    subspace_type,
                    train_basis_dir,
                    train_output_dir,
                    config,
                    {
                        **metadata_base,
                        "held_out_subject": int(held_out_sub),
                        "subjects_in_unit": train_subs,
                    },
                )
            else:
                print(
                    "  No training-subject significant electrodes; skipping train unit."
                )

            transfer_cached = _should_load_cached_oof(transfer_output_dir, config)
            baseline_cached = _should_load_cached_oof(baseline_output_dir, config)
            held_xy = None
            if not (transfer_cached and baseline_cached):
                held_xy = _build_xy(
                    [held_out_sub],
                    enc,
                    sig_electrodes,
                    aligned_embeddings,
                    word_in_glove,
                )
            if held_xy is not None:
                _run_sufficiency_unit(
                    f"held_out_sub_{held_out_sub}/transfer",
                    held_xy[0],
                    held_xy[1],
                    test_idxs,
                    subspace_type,
                    transfer_basis_dir,
                    transfer_output_dir,
                    config,
                    {
                        **metadata_base,
                        "held_out_subject": int(held_out_sub),
                        "subjects_in_unit": [held_out_sub],
                        "transfer_basis_subjects": train_subs,
                    },
                )
                _run_sufficiency_unit(
                    f"held_out_sub_{held_out_sub}/baseline",
                    held_xy[0],
                    held_xy[1],
                    test_idxs,
                    subspace_type,
                    baseline_basis_dir,
                    baseline_output_dir,
                    config,
                    {
                        **metadata_base,
                        "held_out_subject": int(held_out_sub),
                        "subjects_in_unit": [held_out_sub],
                    },
                )
                del held_xy
            elif transfer_cached and baseline_cached:
                _run_sufficiency_unit(
                    f"held_out_sub_{held_out_sub}/transfer",
                    None,
                    None,
                    test_idxs,
                    subspace_type,
                    transfer_basis_dir,
                    transfer_output_dir,
                    config,
                    {
                        **metadata_base,
                        "held_out_subject": int(held_out_sub),
                        "subjects_in_unit": [held_out_sub],
                        "transfer_basis_subjects": train_subs,
                    },
                )
                _run_sufficiency_unit(
                    f"held_out_sub_{held_out_sub}/baseline",
                    None,
                    None,
                    test_idxs,
                    subspace_type,
                    baseline_basis_dir,
                    baseline_output_dir,
                    config,
                    {
                        **metadata_base,
                        "held_out_subject": int(held_out_sub),
                        "subjects_in_unit": [held_out_sub],
                    },
                )
            else:
                print(
                    "  Held-out subject has no significant electrodes; skipping held-out units."
                )


def run_config(config: RRRSufficiencyConfig) -> None:
    start = time.time()
    enc = config.encoding
    subspace_types = _validate_subspace_types(config.subspace_types)
    subject_ids = enc.subjects or list(range(1, 10))
    held_out_subjects = (
        _resolve_held_out_subjects(config, subject_ids)
        if config.mode == ScriptMode.LEAVE_ONE_SUBJECT_OUT
        else None
    )
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
    sig_dir = config.sig_dir or output_dir
    rrr_source_dir = _resolve_rrr_source_dir(config, output_dir)

    print(f"Output dir:       {output_dir}")
    print(f"RRR source dir:   {rrr_source_dir}")
    print(f"Mode:             {config.mode.value}")
    print(f"Subspace types:   {subspace_types}")
    if held_out_subjects is not None:
        print(f"Held-out subjects:{held_out_subjects}")

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
    aligned_embeddings, word_df = load_embeddings(
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
            enc.data_root, word_df, **enc.per_model_config
        )[1]["in_glove"]

    metadata_base = {
        "config": asdict(config),
        "output_dir": output_dir,
        "sig_dir": sig_dir,
        "rrr_source_dir": rrr_source_dir,
        "mode": config.mode.value,
        "subject_ids": subject_ids,
        "held_out_subjects_requested": config.held_out_subjects,
        "held_out_subjects_run": held_out_subjects,
    }

    if config.mode == ScriptMode.LEAVE_ONE_SUBJECT_OUT:
        _run_loso_sufficiency(
            config,
            enc,
            output_dir,
            rrr_source_dir,
            subspace_types,
            subject_ids,
            sig_electrodes,
            aligned_embeddings,
            word_in_glove,
            metadata_base,
        )
    elif config.mode == ScriptMode.CROSS_SUBJECT:
        _run_cross_subject_sufficiency(
            config,
            enc,
            output_dir,
            rrr_source_dir,
            subspace_types,
            subject_ids,
            sig_electrodes,
            aligned_embeddings,
            word_in_glove,
            metadata_base,
        )
    else:
        raise ValueError(
            "run_rrr_sufficiency.py currently supports cross_subject and "
            "leave_one_subject_out modes only."
        )

    print(f"\nDone in {time.time() - start:.1f}s.")


def main() -> None:
    args, overrides = parse_args("RRR sufficiency paired block bootstrap")
    config: RRRSufficiencyConfig = load_config(
        args.config, overrides, RRRSufficiencyConfig
    )
    configs = expand_per_model_layer_configs(config)
    for layer_config in configs:
        if len(configs) > 1:
            print(
                f"\nRunning RRR sufficiency for layer "
                f"{layer_config.encoding.per_model_config['layer']}"
            )
        run_config(layer_config)


if __name__ == "__main__":
    main()
