"""
train_encoding_models.py — Joint or batched encoding model training.

Trains ridge regression encoding models (GPT-2 embeddings → ECoG neural data).
Layer is specified as an integer in the config (24=mid, 48=final).

Writes per-subject results to:
  {run_name}/subjects/sub_{id}/
    encoding_results.joblib
    corrs.npy
    arbitrary_corrs.npy
    corrs_plot.png
"""

# TODO: FIX THIS FILE FOLD PROCESSING

import gc
import os

import numpy as np
from src.utils.data_loaders import load_data_from_cache
import torch
from himalaya.backend import set_backend

from src.utils import (
    train_encoding,
    get_and_offset_arbitrary_embeddings,
    get_models_coefficients,
    load_embeddings,
    get_glove_embeddings,
    apply_offset,
    save_subject_results,
    read_electrode_file,
    RIDGE_ALPHAS,
    load_precomputed_epochs,
    epochs_from_cache,
)
from src.core.config import EncodingConfig, get_results_dir
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.config_utils import parse_args, load_config
from src.utils.constants import check_alpha_grid_boundaries
from src.utils.encoding import (
    get_and_offset_arbitrary_embeddings,
    get_ridge_model_constructor,
    extract_best_alphas,
    train_encoding_two_pass,
)
from src.utils.plotting import plot_alpha_diagnostics


# ---------------------------------------------------------------------------
# Script-local helpers
# ---------------------------------------------------------------------------


def _train_with_arbitrary(
    X_arr,
    Y,
    n_elecs,
    n_lags,
    model_constructor,
    selection,
    word_df,
    offset,
    config: EncodingConfig = None,
    alphas=None,
):
    """Train real encoding model then null (arbitrary) model on same folds.

    When config.refine_alphas is True, uses a two-pass coarse→refined sweep
    for the real model. The arbitrary model always uses a single pass.
    """
    num_folds = config.num_folds if config is not None else 2
    X = [X_arr] * num_folds

    if config is not None and config.refine_alphas:
        (corrs, models, tr_idxs, test_idxs, scalers, coarse_best, refined_grid) = (
            train_encoding_two_pass(X, Y, alphas=alphas, epochs_shape=(n_elecs, n_lags))
        )
        print(
            f"  Two-pass refinement: coarse → refined grid "
            f"[{refined_grid[0]:.2e} .. {refined_grid[-1]:.2e}] ({len(refined_grid)} pts)"
        )
    else:
        corrs, models, tr_idxs, test_idxs, scalers = train_encoding(
            X,
            Y,
            model_constructor,
            epochs_shape=(n_elecs, n_lags),
            num_folds=config.num_folds if config is not None else 2,
        )

    original_folds = list(zip(tr_idxs, test_idxs))

    arb = get_and_offset_arbitrary_embeddings(
        X_arr, selection, word_df, offset, config.arbitrary_seed
    )

    corrs_arb, _, _, _, _ = train_encoding(
        [arb] * num_folds,
        Y,
        model_constructor,
        epochs_shape=(n_elecs, n_lags),
        provided_idxs=original_folds,
        num_folds=config.num_folds if config is not None else 2,
    )
    return corrs, corrs_arb, models, tr_idxs, test_idxs, scalers


def _log_alpha_diagnostics(models, alpha_grid, sub_dir, sub_label):
    """Extract best alphas from fitted models and save diagnostics."""
    per_fold_alphas = [extract_best_alphas(m) for m in models]
    os.makedirs(sub_dir, exist_ok=True)
    np.savez(
        os.path.join(sub_dir, "alpha_diagnostics.npz"),
        **{f"fold_{i}": a for i, a in enumerate(per_fold_alphas)},
    )
    plot_alpha_diagnostics(
        per_fold_alphas,
        alpha_grid,
        title=f"{sub_label}: alpha diagnostics",
        save_path=os.path.join(sub_dir, "alpha_diagnostics.png"),
    )
    all_alphas = np.concatenate(per_fold_alphas)
    boundary = check_alpha_grid_boundaries(all_alphas, alpha_grid)
    if boundary["hit_lower"] or boundary["hit_upper"]:
        print(
            f"  WARNING: Alpha grid boundary hit for {sub_label}: "
            f"lower={boundary['frac_lower']:.1%}, upper={boundary['frac_upper']:.1%}"
        )


# ---------------------------------------------------------------------------
# Joint mode
# ---------------------------------------------------------------------------


def _run_joint(
    sig_elecs,
    aligned_embeddings,
    word_df,
    data_root,
    offset,
    run_name,
    model_constructor,
    config: EncodingConfig,
    dataset_context,
    alphas=None,
    alpha_grid=None,
    word_in_glove=None,
):
    all_subject_ids = list(sig_elecs.keys())
    X_arr, Y, selection, elec_names = load_data_from_cache(
        all_subject_ids,
        data_root,
        offset,
        aligned_embeddings,
        sig_elecs,
        valid_rows=word_in_glove,
        dataset=config.dataset,
        per_dataset_config=config.per_dataset_config,
        dataset_context=dataset_context,
    )
    if word_in_glove is not None:
        selection = selection[word_in_glove[selection]]

    elec_counts = [len(x) for x in elec_names]
    total_elecs = sum(elec_counts)
    n_lags = Y.shape[1] // total_elecs

    lags = np.linspace(-2.0, 2.0, n_lags, endpoint=False)

    print(f"Training joint model: {total_elecs} electrodes, {X_arr.shape[0]} words...")
    corrs, corrs_arb, models, tr_idxs, test_idxs, scalers = _train_with_arbitrary(
        X_arr,
        Y,
        total_elecs,
        n_lags,
        model_constructor,
        selection,
        word_df,
        offset,
        config=config,
        alphas=alphas,
    )
    coefficients, intercepts = get_models_coefficients(
        models, total_elecs, n_lags, scaled_model=False
    )

    if config is not None and config.log_alphas and alpha_grid is not None:
        joint_dir = os.path.join(run_name, "joint")
        _log_alpha_diagnostics(models, alpha_grid, joint_dir, "Joint")

    cursor = 0
    for sub_id, n_elecs in zip(all_subject_ids, elec_counts):
        sl = slice(cursor, cursor + n_elecs)
        cursor += n_elecs
        coef_slices = [c[sl] for c in coefficients]
        int_slices = [i.reshape(total_elecs, n_lags)[sl].ravel() for i in intercepts]
        save_subject_results(
            run_name,
            sub_id,
            corrs[:, sl, :],
            coef_slices,
            int_slices,
            models,
            tr_idxs,
            test_idxs,
            scalers,
            sig_elecs[sub_id],
            offset,
            lags,
            corrs_arb=corrs_arb[:, sl, :],
            per_model_config=config.per_model_config,
        )


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------


def _run_batch(
    sig_elecs,
    subject_ids,
    aligned_embeddings,
    word_df,
    data_root,
    offset,
    run_name,
    model_constructor,
    config: EncodingConfig,
    dataset_context,
    alphas=None,
    alpha_grid=None,
    word_in_glove=None,
):
    for sub_id in subject_ids:
        if sub_id not in sig_elecs:
            print(f"Subject {sub_id}: no significant electrodes, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Processing subject {sub_id}")
        print(f"{'='*60}")
        X_arr, Y, selection, elec_names = load_data_from_cache(
            [sub_id],
            data_root,
            offset,
            aligned_embeddings,
            sig_elecs,
            valid_rows=word_in_glove,
            dataset=config.dataset,
            per_dataset_config=config.per_dataset_config,
            dataset_context=dataset_context,
        )
        elec_counts = [len(x) for x in elec_names]
        n_elecs = sum(elec_counts)
        n_lags = Y.shape[1] // n_elecs
        lags = np.linspace(-2.0, 2.0, n_lags, endpoint=False)

        corrs, corrs_arb, models, tr_idxs, test_idxs, scalers = _train_with_arbitrary(
            X_arr,
            Y,
            n_elecs,
            n_lags,
            model_constructor,
            selection,
            word_df,
            offset,
            config=config,
            alphas=alphas,
        )
        coefficients, intercepts = get_models_coefficients(
            models, n_elecs, n_lags, scaled_model=False
        )

        if config is not None and config.log_alphas and alpha_grid is not None:
            sub_dir = os.path.join(run_name, "subjects", f"sub_{sub_id}")
            _log_alpha_diagnostics(models, alpha_grid, sub_dir, f"Sub {sub_id}")

        save_subject_results(
            run_name,
            sub_id,
            corrs,
            coefficients,
            intercepts,
            models,
            tr_idxs,
            test_idxs,
            scalers,
            sig_elecs[sub_id],
            offset,
            lags,
            corrs_arb=corrs_arb,
            per_model_config=config.per_model_config,
        )

        del X_arr, Y, models, scalers, corrs, corrs_arb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args, overrides = parse_args("Encoding model training")
    config: EncodingConfig = load_config(args.config, overrides, EncodingConfig)

    subject_ids = config.subjects or list(range(1, 10))
    offset = config.offset
    dataset_context = resolve_dataset_context(
        dataset=config.dataset,
        subject_ids=subject_ids,
        per_dataset_config=config.per_dataset_config,
        data_root=config.data_root,
    )
    get_dataset_adapter(config.dataset).validate_subjects_share_stimulus(
        subject_ids, config.per_dataset_config, config.data_root
    )
    run_name = config.run_name or get_results_dir(
        config.model,
        config.offset,
        config.per_model_config,
        dataset_context=dataset_context,
    )

    if torch.cuda.is_available():
        set_backend("torch_cuda")
        print("Using torch_cuda backend.")
    else:
        print("CUDA not available, using CPU.")

    print(f"Results directory: {run_name}")

    sig_dir = config.sig_dir or run_name
    sig_elec_path = os.path.join(sig_dir, "all_subject_sig.csv")
    sig_elecs = read_electrode_file(sig_elec_path)
    sig_elecs = {s: sig_elecs[s] for s in subject_ids if s in sig_elecs}
    if not sig_elecs:
        raise ValueError("No significant electrodes found for the requested subjects.")

    print(f"Loading embeddings...")
    aligned_embeddings, word_df = load_embeddings(
        config.data_root,
        embedding_getter_name=config.model,
        dataset=config.dataset,
        per_dataset_config=config.per_dataset_config,
        dataset_context=dataset_context,
        **config.per_model_config,
    )
    word_in_glove = None
    if config.limit_to_glove or config.model == "glove":
        word_in_glove = get_glove_embeddings(
            config.data_root, word_df, **config.per_model_config
        )[1]["in_glove"]

    alphas = (
        np.array(config.ridge_alphas, dtype=float)
        if config.ridge_alphas is not None
        else None
    )
    alpha_grid = alphas if alphas is not None else RIDGE_ALPHAS
    model_constructor = get_ridge_model_constructor(alphas=alphas)

    if config.batch:
        _run_batch(
            sig_elecs,
            subject_ids,
            aligned_embeddings,
            word_df,
            config.data_root,
            offset,
            run_name,
            model_constructor,
            config=config,
            dataset_context=dataset_context,
            alphas=alphas,
            alpha_grid=alpha_grid,
            word_in_glove=word_in_glove,
        )
    else:
        _run_joint(
            sig_elecs,
            aligned_embeddings,
            word_df,
            config.data_root,
            offset,
            run_name,
            model_constructor,
            config=config,
            dataset_context=dataset_context,
            alphas=alphas,
            alpha_grid=alpha_grid,
            word_in_glove=word_in_glove,
        )


if __name__ == "__main__":
    main()
