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
from src.utils.data_loaders import (
    build_word_selection_metadata,
    compute_shared_word_selection,
    load_data_from_cache,
)
import torch
from himalaya.backend import set_backend

from src.utils import (
    train_encoding,
    load_embeddings,
    get_glove_embeddings,
    save_subject_results,
    read_electrode_file,
)
from src.core.config import EncodingConfig, get_results_dir
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.config_utils import (
    parse_args,
    load_config,
    expand_per_model_layer_configs,
)
from src.utils.constants import check_alpha_grid_boundaries
from src.utils.alpha_grids import resolve_ridge_alphas, save_alpha_diagnostics
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
            train_encoding_two_pass(
                X,
                Y,
                alphas=alphas,
                ridge_cv_folds=config.ridge_cv_folds,
                epochs_shape=(n_elecs, n_lags),
                solver_params=config.ridge_solver_params,
                y_in_cpu=config.ridge_y_in_cpu,
                force_cpu=config.ridge_force_cpu,
            )
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


def _log_alpha_diagnostics(models, alpha_grid, sub_dir, sub_label, context=None, policy=None):
    """Extract best alphas from fitted models and save diagnostics."""
    per_fold_alphas = [extract_best_alphas(m) for m in models]
    os.makedirs(sub_dir, exist_ok=True)
    save_alpha_diagnostics(
        sub_dir,
        alpha_grid,
        best_alphas=per_fold_alphas,
        context=context,
        policy=policy,
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
    shared_selection=None,
    selection_subject_ids=None,
):
    all_subject_ids = list(sig_elecs.keys())
    selection_subject_ids = list(selection_subject_ids or all_subject_ids)
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
        shared_selection=shared_selection,
    )
    epoch_selection_for_metadata = selection
    if word_in_glove is not None:
        selection = selection[word_in_glove[selection]]

    elec_counts = [len(x) for x in elec_names]
    total_elecs = sum(elec_counts)
    n_lags = Y.shape[1] // total_elecs

    lags = np.linspace(-2.0, 2.0, n_lags, endpoint=False)

    print(f"Training joint model: {total_elecs} electrodes, {X_arr.shape[0]} words...")
    joint_dir = os.path.join(run_name, "joint")
    alpha_context = {
        "script": "train_encoding_models",
        "role": "joint",
        "model": config.model,
        "per_model_config": config.per_model_config,
        "offset": config.offset,
        "n_features": int(X_arr.shape[1]),
        "standardize": True,
    }
    alpha_grid = resolve_ridge_alphas(
        alphas,
        X_arr,
        standardize=True,
        context=alpha_context,
        cache_dir=joint_dir,
        policy=config.alpha_grid,
    )
    model_constructor = get_ridge_model_constructor(
        alphas=alpha_grid,
        ridge_cv_folds=config.ridge_cv_folds,
        solver_params=config.ridge_solver_params,
        y_in_cpu=config.ridge_y_in_cpu,
        force_cpu=config.ridge_force_cpu,
    )
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
        alphas=alpha_grid,
    )
    if config is not None and config.log_alphas and alpha_grid is not None:
        _log_alpha_diagnostics(
            models,
            alpha_grid,
            joint_dir,
            "Joint",
            context=alpha_context,
            policy=config.alpha_grid,
        )

    cursor = 0
    for sub_id, n_elecs in zip(all_subject_ids, elec_counts):
        word_selection_metadata = None
        if config.shared_word_selection:
            word_selection_metadata = build_word_selection_metadata(
                "shared_intersection",
                selection_subject_ids,
                epoch_selection_for_metadata,
                offset,
                valid_rows=(
                    word_in_glove[epoch_selection_for_metadata]
                    if word_in_glove is not None
                    else None
                ),
            )
        sl = slice(cursor, cursor + n_elecs)
        cursor += n_elecs
        save_subject_results(
            run_name,
            sub_id,
            corrs[:, sl, :],
            tr_idxs,
            test_idxs,
            scalers,
            sig_elecs[sub_id],
            offset,
            lags,
            corrs_arb=corrs_arb[:, sl, :],
            per_model_config=config.per_model_config,
            word_selection_metadata=word_selection_metadata,
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
    shared_selection=None,
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
            shared_selection=shared_selection,
        )
        word_selection_metadata = None
        if config.shared_word_selection:
            word_selection_metadata = build_word_selection_metadata(
                "shared_intersection",
                subject_ids,
                selection,
                offset,
                valid_rows=(
                    word_in_glove[selection] if word_in_glove is not None else None
                ),
            )
        elec_counts = [len(x) for x in elec_names]
        n_elecs = sum(elec_counts)
        n_lags = Y.shape[1] // n_elecs
        lags = np.linspace(-2.0, 2.0, n_lags, endpoint=False)
        sub_dir = os.path.join(run_name, "subjects", f"sub_{sub_id}")
        alpha_context = {
            "script": "train_encoding_models",
            "role": "subject",
            "subject": int(sub_id),
            "model": config.model,
            "per_model_config": config.per_model_config,
            "offset": config.offset,
            "n_features": int(X_arr.shape[1]),
            "standardize": True,
        }
        alpha_grid = resolve_ridge_alphas(
            alphas,
            X_arr,
            standardize=True,
            context=alpha_context,
            cache_dir=sub_dir,
            policy=config.alpha_grid,
        )
        model_constructor = get_ridge_model_constructor(
            alphas=alpha_grid,
            ridge_cv_folds=config.ridge_cv_folds,
            solver_params=config.ridge_solver_params,
            y_in_cpu=config.ridge_y_in_cpu,
            force_cpu=config.ridge_force_cpu,
        )

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
            alphas=alpha_grid,
        )
        if config is not None and config.log_alphas and alpha_grid is not None:
            _log_alpha_diagnostics(
                models,
                alpha_grid,
                sub_dir,
                f"Sub {sub_id}",
                context=alpha_context,
                policy=config.alpha_grid,
            )

        save_subject_results(
            run_name,
            sub_id,
            corrs,
            tr_idxs,
            test_idxs,
            scalers,
            sig_elecs[sub_id],
            offset,
            lags,
            corrs_arb=corrs_arb,
            per_model_config=config.per_model_config,
            word_selection_metadata=word_selection_metadata,
        )

        del X_arr, Y, models, scalers, corrs, corrs_arb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_config(config: EncodingConfig):
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
    shared_selection = None
    if config.shared_word_selection:
        shared_selection = compute_shared_word_selection(
            subject_ids,
            config.data_root,
            dataset=config.dataset,
            per_dataset_config=config.per_dataset_config,
            dataset_context=dataset_context,
        )
        print(
            "Using shared word-selection intersection: "
            f"{len(shared_selection):,} cached words across {len(subject_ids)} subjects"
        )

    explicit_alphas = (
        np.array(config.ridge_alphas, dtype=float)
        if config.ridge_alphas is not None
        else None
    )

    if config.batch:
        _run_batch(
            sig_elecs,
            subject_ids,
            aligned_embeddings,
            word_df,
            config.data_root,
            offset,
            run_name,
            None,
            config=config,
            dataset_context=dataset_context,
            alphas=explicit_alphas,
            alpha_grid=None,
            word_in_glove=word_in_glove,
            shared_selection=shared_selection,
        )
    else:
        _run_joint(
            sig_elecs,
            aligned_embeddings,
            word_df,
            config.data_root,
            offset,
            run_name,
            None,
            config=config,
            dataset_context=dataset_context,
            alphas=explicit_alphas,
            alpha_grid=None,
            word_in_glove=word_in_glove,
            shared_selection=shared_selection,
            selection_subject_ids=subject_ids,
        )


def main():
    args, overrides = parse_args("Encoding model training")
    config: EncodingConfig = load_config(args.config, overrides, EncodingConfig)
    configs = expand_per_model_layer_configs(config)
    for layer_config in configs:
        if len(configs) > 1:
            print(
                f"\nRunning encoding training for layer "
                f"{layer_config.per_model_config['layer']}"
            )
        run_config(layer_config)


if __name__ == "__main__":
    main()
