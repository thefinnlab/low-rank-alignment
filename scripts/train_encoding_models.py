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

import gc
import os

import numpy as np
import torch
from himalaya.backend import set_backend

from utils import (
    load_raws,
    train_encoding,
    get_arbitrary_embeddings,
    get_models_coefficients,
    load_embeddings,
    build_word_epochs,
    apply_offset,
    save_subject_results,
    read_electrode_file,
)
from utils.config import EncodingConfig, get_results_dir
from utils.config_utils import parse_args, load_config
from utils.encoding import get_ridge_model_constructor


# ---------------------------------------------------------------------------
# Script-local helpers
# ---------------------------------------------------------------------------

def _train_with_arbitrary(X_arr, Y, n_elecs, n_lags, model_constructor,
                           averaged_embeddings, epochs, df_word, offset):
    """Train real encoding model then null (arbitrary) model on same folds."""
    X = [X_arr, X_arr]
    corrs, models, tr_idxs, test_idxs, scalers = train_encoding(
        X, Y, model_constructor, epochs_shape=(n_elecs, n_lags)
    )
    original_folds = list(zip(tr_idxs, test_idxs))

    arb = get_arbitrary_embeddings(averaged_embeddings, epochs, df_word).astype(np.float32)
    # Apply same length trim as _apply_offset does for X_arr
    if offset < 0:
        arb = arb[:offset]
    elif offset > 0:
        arb = arb[offset:]

    corrs_arb, _, _, _, _ = train_encoding(
        [arb, arb], Y, model_constructor,
        epochs_shape=(n_elecs, n_lags), provided_idxs=original_folds,
    )
    return corrs, corrs_arb, models, tr_idxs, test_idxs, scalers


# ---------------------------------------------------------------------------
# Joint mode
# ---------------------------------------------------------------------------

def _run_joint(sig_elecs, aligned_embeddings, df_word,
               data_root, offset, layer_int, run_name, model_constructor):
    raws = load_raws(list(sig_elecs.keys()), data_root,
                     use_high_gamma=True, per_subject_electrodes=sig_elecs)
    elec_counts = [len(r.ch_names) for r in raws]

    all_datas, epochs = [], None
    for raw in raws:
        ep, ep_data = build_word_epochs(raw, df_word)
        all_datas.append(ep_data)
        epochs = ep

    datas = np.concatenate(all_datas, axis=1)  # (n_words, total_elecs, n_lags)
    n_words, total_elecs, n_lags = datas.shape
    lags = np.linspace(-2.0, 2.0, n_lags, endpoint=False)

    avg_embs = aligned_embeddings[epochs.selection]
    X_arr, Y = apply_offset(avg_embs, datas, offset)

    print(f"Training joint model: {total_elecs} electrodes, {X_arr.shape[0]} words...")
    corrs, corrs_arb, models, tr_idxs, test_idxs, scalers = _train_with_arbitrary(
        X_arr, Y, total_elecs, n_lags, model_constructor, avg_embs, epochs, df_word, offset
    )
    coefficients, intercepts = get_models_coefficients(models, total_elecs, n_lags, scaled_model=False)

    cursor = 0
    for sub_id, raw, n_elecs in zip(sig_elecs.keys(), raws, elec_counts):
        sl = slice(cursor, cursor + n_elecs)
        cursor += n_elecs
        coef_slices = [c[sl] for c in coefficients]
        int_slices = [i.reshape(total_elecs, n_lags)[sl].ravel() for i in intercepts]
        save_subject_results(
            run_name, sub_id, corrs[:, sl, :],
            coef_slices, int_slices, models, tr_idxs, test_idxs,
            scalers, raw.ch_names, offset, layer_int, lags,
            corrs_arb=corrs_arb[:, sl, :],
        )


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def _run_batch(sig_elecs, subject_ids, aligned_embeddings, df_word,
               data_root, offset, layer_int, run_name, model_constructor):
    for sub_id in subject_ids:
        if sub_id not in sig_elecs:
            print(f"Subject {sub_id}: no significant electrodes, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Processing subject {sub_id}")
        print(f"{'='*60}")

        raws = load_raws([sub_id], data_root, use_high_gamma=True,
                         per_subject_electrodes={sub_id: sig_elecs[sub_id]})
        raw = raws[0]

        epochs, data = build_word_epochs(raw, df_word)
        n_words, n_elecs, n_lags = data.shape
        lags = np.linspace(-2.0, 2.0, n_lags, endpoint=False)

        avg_embs = aligned_embeddings[epochs.selection]
        X_arr, Y = apply_offset(avg_embs, data, offset)

        corrs, corrs_arb, models, tr_idxs, test_idxs, scalers = _train_with_arbitrary(
            X_arr, Y, n_elecs, n_lags, model_constructor, avg_embs, epochs, df_word, offset
        )
        coefficients, intercepts = get_models_coefficients(models, n_elecs, n_lags, scaled_model=False)

        save_subject_results(
            run_name, sub_id, corrs,
            coefficients, intercepts, models, tr_idxs, test_idxs,
            scalers, raw.ch_names, offset, layer_int, lags,
            corrs_arb=corrs_arb,
        )

        del raws, raw, epochs, data, X_arr, Y, models, scalers, corrs, corrs_arb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args, overrides = parse_args("Encoding model training")
    config: EncodingConfig = load_config(args.config, overrides, EncodingConfig)

    layer_int = config.layer
    subject_ids = config.subjects or list(range(1, 10))
    offset = config.offset
    run_name = config.run_name or get_results_dir(config.model, config.layer, config.offset)

    if torch.cuda.is_available():
        set_backend("torch_cuda")
        print("Using torch_cuda backend.")
    else:
        print("CUDA not available, using CPU.")

    print(f"Results directory: {run_name}")

    sig_elec_path = os.path.join(run_name, "all_subject_sig.csv")
    sig_elecs = read_electrode_file(sig_elec_path)
    sig_elecs = {s: sig_elecs[s] for s in subject_ids if s in sig_elecs}
    if not sig_elecs:
        raise ValueError("No significant electrodes found for the requested subjects.")

    print(f"Loading embeddings (layer {layer_int})...")
    aligned_embeddings, df_word = load_embeddings(config.data_root, layer_int)

    alphas = np.array(config.ridge_alphas) if config.ridge_alphas is not None else None
    model_constructor = get_ridge_model_constructor(alphas=alphas)

    if config.batch:
        _run_batch(sig_elecs, subject_ids, aligned_embeddings, df_word,
                   config.data_root, offset, layer_int, run_name, model_constructor)
    else:
        _run_joint(sig_elecs, aligned_embeddings, df_word,
                   config.data_root, offset, layer_int, run_name, model_constructor)


if __name__ == "__main__":
    main()
