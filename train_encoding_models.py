"""
train_encoding_models.py — Joint or batched encoding model training.

Trains ridge regression encoding models (GPT-2 embeddings → ECoG neural data).

Writes per-subject results to:
  {run_name}/subjects/sub_{id}/
    encoding_results.joblib
    corrs.npy
    arbitrary_corrs.npy
    corrs_plot.png
"""

import argparse
import gc
import os

import numpy as np
import torch
from himalaya.backend import set_backend

from utils import (
    load_raws,
    train_encoding,
    get_arbitrary_embeddings,
    absorb_scaler,
    load_embeddings,
    get_ridge_model_constructor,
    build_word_epochs,
    save_encoding_results,
    plot_corrs_vs_arbitrary,
    read_electrode_file,
)


LAYER_MAP = {"mid": 24, "final": 48}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Encoding model training script")
    parser.add_argument(
        "--layer", choices=["mid", "final"], required=True,
        help="GPT-2-XL layer: 'mid' (24) or 'final' (48)",
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Word offset to apply when aligning embeddings to neural data",
    )
    parser.add_argument(
        "--subjects", nargs="+", type=int, default=None,
        help="Subject BIDS IDs to process (default: 1–9). E.g. --subjects 1 3 5",
    )
    parser.add_argument(
        "--run-name", required=True,
        help="Run directory name; {run_name}/all_subject_sig.csv must exist",
    )
    parser.add_argument("--data-root", default="data/", help="Root of BIDS dataset")
    parser.add_argument(
        "--batch", action="store_true",
        help="Process one subject at a time to reduce peak RAM usage",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Script-local helpers
# ---------------------------------------------------------------------------

def _apply_offset(averaged_embeddings, data, offset):
    """Apply word offset: trim embeddings and neural data to align them."""
    n_words = data.shape[0]
    Y = data.reshape(n_words, -1).astype(np.float32)
    if offset < 0:
        return averaged_embeddings[:offset].astype(np.float32), Y[-offset:]
    elif offset > 0:
        return averaged_embeddings[offset:].astype(np.float32), Y[:-offset]
    return averaged_embeddings.astype(np.float32), Y


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


def _extract_coefficients(models, n_elecs, n_lags):
    """Extract raw-space coefficients and intercepts via absorb_scaler."""
    coefficients, intercepts = [], []
    for fold_model in models:
        coef, intercept = absorb_scaler(fold_model, n_elecs, n_lags)
        coefficients.append(coef)
        intercepts.append(intercept)
    return coefficients, intercepts


def _save_subject(run_name, sub_id, corrs, corrs_arb, coefficients, intercepts,
                  models, tr_idxs, test_idxs, scalers, elec_names,
                  offset, layer_int, lags):
    """Build results dict and write all outputs to subjects/sub_{id}/."""
    results = {
        "corrs": corrs,
        "arb_corrs": corrs_arb,
        "coefficients": coefficients,
        "intercepts": intercepts,
        "models": models,
        "train_idxs": tr_idxs,
        "test_idxs": test_idxs,
        "scalers": scalers,
        "elec_names": list(elec_names),
        "subject_id": sub_id,
        "offset": offset,
        "layer": layer_int,
        "lags": lags,
    }
    sub_dir = os.path.join(run_name, "subjects", f"sub_{sub_id}")
    os.makedirs(sub_dir, exist_ok=True)
    save_encoding_results(results, os.path.join(sub_dir, "encoding_results.joblib"))
    np.save(os.path.join(sub_dir, "corrs.npy"), corrs)
    np.save(os.path.join(sub_dir, "arbitrary_corrs.npy"), corrs_arb)
    plot_corrs_vs_arbitrary(corrs, corrs_arb, lags,
                            save_path=os.path.join(sub_dir, "corrs_plot.png"))
    print(f"  Subject {sub_id}: {corrs.shape[1]} electrodes → {sub_dir}")


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
    X_arr, Y = _apply_offset(avg_embs, datas, offset)

    print(f"Training joint model: {total_elecs} electrodes, {X_arr.shape[0]} words...")
    corrs, corrs_arb, models, tr_idxs, test_idxs, scalers = _train_with_arbitrary(
        X_arr, Y, total_elecs, n_lags, model_constructor, avg_embs, epochs, df_word, offset
    )
    coefficients, intercepts = _extract_coefficients(models, total_elecs, n_lags)

    cursor = 0
    for sub_id, raw, n_elecs in zip(sig_elecs.keys(), raws, elec_counts):
        sl = slice(cursor, cursor + n_elecs)
        cursor += n_elecs
        coef_slices = [c[sl] for c in coefficients]
        int_slices = [i.reshape(total_elecs, n_lags)[sl].ravel() for i in intercepts]
        _save_subject(run_name, sub_id, corrs[:, sl, :], corrs_arb[:, sl, :],
                      coef_slices, int_slices, models, tr_idxs, test_idxs,
                      scalers, raw.ch_names, offset, layer_int, lags)


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
        X_arr, Y = _apply_offset(avg_embs, data, offset)

        corrs, corrs_arb, models, tr_idxs, test_idxs, scalers = _train_with_arbitrary(
            X_arr, Y, n_elecs, n_lags, model_constructor, avg_embs, epochs, df_word, offset
        )
        coefficients, intercepts = _extract_coefficients(models, n_elecs, n_lags)

        _save_subject(run_name, sub_id, corrs, corrs_arb,
                      coefficients, intercepts, models, tr_idxs, test_idxs,
                      scalers, raw.ch_names, offset, layer_int, lags)

        del raws, raw, epochs, data, X_arr, Y, models, scalers, corrs, corrs_arb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    layer_int = LAYER_MAP[args.layer]
    subject_ids = args.subjects or list(range(1, 10))
    offset = args.offset

    if torch.cuda.is_available():
        set_backend("torch_cuda")
        print("Using torch_cuda backend.")
    else:
        print("CUDA not available, using CPU.")

    sig_elec_path = os.path.join(args.run_name, "all_subject_sig.csv")
    sig_elecs = read_electrode_file(sig_elec_path)
    sig_elecs = {s: sig_elecs[s] for s in subject_ids if s in sig_elecs}
    if not sig_elecs:
        raise ValueError("No significant electrodes found for the requested subjects.")

    print(f"Loading embeddings (layer {layer_int})...")
    aligned_embeddings, df_word = load_embeddings(args.data_root, layer_int)

    model_constructor = get_ridge_model_constructor()

    if args.batch:
        _run_batch(sig_elecs, subject_ids, aligned_embeddings, df_word,
                   args.data_root, offset, layer_int, args.run_name, model_constructor)
    else:
        _run_joint(sig_elecs, aligned_embeddings, df_word,
                   args.data_root, offset, layer_int, args.run_name, model_constructor)


if __name__ == "__main__":
    main()
