"""Fit transferable input bases on an entire iEEG source stimulus."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import RRRConfig
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.rrr.core import canonical_qr, fit_dense_rrr
from src.utils.rrr.evaluation import fit_pca_input_basis
from src.utils.rrr.inputs import _get_decomposition_inputs
from src.utils.rrr.significance import load_significance_selection
from src.utils.shared.config import dict_to_dataclass
from src.utils.shared.embeddings import load_embeddings
from src.utils.shared.encoding import fit_encoding_baseline


def _fit_bases(X, Y, enc, ranks, *, standardized_rrr_coordinates=False):
    """Fit rank-specific orthonormal RRR/PCA bases from complete source arrays.

    By default the upstream ridge absorbs its input scaler. fMRI targets instead
    request coefficients in standardized input coordinates so the resulting RRR
    factor matches their target-side normalized embeddings.
    """
    requested = sorted(set(map(int, ranks)))
    max_rank = max(requested)
    available = min(X.shape[1], Y.shape[1], X.shape[0])
    if max_rank > available:
        raise ValueError(
            f"Requested source rank {max_rank} exceeds whole-stimulus iEEG rank {available}"
        )

    baseline = fit_encoding_baseline(
        X,
        Y,
        alphas=(None if enc.ridge_alphas is None else np.asarray(enc.ridge_alphas)),
        ridge_cv_folds=enc.ridge_cv_folds,
        ridge_solver_params=enc.ridge_solver_params,
        ridge_y_in_cpu=enc.ridge_y_in_cpu,
        ridge_force_cpu=enc.ridge_force_cpu,
        alpha_policy=enc.alpha_grid,
        alpha_context={
            "script": "run_rrr",
            "role": "whole_stimulus_source",
            "model": enc.model,
            "per_model_config": enc.per_model_config,
            "offset": enc.offset,
        },
        absorb_input_scaler=not standardized_rrr_coordinates,
    )
    fitted = fit_dense_rrr(
        X,
        baseline["Y_train"],
        k=max_rank,
        selected_idx=np.arange(X.shape[1]),
        precomputed_Y_hat_ols=baseline["Y_hat_full_train"],
        precomputed_beta_full=baseline["beta_full_input_coordinates"],
        return_b=False,
    )["A_dense"]
    pca = fit_pca_input_basis(X, max_rank=max_rank)
    return {
        "rrr": {rank: canonical_qr(fitted[:, :rank])[0] for rank in requested},
        "pca": {rank: pca[:, :rank] for rank in requested},
    }


def fit_whole_stimulus_ieeg_bases(
    source,
    ranks: list[int],
    *,
    standardized_rrr_coordinates: bool = False,
) -> dict[str, dict[int, np.ndarray]]:
    """Refit whole-stimulus RRR and PCA bases from an iEEG source Artifact config."""
    saved = source.load_field("effective_config")
    config = dict_to_dataclass(saved, RRRConfig)
    enc = config.encoding
    subject_ids = enc.subjects or list(range(1, 10))
    context = resolve_dataset_context(
        dataset=enc.dataset, subject_ids=subject_ids,
        per_dataset_config=enc.per_dataset_config, data_root=enc.data_root,
    )
    get_dataset_adapter(enc.dataset).validate_subjects_share_stimulus(
        subject_ids, enc.per_dataset_config, enc.data_root
    )
    significance = load_significance_selection(
        config.significance_selection, results_root=config.results_root,
        subjects=subject_ids, modality="ieeg",
    )
    target_frame = pd.concat(significance.target_significance.values(), ignore_index=True)
    significant = target_frame[target_frame["significant"].astype(bool)]
    sig_electrodes = {
        int(subject): group["target_id"].astype(str).tolist()
        for subject, group in significant.groupby("subject")
    }
    significance.encoding_results.update({
        int(subject): value for subject, value in list(significance.encoding_results.items())
    })
    detailed = pd.concat(significance.significance_details.values(), ignore_index=True)
    embeddings, word_df = load_embeddings(
        enc.data_root, enc.model, dataset=enc.dataset,
        per_dataset_config=enc.per_dataset_config, dataset_context=context,
        limit_to_glove=enc.limit_to_glove or enc.model == "glove",
        glove_dim=enc.per_model_config.get("dim", 300),
        **enc.per_model_config,
    )
    word_in_glove = None
    if enc.limit_to_glove or enc.model == "glove":
        word_in_glove = word_df["in_glove"]
    shared_selection = None
    if enc.shared_word_selection:
        from src.utils.rrr.ieeg_rrr_mode.common import _load_shared_word_selection_metadata
        shared_selection = _load_shared_word_selection_metadata(significance, subject_ids)
    prepared = _get_decomposition_inputs(
        subject_ids, sig_electrodes, enc, embeddings,
        word_in_glove=word_in_glove,
        shared_selection_metadata=shared_selection,
        encoding_results=significance.encoding_results[int(subject_ids[0])],
        significance=detailed,
    )
    if prepared is None:
        raise ValueError("The iEEG source has no significant electrodes")
    return _fit_bases(
        np.asarray(next(iter(prepared.embeddings.values()))),
        np.asarray(prepared.targets),
        enc,
        ranks,
        standardized_rrr_coordinates=standardized_rrr_coordinates,
    )


__all__ = ["fit_whole_stimulus_ieeg_bases"]
