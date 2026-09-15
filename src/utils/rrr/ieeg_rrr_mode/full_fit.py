"""Whole-dataset iEEG factor fitting without held-out evaluation."""

from __future__ import annotations

import numpy as np

from scripts.run_rrr import Artifact
from src.utils.rrr.ieeg_source_bases import fit_full_dataset_ieeg_factors
from src.utils.rrr.inputs import _get_decomposition_inputs

from .common import _reference_encoding_results


def run_full_fit(
    config,
    enc,
    artifact: Artifact,
    significance,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    word_df,
    word_in_glove=None,
    shared_selection_metadata=None,
):
    """Fit and save one RRR/PCA factor set using every selected stimulus row."""
    methods = ["rrr", *(("pca",) if config.run_pca_baseline else ())]
    required = {
        "target_manifest": [],
        "input_factors": [
            artifact._selector(method=method) for method in methods
        ],
        "input_scalers": [
            artifact._selector(method=method) for method in methods
        ],
    }
    if config.save_b_matrices:
        required["target_factors"] = [artifact._selector(method="rrr")]
    if artifact.complete and not config.refresh:
        return artifact

    prepared = _get_decomposition_inputs(
        subject_ids,
        sig_electrodes,
        enc,
        aligned_embeddings,
        word_in_glove=word_in_glove,
        shared_selection_metadata=shared_selection_metadata,
        encoding_results=_reference_encoding_results(significance, subject_ids),
        significance=significance.detailed_significance,
    )
    if prepared is None:
        raise ValueError("The iEEG full_fit source has no significant electrodes")
    X = np.asarray(next(iter(prepared.embeddings.values())))
    factors = fit_full_dataset_ieeg_factors(
        X,
        np.asarray(prepared.targets),
        enc,
        config.rank_grid,
        return_b=config.save_b_matrices,
    )
    artifact.save_field(
        "target_manifest", prepared.target_manifest, overwrite=config.refresh
    )
    for method in methods:
        artifact.save_factors(
            method=method,
            input_factors=factors[method],
            input_scaler=factors["input_scaler"],
            target_factors=(factors["target"] if method == "rrr" else None),
            overwrite=config.refresh,
        )
    artifact.save_field(
        "metadata",
        {
            "mode": "full_fit",
            "fit_rows": int(X.shape[0]),
            "embedding_dim": int(X.shape[1]),
            "target_count": int(prepared.targets.shape[1]),
            "factor_coordinates": "standardized-v1",
            "evaluation": None,
            "pca_available": "pca" in methods,
        },
        overwrite=config.refresh,
    )
    artifact.mark_complete(required)
    return artifact


__all__ = ["run_full_fit"]
