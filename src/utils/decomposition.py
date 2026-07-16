"""Shared preparation of aligned embedding and neural decomposition inputs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.utils.analysis import build_sig_mask_flat_for_elecs
from src.utils.data_loaders import build_XY_from_subjects
from src.utils.encoding import apply_offset, load_encoding_results


@dataclass
class DecompositionInputs:
    """Word-aligned inputs, raw targets, folds, and target identities."""

    embeddings: dict[str, np.ndarray]
    targets: np.ndarray
    test_idxs: list[np.ndarray]
    selection: np.ndarray
    target_manifest: pd.DataFrame
    sig_mask_flat: np.ndarray | None = None


def _embedding_mapping(
    embeddings: np.ndarray | Sequence[np.ndarray] | Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if isinstance(embeddings, Mapping):
        result = {str(name): np.asarray(values) for name, values in embeddings.items()}
    elif isinstance(embeddings, np.ndarray):
        result = {"embedding": np.asarray(embeddings)}
    else:
        result = {str(index): np.asarray(values) for index, values in enumerate(embeddings)}
    if not result:
        raise ValueError("At least one embedding matrix is required")
    return result


def build_target_manifest(subject_ids, sig_electrodes, electrode_groups, lags):
    rows = []
    loaded_subjects = [
        subject for subject in subject_ids
        if subject in sig_electrodes and sig_electrodes[subject]
    ]
    for subject, electrodes in zip(loaded_subjects, electrode_groups):
        for electrode in electrodes:
            for lag_index, lag in enumerate(lags):
                rows.append({
                    "subject": int(subject),
                    "electrode": str(electrode),
                    "lag_index": int(lag_index),
                    "lag": float(lag),
                })
    return pd.DataFrame(rows)


def _get_decomposition_inputs(
    subject_ids,
    output_dir,
    sig_electrodes,
    encoding_config,
    embeddings,
    *,
    compute_sig_nonsig=False,
    sig_dir=None,
    word_in_glove=None,
    shared_selection_metadata=None,
) -> DecompositionInputs | None:
    """Prepare one or more embedding matrices against one common neural problem.

    This helper intentionally does not fit a full-embedding baseline. Callers such
    as RRR that require one must construct it explicitly after this function.
    """
    enc_results = load_encoding_results(
        os.path.join(
            output_dir, "subjects", f"sub_{subject_ids[0]}", "encoding_results.joblib"
        )
    )
    test_idxs = [np.asarray(index, dtype=int) for index in enc_results["test_idxs"]]
    subs_with_sig = [
        subject for subject in subject_ids
        if subject in sig_electrodes and sig_electrodes[subject]
    ]
    if not subs_with_sig:
        print("  No significant electrodes found; skipping.")
        return None

    raw_embeddings = _embedding_mapping(embeddings)
    first_name = next(iter(raw_embeddings))
    X_first, targets, selection, electrode_groups = build_XY_from_subjects(
        subject_ids,
        encoding_config,
        sig_electrodes,
        raw_embeddings[first_name],
        valid_rows=word_in_glove,
        shared_selection=(
            shared_selection_metadata["epoch_selection"]
            if shared_selection_metadata is not None
            else None
        ),
    )
    aligned = {first_name: np.asarray(X_first)}
    selection = np.asarray(selection, dtype=int)
    valid_on_selection = (
        np.asarray(word_in_glove)[selection] if word_in_glove is not None else None
    )
    dummy_targets = np.zeros((len(selection), 1), dtype=np.float32)
    for name, values in raw_embeddings.items():
        if name == first_name:
            continue
        X_current, _ = apply_offset(
            values[selection],
            dummy_targets,
            int(encoding_config.offset),
            valid_rows=valid_on_selection,
        )
        if X_current.shape[0] != X_first.shape[0]:
            raise ValueError(
                f"Embedding {name!r} produced {X_current.shape[0]} rows; "
                f"expected {X_first.shape[0]}"
            )
        aligned[name] = np.asarray(X_current)

    if shared_selection_metadata is not None:
        expected_n = len(shared_selection_metadata["analysis_selection"])
        if X_first.shape[0] != expected_n:
            raise ValueError(
                "Loaded decomposition rows do not match saved shared word-selection "
                f"metadata: X has {X_first.shape[0]} rows, metadata has {expected_n}."
            )
    nonempty_test_idxs = [index for index in test_idxs if len(index)]
    if nonempty_test_idxs and max(int(np.max(index)) for index in nonempty_test_idxs) >= X_first.shape[0]:
        raise ValueError("Saved fold indices are incompatible with loaded decomposition rows")

    sig_mask_flat = None
    if compute_sig_nonsig:
        sig_mask_flat = build_sig_mask_flat_for_elecs(
            output_dir,
            {subject: sig_electrodes[subject] for subject in subs_with_sig},
            enc_results,
            sig_dir=sig_dir,
        )
    lags = np.asarray(enc_results.get("lags", np.arange(targets.shape[1])), dtype=float)
    manifest = build_target_manifest(subject_ids, sig_electrodes, electrode_groups, lags)
    if len(manifest) != targets.shape[1]:
        raise ValueError(
            f"Target manifest has {len(manifest)} rows for {targets.shape[1]} targets"
        )
    return DecompositionInputs(
        embeddings=aligned,
        targets=np.asarray(targets),
        test_idxs=test_idxs,
        selection=selection,
        target_manifest=manifest,
        sig_mask_flat=sig_mask_flat,
    )
