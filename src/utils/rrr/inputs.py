"""Artifact-backed preparation of aligned iEEG decomposition inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.utils.shared.encoding import apply_offset
from src.utils.shared.ieeg import build_XY_from_subjects


@dataclass
class DecompositionInputs:
    embeddings: dict[str, np.ndarray]
    targets: np.ndarray
    test_idxs: list[np.ndarray]
    selection: np.ndarray
    target_manifest: pd.DataFrame
    sig_mask_flat: np.ndarray | None = None


def _embedding_mapping(embeddings):
    if isinstance(embeddings, Mapping):
        result = {str(name): np.asarray(values) for name, values in embeddings.items()}
    elif isinstance(embeddings, np.ndarray):
        result = {"embedding": np.asarray(embeddings)}
    elif isinstance(embeddings, Sequence):
        result = {str(index): np.asarray(values) for index, values in enumerate(embeddings)}
    else:
        raise TypeError("embeddings must be an array, sequence, or mapping")
    if not result:
        raise ValueError("At least one embedding matrix is required")
    return result


def build_target_manifest(subject_ids, sig_electrodes, electrode_groups, lags):
    rows = []
    loaded = [s for s in subject_ids if sig_electrodes.get(s)]
    for subject, electrodes in zip(loaded, electrode_groups):
        for electrode in electrodes:
            for lag_index, lag in enumerate(lags):
                rows.append({"subject": int(subject), "electrode": str(electrode),
                             "lag_index": int(lag_index), "lag": float(lag)})
    return pd.DataFrame(rows)


def _significance_mask(manifest, significance):
    if significance is None or len(significance) == 0:
        return None
    details = significance.copy()
    if "lag_idx" in details and "lag_index" not in details:
        details = details.rename(columns={"lag_idx": "lag_index"})
    if "sig" in details and "significant" not in details:
        details = details.rename(columns={"sig": "significant"})
    keys = ["subject", "electrode", "lag_index"]
    if not set(keys + ["significant"]).issubset(details.columns):
        return None
    left = manifest.copy()
    left["subject"] = left["subject"].astype(str)
    details["subject"] = details["subject"].astype(str)
    left["electrode"] = left["electrode"].astype(str)
    details["electrode"] = details["electrode"].astype(str)
    merged = left.merge(details[keys + ["significant"]], on=keys, how="left")
    return merged["significant"].fillna(False).astype(bool).to_numpy()


def _get_decomposition_inputs(subject_ids, sig_electrodes, *args,
                              compute_sig_nonsig=False, word_in_glove=None,
                              shared_selection_metadata=None,
                              encoding_results=None, significance=None):
    """Load neural targets and align one or more embedding views.

    Accepts both the refactored call shape and the transient positional shape used
    by the temporal-window runner during the unification commit.
    """
    if len(args) == 2:
        encoding_config, embeddings = args
    elif len(args) == 4:
        encoding_results, significance, encoding_config, embeddings = args
    else:
        raise TypeError("Unexpected decomposition input arguments")
    if encoding_results is None:
        raise ValueError("encoding_results are required")

    reference = encoding_results
    test_idxs = [np.asarray(index, dtype=int) for index in reference["test_idxs"]]
    loaded_subjects = [s for s in subject_ids if sig_electrodes.get(s)]
    if not loaded_subjects:
        print("  No significant electrodes found; skipping.")
        return None

    raw = _embedding_mapping(embeddings)
    first_name = next(iter(raw))
    X_first, targets, selection, electrode_groups = build_XY_from_subjects(
        subject_ids, encoding_config, sig_electrodes, raw[first_name],
        valid_rows=word_in_glove,
        shared_selection=(shared_selection_metadata["epoch_selection"]
                          if shared_selection_metadata is not None else None),
    )
    aligned = {first_name: np.asarray(X_first)}
    selection = np.asarray(selection, dtype=int)
    valid = np.asarray(word_in_glove)[selection] if word_in_glove is not None else None
    dummy = np.zeros((len(selection), 1), dtype=np.float32)
    for name, values in raw.items():
        if name == first_name:
            continue
        current, _ = apply_offset(values[selection], dummy,
                                  int(encoding_config.offset), valid_rows=valid)
        if current.shape[0] != X_first.shape[0]:
            raise ValueError(f"Embedding {name!r} produced {current.shape[0]} rows; "
                             f"expected {X_first.shape[0]}")
        aligned[name] = np.asarray(current)

    if shared_selection_metadata is not None:
        expected = len(shared_selection_metadata["analysis_selection"])
        if X_first.shape[0] != expected:
            raise ValueError("Loaded decomposition rows do not match saved shared "
                             "word-selection metadata")
    nonempty = [index for index in test_idxs if len(index)]
    if nonempty and max(int(np.max(index)) for index in nonempty) >= X_first.shape[0]:
        raise ValueError("Saved fold indices are incompatible with loaded decomposition rows")

    lags = np.asarray(reference.get("lags", np.arange(targets.shape[1])), dtype=float)
    manifest = build_target_manifest(subject_ids, sig_electrodes, electrode_groups, lags)
    if len(manifest) != targets.shape[1]:
        raise ValueError(f"Target manifest has {len(manifest)} rows for "
                         f"{targets.shape[1]} targets")
    mask = _significance_mask(manifest, significance) if compute_sig_nonsig else None
    return DecompositionInputs(aligned, np.asarray(targets), test_idxs, selection,
                               manifest, mask)
