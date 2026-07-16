"""Small reusable transformations for downstream GCCA encoding."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Mapping

import numpy as np

ENCODING_SCHEMA_VERSION = 2


def get_rrr_embeds(embeddings: np.ndarray, artifact: Mapping[str, np.ndarray]) -> np.ndarray:
    """Project raw embeddings through the exact ranked RRR basis used by GCCA."""
    values = np.asarray(embeddings, dtype=float)
    basis = np.asarray(artifact["basis"], dtype=float)
    if values.shape[1] != basis.shape[0]:
        raise ValueError(
            f"Embedding width {values.shape[1]} does not match RRR basis width {basis.shape[0]}"
        )
    return values @ basis


def transform_rrr_to_gcca(
    rrr_embeddings: np.ndarray,
    artifact: Mapping[str, np.ndarray],
    *,
    verify: bool = True,
) -> np.ndarray:
    """Transform ranked RRR coordinates into calibrated GCCA coordinates."""
    projected = np.asarray(rrr_embeddings, dtype=float)
    scale = np.asarray(artifact["projected_scale"], dtype=float)
    safe_scale = np.where(np.abs(scale) > np.finfo(float).eps, scale, 1.0)
    raw = (
        (projected - np.asarray(artifact["projected_mean"], dtype=float))
        / safe_scale
    ) @ np.asarray(artifact["rotation"], dtype=float)
    if verify and "all_anchor_scores" in artifact:
        np.testing.assert_allclose(
            raw,
            np.asarray(artifact["all_anchor_scores"]),
            rtol=1e-6,
            atol=1e-7,
            err_msg="Saved GCCA transform does not reproduce its raw scores",
        )
    calibrated = (
        raw * np.asarray(artifact["calibration_gain"], dtype=float)
        + np.asarray(artifact["calibration_intercept"], dtype=float)
    )
    if verify and "calibrated_all_anchor_scores" in artifact:
        np.testing.assert_allclose(
            calibrated,
            np.asarray(artifact["calibrated_all_anchor_scores"]),
            rtol=1e-6,
            atol=1e-7,
            err_msg="Saved GCCA calibration does not reproduce calibrated scores",
        )
    return calibrated


def component_weights(
    method: str,
    contributions: np.ndarray,
    *,
    inner_agreement=None,
    calibration_error=None,
) -> np.ndarray:
    """Return non-negative per-component view weights whose columns sum to one."""
    contribution = np.asarray(contributions, dtype=float)
    if contribution.ndim != 2:
        raise ValueError("contributions must have shape (views, components)")
    if method == "equal":
        raw = np.ones_like(contribution)
    elif method == "contribution":
        raw = contribution.copy()
    elif method == "inner_agreement":
        if inner_agreement is None:
            raise ValueError("inner_agreement weights require inner reliability data")
        raw = np.maximum(np.asarray(inner_agreement, dtype=float), 0)
    elif method == "inverse_error":
        if calibration_error is None:
            raise ValueError("inverse_error weights require inner reliability data")
        raw = 1.0 / np.maximum(
            np.asarray(calibration_error, dtype=float), np.finfo(float).eps
        )
    elif method == "hybrid":
        if inner_agreement is None or calibration_error is None:
            raise ValueError("hybrid weights require inner reliability data")
        raw = (
            contribution
            * np.maximum(np.asarray(inner_agreement, dtype=float), 0)
            / np.maximum(np.asarray(calibration_error, dtype=float), np.finfo(float).eps)
        )
    else:
        raise ValueError(f"Unknown averaging method {method!r}")
    raw = np.where(np.isfinite(raw) & (raw >= 0), raw, 0.0)
    invalid = raw.sum(axis=0) <= np.finfo(float).eps
    raw[:, invalid] = 1.0
    return raw / raw.sum(axis=0, keepdims=True)


def get_gcca_embeddings(
    artifacts: Mapping[str, Mapping[str, np.ndarray]],
    embeddings: Mapping[str, np.ndarray],
    *,
    averaging_method: str = "contribution",
    inner_agreement=None,
    calibration_error=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform every configured view and combine it in calibrated GCCA space."""
    views = list(artifacts)
    if set(views) != set(embeddings):
        raise ValueError("GCCA artifacts and embedding views do not match")
    scores = []
    contributions = []
    for view in views:
        artifact = artifacts[view]
        scores.append(
            transform_rrr_to_gcca(get_rrr_embeds(embeddings[view], artifact), artifact)
        )
        contributions.append(np.asarray(artifact["contribution_share"], dtype=float))
    weights = component_weights(
        averaging_method,
        np.stack(contributions),
        inner_agreement=inner_agreement,
        calibration_error=calibration_error,
    )
    combined = np.sum(np.stack(scores) * weights[:, None, :], axis=0)
    return combined, weights


def atomic_replace_directory(destination: Path, build) -> None:
    """Build a complete directory beside destination and atomically install it."""
    destination = Path(destination)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    backup = destination.with_name(f".{destination.name}.old-{os.getpid()}")
    shutil.rmtree(temporary, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        build(temporary)
        if destination.exists():
            destination.rename(backup)
        temporary.rename(destination)
        shutil.rmtree(backup, ignore_errors=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
