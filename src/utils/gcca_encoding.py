"""Reusable pieces for contribution-weighted encoding of saved GCCA scores.

The baseline cache deliberately depends only on the neural evaluation problem;
no averaging option is accepted by :func:`baseline_fingerprint`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.utils.correlation import fisher_z_mean

ENCODING_SCHEMA_VERSION = 1


def embedding_identity(run: Mapping) -> str:
    """Stable identity for a controlled raw embedding, independent of view name."""
    payload = {
        "model": run.get("model"),
        "layer": (run.get("per_model_config") or {}).get("layer"),
        "pooling": (run.get("per_model_config") or {}).get("pooling", "mean"),
        "per_model_config": run.get("per_model_config") or {},
        "offset": int(run.get("offset", 0)),
    }
    digest = hashlib.sha256(_canonical(payload)).hexdigest()[:12]
    model = str(payload["model"] or "embedding").replace("/", "-")
    return f"{model}-{digest}"


def deduplicate_embedding_runs(runs: Mapping[str, Mapping]):
    """Return identity->run and view->identity maps, preserving first occurrence."""
    unique, view_map = {}, {}
    for view, run in runs.items():
        identity = embedding_identity(run)
        view_map[view] = identity
        unique.setdefault(identity, run)
    return unique, view_map


def _canonical(value) -> bytes:
    def default(item):
        if isinstance(item, Path):
            return str(item.resolve())
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=default).encode()


def baseline_fingerprint(*, target_manifest, selection, folds, embeddings, evaluation_policy, source_paths) -> str:
    """Hash all baseline-relevant inputs (and intentionally no averaging inputs)."""
    targets = target_manifest.to_dict("records") if isinstance(target_manifest, pd.DataFrame) else target_manifest
    payload = {
        "schema": ENCODING_SCHEMA_VERSION, "targets": targets,
        "selection": np.asarray(selection),
        "folds": [np.asarray(fold) for fold in folds],
        "embeddings": embeddings, "evaluation_policy": evaluation_policy,
        "source_paths": [str(Path(path).resolve()) for path in source_paths],
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


def union_significant_targets(frames: list[pd.DataFrame], *, anchor_dataset: str, anchor_stimulus: str | None = None) -> pd.DataFrame:
    """Union significant electrode targets, excluding non-anchor neural datasets."""
    kept = []
    for frame in frames:
        current = frame.copy()
        if "dataset" in current:
            current = current[current.dataset.astype(str) == str(anchor_dataset)]
        if anchor_stimulus is not None and "stimulus_id" in current:
            current = current[current.stimulus_id.astype(str) == str(anchor_stimulus)]
        kept.append(current)
    if not kept:
        return pd.DataFrame()
    result = pd.concat(kept, ignore_index=True)
    keys = [key for key in ("subject", "subject_id", "electrode", "electrode_name", "lag") if key in result]
    if not keys:
        raise ValueError("Significance tables contain no target identity columns")
    return result.drop_duplicates(keys).sort_values(keys, kind="stable").reset_index(drop=True)


def reconstruct_view_scores(artifact: Mapping[str, np.ndarray], embeddings: np.ndarray, *, calibrated: bool = True, verify: bool = True) -> np.ndarray:
    """Apply saved A, projected normalization, GCCA rotation, and calibration."""
    x = np.asarray(embeddings, float)
    projected = x @ np.asarray(artifact["basis"])
    scale = np.asarray(artifact["projected_scale"])
    safe_scale = np.where(np.abs(scale) > np.finfo(float).eps, scale, 1.0)
    raw = ((projected - np.asarray(artifact["projected_mean"])) / safe_scale) @ np.asarray(artifact["rotation"])
    # maxvar_ambient_map includes the same affine projection; this catches stale schemas.
    if verify:
        saved = np.asarray(artifact["all_anchor_scores"])
        ambient = x @ np.asarray(artifact["ambient_map"]) + np.asarray(artifact["intercept"])
        if not np.allclose(raw, saved, rtol=1e-6, atol=1e-7) or not np.allclose(ambient, saved, rtol=1e-6, atol=1e-7):
            raise ValueError("Saved GCCA basis/rotation does not reproduce ambient-map scores")
    if calibrated:
        raw = raw * np.asarray(artifact["calibration_gain"]) + np.asarray(artifact["calibration_intercept"])
        if verify and "calibrated_all_anchor_scores" in artifact and not np.allclose(raw, artifact["calibrated_all_anchor_scores"], rtol=1e-6, atol=1e-7):
            raise ValueError("Saved GCCA calibration does not reproduce calibrated scores")
    return raw


def component_weights(method: str, contributions: np.ndarray, *, inner_agreement=None, calibration_error=None) -> np.ndarray:
    """Return fixed per-component view weights; columns sum to one."""
    contribution = np.asarray(contributions, float)
    if contribution.ndim != 2:
        raise ValueError("contributions must have shape (views, components)")
    if method == "equal":
        raw = np.ones_like(contribution)
    elif method == "contribution":
        raw = contribution.copy()
    elif method == "inner_agreement":
        if inner_agreement is None: raise ValueError("inner_agreement weights require inner reliability data")
        raw = np.maximum(np.asarray(inner_agreement, float), 0)
    elif method == "inverse_error":
        if calibration_error is None: raise ValueError("inverse_error weights require inner reliability data")
        raw = 1.0 / np.maximum(np.asarray(calibration_error, float), np.finfo(float).eps)
    elif method == "hybrid":
        if inner_agreement is None or calibration_error is None: raise ValueError("hybrid weights require inner reliability data")
        raw = contribution * np.maximum(np.asarray(inner_agreement, float), 0) / np.maximum(np.asarray(calibration_error, float), np.finfo(float).eps)
    else:
        raise ValueError(f"Unknown averaging method {method!r}")
    valid = np.isfinite(raw) & (raw >= 0)
    raw = np.where(valid, raw, 0.0)
    totals = raw.sum(0)
    invalid = ~np.isfinite(totals) | (totals <= np.finfo(float).eps)
    raw[:, invalid] = 1.0
    return raw / raw.sum(0, keepdims=True)


def paired_improvements(model_corr, baselines: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Summarize paired targetwise improvements against each and all baselines."""
    model = np.asarray(model_corr, float)
    rows = []
    stack = np.stack([np.asarray(value, float) for value in baselines.values()])
    for name, base in baselines.items():
        base = np.asarray(base, float); delta = model - base
        rows.append({"baseline": name, "mean_delta": float(np.nanmean(delta)),
                     "fisher_z_mean_delta": float(np.nanmean(np.arctanh(np.clip(model, -.999999, .999999)) - np.arctanh(np.clip(base, -.999999, .999999)))),
                     "fraction_improved": float(np.nanmean(delta > 0)),
                     "fraction_beating_every_baseline": float(np.nanmean(model > np.nanmax(stack, axis=0)))})
    return pd.DataFrame(rows)


def atomic_replace_directory(destination: Path, build) -> None:
    """Build a complete directory beside destination and atomically install it."""
    destination = Path(destination); temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.rmtree(temporary, ignore_errors=True); temporary.mkdir(parents=True)
    try:
        build(temporary)
        backup = destination.with_name(f".{destination.name}.old-{os.getpid()}")
        if destination.exists(): destination.rename(backup)
        temporary.rename(destination)
        shutil.rmtree(backup, ignore_errors=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
