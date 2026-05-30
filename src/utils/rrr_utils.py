"""Shared helpers for RRR analysis."""

import os

import numpy as np
import pandas as pd

from src.core.config import ScriptMode


def add_sig_nonsig_corrs(
    metrics: dict,
    suf: dict,
    sig_mask_flat: np.ndarray,
    pairs: list[tuple[str, str]],
) -> None:
    """Add sig/nonsig split correlations to metrics dict in-place.

    Args:
        metrics: Dict to update.
        suf: Sufficiency test result dict containing per-target arrays.
        sig_mask_flat: Boolean mask over targets (True = significant).
        pairs: List of (metric_col_name, suf_per_target_key) tuples.
    """
    for col_name, suf_key in pairs:
        arr = suf[suf_key]
        metrics[f"{col_name}_sig"] = (
            float(arr[sig_mask_flat].mean()) if sig_mask_flat.any() else float("nan")
        )
        metrics[f"{col_name}_nonsig"] = (
            float(arr[~sig_mask_flat].mean())
            if (~sig_mask_flat).any()
            else float("nan")
        )


def make_sig_split_df(df: "pd.DataFrame", suffix: str) -> "pd.DataFrame":
    """Return copy of df where {col}_{suffix} columns replace {col}.

    For every scalar column (not ending in _sig or _nonsig) that has a
    {col}_{suffix} partner, replace col values with the partner values.
    Then drop all _sig and _nonsig suffixed columns.

    Used by scripts to produce sig-only / nonsig-only DataFrames for plotting.

    Args:
        df: Source DataFrame (summary or grid_results).
        suffix: "sig" or "nonsig".

    Returns:
        New DataFrame with sig/nonsig suffix columns replaced and removed.
    """
    opp = "nonsig" if suffix == "sig" else "sig"
    result = df.copy()
    scalar_cols = [
        c for c in df.columns if not c.endswith("_sig") and not c.endswith("_nonsig")
    ]
    for col in scalar_cols:
        partner = f"{col}_{suffix}"
        if partner in df.columns:
            result[col] = df[partner]
    to_drop = [
        c for c in result.columns if c.endswith(f"_{suffix}") or c.endswith(f"_{opp}")
    ]
    return result.drop(columns=to_drop, errors="ignore")


def apply_sig_nonsig_splits_and_strip(
    metrics: dict,
    sig_mask_flat: "np.ndarray | None",
    sig_mask_shape=None,
) -> None:
    """Split any *_per_target arrays into *_sig/*_nonsig scalar means, then delete them.

    Modifies *metrics* in-place.  Safe to call when *sig_mask_flat* is None — arrays
    are still stripped; no split keys are added.
    """
    _per_target_keys = [k for k in metrics if k.endswith("_per_target")]
    if sig_mask_flat is not None:
        for pt_key in _per_target_keys:
            base = pt_key[: -len("_per_target")]
            arr = metrics[pt_key]
            metrics[f"{base}_sig"] = (
                float(arr[sig_mask_flat].mean())
                if sig_mask_flat.any()
                else float("nan")
            )
            metrics[f"{base}_nonsig"] = (
                float(arr[~sig_mask_flat].mean())
                if (~sig_mask_flat).any()
                else float("nan")
            )
    for pt_key in _per_target_keys:
        if (
            pt_key == "nr_corr_per_target"
        ):  # Keep this one for later sig/nonsig split analysis
            continue
        del metrics[pt_key]


def resolve_rrr_source_dir(
    mode: ScriptMode, output_dir: str, override: str | None = None
) -> str:
    """Resolve the RRR source directory given the mode.

    Args:
        mode: ScriptMode (cross_subject, per_subject, per_region).
        output_dir: Root results directory.
        override: Explicit override; returned as-is if not None.

    Returns:
        Path to the RRR source directory.
    """
    if override:
        return override
    if mode == ScriptMode.PER_REGION:
        return os.path.join(output_dir, "per_region_rrr")
    if mode == ScriptMode.PER_SUBJECT:
        return os.path.join(output_dir, "per_subject_rrr")
    return os.path.join(output_dir, "rrr")
