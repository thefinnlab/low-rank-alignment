"""Shared helpers for RRR-based scripts (run_rrr, run_ablation, etc.)."""

import os

import numpy as np

from utils.config import SVDMode
from utils.srrr import (
    compute_cca_similarity,
    compute_procrustes_similarity,
    run_sufficiency_test,
)


def run_prediction_metrics(
    Y_hat_min: np.ndarray,
    Y_hat_min_train: np.ndarray,
    Y_train: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    Y_hat_full: np.ndarray,
    alphas: np.ndarray | None = None,
    regression_type: str = "ridge",
    compute_cca: bool = False,
    compute_procrustes: bool = False,
) -> tuple[dict, float | None, float | None]:
    """Run sufficiency test and optional alignment metrics for a minimal model.

    Returns:
        (suf, cca_sim, proc_sim) where suf is the run_sufficiency_test result dict.
    """
    suf = run_sufficiency_test(
        X_train,
        Y_train - Y_hat_min_train,
        X_test,
        Y_test,
        Y_hat_full=Y_hat_full,
        Y_hat_min=Y_hat_min,
        null_r2=None,
        regression_type=regression_type,
        alphas=alphas,
    )
    cca_sim = compute_cca_similarity(Y_hat_full, Y_hat_min) if compute_cca else None
    proc_sim = (
        compute_procrustes_similarity(Y_hat_full, Y_hat_min)
        if compute_procrustes
        else None
    )
    return suf, cca_sim, proc_sim



def resolve_rrr_source_dir(
    mode: SVDMode, output_dir: str, override: str | None = None
) -> str:
    """Resolve the RRR source directory given the mode.

    Args:
        mode: SVDMode (cross_subject, per_subject, per_region).
        output_dir: Root results directory.
        override: Explicit override; returned as-is if not None.

    Returns:
        Path to the RRR source directory.
    """
    if override:
        return override
    if mode == SVDMode.PER_REGION:
        return os.path.join(output_dir, "per_region_rrr")
    if mode == SVDMode.PER_SUBJECT:
        return os.path.join(output_dir, "per_subject_rrr")
    return os.path.join(output_dir, "rrr")
