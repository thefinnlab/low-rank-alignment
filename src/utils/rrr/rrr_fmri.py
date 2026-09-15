"""Compatibility facade for the fMRI reduced-rank regression runner.

The implementation is organized under :mod:`src.utils.rrr.fmri_rrr_mode`.
Legacy helper imports remain available here for downstream callers.
"""

import torch
from himalaya.backend import set_backend

from scripts.run_rrr import Artifact as IEEGRRRArtifact  # compatibility
from src.core.config import FMRICrossDatasetRRRConfig
from src.utils.rrr.fmri_rrr_mode import common as _common
from src.utils.rrr.fmri_rrr_mode.common import *  # noqa: F401,F403
from src.utils.rrr.fmri_rrr_mode.common import prepare_fmri_run
from src.utils.rrr.significance import load_significance_selection  # compatibility


# Legacy test and downstream patch points. The wrappers below synchronize these
# names into the implementation module before invoking it.
fit_dense_rrr = _common.fit_dense_rrr
resolve_ridge_alphas = _common.resolve_ridge_alphas
get_ridge_model_constructor = _common.get_ridge_model_constructor
extract_best_alphas = _common.extract_best_alphas
load_story_embeddings = _common.load_story_embeddings


def _sync_legacy_overrides():
    for name in (
        "fit_dense_rrr",
        "resolve_ridge_alphas",
        "get_ridge_model_constructor",
        "extract_best_alphas",
        "load_story_embeddings",
    ):
        setattr(_common, name, globals()[name])


def _fit_within_fmri_bases(*args, **kwargs):
    _sync_legacy_overrides()
    return _common._fit_within_fmri_bases(*args, **kwargs)


def _fit_final_ridge(*args, **kwargs):
    _sync_legacy_overrides()
    return _common._fit_final_ridge(*args, **kwargs)


def run_fmri(
    config: FMRICrossDatasetRRRConfig,
    target_pca_dim: int | None = None,
):
    """Initialize one fMRI run and dispatch its mode-specific flow."""
    if torch.cuda.is_available():
        set_backend("torch_cuda")
    backend = "torch_cuda" if torch.cuda.is_available() else "numpy"
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Backend: {backend} ({device})", flush=True)

    _sync_legacy_overrides()
    setup = prepare_fmri_run(config, target_pca_dim=target_pca_dim)
    print(f"Artifact:    {setup.artifact.identity_record()}", flush=True)
    print(f"Mode:        {config.source_mode}", flush=True)
    if config.source_mode == "within_fmri":
        from src.utils.rrr.fmri_rrr_mode.internal import run_internal

        return run_internal(setup, target_pca_dim=target_pca_dim)
    if config.source_mode == "cross_dataset":
        from src.utils.rrr.fmri_rrr_mode.cross_dataset import run_cross_dataset

        return run_cross_dataset(setup)
    if config.source_mode == "leave_one_subject_out":
        from src.utils.rrr.fmri_rrr_mode.leave_one_subject_out import (
            run_leave_one_subject_out,
        )

        return run_leave_one_subject_out(setup)
    raise ValueError(f"Unsupported fMRI source_mode {config.source_mode!r}")


__all__ = ["run_fmri", *_common.__all__]
