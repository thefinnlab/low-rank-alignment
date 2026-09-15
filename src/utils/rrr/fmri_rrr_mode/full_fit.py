"""Whole-dataset fMRI factor fitting without held-out evaluation."""

from __future__ import annotations

from scripts.run_rrr import UnitSelector
from src.utils.shared.fmri import (
    build_fmri_stimulus,
    load_fmri_responses,
    session_cv_splits,
)

from .common import (
    _fit_final_ridge,
    _fit_within_fmri_bases,
    load_stimulus_data,
    save_pca_factors,
    save_rrr_factors,
)


def run_full_fit(setup):
    """Fit RRR and PCA factors using all configured fMRI stories and subjects."""
    config, artifact = setup.config, setup.artifact
    ranks = sorted(set(map(int, config.rank_grid)))
    required = {
        "target_manifest": [],
        "input_factors": [
            artifact._selector(method=method, rank=rank)
            for method in ("rrr", "pca")
            for rank in ranks
        ],
    }
    if artifact.complete and not config.refresh:
        return artifact

    data = load_stimulus_data(
        config, setup.train_stories, setup.test_stories, setup.groups
    )
    stories = setup.train_stories + setup.test_stories
    full = build_fmri_stimulus(
        data.aligned, stories, ndelays=int(config.fmri.ndelays)
    )
    with setup.timed(
        f"load selected full-fit responses ({len(config.fmri.subjects)} subject(s))"
    ):
        full.responses = load_fmri_responses(
            config.fmri.fmri_repo_root,
            stories,
            full.story_slices,
            config.fmri.subjects,
            setup.masks,
        )
    cv_groups = dict(setup.groups)
    cv_groups[max(cv_groups, default=0) + 1] = list(setup.test_stories)
    cv = session_cv_splits(full.story_slices, cv_groups)
    with setup.timed("fit full-dataset fMRI ridge"):
        ridge_fit = _fit_final_ridge(
            full.stimulus,
            full.responses,
            None,
            cv,
            config,
            context={
                "script": "fmri_rrr",
                "mode": "full_fit",
                "method": "full",
                "n_features": full.stimulus.shape[1],
                "standardize": False,
            },
            return_train_prediction=True,
            return_coefficients=True,
        )
    with setup.timed("fit full-dataset fMRI RRR and PCA factors"):
        bases, factors = _fit_within_fmri_bases(
            full, ridge_fit, data.ambient_dim, config
        )
    save_rrr_factors(
        artifact,
        config,
        UnitSelector(method="rrr"),
        bases["rrr"],
        factors,
        root_compat=True,
    )
    save_pca_factors(
        artifact,
        config,
        UnitSelector(method="pca"),
        bases["pca"],
        factors["pca_full"],
        root_compat=True,
    )
    artifact.save_field(
        "target_manifest", setup.target_manifest, overwrite=config.refresh
    )
    metadata = setup.base_metadata(data=data)
    metadata.update(
        {
            "source_mode": "full_fit",
            "fit_stories": stories,
            "fit_rows": int(full.stimulus.shape[0]),
            "full_pre_fir_shape": list(full.pre_fir_stimulus.shape),
            "evaluation": None,
            "factor_coordinates": "per_story_standardized_pre_fir-v1",
            "pca_available": True,
        }
    )
    artifact.save_field("metadata", metadata, overwrite=config.refresh)
    artifact.mark_complete(required)
    return artifact


__all__ = ["run_full_fit"]
