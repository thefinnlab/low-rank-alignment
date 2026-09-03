"""Podcast atlas-region items for cross-condition RRR."""

from __future__ import annotations

from copy import deepcopy
from typing import Protocol

import numpy as np

from src.core.registry import resolve_dataset_context
from src.utils.atlas.core import build_electrode_region_map
from src.utils.rrr.cross_condition import ComparisonGroup, ComparisonItem
from src.utils.rrr.inputs import build_target_manifest
from src.utils.shared.embeddings import load_embeddings
from src.utils.shared.ieeg import load_data_from_cache, load_precomputed_epochs


class RegionItemConfig(Protocol):
    """Configuration fields consumed by the region item provider."""

    data_root: str
    model: str
    per_model_config: dict
    subjects: list[int] | None
    offset: int
    atlas_path: str | None
    atlas_region_groups: dict[str, list[str]]
    limit_to_glove: bool
    glove_dim: int


def get_region_items(config: RegionItemConfig) -> ComparisonGroup:
    """Build one item per configured podcast region using all mapped electrodes."""
    subjects = list(config.subjects or range(1, 10))
    context = resolve_dataset_context("podcast", subjects, {}, config.data_root)
    embeddings, word_df = load_embeddings(
        config.data_root,
        config.model,
        dataset="podcast",
        per_dataset_config={},
        dataset_context=context,
        limit_to_glove=config.limit_to_glove,
        glove_dim=config.glove_dim,
        **deepcopy(config.per_model_config),
    )
    valid_rows = (
        word_df["in_glove"].to_numpy(dtype=bool)
        if config.limit_to_glove
        else None
    )
    electrode_region_map = build_electrode_region_map(
        subjects,
        config.data_root,
        config.atlas_region_groups,
        config.atlas_path,
    )

    items = []
    target_manifests = {}
    eligibility = []
    exclusions = []
    reference_X = None
    for region in config.atlas_region_groups:
        per_subject = electrode_region_map.get(region, {})
        region_subjects = [subject for subject in subjects if per_subject.get(subject)]
        if not region_subjects:
            exclusions.append({
                "item": region,
                "stage": "electrodes",
                "reason": "No atlas-mapped electrodes found in the configured region",
            })
            continue
        X, Y, _, electrode_groups = load_data_from_cache(
            region_subjects,
            config.data_root,
            config.offset,
            embeddings,
            per_subject,
            valid_rows=valid_rows,
            dataset="podcast",
            per_dataset_config={},
            dataset_context=context,
        )
        first_cache = load_precomputed_epochs(
            region_subjects[0],
            config.data_root,
            dataset="podcast",
            per_dataset_config={},
            dataset_context=context,
        )
        lags = np.linspace(
            float(first_cache["tmin"]),
            float(first_cache["tmax"]),
            int(first_cache["n_lags"]),
            endpoint=False,
        )
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        if reference_X is None:
            reference_X = X
        elif not np.array_equal(reference_X, X):
            raise ValueError(
                "Podcast region items must use exactly the same aligned X rows"
            )
        manifest = build_target_manifest(
            region_subjects, per_subject, electrode_groups, lags
        )
        if len(manifest) != Y.shape[1]:
            raise ValueError(
                f"Target manifest has {len(manifest)} rows for {Y.shape[1]} targets"
            )

        n_electrodes = sum(len(per_subject[subject]) for subject in region_subjects)
        items.append(ComparisonItem(
            name=str(region),
            X=X,
            Y=Y,
            target_set=f"region-{region}-targets",
        ))
        target_manifests[str(region)] = manifest
        eligibility.append({
            "item": region,
            "n_words": len(X),
            "n_subjects": len(region_subjects),
            "n_electrodes": n_electrodes,
            "n_lags": len(lags),
            "eligible": True,
        })

    if len(items) < 2:
        exclusions.append({
            "item": "*",
            "stage": "eligibility",
            "reason": "Fewer than two fully loadable regions remain",
        })
    return ComparisonGroup(
        items=items,
        target_manifests=target_manifests,
        eligibility=eligibility,
        exclusions=exclusions,
    )


__all__ = ["get_region_items"]
