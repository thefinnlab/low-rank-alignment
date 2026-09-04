"""Standalone pairwise cross-condition orthogonal RRR analysis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import ClassVar

import numpy as np
import pandas as pd

from src.core.artifacts import ArtifactStore, normalize_model, normalized_config
from src.utils.rrr.cross_condition import (
    ComparisonGroup,
    ItemProvider,
    evaluate_projected_ridge,
    validate_comparison_items,
)
from src.utils.atlas.core import REGION_GROUPS
from src.utils.rrr.cross_condition_items.stimulus import (
    discover_subject_movies,
    get_stimulus_items,
)
from src.utils.rrr.cross_condition_items.region import get_region_items
from src.utils.rrr.core import fit_full_rrr, orthogonal_rank_factors
from src.utils.rrr.evaluation import fit_pca_input_basis, score_predictions
from src.utils.shared.encoding import contiguous_folds
from src.utils.shared.config import load_config, parse_args


ARTIFACT_CONTRACT_VERSION = 4


@dataclass
class Config:
    """Configuration for one model/layer cross-condition study."""

    STRICT_CONFIG: ClassVar[bool] = True

    dataset: str = "brain_treebank"
    item_getter: str = "stimulus"
    data_root: str = "data/brain-treebank"
    results_root: str = "results"
    model: str = "transformer"
    per_model_config: dict = field(
        default_factory=lambda: {"model_dir": "gpt2", "layer": 12}
    )
    subjects: list[int] | None = None
    movies: dict = field(default_factory=dict)
    offset: int = 0
    limit_to_glove: bool = True
    glove_dim: int = 300
    rank_grid: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    num_folds: int = 5
    rereference: str = "laplacian"
    atlas_path: str | None = None
    atlas_region_groups: dict[str, list[str]] = field(
        default_factory=lambda: {
            name: list(labels) for name, labels in REGION_GROUPS.items()
        }
    )
    ridge_alphas: list[float] | None = None
    ridge_cv_folds: int = 5
    ridge_solver_params: dict | None = None
    ridge_y_in_cpu: bool | None = None
    ridge_force_cpu: bool | None = None
    alpha_grid: dict = field(default_factory=dict)
    log_alphas: bool = False
    use_low_rank_svd: bool = True
    low_rank_svd_seed: int = 0
    variant_name: str | None = None
    refresh: bool = False

    def validate(self):
        if self.item_getter not in {"stimulus", "region"}:
            raise ValueError("item_getter must be 'stimulus' or 'region'")
        if self.item_getter == "stimulus":
            if self.dataset not in {"brain_treebank", "brain-treebank"}:
                raise ValueError("item_getter='stimulus' requires dataset='brain_treebank'")
            if self.rereference != "laplacian":
                raise ValueError(
                    "Brain Treebank stimulus items require rereference='laplacian'"
                )
            if self.num_folds != 5:
                raise ValueError(
                    "Brain Treebank stimulus items currently require five target folds"
                )
        else:
            if self.dataset != "podcast":
                raise ValueError("item_getter='region' requires dataset='podcast'")
            if not self.atlas_region_groups:
                raise ValueError(
                    "item_getter='region' requires non-empty atlas_region_groups"
                )
            if self.subjects is not None and not self.subjects:
                raise ValueError("subjects must be null or a non-empty list")
            if self.num_folds < 2:
                raise ValueError("num_folds must be at least two")
        if not self.rank_grid or any(int(rank) < 1 for rank in self.rank_grid):
            raise ValueError("rank_grid must contain positive integers")
        if len(set(map(int, self.rank_grid))) != len(self.rank_grid):
            raise ValueError("rank_grid cannot contain duplicate ranks")
        if self.ridge_cv_folds < 2:
            raise ValueError("ridge_cv_folds must be at least two")
        layer = self.per_model_config.get("layer")
        if layer is None and self.model != "glove":
            raise ValueError("per_model_config.layer is required")
        if not self.per_model_config.get("model_dir") and self.model != "glove":
            raise ValueError("per_model_config.model_dir is required")


class Artifact(ArtifactStore):
    """Storage contract owned only by the cross-condition producer."""

    SCRIPT = "run_cross_condition_rrr"
    COMPACT_STORAGE = True
    SELECTOR_ORDER = (
        "subject", "source_item", "target_item", "method", "rank", "fold",
    )
    FIELD_CONTRACT = {
        "effective_config": True,
        "eligibility": True,
        "exclusions": True,
        "target_manifest": False,
        "source_factors": False,
        "rank_metrics": False,
        "fold_results": True,
        "summary": True,
        "diagnostics": False,
    }

    def __init__(self, *, model="gpt2", layer=12, offset=0, variant=None,
                 results_root="results", dataset="brain-treebank"):
        self._init_store(
            mode="cross_condition", dataset=dataset, offset=offset,
            model=normalize_model(model), layer=layer, variant=variant,
            results_root=results_root,
        )

    @classmethod
    def from_config(cls, config: Config):
        model = config.per_model_config.get("model_dir", config.model)
        model = normalize_model(model, glove_dim=config.per_model_config.get("dim"))
        layer = None if config.model == "glove" else config.per_model_config.get("layer")
        digest = hashlib.sha256(json.dumps(
            _artifact_config(config), sort_keys=True
        ).encode("utf-8")).hexdigest()[:10]
        if config.variant_name:
            variant = config.variant_name
        elif config.item_getter == "stimulus":
            variant = f"rereference-{config.rereference}-{digest}"
        else:
            variant = f"region-{digest}"
        dataset = "podcast" if config.dataset == "podcast" else "brain-treebank"
        return cls(model=model, layer=layer, offset=config.offset, variant=variant,
                   results_root=config.results_root, dataset=dataset)


def _artifact_config(config: Config) -> dict:
    """Version the persisted selector/field contract independently of YAML."""
    return {
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "config": normalized_config(config),
    }


def _ridge_kwargs(config: Config) -> dict:
    return {
        "alphas": config.ridge_alphas,
        "alpha_policy": config.alpha_grid,
        "ridge_cv_folds": config.ridge_cv_folds,
        "ridge_solver_params": config.ridge_solver_params,
        "ridge_y_in_cpu": config.ridge_y_in_cpu,
        "ridge_force_cpu": config.ridge_force_cpu,
        "use_low_rank_svd": config.use_low_rank_svd,
        "low_rank_svd_seed": config.low_rank_svd_seed,
    }


def _metric_payload(result: dict, *, log_alphas: bool) -> dict:
    """Drop sample predictions while retaining primitive per-target metrics."""
    payload = {}
    for key, value in result.items():
        if key.endswith("Y_hat") or key == "Y_hat":
            continue
        if not log_alphas and key.endswith("best_alphas"):
            continue
        payload[key] = value
    return payload


def _metric_row(selector: dict, payload: dict, source_words: int, target_words: int) -> dict:
    return {
        **selector,
        "source_words": int(source_words), "target_words": int(target_words),
        "corr": float(np.asarray(payload["corr"])),
        "r2": float(np.asarray(payload["r2"])),
    }


def _selected_movies(config: Config) -> dict[int, list[str]]:
    discovered = discover_subject_movies(config.data_root, config.subjects)
    if not config.movies:
        return discovered
    selected = {}
    for subject, available in discovered.items():
        requested = config.movies.get(str(subject), config.movies.get(subject, available))
        unknown = sorted(set(requested) - set(available))
        if unknown:
            raise ValueError(f"Subject {subject} does not have requested movies {unknown}")
        if len(requested) >= 2:
            selected[subject] = list(requested)
    return selected


def _save_metric(artifact: Artifact, selector: dict, result: dict, refresh: bool,
                 log_alphas: bool):
    payload = _metric_payload(result, log_alphas=log_alphas)
    artifact.save_field(
        "rank_metrics", payload, selector=selector, overwrite=refresh
    )
    return payload


def run_item_comparisons(
    config: Config, artifact: Artifact, item_provider: ItemProvider,
):
    """Run shared within-item and directed cross-item comparisons for one group."""
    group = item_provider()
    if not isinstance(group, ComparisonGroup):
        raise TypeError("An item provider must return a ComparisonGroup")
    if len(group.items) < 2:
        return [], group.eligibility, group.exclusions, []

    validate_comparison_items(group.items, config.rank_grid, config.num_folds)
    context = dict(group.context)
    reserved = {
        "source_item", "target_item", "method", "rank", "fold",
        "source_words", "target_words", "corr", "r2",
    }
    overlap = sorted(reserved.intersection(context))
    if overlap:
        raise ValueError(f"Comparison group context uses reserved keys: {overlap}")
    for item in group.items:
        manifest = group.target_manifests.get(item.name)
        if manifest is not None:
            artifact.save_field(
                "target_manifest", manifest,
                selector={**context, "target_item": item.name},
                overwrite=config.refresh,
            )

    max_rank = max(map(int, config.rank_grid))

    ridge_kwargs = _ridge_kwargs(config)
    source_models = {}
    for item in group.items:
        model = fit_full_rrr(
            item.X, item.Y, max_rank,
            alpha_context={"script": "run_cross_condition_rrr", "role": "source_full"},
            **ridge_kwargs,
        )
        model["pca_Q"] = fit_pca_input_basis(item.X, max_rank)
        model["rank_factors"] = {}
        source_models[item.name] = model
        base_selector = {**context, "source_item": item.name}
        artifact.save_field(
            "source_factors", model["A_raw"],
            selector={**base_selector, "method": "rrr"},
            overwrite=config.refresh,
        )
        for rank in map(int, config.rank_grid):
            factors = orthogonal_rank_factors(
                model["A_raw"], model["B_raw"], rank
            )
            model["rank_factors"][rank] = factors

    diagnostics = []
    for target in group.items:
        for source in group.items:
            if source.name == target.name:
                continue
            if source.target_set is None or target.target_set is None:
                reason = "target_set_missing"
            elif source.target_set != target.target_set:
                reason = "target_set_mismatch"
            else:
                continue
            diagnostics.append({
                **context, "source_item": source.name, "target_item": target.name,
                "method": "transfer_rrr_full", "reason": reason,
            })

    rows = []
    for target in group.items:
        folds = contiguous_folds(len(target.X), config.num_folds)
        for fold, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = target.X[train_idx], target.X[test_idx]
            raw_train, raw_test = target.Y[train_idx], target.Y[test_idx]
            oracle = fit_full_rrr(
                X_train, raw_train, max_rank, X_test=X_test, Y_test=raw_test,
                alpha_context={
                    "script": "run_cross_condition_rrr", "role": "source_full"
                },
                **ridge_kwargs,
            )
            Y_train, Y_test = oracle["Y_train"], oracle["Y_test"]
            oracle_pca = fit_pca_input_basis(X_train, max_rank)
            full_prediction = oracle["full_prediction_test"]
            full_score = score_predictions(Y_test, full_prediction)
            full_selector = {
                **context, "source_item": target.name,
                "target_item": target.name, "method": "target_full_ridge",
                "fold": fold,
            }
            payload = _save_metric(
                artifact, full_selector, full_score, config.refresh, config.log_alphas
            )
            rows.append(_metric_row(full_selector, payload, len(target.X), len(target.X)))

            for rank in map(int, config.rank_grid):
                oracle_factors = orthogonal_rank_factors(
                    oracle["A_raw"], oracle["B_raw"], rank
                )
                for method, Q in (
                    ("within_rrr", oracle_factors["Q"]),
                    ("within_pca", oracle_pca),
                ):
                    result = evaluate_projected_ridge(
                        Q, rank, X_train, X_test, Y_train, Y_test, **ridge_kwargs
                    )
                    selector = {
                        **context, "source_item": target.name,
                        "target_item": target.name, "method": method,
                        "rank": rank, "fold": fold,
                    }
                    payload = _save_metric(
                        artifact, selector, result, config.refresh, config.log_alphas
                    )
                    rows.append(_metric_row(
                        selector, payload, len(target.X), len(target.X)
                    ))

                for source in group.items:
                    if source.name == target.name:
                        continue
                    source_model = source_models[source.name]
                    source_factors = source_model["rank_factors"][rank]
                    for method, Q in (
                        ("transfer_rrr_basis", source_factors["Q"]),
                        ("transfer_pca_basis", source_model["pca_Q"]),
                    ):
                        result = evaluate_projected_ridge(
                            Q, rank, X_train, X_test, Y_train, Y_test, **ridge_kwargs
                        )
                        selector = {
                            **context, "source_item": source.name,
                            "target_item": target.name, "method": method,
                            "rank": rank, "fold": fold,
                        }
                        payload = _save_metric(
                            artifact, selector, result, config.refresh, config.log_alphas
                        )
                        rows.append(_metric_row(
                            selector, payload, len(source.X), len(target.X)
                        ))

                    if (source.target_set is None or target.target_set is None
                            or source.target_set != target.target_set):
                        continue
                    B_raw = source_factors["B_raw"]
                    B_orth = source_factors["B_orth"]
                    intercept = source_model["ridge_intercept"] @ B_raw @ B_raw.T
                    prediction = X_test @ source_factors["Q"] @ B_orth.T + intercept
                    result = score_predictions(Y_test, prediction)
                    selector = {
                        **context, "source_item": source.name,
                        "target_item": target.name, "method": "transfer_rrr_full",
                        "rank": rank, "fold": fold,
                    }
                    payload = _save_metric(
                        artifact, selector, result, config.refresh, config.log_alphas
                    )
                    rows.append(_metric_row(
                        selector, payload, len(source.X), len(target.X)
                    ))

    return rows, group.eligibility, group.exclusions, diagnostics


def _run_subject(config: Config, artifact: Artifact, subject: int, movies: list[str]):
    """Adapt the Brain Treebank stimulus provider to the shared comparison loop."""
    return run_item_comparisons(
        config, artifact,
        lambda: get_stimulus_items(config, subject, movies),
    )


def _collect_provider_results(config: Config, artifact: Artifact):
    """Run the configured item provider and return its combined records."""
    if config.item_getter == "region":
        return run_item_comparisons(
            config, artifact, lambda: get_region_items(config)
        )

    all_rows, eligibility, exclusions, diagnostics = [], [], [], []
    all_discovered = discover_subject_movies(
        config.data_root, config.subjects, min_movies=1
    )
    selected = _selected_movies(config)
    for subject, movies in all_discovered.items():
        if subject not in selected:
            exclusions.append({
                "subject": subject, "item": movies[0] if movies else "*",
                "stage": "eligibility",
                "reason": "Subject has fewer than two selected movies",
            })
    for subject, movies in selected.items():
        rows, eligible, excluded, skipped = _run_subject(
            config, artifact, subject, movies
        )
        all_rows.extend(rows)
        eligibility.extend(eligible)
        exclusions.extend(excluded)
        diagnostics.extend(skipped)
    return all_rows, eligibility, exclusions, diagnostics


def _records_frame(records: list[dict], base_columns: list[str]) -> pd.DataFrame:
    """Build a stable empty table while retaining provider-specific columns."""
    if not records:
        return pd.DataFrame(columns=base_columns)
    frame = pd.DataFrame(records)
    ordered = [column for column in base_columns if column in frame]
    ordered.extend(column for column in frame if column not in ordered)
    return frame[ordered]


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    metric_columns = {
        "source_item", "target_item", "method", "rank", "fold",
        "source_words", "target_words", "corr", "r2",
    }
    context_keys = [column for column in frame.columns if column not in metric_columns]
    keys = [
        *context_keys, "source_item", "target_item", "method", "rank",
        "source_words", "target_words",
    ]
    summary = frame.groupby(keys, as_index=False, dropna=False)[["corr", "r2"]].mean()
    oracle = summary[summary["method"] == "within_rrr"][
        [*context_keys, "target_item", "rank", "corr", "r2"]
    ].rename(columns={"corr": "oracle_corr", "r2": "oracle_r2"})
    summary = summary.merge(
        oracle, on=[*context_keys, "target_item", "rank"], how="left"
    )
    summary["target_oracle_corr_gap"] = summary["corr"] - summary["oracle_corr"]
    summary["target_oracle_r2_gap"] = summary["r2"] - summary["oracle_r2"]
    return summary


def run_config(config: Config):
    config.validate()
    layers = config.per_model_config.get("layer")
    if isinstance(layers, list):
        outputs = []
        for layer in layers:
            current = replace(config, per_model_config={**config.per_model_config, "layer": layer})
            outputs.extend(run_config(current))
        return outputs

    artifact = Artifact.from_config(config)
    artifact.begin(config=_artifact_config(config), refresh=config.refresh)
    if artifact.complete and not config.refresh:
        print(f"Artifact complete; reusing {artifact.identity_record()}")
        return [artifact]
    artifact.save_field("effective_config", asdict(config), overwrite=config.refresh)

    all_rows, eligibility, exclusions, diagnostics = _collect_provider_results(
        config, artifact
    )
    eligibility_frame = _records_frame(
        eligibility, ["subject", "item", "n_words", "n_lags", "eligible"]
    )
    exclusion_frame = _records_frame(
        exclusions, ["subject", "item", "stage", "reason"]
    )
    diagnostic_frame = _records_frame(
        diagnostics,
        ["subject", "source_item", "target_item", "method", "reason"],
    )
    fold_frame = pd.DataFrame(all_rows)
    summary = _summary(fold_frame)
    artifact.save_field("eligibility", eligibility_frame, overwrite=config.refresh)
    artifact.save_field("exclusions", exclusion_frame, overwrite=config.refresh)
    artifact.save_field("fold_results", fold_frame, overwrite=config.refresh)
    artifact.save_field("summary", summary, overwrite=config.refresh)
    artifact.save_field("diagnostics", diagnostic_frame, overwrite=config.refresh)
    if fold_frame.empty:
        raise ValueError(
            "Cross-condition preflight found fewer than two fully loadable items; "
            "inspect the saved exclusions table"
        )
    artifact.mark_complete()
    print(f"Artifact: {artifact.identity_record()}")
    return [artifact]


def main():
    args, overrides = parse_args(__doc__)
    config = load_config(args.config, overrides, Config)
    run_config(config)


if __name__ == "__main__":
    main()
