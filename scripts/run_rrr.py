"""Unified client entry point for dataset-specific reduced-rank regression."""

from __future__ import annotations

import io
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from typing import ClassVar

import numpy as np
import pandas as pd

from src.core.artifacts import (
    ArtifactError,
    ArtifactStore,
    canonical_significance_selection,
    normalize_dataset,
    normalize_model,
    readable_variant,
)
from src.core.config import (
    FMRICrossDatasetRRRConfig,
    RRRConfig,
    ScriptMode,
    SignificanceSelectionConfig,
)
from src.core.registry import resolve_dataset_context
from src.utils.shared.config import (
    expand_fmri_layer_configs,
    expand_per_model_layer_configs,
    load_config,
    parse_args,
)


FMRI_DATASETS = {"fmri", "deep_fmri", "deep-fmri"}


def _is_fmri(dataset):
    return dataset in FMRI_DATASETS


def _strict_branch(payload: dict, cls):
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    from src.utils.shared.config import dict_to_dataclass
    return dict_to_dataclass(payload, cls)


@dataclass
class Config:
    """One client-facing RRR configuration for every neural dataset."""

    STRICT_CONFIG: ClassVar[bool] = True

    dataset: str = "podcast"
    mode: ScriptMode = ScriptMode.INTERNAL
    results_root: str = "results"
    plot: bool = True
    refresh: bool = False
    rank_grid: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50])
    significance_selection: SignificanceSelectionConfig = field(
        default_factory=SignificanceSelectionConfig
    )
    rrr_source: dict | None = None
    views: list[dict] = field(default_factory=list)
    subspace_aggregation: str = "extrinsic"
    ieeg: dict | None = field(default_factory=dict)
    fmri: dict | None = None

    @property
    def modality(self):
        return "fmri" if _is_fmri(self.dataset) else "ieeg"

    def validate(self):
        if isinstance(self.mode, str):
            self.mode = ScriptMode(self.mode)
        selected = self.fmri if self.modality == "fmri" else self.ieeg
        inactive = self.ieeg if self.modality == "fmri" else self.fmri
        if selected is None:
            raise ValueError(f"dataset={self.dataset!r} requires a {self.modality} block")
        if inactive not in (None, {}):
            raise ValueError("Only the dataset-selected modality block may be configured")
        shared = {ScriptMode.INTERNAL, ScriptMode.CROSS_DATASET}
        if self.modality == "fmri" and self.mode not in shared:
            raise ValueError(f"RRR mode {self.mode.value!r} is unsupported for fMRI")
        if self.mode in {
            ScriptMode.CROSS_DATASET, ScriptMode.CROSS_TEMPORAL_WINDOWS
        } and not self.rrr_source:
            raise ValueError(f"mode={self.mode.value} requires rrr_source")
        if not self.rank_grid or any(int(rank) < 1 for rank in self.rank_grid):
            raise ValueError("rank_grid must contain positive integers")

    @staticmethod
    def _legacy_source(source):
        return deepcopy(source) if source else None

    def selected_config(self):
        self.validate()
        payload = deepcopy(self.fmri if self.modality == "fmri" else self.ieeg) or {}
        if self.modality == "fmri" and "data" in payload:
            payload["fmri"] = payload.pop("data")
        payload.update({
            "results_root": self.results_root,
            "plot": self.plot,
            "refresh": self.refresh,
            "rank_grid": self.rank_grid,
            "significance_selection": deepcopy(self.significance_selection),
            "subspace_aggregation": self.subspace_aggregation,
        })
        if self.modality == "fmri":
            payload["source_mode"] = (
                "within_fmri" if self.mode == ScriptMode.INTERNAL else "cross_dataset"
            )
            if self.rrr_source:
                if _is_fmri(self.rrr_source.get("dataset")):
                    raise ValueError("fMRI cross-dataset RRR requires an iEEG rrr_source")
                payload["ieeg_rrr_source"] = self._legacy_source(self.rrr_source)
            return _strict_branch(payload, FMRICrossDatasetRRRConfig)

        payload.setdefault("encoding", {})["dataset"] = self.dataset
        payload["mode"] = (
            ScriptMode.CROSS_SUBJECT if self.mode == ScriptMode.INTERNAL else self.mode
        )
        if self.rrr_source:
            key = "fmri_rrr_source" if _is_fmri(self.rrr_source.get("dataset")) else "ieeg_rrr_source"
            payload[key] = self._legacy_source(self.rrr_source)
            payload["cross_source_format"] = (
                "rank_bases" if key == "fmri_rrr_source" else "fold_factors"
            )
        converted_views = []
        for view in self.views:
            item = {key: deepcopy(value) for key, value in view.items() if key != "source"}
            source = view.get("source")
            if not source:
                raise ValueError(f"RRR view {view.get('name')!r} requires source")
            item["fmri_rrr_source" if _is_fmri(source.get("dataset"))
                 else "ieeg_rrr_source"] = deepcopy(source)
            converted_views.append(item)
        payload["views"] = converted_views
        return _strict_branch(payload, RRRConfig)


@dataclass(frozen=True)
class UnitSelector:
    method: str = "rrr"
    subject: int | str | None = None
    region: str | None = None
    temporal_window: str | None = None
    source_temporal_window: str | None = None
    target_temporal_window: str | None = None
    held_out_subject: int | str | None = None
    loso_phase: str | None = None
    cross_dataset_source: str | None = None

    def mapping(self):
        return {key: value for key, value in asdict(self).items() if value is not None}


class Artifact(ArtifactStore):
    """Unified storage and loading contract for iEEG and fMRI RRR results."""

    SCRIPT = "run_rrr"
    SELECTOR_ORDER = (
        "subject", "method", "region", "temporal_window", "source_temporal_window",
        "target_temporal_window", "held_out_subject",
        "loso_phase", "cross_dataset_source", "basis", "delay", "rank", "fold",
        "name",
    )
    FIELD_CONTRACT = {
        "effective_config": True, "upstream_identities": True,
        "target_manifest": True, "input_factors": False,
        "target_factors": False, "full_metrics": True,
        "rank_metrics": True, "fold_results": False, "mean_results": True,
        "metadata": False, "unit_metadata": False, "diagnostics": False,
        "plots": False, "mode_metadata": False
    }

    def __init__(self, *, mode="internal", dataset="podcast", offset=0,
                 model="gpt2-xl", layer=24, variant=None,
                 significance_selection=None, results_root="results"):
        if variant is not None and significance_selection is not None:
            raise ValueError(
                "RRR Artifact variant and significance_selection are mutually exclusive"
            )
        if significance_selection is not None:
            variant = readable_variant(
                {"significance_selection": significance_selection}
            )
        self._init_store(mode=mode, dataset=dataset, offset=offset, model=model,
                         layer=layer, variant=variant, results_root=results_root)
        self.requested_significance_selection = significance_selection

    def _manifest(self):
        manifest = super()._manifest()
        if self.requested_significance_selection is not None:
            saved = manifest.get("configuration", {}).get("significance_selection")
            if saved is None:
                raise ArtifactError(
                    "Selected RRR Artifact does not record significance_selection"
                )
            requested = canonical_significance_selection(
                self.requested_significance_selection
            )
            if canonical_significance_selection(saved) != requested:
                raise ArtifactError(
                    "Selected RRR Artifact significance_selection does not match "
                    "the requested selection"
                )
        return manifest

    @classmethod
    def from_ieeg_config(cls, config, dataset_context):
        enc = config.encoding
        model = enc.per_model_config.get("model_dir", enc.model)
        mode = "internal" if config.mode == ScriptMode.CROSS_SUBJECT else config.mode.value
        artifact = cls(
            mode=mode,
            dataset=normalize_dataset(enc.dataset, dataset_context.stimulus_id),
            offset=enc.offset,
            model=normalize_model(model, glove_dim=enc.per_model_config.get("dim")),
            layer=None if enc.model == "glove" else enc.per_model_config.get("layer"),
            results_root=config.results_root,
        )
        artifact.variant_extras = {}
        rereference = getattr(dataset_context, "rereference", None)
        if rereference not in (None, "common_average"):
            artifact.variant_extras["rereference"] = rereference
        if config.atlas_path:
            artifact.variant_extras["atlas"] = (
                config.atlas_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            )
        return artifact

    @classmethod
    def from_fmri_config(cls, config):
        mode = "internal" if config.source_mode == "within_fmri" else config.source_mode
        return cls(mode=mode, dataset="deep-fmri", offset=0,
                   model=normalize_model(config.model_dir), layer=config.layer,
                   results_root=config.results_root)

    @staticmethod
    def _selector(unit=None, **extra):
        result = unit.mapping() if isinstance(unit, UnitSelector) else dict(unit or {})
        result.update({key: value for key, value in extra.items() if value is not None})
        return result

    def save_factors(self, *, unit=None, fold=None, method=None, rank=None,
                     input_factors=None, factors=None, target_factors=None,
                     delay=None, basis=None, overwrite=False):
        value = input_factors if input_factors is not None else factors
        selector = self._selector(
            unit, fold=fold, method=method, rank=rank, delay=delay, basis=basis
        )
        self.save_field("input_factors", value, selector=selector, overwrite=overwrite)
        if target_factors is not None:
            self.save_field("target_factors", target_factors, selector=selector,
                            overwrite=overwrite)

    def load_input_factors(self, method="rrr", rank=None, selectors=None):
        filters = dict(selectors or {})
        filters["method"] = method
        entries = self.load_collection("input_factors", filters)
        ranked = [(dict(selector), value) for selector, value in entries.items()
                  if dict(selector).get("rank") == rank]
        if ranked:
            return [np.asarray(value) for _, value in ranked]
        values = []
        for selector, value in entries.items():
            mapping = dict(selector)
            if "rank" in mapping or "fold" not in mapping:
                continue
            array = np.asarray(value)
            values.append(array if rank is None else array[:, :int(rank)])
        return values

    def save_rank_metrics(self, *, unit=None, method=None, rank, fold=None,
                          metrics, overwrite=False):
        self.save_field("rank_metrics", metrics,
                        selector=self._selector(unit, method=method, rank=int(rank), fold=fold),
                        overwrite=overwrite)

    def load_rank_metrics(self, *, unit=None, method=None, rank, fold=None):
        return self.load_field("rank_metrics",
                               selector=self._selector(unit, method=method,
                                                       rank=int(rank), fold=fold))

    def load_factors(self, *, unit, fold):
        selector = self._selector(unit, fold=int(fold))
        result = {"input_factors": self.load_field("input_factors", selector=selector)}
        if self.has_field("target_factors", selector=selector):
            result["target_factors"] = self.load_field("target_factors", selector=selector)
        return result

    def save_full_metrics(self, *, metrics, unit=None, fold=None, overwrite=False):
        self.save_field("full_metrics", metrics,
                        selector=self._selector(unit, fold=fold), overwrite=overwrite)

    def save_summary(self, *, summary, unit=None, overwrite=False):
        frame = summary.copy() if isinstance(summary, pd.DataFrame) else pd.DataFrame(summary)
        if "k" in frame and "rank" not in frame:
            frame = frame.rename(columns={"k": "rank"})
        method = unit.method if isinstance(unit, UnitSelector) else None
        if method and "method" not in frame:
            frame["method"] = method
        primary = "pca" if method == "pca" else "nr"
        if "corr" not in frame and f"{primary}_corr" in frame:
            frame["corr"] = frame[f"{primary}_corr"]
        if "r2" not in frame and f"{primary}_r2" in frame:
            frame["r2"] = frame[f"{primary}_r2"]
        self.save_mean_results(results=frame, unit=unit, overwrite=overwrite)

    def save_fold_results(self, *, results, unit=None, overwrite=False):
        self.save_field("fold_results", results, selector=self._selector(unit),
                        overwrite=overwrite)

    def load_fold_results(self, selectors=None):
        frames = []
        for selector, value in self.load_collection("fold_results", selectors).items():
            frame = value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
            for key, item in dict(selector).items():
                if key not in frame:
                    frame[key] = item
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def load_summary(self, *, unit):
        return self.load_field("mean_results", selector=self._selector(unit))

    def save_target_manifest(self, *, target_manifest, unit=None, overwrite=False):
        self.save_field("target_manifest", target_manifest,
                        selector=self._selector(unit), overwrite=overwrite)

    def save_diagnostics(self, *, diagnostics, unit=None, rank=None, fold=None,
                         name=None, overwrite=False):
        self.save_field("diagnostics", diagnostics,
                        selector=self._selector(
                            unit, rank=rank, fold=fold, name=name
                        ),
                        overwrite=overwrite)

    def save_unit_metadata(self, *, metadata, unit=None, overwrite=False):
        self.save_field("unit_metadata", metadata, selector=self._selector(unit),
                        overwrite=overwrite)

    def resume_plan(self, *, unit, ranks, folds, require_target_factors=False,
                    require_input_factors=True, required_metric_keys=None, refresh=False):
        completed = set()
        for fold in folds:
            factor_selector = self._selector(unit, fold=int(fold))
            factors = (not require_input_factors or
                       self.has_field("input_factors", selector=factor_selector))
            targets = (not require_target_factors or
                       self.has_field("target_factors", selector=factor_selector))
            for rank in ranks:
                metric_selector = self._selector(unit, rank=int(rank), fold=int(fold))
                metrics_ok = self.has_field("rank_metrics", selector=metric_selector)
                if metrics_ok and required_metric_keys:
                    metrics_ok = set(required_metric_keys).issubset(
                        self.load_field("rank_metrics", selector=metric_selector)
                    )
                if not refresh and factors and targets and metrics_ok:
                    completed.add((int(rank), int(fold)))
        ranks_by_fold = {
            int(fold): [int(rank) for rank in ranks
                        if (int(rank), int(fold)) not in completed]
            for fold in folds
        }
        return {"completed_pairs": completed, "ranks_by_fold": ranks_by_fold,
                "folds_to_build": {fold for fold, pending in ranks_by_fold.items() if pending}}

    def resume_method_plan(self, methods, ranks, refresh=False):
        completed = set()
        for method in methods:
            for rank in ranks:
                selector = self._selector(method=method, rank=int(rank))
                if not refresh and self.has_field("rank_metrics", selector=selector):
                    completed.add((method, int(rank)))
        return {"completed": completed,
                "pending": [(method, int(rank)) for method in methods for rank in ranks
                            if (method, int(rank)) not in completed]}

    def load_artifacts(self, fields, selectors=None):
        selectors = selectors or {}
        scalar = {"effective_config", "upstream_identities", "metadata"}
        for name in fields:
            setattr(self, name, self.load_field(name) if name in scalar
                    else self.load_collection(name, selectors))
        return self

    def save_artifacts(self, *, effective_config, upstream_identities,
                       plot=True, overwrite=False):
        self.save_field("effective_config", effective_config, overwrite=overwrite)
        self.save_field("upstream_identities", upstream_identities, overwrite=overwrite)

    def save_mean_results(self, *, results, unit=None, overwrite=False):
        self.save_field("mean_results", results, selector=self._selector(unit),
                        overwrite=overwrite)

    def load_mean_results(self, selectors=None):
        frames = []
        for selector, value in self.load_collection("mean_results", selectors).items():
            frame = value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
            for key, item in dict(selector).items():
                if key not in frame:
                    frame[key] = item
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def save_plot(self, *, name, figure, unit=None, overwrite=False):
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
        self.save_field("plots", buffer.getvalue(),
                        selector=self._selector(unit, name=name), overwrite=overwrite)


def run_config(config: Config):
    selected = config.selected_config()
    if config.modality == "fmri":
        from src.utils.rrr.rrr_fmri import run_fmri
        return [run_fmri(item) for item in expand_fmri_layer_configs(selected)]
    from src.utils.rrr.rrr_ieeg import run_ieeg
    return [run_ieeg(item) for item in expand_per_model_layer_configs(selected)]


def main():
    args, overrides = parse_args(__doc__)
    config = load_config(args.config, overrides, Config)
    run_config(config)


if __name__ == "__main__":
    main()
