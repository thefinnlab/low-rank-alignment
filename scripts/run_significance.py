"""Unified client entry point for dataset-specific significance analyses."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from typing import ClassVar

from src.core.artifacts import ArtifactStore, normalize_dataset, normalize_model
from src.core.config import FMRISignificanceConfig, SignificanceConfig
from src.core.registry import resolve_dataset_context
from src.utils.shared.config import (
    expand_fmri_layer_configs,
    expand_per_model_layer_configs,
    load_config,
    parse_args,
)


FMRI_DATASETS = {"fmri", "deep_fmri", "deep-fmri"}


def _strict_branch(payload: dict, cls):
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    from src.utils.shared.config import dict_to_dataclass
    return dict_to_dataclass(payload, cls)


@dataclass
class Config:
    """One client-facing significance configuration for every neural dataset."""

    STRICT_CONFIG: ClassVar[bool] = True

    dataset: str = "podcast"
    mode: str = "sig_lags"
    results_root: str = "results"
    plot: bool = True
    refresh: bool = False
    num_null_shifts: int = 10000
    null_seed: int = 42
    null_batch_size: int = 16
    minimal_shift: int = 500
    fdr_q: float = 0.01
    ieeg: dict | None = field(default_factory=dict)
    fmri: dict | None = None

    @property
    def modality(self) -> str:
        return "fmri" if self.dataset in FMRI_DATASETS else "ieeg"

    def validate(self):
        selected = self.fmri if self.modality == "fmri" else self.ieeg
        inactive = self.ieeg if self.modality == "fmri" else self.fmri
        if selected is None:
            raise ValueError(f"dataset={self.dataset!r} requires a {self.modality} block")
        if inactive not in (None, {}):
            raise ValueError("Only the dataset-selected modality block may be configured")
        expected = "held_out_session" if self.modality == "fmri" else "sig_lags"
        if self.mode != expected:
            raise ValueError(
                f"mode={self.mode!r} is unsupported for {self.modality}; expected {expected!r}"
            )
        if self.num_null_shifts < 1 or self.null_batch_size < 1:
            raise ValueError("Null counts and batch size must be positive")
        if self.minimal_shift < 1 or not 0 < self.fdr_q < 1:
            raise ValueError("minimal_shift must be positive and fdr_q must lie in (0, 1)")

    def selected_config(self):
        self.validate()
        payload = deepcopy(self.fmri if self.modality == "fmri" else self.ieeg) or {}
        if self.modality == "fmri" and "data" in payload:
            payload["fmri"] = payload.pop("data")
        payload.update({
            "results_root": self.results_root,
            "plot": self.plot,
            "refresh": self.refresh,
            "num_null_shifts": self.num_null_shifts,
            "null_seed": self.null_seed,
            "null_batch_size": self.null_batch_size,
            "minimal_shift": self.minimal_shift,
            "fdr_q": self.fdr_q,
        })
        cls = FMRISignificanceConfig if self.modality == "fmri" else SignificanceConfig
        if self.modality == "ieeg":
            payload.setdefault("encoding", {})["dataset"] = self.dataset
        return _strict_branch(payload, cls)


class Artifact(ArtifactStore):
    """Unified storage contract for significant neural targets."""

    SCRIPT = "run_significance"
    SELECTOR_ORDER = ("subject", "name")
    FIELD_CONTRACT = {
        "effective_config": True,
        "target_significance": False,
        "significance_details": False,
        "run_metadata": True,
        "encoding_results": False,
        "correlations": False,
        "null_distribution": False,
        "diagnostics": False,
        "checkpoints": False,
        "plots": False,
    }

    def __init__(self, *, mode="sig_lags", dataset="podcast", offset=0,
                 model="gpt2-xl", layer=24, variant=None, results_root="results"):
        self._init_store(mode=mode, dataset=dataset, offset=offset, model=model,
                         layer=layer, variant=variant, results_root=results_root)

    @classmethod
    def from_ieeg_config(cls, config, dataset_context):
        enc = config.encoding
        model = enc.per_model_config.get("model_dir", enc.model)
        return cls(
            mode="sig_lags",
            dataset=normalize_dataset(enc.dataset, dataset_context.stimulus_id),
            offset=enc.offset,
            model=normalize_model(model, glove_dim=enc.per_model_config.get("dim")),
            layer=None if enc.model == "glove" else enc.per_model_config.get("layer"),
            results_root=config.results_root,
        )

    @classmethod
    def from_fmri_config(cls, config):
        return cls(mode="held_out_session", dataset="deep-fmri", offset=0,
                   model=normalize_model(config.model_dir), layer=config.layer,
                   results_root=config.results_root)

    def save_subject_field(self, field, subject, value, *, name=None, overwrite=False):
        selector = {"subject": str(subject)}
        if name is not None:
            selector["name"] = str(name)
        self.save_field(field, value, selector=selector,
                        overwrite=overwrite)

    def load_subject_field(self, field, subjects=None):
        filters = {"subject": [str(subject) for subject in subjects]} if subjects else None
        return {
            dict(selector)["subject"]: value
            for selector, value in self.load_collection(field, filters).items()
        }

    def load_targets(self, subjects=None):
        return self.load_subject_field("target_significance", subjects)

    def load_artifacts(self, fields, selectors=None):
        selectors = selectors or {}
        subject_fields = {
            "target_significance", "significance_details", "encoding_results",
            "correlations", "null_distribution", "diagnostics", "checkpoints", "plots",
        }
        for name in fields:
            value = (self.load_subject_field(name, selectors.get("subjects"))
                     if name in subject_fields else self.load_field(name))
            setattr(self, name, value)
        self.complete = bool(self.manifest().get("complete"))
        return self


def run_config(config: Config):
    selected = config.selected_config()
    if config.modality == "fmri":
        from src.utils.significance.significance_fmri import run_fmri
        expanded = expand_fmri_layer_configs(selected)
        return [run_fmri(item) for item in expanded]
    from src.utils.significance.significance_ieeg import run_ieeg
    expanded = expand_per_model_layer_configs(selected)
    return [run_ieeg(item) for item in expanded]


def main():
    args, overrides = parse_args(__doc__)
    config = load_config(args.config, overrides, Config)
    run_config(config)


if __name__ == "__main__":
    main()
