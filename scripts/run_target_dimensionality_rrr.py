"""Sweep internal RRR over fold-safe target-PCA dimensionalities."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.run_rrr import Artifact as RRRArtifact
from scripts.run_rrr import Config as RRRClientConfig
from src.core.artifacts import readable_variant
from src.core.config import ScriptMode
from src.utils.shared.config import (
    expand_fmri_layer_configs,
    expand_per_model_layer_configs,
    load_config,
    parse_args,
)


def _power_grid(cap: int) -> list[int]:
    """Return powers of two through ``cap``, including an exact final endpoint."""
    cap = int(cap)
    if cap < 1:
        raise ValueError("dimension cap must be positive")
    values = []
    value = 1
    while value <= cap:
        values.append(value)
        value *= 2
    if values[-1] != cap:
        values.append(cap)
    return values


@dataclass
class Config(RRRClientConfig):
    """Target-dimensionality sweep configuration."""

    STRICT_CONFIG: ClassVar[bool] = True
    target_pca_dims: list[int] | None = field(default=None)

    def validate(self):
        super().validate()
        if self.mode != ScriptMode.INTERNAL:
            raise ValueError("target-dimensionality RRR supports only mode=internal")
        if self.target_pca_dims is not None:
            dims = [int(value) for value in self.target_pca_dims]
            if not dims or any(value < 1 for value in dims):
                raise ValueError("target_pca_dims must contain positive integers")
            if len(set(dims)) != len(dims):
                raise ValueError("target_pca_dims must not contain duplicates")
            self.target_pca_dims = sorted(dims)


class Artifact(RRRArtifact):
    """RRR-compatible outputs for one target-PCA dimensionality variant."""

    SCRIPT = "run_target_dimensionality_rrr"
    FIELD_CONTRACT = {**RRRArtifact.FIELD_CONTRACT, "sweep_results": False}

    def __init__(self, *, target_pca_dim=None, **kwargs):
        significance_selection = kwargs.get("significance_selection")
        variant = kwargs.get("variant")
        if target_pca_dim is None or variant is not None:
            super().__init__(**kwargs)
        else:
            kwargs = dict(kwargs)
            kwargs.pop("significance_selection", None)
            kwargs["variant"] = readable_variant(
                {"significance_selection": significance_selection}
                if significance_selection is not None else {},
                extras={"target_pca_dim": int(target_pca_dim)},
            )
            super().__init__(**kwargs)
            self.requested_significance_selection = significance_selection
        self.target_pca_dim = (
            None if target_pca_dim is None else int(target_pca_dim)
        )

    def save_sweep_results(self, frame: pd.DataFrame, *, overwrite=False):
        self.save_field("sweep_results", frame, overwrite=overwrite)

    def load_sweep_results(self) -> pd.DataFrame:
        return self.load_field("sweep_results")


def _target_dims(config: Config, cap: int) -> list[int]:
    if config.target_pca_dims is None:
        return _power_grid(cap)
    dims = list(map(int, config.target_pca_dims))
    too_large = [value for value in dims if value > cap]
    if too_large:
        raise ValueError(
            f"target_pca_dims {too_large} exceed effective PCA cap {cap}"
        )
    if cap not in dims:
        raise ValueError(
            f"target_pca_dims must include the effective full endpoint {cap}"
        )
    return dims


def _inspect_ieeg_dimensions(config):
    import h5py

    from src.core.registry import DatasetContext, get_dataset_adapter
    from src.utils.rrr.significance import (
        ieeg_target_dimension_summary,
        load_significance_selection,
    )

    enc = config.encoding
    subject_ids = enc.subjects or list(range(1, 10))
    significance = load_significance_selection(
        config.significance_selection, results_root=config.results_root,
        subjects=subject_ids, modality="ieeg",
    )
    original_targets, cap = ieeg_target_dimension_summary(
        significance, subject_ids
    )

    if enc.model == "glove":
        embedding_dim = int(enc.per_model_config["dim"])
    else:
        model_dir = enc.per_model_config.get("model_dir", "gpt2-xl")
        artifact_dataset = significance.reference.dataset
        prefix = "brain-treebank-"
        stimulus_id = (
            artifact_dataset.removeprefix(prefix)
            if artifact_dataset.startswith(prefix) else artifact_dataset
        )
        context = DatasetContext(
            dataset=enc.dataset,
            stimulus_id=stimulus_id,
            stimulus_label=stimulus_id,
        )
        embedding_dir = enc.per_model_config.get("embedding_dir")
        if embedding_dir is None:
            embedding_dir = get_dataset_adapter(enc.dataset).get_embedding_dir(
                enc.data_root, context, model_dir
            )
        path = f"{embedding_dir}/features.hdf5"
        layer = int(enc.per_model_config["layer"])
        with h5py.File(path, "r") as handle:
            embedding_dim = int(handle[f"layer-{layer}"].shape[-1])
    return original_targets, cap, embedding_dim


def _inspect_fmri_dimensions(config):
    from src.utils.rrr.significance import load_significance_selection
    from src.utils.shared.fmri import (
        align_fmri_stimuli,
        build_fmri_stimulus,
        load_story_embeddings,
        resolve_fmri_stories,
    )

    significance = load_significance_selection(
        config.significance_selection, results_root=config.results_root,
        subjects=config.fmri.subjects, modality="fmri",
    )
    original_targets = sum(
        int(frame["significant"].astype(bool).sum())
        for frame in significance.target_significance.values()
    )
    train_stories, test_stories, _ = resolve_fmri_stories(
        config.fmri.fmri_repo_root, config.fmri.sessions
    )
    all_stories = train_stories + test_stories
    embeddings = load_story_embeddings(
        config.fmri.embedding_root, config.model_dir, all_stories,
        config.layer, config.pooling,
    )
    embedding_dim = next(iter(embeddings.values())).shape[1]
    aligned = align_fmri_stimuli(
        config.fmri.fmri_repo_root, all_stories, embeddings,
        trim=config.fmri.trim,
    )
    train = build_fmri_stimulus(
        aligned, train_stories, ndelays=config.fmri.ndelays
    )
    cap = min(original_targets, train.stimulus.shape[0])
    return int(original_targets), int(cap), int(embedding_dim)


def _plot_heatmap(frame: pd.DataFrame, metric: str):
    methods = list(dict.fromkeys(frame["method"].astype(str)))
    figure, axes = plt.subplots(
        1, len(methods), figsize=(5 * len(methods), 4), squeeze=False
    )
    for axis, method in zip(axes[0], methods):
        current = frame[frame["method"] == method]
        pivot = current.pivot(index="target_pca_dim", columns="rank", values=metric)
        image = axis.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
        axis.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
        axis.set_yticks(range(len(pivot.index)), labels=pivot.index)
        axis.set_xlabel("RRR rank")
        axis.set_ylabel("Target PCA dimension")
        axis.set_title(f"{method.upper()} {metric}")
        figure.colorbar(image, ax=axis)
    figure.tight_layout()
    return figure


def _compile(artifacts, original_targets: int, cap: int):
    frames = []
    for artifact in artifacts:
        frame = artifact.load_mean_results()
        frame["target_pca_dim"] = int(artifact.target_pca_dim)
        frame["original_target_count"] = int(original_targets)
        frame["effective_target_pca_cap"] = int(cap)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["target_pca_dim", "method", "rank"]
    )
    owner = max(artifacts, key=lambda item: int(item.target_pca_dim))
    owner.save_sweep_results(combined, overwrite=True)
    for metric in ("corr", "r2"):
        figure = _plot_heatmap(combined, metric)
        owner.save_plot(
            name=f"target_dimensionality_{metric}", figure=figure, overwrite=True
        )
        plt.close(figure)
    manifest = owner.manifest()
    required = {
        field: [entry["selector"] for entry in manifest["entries"].get(field, {}).values()]
        for field, is_required in owner.FIELD_CONTRACT.items()
        if is_required
    }
    owner.mark_complete(required)
    return combined


def _run_expanded(config: Config, selected):
    if config.modality == "fmri":
        from src.utils.rrr.rrr_fmri import run_fmri
        inspect = _inspect_fmri_dimensions
        runner = run_fmri
        expanded = expand_fmri_layer_configs(selected)
    else:
        from src.utils.rrr.rrr_ieeg import run_ieeg
        inspect = _inspect_ieeg_dimensions
        runner = run_ieeg
        expanded = expand_per_model_layer_configs(selected)

    all_artifacts = []
    for layer_config in expanded:
        original_targets, cap, embedding_dim = inspect(layer_config)
        dims = _target_dims(config, cap)
        artifacts = []
        for target_dim in dims:
            current = deepcopy(layer_config)
            current.rank_grid = _power_grid(min(target_dim, embedding_dim))
            if config.modality == "ieeg":
                current.compute_sig_nonsig = False
            artifact = runner(current, target_pca_dim=target_dim)
            artifact.target_pca_dim = int(target_dim)
            artifacts.append(artifact)
        _compile(artifacts, original_targets, cap)
        all_artifacts.extend(artifacts)
    return all_artifacts


def run_config(config: Config):
    config.validate()
    return _run_expanded(config, config.selected_config())


def main():
    args, overrides = parse_args(__doc__)
    config = load_config(args.config, overrides, Config)
    run_config(config)


if __name__ == "__main__":
    main()
