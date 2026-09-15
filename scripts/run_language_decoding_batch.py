#!/usr/bin/env python
"""Run every configured layer for one model in a language-decoding dataset pair."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_language_decoding import run_config
from scripts.run_rrr import Config as RRRClientConfig
from src.core.config import EncodingConfig, LanguageDecodingConfig
from src.utils.shared.config import (
    apply_overrides,
    expand_fmri_layer_configs,
    expand_per_model_layer_configs,
    load_config,
    parse_override_args,
)


FMRI_DATASETS = {"fmri", "deep-fmri", "deep_fmri"}


def _rrr_config_path(dataset: str, model: str, *, source: bool, cross: bool) -> str:
    if dataset in FMRI_DATASETS:
        return f"configs/fmri/within_fmri_cross_model_layer/{model}.yml"
    suffix = (
        "full_fit.yml" if source and cross else
        "cross_model_layer.yml" if source else
        "cross_subject.yml"
    )
    if dataset == "podcast":
        return f"configs/rrr/{model}/{suffix}"
    if dataset == "cars-2":
        return f"configs/brain_treebank/cars_2/rrr/{model}/{suffix}"
    raise ValueError(f"Unsupported language-decoding dataset {dataset!r}")


def _expanded_rrr(path: str):
    public = load_config(path, {}, RRRClientConfig)
    selected = public.selected_config()
    expand = (
        expand_fmri_layer_configs
        if public.modality == "fmri"
        else expand_per_model_layer_configs
    )
    return public, expand(selected)


def _layer(config) -> int | None:
    if hasattr(config, "encoding"):
        return config.encoding.per_model_config.get("layer")
    return config.layer


def _model(config) -> str:
    if hasattr(config, "encoding"):
        return config.encoding.per_model_config.get("model_dir", config.encoding.model)
    return config.model_dir


def _source_request(public, selected) -> dict:
    layer = _layer(selected)
    request = {
        "mode": "internal" if public.mode.value == "internal" else public.mode.value,
        "dataset": "deep-fmri" if public.modality == "fmri" else public.dataset,
        "offset": 0 if public.modality == "fmri" else selected.encoding.offset,
        "model": _model(selected),
        "layer": layer,
        "significance_selection": asdict(selected.significance_selection),
        "results_root": public.results_root,
    }
    if public.modality == "fmri":
        request["subspace_aggregation"] = selected.subspace_aggregation
    else:
        rereference = selected.encoding.per_dataset_config.get("rereference")
        if rereference:
            request["rereference"] = rereference
    return request


def _target_encoding(public, selected) -> EncodingConfig:
    if public.modality == "ieeg":
        return deepcopy(selected.encoding)
    model = selected.model_dir
    return EncodingConfig(
        dataset="deep-fmri",
        model="glove" if model.startswith("glove") else "transformer",
        offset=0,
        per_model_config={
            "model_dir": model,
            "layer": selected.layer,
            **({"dim": 300} if model.startswith("glove") else {}),
        },
    )


def build_configs(manifest_path: str, model: str) -> list[LanguageDecodingConfig]:
    payload = yaml.safe_load(Path(manifest_path).read_text()) or {}
    target = str(payload["target_dataset"])
    source = str(payload.get("source_dataset", target))
    cross = source != target
    target_public, targets = _expanded_rrr(
        _rrr_config_path(target, model, source=False, cross=cross)
    )
    source_public, sources = _expanded_rrr(
        _rrr_config_path(source, model, source=True, cross=cross)
    )
    source_by_layer = {_layer(item): item for item in sources}
    configs = []
    for target_item in targets:
        layer = _layer(target_item)
        if layer not in source_by_layer:
            raise ValueError(
                f"Source and target layer grids differ for {model}: missing {layer}"
            )
        source_item = source_by_layer[layer]
        target_is_fmri = target in FMRI_DATASETS
        kwargs = {
            "encoding": _target_encoding(target_public, target_item),
            "rrr_source": _source_request(source_public, source_item),
            "mode": "fmri_tr_pre_fir" if target_is_fmri else (
                "cross_dataset" if cross else "within_dataset"
            ),
            "source_name": source,
            "rank_grid": payload.get("rank_grid", [1, 2, 4, 8, 16, 32, 64, 128, 256]),
            "tasks": list(payload["tasks"]) + (list(payload.get("tr_tasks", [])) if target_is_fmri else []),
            "task_data": dict(payload["task_data"]),
            "device": payload.get("device", "auto"),
            "results_root": payload.get("results_root", "results"),
            "refresh": bool(payload.get("refresh", False)),
        }
        if target_is_fmri and cross:
            kwargs["fmri_target_artifact"] = _source_request(target_public, target_item)
        configs.append(LanguageDecodingConfig(**kwargs))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Run only this layer (the default runs every configured layer)",
    )
    args, unknown = parser.parse_known_args()
    overrides = parse_override_args(unknown)
    payload = yaml.safe_load(Path(args.config).read_text())
    for config in build_configs(args.config, args.model):
        if args.layer is not None and _layer(config) != args.layer:
            continue
        config = apply_overrides(config, overrides)
        if config.mode != "fmri_tr_pre_fir":
            tr_tasks = payload.get("tr_tasks", [])
            config.tasks = [task for task in config.tasks if task not in tr_tasks]
        offsets = ([config.label_offset] if "label_offset" in overrides
                   or config.mode == "fmri_tr_pre_fir"
                   else payload.get("label_offsets", [0]))
        for offset in offsets:
            run = deepcopy(config)
            run.label_offset = offset
            run.validate()
            print(
                f"Language decoding {run.source_name} -> {run.encoding.dataset}, "
                f"{args.model}, layer {_layer(run)}, label offset {offset}",
                flush=True,
            )
            artifact = run_config(run)
            print(f"Done: {artifact._variant_dir()}", flush=True)


if __name__ == "__main__":
    main()
