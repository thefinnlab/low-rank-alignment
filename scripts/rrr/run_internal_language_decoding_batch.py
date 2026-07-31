#!/usr/bin/env python
"""Decode one model's completed internal RRR/PCA layers for one dataset."""

from __future__ import annotations

import argparse
import os
import re
import sys
from copy import deepcopy
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.config import LanguageDecodingConfig, LanguageDecodingMode, RRRConfig
from src.utils.config_utils import expand_per_model_layer_configs, load_config
from src.utils.language_decoding import run_language_decoding


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Dataset batch manifest")
    parser.add_argument("--model", required=True, help="Model key in the manifest")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--max-rank", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_layer_configs(
    manifest_path: str,
    model: str,
    tasks: list[str] | None = None,
    max_rank: int | None = None,
    run_name: str | None = None,
) -> list[LanguageDecodingConfig]:
    payload = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8")) or {}
    models = payload.get("models", {})
    if model not in models:
        raise ValueError(f"Unknown model {model!r}; available={sorted(models)}")
    model_spec = models[model]
    rrr_config = load_config(model_spec["rrr_config"], {}, RRRConfig)
    task_data = dict(payload["task_data"])
    self_surprisal_root = payload.get("self_surprisal_root")
    if self_surprisal_root:
        prediction_metrics_path = str(
            Path(self_surprisal_root)
            / model
            / payload["self_surprisal_dataset"]
            / payload["self_surprisal_stimulus"]
            / "tokens.tsv"
        )
        task_data["self_surprisal"] = prediction_metrics_path
        task_data["self_entropy"] = prediction_metrics_path
    config = LanguageDecodingConfig(
        encoding=deepcopy(rrr_config.encoding),
        rrr_run_path=str(model_spec["result_template"]) + "/rrr",
        pca_run_path=str(model_spec["result_template"]) + "/pca",
        mode=LanguageDecodingMode.WITHIN_DATASET,
        source_name=(
            f"{payload['dataset']}_{model}_internal"
            + (f"__{run_name}" if run_name else "")
        ),
        device=str(payload.get("device", "auto")),
        rank_grid=[
            int(rank)
            for rank in rrr_config.rank_grid
            if max_rank is None or int(rank) <= max_rank
        ],
        tasks=[str(task) for task in (tasks or payload["tasks"])],
        task_data=task_data,
        logistic_cs=[float(value) for value in payload.get(
            "logistic_cs", [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
        )],
        refresh=bool(payload.get("refresh", False)),
    )
    if not config.rank_grid:
        raise ValueError(f"No configured ranks are at or below max_rank={max_rank}")
    return expand_per_model_layer_configs(config)


def _completed_pair(config: LanguageDecodingConfig) -> bool:
    rrr = Path(config.rrr_run_path)
    pca = Path(config.pca_run_path)
    return bool(list(rrr.glob("fold_*/A.npy"))) and bool(list(pca.glob("fold_*/A.npy")))


def main() -> None:
    args = _parse_args()
    if args.max_rank is not None and args.max_rank < 1:
        raise ValueError("--max-rank must be positive")
    if args.run_name and not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_name):
        raise ValueError("--run-name may contain only letters, numbers, '.', '_', and '-'")
    configs = build_layer_configs(
        args.config, args.model, args.tasks, args.max_rank, args.run_name
    )
    if args.dry_run:
        for config in configs:
            layer = config.encoding.per_model_config["layer"]
            status = "completed_pair" if _completed_pair(config) else "missing_pair"
            print(
                f"{args.model}: layer {layer}: {status}: "
                f"rrr={config.rrr_run_path} pca={config.pca_run_path}"
            )
        return
    completed = [config for config in configs if _completed_pair(config)]
    skipped = [
        config.encoding.per_model_config["layer"]
        for config in configs
        if not _completed_pair(config)
    ]
    if skipped:
        print(f"Skipping layers without completed internal RRR+PCA: {skipped}")
    if not completed:
        print(f"No completed internal RRR/PCA layer pairs for {args.model!r}; nothing to do")
        return
    for config in completed:
        if args.device is not None:
            config.device = args.device
        if args.refresh:
            config.refresh = True
        layer = config.encoding.per_model_config["layer"]
        print(f"{args.model}: internal RRR/PCA language decoding at layer {layer}")
        print(f"Done: {run_language_decoding(config)}")


if __name__ == "__main__":
    main()
