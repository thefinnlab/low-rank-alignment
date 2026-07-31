#!/usr/bin/env python
"""Decode fMRI internal RRR/PCA spaces on the shared held-out story."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.config import FMRICrossDatasetRRRConfig, LanguageDecodingConfig
from src.utils.config_utils import expand_fmri_layer_configs, load_config
from src.utils.fmri import load_story_embeddings, load_story_word_data, resolve_fmri_stories
from src.utils.language_decoding import (
    _configure_decoder_backend,
    _decoder_metrics,
    _fit_decoder,
    _weight_payload,
)
from src.utils.language_tasks import get_language_task


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--max-rank", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _layer_configs(path: str, model: str):
    manifest = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if model not in manifest["models"]:
        raise ValueError(f"Unknown model {model!r}; available={sorted(manifest['models'])}")
    spec = manifest["models"][model]
    source = load_config(spec["rrr_config"], {}, FMRICrossDatasetRRRConfig)
    source.output_dir = str(spec["result_template"])
    return manifest, expand_fmri_layer_configs(source)


def _complete(config: FMRICrossDatasetRRRConfig) -> bool:
    root = Path(config.output_dir)
    return any((root / "rrr" / f"k{k}" / "A.npy").is_file() and
               (root / "pca" / f"k{k}" / "A.npy").is_file()
               for k in config.rank_grid)


def _load_labels(stories, feature_root: str, tasks: list[str], self_surprisal_root=None):
    tables = {}
    for story in stories:
        path = Path(feature_root) / story / "tokens.tsv"
        table = pd.read_csv(path, sep="\t", keep_default_na=False, low_memory=False)
        self_tasks = {"self_surprisal", "self_entropy"} & set(tasks)
        if self_tasks:
            if self_surprisal_root is None:
                raise ValueError(
                    f"{sorted(self_tasks)} require self_surprisal_root"
                )
            self_path = Path(self_surprisal_root) / story / "tokens.tsv"
            self_table = pd.read_csv(
                self_path, sep="\t", keep_default_na=False, low_memory=False
            )
            if len(self_table) != len(table) or not np.array_equal(
                self_table["word"].astype(str), table["word"].astype(str)
            ):
                raise ValueError(f"Self-surprisal alignment failed for {story}")
            missing_self = self_tasks - set(self_table)
            if missing_self:
                raise ValueError(
                    f"{self_path} is missing columns {sorted(missing_self)}"
                )
            for task in self_tasks:
                table[task] = pd.to_numeric(self_table[task], errors="coerce")
        missing = set(tasks) - set(table)
        if missing:
            raise ValueError(f"{story} is missing feature columns {sorted(missing)}")
        tables[story] = table
    targets = {}
    for task in tasks:
        spec = get_language_task(task)
        values = pd.concat([tables[story][task] for story in stories], ignore_index=True)
        if spec.kind == "continuous":
            targets[task] = pd.to_numeric(values, errors="coerce").to_numpy(float)
        else:
            present = values.notna()
            labels = values[present].astype(str)
            mapping = {label: i for i, label in enumerate(sorted(labels.unique()))}
            target = np.full(len(values), np.nan)
            target[present] = labels.map(mapping).to_numpy(float)
            targets[task] = target
    return tables, targets


def _session_cv(stories, tables, groups, valid):
    slices = {}
    cursor = 0
    for story in stories:
        slices[story] = np.arange(cursor, cursor + len(tables[story]))
        cursor += len(tables[story])
    retained = np.flatnonzero(valid)
    position = np.full(len(valid), -1, dtype=int)
    position[retained] = np.arange(len(retained))
    splits = []
    for session_stories in groups.values():
        validation = np.concatenate([slices[story] for story in session_stories])
        validation = validation[valid[validation]]
        validation_positions = position[validation]
        train_positions = np.setdiff1d(np.arange(len(retained)), validation_positions)
        splits.append((train_positions, validation_positions))
    return splits


def _run(manifest, config, model, device, refresh, run_name=None):
    train_stories, test_stories, groups = resolve_fmri_stories(
        config.fmri.fmri_repo_root, config.fmri.sessions
    )
    if test_stories != ["wheretheressmoke"]:
        raise ValueError(f"Expected held-out wheretheressmoke, got {test_stories}")
    stories = train_stories + test_stories
    embeddings = load_story_embeddings(
        config.fmri.embedding_root, config.model_dir, stories, config.layer, config.pooling
    )
    frames, _ = load_story_word_data(config.fmri.fmri_repo_root, stories)
    self_root = manifest.get("self_surprisal_root")
    if self_root:
        self_root = Path(self_root) / model / "fmri"
    tables, targets = _load_labels(
        stories, manifest["feature_root"], manifest["tasks"], self_root
    )
    for story in stories:
        if len(embeddings[story]) != len(tables[story]) or not np.array_equal(
            frames[story]["word"].astype(str), tables[story]["word"].astype(str)
        ):
            raise ValueError(f"Embedding/feature alignment failed for {story}")
    X_train = np.vstack([embeddings[story] for story in train_stories]).astype(np.float32)
    X_test = np.asarray(embeddings[test_stories[0]], dtype=np.float32)
    n_train = len(X_train)
    decoder = LanguageDecodingConfig(
        rrr_run_path="unused", device=device or manifest.get("device", "auto"),
        tasks=list(manifest["tasks"]), refresh=refresh,
    )
    solver_params = dict(decoder.encoding.ridge_solver_params or {})
    solver_params.setdefault("n_alphas_batch", 1)
    decoder.encoding.ridge_solver_params = solver_params
    _configure_decoder_backend(decoder.device)
    output_identity = "heldout_wheretheressmoke"
    if run_name:
        output_identity += f"__{run_name}"
    output = Path(config.output_dir) / "language_decoding" / output_identity
    if (output / "metrics.csv").is_file() and not refresh:
        print(f"Reusing {output}")
        return
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    designs = []
    paired_ranks = [
        int(rank)
        for rank in config.rank_grid
        if (Path(config.output_dir) / "rrr" / f"k{rank}" / "A.npy").is_file()
        and (Path(config.output_dir) / "pca" / f"k{rank}" / "A.npy").is_file()
    ]
    for method in ("rrr", "pca"):
        for rank in paired_ranks:
            factor = Path(config.output_dir) / method / f"k{rank}" / "A.npy"
            designs.append((method, rank, np.load(factor)))
    for method, rank, factor in designs:
        train_design = X_train if factor is None else X_train @ factor
        test_design = X_test if factor is None else X_test @ factor
        for task in manifest["tasks"]:
            spec = get_language_task(task)
            y = targets[task]
            train_y, test_y = y[:n_train], y[n_train:]
            train_valid, test_valid = np.isfinite(train_y), np.isfinite(test_y)
            cv = _session_cv(train_stories, tables, groups, train_valid)
            pipeline = _fit_decoder(
                train_design[train_valid], train_y[train_valid], spec.kind, decoder, cv=cv
            )
            prediction = pipeline.predict(test_design[test_valid])
            metrics = {
                "dataset": "fmri", "model": model, "layer": int(config.layer),
                "test_story": test_stories[0], "method": method, "rank": rank,
                "task": task, "kind": spec.kind,
                "n_train": int(train_valid.sum()), "n_test": int(test_valid.sum()),
                **_decoder_metrics(spec.kind, test_y[test_valid], prediction),
            }
            rows.append(metrics)
            artifact = output / "tasks" / task / method / ("full" if rank is None else f"k{rank}")
            artifact.mkdir(parents=True, exist_ok=True)
            np.savez(artifact / "weights.npz", **_weight_payload(pipeline, factor))
            np.savez_compressed(
                artifact / "heldout_predictions.npz",
                y_true=np.asarray(test_y[test_valid]),
                y_pred=np.asarray(prediction),
                row_index=np.flatnonzero(test_valid),
                block_id=tables[test_stories[0]].loc[
                    test_valid, "sentence_id"
                ].astype(str).to_numpy(dtype=str),
            )
            (artifact / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True))
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    (output / "metadata.json").write_text(json.dumps({
        "dataset": "fmri", "model": model, "layer": int(config.layer),
        "train_stories": train_stories, "test_story": test_stories[0],
        "inner_cv": "leave_one_session_out", "tasks": manifest["tasks"],
        "paired_ranks": paired_ranks,
    }, indent=2))
    print(f"Done: {output}")


def main():
    args = _args()
    if args.run_name and not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_name):
        raise ValueError("--run-name may contain only letters, numbers, '.', '_', and '-'")
    manifest, configs = _layer_configs(args.config, args.model)
    if args.max_rank is not None:
        if args.max_rank < 1:
            raise ValueError("--max-rank must be positive")
        for config in configs:
            config.rank_grid = [
                int(rank) for rank in config.rank_grid if int(rank) <= args.max_rank
            ]
            if not config.rank_grid:
                raise ValueError(
                    f"No configured ranks are at or below max_rank={args.max_rank}"
                )
    if args.tasks is not None:
        unknown = sorted(set(args.tasks) - set(manifest["tasks"]))
        if unknown:
            raise ValueError(
                f"Unknown task(s) {unknown}; available={manifest['tasks']}"
            )
        manifest["tasks"] = list(args.tasks)
    if args.dry_run:
        for config in configs:
            print(f"{args.model}: layer {config.layer}: " + ("completed_pair" if _complete(config) else "missing_pair"))
        return
    completed = [config for config in configs if _complete(config)]
    if not completed:
        print(f"No completed fMRI internal RRR/PCA layers for {args.model}; nothing to do")
        return
    for config in completed:
        _run(
            manifest, config, args.model, args.device, args.refresh, args.run_name
        )


if __name__ == "__main__":
    main()
