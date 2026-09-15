"""Decode word-aligned language features from saved RRR/PCA input subspaces."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.run_rrr import Artifact as RRRArtifact
from scripts.run_significance import Artifact as SignificanceArtifact
from src.core.artifacts import ArtifactStore, normalize_dataset, normalize_model
from src.core.config import LanguageDecodingConfig, LanguageDecodingMode
from src.core.registry import resolve_dataset_context
from src.utils.shared.config import expand_per_model_layer_configs, load_config, parse_args


class Artifact(ArtifactStore):
    """Storage and downstream-loading contract for language-decoding results."""

    SCRIPT = "run_language_decoding"
    SELECTOR_ORDER = ("task", "subspace", "rank", "fold", "name")
    FIELD_CONTRACT = {
        "effective_config": True, "upstream_identities": True,
        "sample_selection": True, "word_selection": False,
        "run_metadata": True, "fit_metrics": True,
        "decoder_weights": True, "heldout_predictions": True,
        "grid_results": True, "summary": True, "iu_alignment": False,
        "prediction_agreement": False, "prediction_agreement_summary": False,
        "target_diagnostics": False, "channel_metrics": False,
        "plots": False,
    }

    def __init__(self, *, mode="within_dataset", dataset="podcast", offset=0,
                 model="gpt2-xl", layer=24, variant=None, results_root="results"):
        self._init_store(mode=mode, dataset=dataset, offset=offset, model=model,
                         layer=layer, variant=variant, results_root=results_root)

    @classmethod
    def from_config(cls, config, source=None):
        if config.mode in {
            LanguageDecodingMode.FMRI_WORD,
            LanguageDecodingMode.FMRI_TR_PRE_FIR,
        }:
            if source is None:
                raise ValueError("fMRI decoding Artifact identity requires its RRR source")
            identity = source.identity
            return cls(
                mode=config.mode.value,
                dataset=identity["dataset"],
                offset=identity["offset"],
                model=identity["model"],
                layer=identity["layer"],
                variant=config.variant_name,
                results_root=config.results_root,
            )
        enc = config.encoding
        context = resolve_dataset_context(
            enc.dataset, enc.subjects or list(range(1, 10)),
            enc.per_dataset_config, enc.data_root,
        )
        model = enc.per_model_config.get("model_dir", enc.model)
        return cls(
            mode=config.mode.value,
            dataset=normalize_dataset(enc.dataset, context.stimulus_id),
            offset=enc.offset,
            model=normalize_model(model, glove_dim=enc.per_model_config.get("dim")),
            layer=None if enc.model == "glove" else enc.per_model_config.get("layer"),
            variant=config.variant_name,
            results_root=config.results_root,
        )

    def load_artifacts(self, fields, selectors=None):
        scalar = {
            "effective_config", "upstream_identities", "sample_selection",
            "word_selection",
            "run_metadata", "grid_results", "summary", "iu_alignment",
            "prediction_agreement", "prediction_agreement_summary",
        }
        for field in fields:
            setattr(self, field, self.load_field(field) if field in scalar
                    else self.load_collection(field, selectors))
        self.complete = bool(self.manifest().get("complete"))
        return self

    def save_artifacts(self, *, output, config, upstream):
        _publish(self, Path(output), config, upstream)
        return self


def _source_artifact(config) -> RRRArtifact:
    if not config.rrr_source:
        raise ValueError("The executable language-decoding client requires rrr_source")
    kwargs = dict(config.rrr_source)
    if config.mode in {
        LanguageDecodingMode.FMRI_WORD,
        LanguageDecodingMode.FMRI_TR_PRE_FIR,
    } and kwargs.get("mode", "internal") != "internal":
        raise ValueError("fMRI decoding requires a normal internal RRR source Artifact")
    kwargs.setdefault("results_root", config.results_root)
    source = RRRArtifact(**kwargs)
    if source.variant is None:
        source._resolve_variant()
    return source


def _factor_entries(source, method, selectors):
    entries = source.load_collection(
        "input_factors", {**dict(selectors or {}), "method": method}
    )
    return sorted(
        ((dict(selector), np.asarray(value)) for selector, value in entries.items()),
        key=lambda item: (item[0].get("fold", -1), item[0].get("rank", -1)),
    )


def _materialize_method(source, method, root, selectors):
    entries = _factor_entries(source, method, selectors)
    if not entries:
        return None
    destination = root / method
    fold_entries = [(selector, value) for selector, value in entries if "fold" in selector]
    if fold_entries:
        for selector, value in fold_entries:
            path = destination / f"fold_{int(selector['fold'])}"
            path.mkdir(parents=True, exist_ok=True)
            np.save(path / "A.npy", value)
            scaler = source.load_field("input_scalers", selector=selector)
            np.savez(path / "input_scaler.npz", **scaler)
    else:
        ranked = [(selector, value) for selector, value in entries if "rank" in selector]
        materialized = ranked or [max(
            entries, key=lambda item: int(item[0].get("rank", 0))
        )]
        for selector, value in materialized:
            directory = (
                f"rank_{int(selector['rank'])}" if "rank" in selector else "fold_0"
            )
            path = destination / directory
            path.mkdir(parents=True, exist_ok=True)
            np.save(path / "A.npy", value)
            if source.has_field("input_scalers", selector=selector):
                scaler = source.load_field("input_scalers", selector=selector)
                np.savez(path / "input_scaler.npz", **scaler)
    return destination


def _materialize_significance(source, root, subjects):
    identity = next(
        (item for item in source.load_field("upstream_identities")
         if item.get("script") == "run_significance"), None,
    )
    if identity is None:
        raise ValueError("RRR source has no significance Artifact dependency")
    kwargs = {key: value for key, value in identity.items() if key != "script"}
    kwargs["results_root"] = source.results_root
    available = SignificanceArtifact(**kwargs).load_collection("encoding_results")
    requested = {str(subject) for subject in subjects} if subjects else None
    for selector, value in available.items():
        subject = str(dict(selector)["subject"])
        if requested is not None and subject not in requested:
            continue
        path = root / "subjects" / f"sub_{subject}"
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(value, path / "encoding_results.joblib")


def _selector_from_fit(path, task_root):
    parts = path.relative_to(task_root).parts
    task, subspace = parts[0], parts[1]
    if subspace == "full":
        rank, fold_part = None, parts[2]
    else:
        rank, fold_part = int(parts[2].removeprefix("k")), parts[3]
    selector = {
        "task": task, "subspace": subspace,
        "fold": int(fold_part.removeprefix("fold_")),
    }
    if rank is not None:
        selector["rank"] = rank
    return selector


def _publish(artifact, output, config, upstream):
    overwrite = bool(config.refresh)
    artifact.save_field("effective_config", asdict(config), overwrite=overwrite)
    artifact.save_field("upstream_identities", [upstream], overwrite=overwrite)
    scalar_files = {
        "sample_selection": "sample_selection.csv",
        "word_selection": "word_selection.csv", "run_metadata": "run_metadata.json",
        "grid_results": "grid_results.csv", "summary": "summary.csv",
        "iu_alignment": "iu_alignment.csv",
        "prediction_agreement": "prediction_agreement.csv",
        "prediction_agreement_summary": "prediction_agreement_summary.csv",
    }
    if not (output / "sample_selection.csv").exists():
        word_selection = output / "word_selection.csv"
        if word_selection.exists():
            pd.read_csv(word_selection).to_csv(
                output / "sample_selection.csv", index=False
            )
    for field, filename in scalar_files.items():
        path = output / filename
        if path.exists():
            value = (json.loads(path.read_text()) if path.suffix == ".json"
                     else pd.read_csv(path))
            if field == "run_metadata":
                for key in (
                    "subspace_unit_dirs", "rrr_unit_dir", "pca_unit_dir",
                    "fold_results_dir", "output_dir", "multiview_source_config",
                ):
                    value.pop(key, None)
                value["rrr_source"] = upstream
            artifact.save_field(field, value, overwrite=overwrite)
    for metrics_path in (output / "tasks").glob("**/metrics.json"):
        selector = _selector_from_fit(metrics_path.parent, output / "tasks")
        artifact.save_field(
            "fit_metrics", pd.DataFrame([json.loads(metrics_path.read_text())]),
            selector=selector, overwrite=overwrite,
        )
        for field, filename in (
            ("decoder_weights", "weights.npz"),
            ("heldout_predictions", "heldout_predictions.npz"),
        ):
            with np.load(metrics_path.parent / filename, allow_pickle=True) as values:
                artifact.save_field(field, dict(values), selector=selector,
                                    overwrite=overwrite)
        classifier_path = metrics_path.parent / "classifier_weights.npz"
        if classifier_path.exists():
            with np.load(classifier_path, allow_pickle=True) as values:
                artifact.save_field(
                    "decoder_weights", dict(values),
                    selector={**selector, "name": "dominant_label_classifier"},
                    overwrite=overwrite,
                )
        channel_path = metrics_path.parent / "channel_metrics.csv"
        if channel_path.exists():
            artifact.save_field(
                "channel_metrics", pd.read_csv(channel_path),
                selector=selector, overwrite=overwrite,
            )
    for metadata_path in (output / "target_diagnostics").glob("*/metadata.json"):
        task = metadata_path.parent.name
        artifact.save_field(
            "target_diagnostics", json.loads(metadata_path.read_text()),
            selector={"task": task, "name": "metadata"}, overwrite=overwrite,
        )
        availability_path = metadata_path.parent / "availability.npz"
        with np.load(availability_path, allow_pickle=True) as values:
            artifact.save_field(
                "target_diagnostics", dict(values),
                selector={"task": task, "name": "availability"},
                overwrite=overwrite,
            )
        with np.load(metadata_path.parent / "targets.npz", allow_pickle=True) as values:
            artifact.save_field(
                "target_diagnostics", dict(values),
                selector={"task": task, "name": "targets"},
                overwrite=overwrite,
            )
    plot = output / "rank_analysis.png"
    if plot.exists():
        artifact.save_field("plots", plot.read_bytes(),
                            selector={"name": "rank_analysis"}, overwrite=overwrite)
    selectors = [entry["selector"] for entry in
                 artifact.manifest()["entries"]["fit_metrics"].values()]
    artifact.mark_complete({
        "fit_metrics": selectors, "decoder_weights": selectors,
        "heldout_predictions": selectors,
    })


def run_config(config: LanguageDecodingConfig) -> Artifact:
    source = _source_artifact(config)
    artifact = Artifact.from_config(config, source=source).begin(
        config=config, upstreams=[source.identity_record()], refresh=config.refresh
    )
    if artifact.complete and not config.refresh:
        return artifact
    with tempfile.TemporaryDirectory(prefix="language-decoding-") as temporary:
        root = Path(temporary)
        if config.mode in {
            LanguageDecodingMode.FMRI_WORD,
            LanguageDecodingMode.FMRI_TR_PRE_FIR,
        }:
            source_config = source.load_field("effective_config")
            factors = {}
            selectors = dict(config.rrr_selectors or {})
            for method in ("rrr", "pca"):
                entries = source.load_collection(
                    "input_factors", {**selectors, "method": method}
                )
                for selector, value in entries.items():
                    mapping = dict(selector)
                    if "rank" in mapping and int(mapping["rank"]) in set(
                        map(int, config.rank_grid)
                    ):
                        key = (method, int(mapping["rank"]))
                        if key in factors:
                            raise ValueError(
                                f"rrr_selectors match multiple fMRI factors for "
                                f"method={method}, rank={mapping['rank']}"
                            )
                        factors[key] = np.asarray(value)
            missing_rrr = sorted(
                set(map(int, config.rank_grid))
                - {rank for method, rank in factors if method == "rrr"}
            )
            if missing_rrr:
                raise FileNotFoundError(
                    f"RRR source Artifact is missing requested fMRI ranks {missing_rrr}"
                )
            from src.utils.language_interpret.fmri_decoding import run_fmri_decoding
            output = run_fmri_decoding(
                config, source_config, factors, root / "output"
            )
            artifact.save_artifacts(
                output=output, config=config, upstream=source.identity_record()
            )
            return artifact
        rrr = _materialize_method(source, "rrr", root, config.rrr_selectors)
        if rrr is None:
            raise FileNotFoundError("RRR source Artifact has no matching rrr factors")
        pca = _materialize_method(source, "pca", root, config.rrr_selectors)
        _materialize_significance(source, root, config.encoding.subjects)
        runtime = deepcopy(config)
        runtime.rrr_run_path, runtime.pca_run_path = str(rrr), None if pca is None else str(pca)
        runtime.fold_results_dir, runtime.output_dir = str(root), str(root / "output")
        from src.utils.language_interpret.decoding import run_language_decoding
        output = run_language_decoding(runtime)
        artifact.save_artifacts(
            output=output, config=config, upstream=source.identity_record()
        )
    return artifact


def main():
    args, overrides = parse_args("Decode linguistic features from RRR/PCA subspaces")
    config = load_config(args.config, overrides, LanguageDecodingConfig)
    for layer_config in expand_per_model_layer_configs(config):
        layer = layer_config.encoding.per_model_config.get("layer")
        print(f"\nLanguage decoding for layer {layer}")
        artifact = run_config(layer_config)
        print(f"Done: {artifact._variant_dir()}")


if __name__ == "__main__":
    main()
