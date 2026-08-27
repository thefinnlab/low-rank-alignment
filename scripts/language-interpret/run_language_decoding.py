"""Decode word-aligned language features from saved RRR/PCA input subspaces."""

from __future__ import annotations

import json
import tempfile
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.run_rrr import Artifact as RRRArtifact
from scripts.run_significance import Artifact as SignificanceArtifact
from src.core.artifacts import ArtifactStore, normalize_dataset, normalize_model
from src.core.config import LanguageDecodingConfig
from src.core.registry import resolve_dataset_context
from src.utils.shared.config import expand_per_model_layer_configs, load_config, parse_args


class Artifact(ArtifactStore):
    """Storage and downstream-loading contract for language-decoding results."""

    SCRIPT = "run_language_decoding"
    SELECTOR_ORDER = ("task", "subspace", "rank", "fold", "name")
    FIELD_CONTRACT = {
        "effective_config": True, "upstream_identities": True,
        "word_selection": True, "run_metadata": True, "fit_metrics": True,
        "decoder_weights": True, "heldout_predictions": True,
        "grid_results": True, "summary": True, "iu_alignment": False,
        "prediction_agreement": False, "prediction_agreement_summary": False,
        "plots": False,
    }

    def __init__(self, *, mode="within_dataset", dataset="podcast", offset=0,
                 model="gpt2-xl", layer=24, variant=None, results_root="results"):
        self._init_store(mode=mode, dataset=dataset, offset=offset, model=model,
                         layer=layer, variant=variant, results_root=results_root)

    @classmethod
    def from_config(cls, config):
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
            "effective_config", "upstream_identities", "word_selection",
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
    else:
        _, value = max(entries, key=lambda item: int(item[0].get("rank", 0)))
        path = destination / "fold_0"
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "A.npy", value)
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
        "word_selection": "word_selection.csv", "run_metadata": "run_metadata.json",
        "grid_results": "grid_results.csv", "summary": "summary.csv",
        "iu_alignment": "iu_alignment.csv",
        "prediction_agreement": "prediction_agreement.csv",
        "prediction_agreement_summary": "prediction_agreement_summary.csv",
    }
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
    artifact = Artifact.from_config(config).begin(
        config=config, upstreams=[source.identity_record()], refresh=config.refresh
    )
    if artifact.complete and not config.refresh:
        return artifact
    with tempfile.TemporaryDirectory(prefix="language-decoding-") as temporary:
        root = Path(temporary)
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
