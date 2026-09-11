"""Predict saved RRR/PCA embedding subspaces from language-task variables."""

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
from src.core.config import LanguageReconstructionConfig
from src.core.registry import resolve_dataset_context
from src.utils.shared.config import expand_per_model_layer_configs, load_config, parse_args


class Artifact(ArtifactStore):
    """Storage and downstream-loading contract for language reconstruction."""

    SCRIPT = "run_language_reconstruction"
    SELECTOR_ORDER = (
        "predictor_set", "subspace", "target_view", "rank", "fold", "name"
    )
    FIELD_CONTRACT = {
        "effective_config": True, "upstream_identities": True,
        "word_selection": True, "task_schema": True, "run_metadata": True,
        "fit_metrics": True, "fitted_models": False, "grid_results": True,
        "fold_results": True, "summary": True,
        "comparison_grid_results": True, "rrr_pca_comparison": True,
        "predictor_comparison_grid": True, "predictor_comparison": True,
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
        scalar = set(self.FIELD_CONTRACT) - {"fit_metrics", "fitted_models", "plots"}
        for field in fields:
            setattr(self, field, self.load_field(field) if field in scalar
                    else self.load_collection(field, selectors))
        self.complete = bool(self.manifest().get("complete"))
        return self

    def save_artifacts(self, *, output, config, upstream):
        _publish(self, Path(output), config, upstream)
        return self


def _source_artifact(config):
    if not config.rrr_source:
        raise ValueError("The executable language-reconstruction client requires rrr_source")
    kwargs = dict(config.rrr_source)
    kwargs.setdefault("results_root", config.results_root)
    source = RRRArtifact(**kwargs)
    if source.variant is None:
        source._resolve_variant()
    return source


def _materialize_method(source, method, destination, selectors):
    entries = source.load_collection(
        "input_factors", {**dict(selectors or {}), "method": method}
    )
    ordered = sorted(
        ((dict(selector), np.asarray(value)) for selector, value in entries.items()),
        key=lambda item: (item[0].get("fold", -1), item[0].get("rank", -1)),
    )
    if not ordered:
        raise FileNotFoundError(f"RRR source Artifact has no matching {method} factors")
    folds = [(selector, value) for selector, value in ordered if "fold" in selector]
    for selector, value in folds or [({"fold": 0}, max(
        ordered, key=lambda item: int(item[0].get("rank", 0))
    )[1])]:
        path = destination / f"fold_{int(selector['fold'])}"
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "A.npy", value)
        if "fold" in selector:
            scaler = source.load_field("input_scalers", selector=selector)
            np.savez(path / "input_scaler.npz", **scaler)


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


def _fit_selector(path, fits_root):
    predictor, subspace, view, rank, fold = path.relative_to(fits_root).parts
    return {
        "predictor_set": predictor, "subspace": subspace, "target_view": view,
        "rank": int(rank.removeprefix("k")), "fold": int(fold.removeprefix("fold_")),
    }


def _publish(artifact, output, config, upstream):
    overwrite = bool(config.refresh)
    artifact.save_field("effective_config", asdict(config), overwrite=overwrite)
    artifact.save_field("upstream_identities", [upstream], overwrite=overwrite)
    files = {
        "word_selection": "word_selection.csv", "task_schema": "task_schema.json",
        "run_metadata": "run_metadata.json", "grid_results": "grid_results.csv",
        "fold_results": "fold_results.csv", "summary": "summary.csv",
        "comparison_grid_results": "comparison_grid_results.csv",
        "rrr_pca_comparison": "rrr_pca_comparison.csv",
        "predictor_comparison_grid": "predictor_comparison_grid.csv",
        "predictor_comparison": "predictor_comparison.csv",
    }
    for field, filename in files.items():
        path = output / filename
        value = json.loads(path.read_text()) if path.suffix == ".json" else pd.read_csv(path)
        if field == "run_metadata":
            for key in ("rrr_unit_dir", "pca_unit_dir", "fold_results_dir"):
                value.pop(key, None)
            value["rrr_source"] = upstream
        artifact.save_field(field, value, overwrite=overwrite)
    for metrics_path in (output / "fits").glob("**/metrics.json"):
        selector = _fit_selector(metrics_path.parent, output / "fits")
        artifact.save_field(
            "fit_metrics", pd.DataFrame([json.loads(metrics_path.read_text())]),
            selector=selector, overwrite=overwrite,
        )
        if config.save_models:
            models = {
                "preprocessor": joblib.load(metrics_path.parent / "preprocessor.joblib"),
                "model": joblib.load(metrics_path.parent / "model.joblib"),
            }
            artifact.save_field("fitted_models", models, selector=selector,
                                overwrite=overwrite)
    for plot in output.glob("*.png"):
        artifact.save_field("plots", plot.read_bytes(),
                            selector={"name": plot.stem}, overwrite=overwrite)
    selectors = [entry["selector"] for entry in
                 artifact.manifest()["entries"]["fit_metrics"].values()]
    artifact.mark_complete({"fit_metrics": selectors})


def run_config(config: LanguageReconstructionConfig) -> Artifact:
    source = _source_artifact(config)
    artifact = Artifact.from_config(config).begin(
        config=config, upstreams=[source.identity_record()], refresh=config.refresh
    )
    if artifact.complete and not config.refresh:
        return artifact
    with tempfile.TemporaryDirectory(prefix="language-reconstruction-") as temporary:
        root = Path(temporary)
        rrr, pca = root / "rrr", root / "pca"
        _materialize_method(source, "rrr", rrr, config.rrr_selectors)
        _materialize_method(source, "pca", pca, config.rrr_selectors)
        _materialize_significance(source, root, config.encoding.subjects)
        runtime = deepcopy(config)
        runtime.rrr_run_path, runtime.fold_results_dir = str(rrr), str(root)
        runtime.output_dir = str(root / "output")
        from src.utils.language_interpret.reconstruction import run_language_reconstruction
        output = run_language_reconstruction(runtime)
        artifact.save_artifacts(
            output=output, config=config, upstream=source.identity_record()
        )
    return artifact


def main():
    args, overrides = parse_args(
        "Reconstruct RRR/PCA subspaces from language-task variables"
    )
    config = load_config(args.config, overrides, LanguageReconstructionConfig)
    for layer_config in expand_per_model_layer_configs(config):
        layer = layer_config.encoding.per_model_config.get("layer")
        print(f"\nLanguage reconstruction for layer {layer}")
        artifact = run_config(layer_config)
        print(f"Done: {artifact._variant_dir()}")


if __name__ == "__main__":
    main()
