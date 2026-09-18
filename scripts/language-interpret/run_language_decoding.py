"""Decode word-aligned language features from saved RRR/PCA input subspaces."""

from __future__ import annotations

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
    COMPACT_STORAGE = True
    SELECTOR_ORDER = ("task", "subspace", "rank", "fold", "name")
    FIELD_CONTRACT = {
        "effective_config": True, "upstream_identities": True,
        "sample_selection": True, "word_selection": False,
        "run_metadata": True, "fit_metrics": True,
        "decoder_weights": True, "heldout_predictions": True,
        "grid_results": True, "summary": True, "iu_alignment": False,
        "prediction_agreement": False, "prediction_agreement_summary": False,
        "target_diagnostics": False, "channel_metrics": False,
        "speaker_class_mapping": False,
        "plots": False,
    }

    def __init__(self, *, mode="within_dataset", dataset="podcast", offset=0,
                 model="gpt2-xl", layer=24, variant=None, results_root="results"):
        self._init_store(mode=mode, dataset=dataset, offset=offset, model=model,
                         layer=layer, variant=variant, results_root=results_root)

    @classmethod
    def from_config(cls, config, source=None, target=None):
        if config.mode in {
            LanguageDecodingMode.FMRI_WORD,
            LanguageDecodingMode.FMRI_TR_PRE_FIR,
        }:
            if source is None:
                raise ValueError("fMRI decoding Artifact identity requires its RRR source")
            identity = (target or source).identity
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

    def finish(self):
        selectors = [entry["selector"] for entry in
                     self.manifest()["entries"]["fit_metrics"].values()]
        self.mark_complete({field: selectors for field in
                            ("fit_metrics", "decoder_weights", "heldout_predictions")})
        return self


def _source_artifact(config) -> RRRArtifact:
    if not config.rrr_source:
        raise ValueError("The executable language-decoding client requires rrr_source")
    kwargs = dict(config.rrr_source)
    if config.mode in {
        LanguageDecodingMode.FMRI_WORD,
        LanguageDecodingMode.FMRI_TR_PRE_FIR,
    } and (
        not config.fmri_target_artifact
        and kwargs.get("mode", "internal") != "internal"
    ):
        raise ValueError("Within-fMRI decoding requires a normal internal RRR source Artifact")
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


def _source_scaler(source, selector):
    try:
        scaler = source.load_input_scaler(selector)
    except KeyError as error:
        raise ValueError(
            f"Source factor {selector} is missing its input scaler"
        ) from error
    if scaler.get("coordinates") != "standardized-v1":
        raise ValueError(
            f"Source factor {selector} has unsupported scaler coordinates "
            f"{scaler.get('coordinates')!r}"
        )
    return scaler


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
            scaler = _source_scaler(source, selector)
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
            scaler = _source_scaler(source, selector)
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


def run_config(config: LanguageDecodingConfig) -> Artifact:
    source = _source_artifact(config)
    target = None
    if config.fmri_target_artifact:
        kwargs = dict(config.fmri_target_artifact)
        kwargs.setdefault("results_root", config.results_root)
        target = RRRArtifact(**kwargs)
        if target.variant is None:
            target._resolve_variant()
    upstreams = [source.identity_record()]
    if target is not None:
        upstreams.append(target.identity_record())
    artifact = Artifact.from_config(config, source=source, target=target).begin(
        config=config, upstreams=upstreams, refresh=config.refresh,
        variant_extras={
            **({"speaker-labels": "outer-fold-shared-other"}
               if {"speaker_identity", "native_speaker"} & set(config.tasks)
               else {}),
            "input-scaling": "source-then-target-reduced-v1",
        },
    )
    if artifact.complete and not config.refresh:
        return artifact
    artifact.save_field("effective_config", asdict(config), overwrite=True)
    artifact.save_field("upstream_identities", upstreams, overwrite=True)
    with tempfile.TemporaryDirectory(prefix="language-decoding-") as temporary:
        root = Path(temporary)
        if config.mode in {
            LanguageDecodingMode.FMRI_WORD,
            LanguageDecodingMode.FMRI_TR_PRE_FIR,
        }:
            source_config = (target or source).load_field("effective_config")
            factors = {}
            # The source transform precedes the RRR/PCA branch, so one scaler
            # serves every method and rank loaded below.
            source_scaler = _source_scaler(source, {})
            selectors = dict(config.rrr_selectors or {})
            for method in ("rrr", "pca"):
                entries = source.load_collection(
                    "input_factors", {**selectors, "method": method}
                )
                unranked = []
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
                    elif "rank" not in mapping:
                        unranked.append((mapping, np.asarray(value)))
                if unranked:
                    if len(unranked) != 1 or any(
                        key_method == method for key_method, _ in factors
                    ):
                        raise ValueError(
                            f"rrr_selectors do not identify one {method} factor set"
                        )
                    _, full = unranked[0]
                    for rank in map(int, config.rank_grid):
                        if full.shape[1] >= rank:
                            factors[(method, rank)] = full[:, :rank]
            missing_rrr = sorted(
                set(map(int, config.rank_grid))
                - {rank for method, rank in factors if method == "rrr"}
            )
            if missing_rrr:
                raise FileNotFoundError(
                    f"RRR source Artifact is missing requested fMRI ranks {missing_rrr}"
                )
            from src.utils.language_interpret.fmri_decoding import run_fmri_decoding
            run_fmri_decoding(
                config, source_config, factors, source_scaler,
                root / "output", artifact=artifact
            )
            return artifact.finish()
        rrr = _materialize_method(source, "rrr", root, config.rrr_selectors)
        if rrr is None:
            raise FileNotFoundError("RRR source Artifact has no matching rrr factors")
        pca = _materialize_method(source, "pca", root, config.rrr_selectors)
        _materialize_significance(source, root, config.encoding.subjects)
        runtime = deepcopy(config)
        runtime.rrr_run_path, runtime.pca_run_path = str(rrr), None if pca is None else str(pca)
        runtime.fold_results_dir, runtime.output_dir = str(root), str(root / "output")
        from src.utils.language_interpret.decoding import run_language_decoding
        run_language_decoding(runtime, artifact=artifact)
        artifact.finish()
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
