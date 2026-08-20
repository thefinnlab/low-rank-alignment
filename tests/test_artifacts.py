import json

import numpy as np
import pytest

from src.core.artifacts import (
    ArtifactAmbiguityError,
    ArtifactError,
    ArtifactIncompleteError,
    normalize_dataset,
    normalize_model,
    readable_variant,
)
from scripts.run_significance import Artifact as SignificanceArtifact
from scripts.run_rrr import Artifact as RRRArtifact, UnitSelector


def test_canonical_identity_and_defaults(tmp_path):
    artifact = SignificanceArtifact(
        dataset="podcast", model="glove-300", layer=None, results_root=tmp_path
    )
    assert artifact.identity == {
        "script": "run_significance", "mode": "sig_lags", "dataset": "podcast",
        "offset": 0, "model": "glove-300", "layer": None,
    }
    assert normalize_dataset("brain_treebank", "cars-2") == "brain-treebank-cars-2"
    assert normalize_dataset("fmri") == "deep-fmri"
    assert normalize_model("gpt2-xl-batch") == "gpt2-xl-batch"
    assert normalize_model("glove", glove_dim=300) == "glove-300"


def test_selective_fields_partial_completion_and_decode_errors(tmp_path):
    artifact = SignificanceArtifact(results_root=tmp_path).begin(
        config={"subjects": [1], "refresh": False}
    )
    artifact.save_subject_field("encoding_results", 1, {"test_idxs": [[0]]})
    artifact.save_subject_field("correlations", 1, np.ones((1, 1, 1)))
    loaded = SignificanceArtifact(results_root=tmp_path).load_artifacts(
        ["encoding_results"], selectors={"subjects": [1]}
    )
    assert loaded.correlations is None
    assert next(iter(loaded.encoding_results.values()))["test_idxs"] == [[0]]
    with pytest.raises(ArtifactIncompleteError):
        artifact.mark_complete({
            "encoding_results": [{"subject": "1"}],
            "correlations": [{"subject": "1"}],
        })

    manifest = artifact.manifest()
    entry = next(iter(manifest["entries"]["encoding_results"].values()))
    payload = artifact._variant_dir() / entry["file"]
    payload.write_bytes(payload.read_bytes() + b"corrupt")
    with pytest.raises(ArtifactError, match="Could not decode"):
        artifact.load_field("encoding_results", selector={"subject": "1"})


def test_variant_ambiguity_and_refresh(tmp_path):
    for value, variant in ((1, "first"), (2, "second")):
        artifact = SignificanceArtifact(results_root=tmp_path, variant=variant).begin(
            config={"scientific_parameter": value}
        )
        artifact.save_field("effective_config", {"value": value})
    with pytest.raises(ArtifactAmbiguityError):
        SignificanceArtifact(results_root=tmp_path).load_artifacts(["effective_config"])

    artifact = SignificanceArtifact(results_root=tmp_path, variant="first").begin(
        config={"scientific_parameter": 1}, refresh=True
    )
    assert not artifact.has_field("effective_config")


def test_readable_variant_collision_requires_explicit_name(tmp_path):
    SignificanceArtifact(results_root=tmp_path).begin(config={"parameter": 1})
    with pytest.raises(ArtifactAmbiguityError, match="variant_name"):
        SignificanceArtifact(results_root=tmp_path).begin(config={"parameter": 2})


def test_rrr_variant_distinguishes_significance_selection_and_source_sets():
    reference = {"dataset": "podcast", "model": "gpt2", "layer": 12}
    within = {
        "significance_selection": {
            "mode": "within-layer", "reference_artifact": reference,
        }
    }
    cross = {
        "significance_selection": {
            "mode": "cross-model-layer",
            "reference_artifact": reference,
            "artifacts": [{"model": "gpt2", "layer": [0, 6, 12]}],
        }
    }
    reordered_equivalent = {
        "significance_selection": {
            "mode": "cross-model-layer",
            "reference_artifact": reference,
            "artifacts": [
                {"model": "gpt2", "layer": 12},
                {"model": "gpt2", "layer": [6, 0]},
            ],
        }
    }

    within_variant = readable_variant(within)
    cross_variant = readable_variant(cross)

    assert within_variant.startswith("significance-within-layer-")
    assert cross_variant.startswith("significance-cross-model-layer-")
    assert within_variant != cross_variant
    assert cross_variant == readable_variant(reordered_equivalent)


def test_rrr_artifact_loads_variant_from_significance_selection(tmp_path):
    reference = {
        "mode": "sig_lags", "dataset": "podcast", "offset": 0,
        "model": "gpt2", "layer": 12,
    }
    selection = {
        "mode": "within-layer", "reference_artifact": reference,
    }
    other_selection = {
        "mode": "cross-model-layer", "reference_artifact": reference,
        "artifacts": [{"dataset": "podcast", "model": "gpt2", "layer": [6, 12]}],
    }
    for value, configured_selection in ((1, selection), (2, other_selection)):
        artifact = RRRArtifact(results_root=tmp_path).begin(config={
            "significance_selection": configured_selection,
            "scientific_parameter": value,
        })
        artifact.save_field("effective_config", {"value": value})

    loaded = RRRArtifact(
        results_root=tmp_path, significance_selection=selection
    ).load_artifacts(["effective_config"])
    assert loaded.effective_config == {"value": 1}
    assert loaded.variant == readable_variant({"significance_selection": selection})


def test_rrr_artifact_rejects_variant_and_significance_selection(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        RRRArtifact(
            results_root=tmp_path,
            variant="manual",
            significance_selection={"reference_artifact": {"dataset": "podcast"}},
        )


def test_rrr_manifest_resume_is_selector_complete(tmp_path):
    artifact = RRRArtifact(results_root=tmp_path).begin(config={"rank_grid": [1, 2]})
    unit = UnitSelector(method="rrr", region="superior-temporal")
    artifact.save_factors(unit=unit, fold=0, input_factors=np.eye(2),
                          target_factors=np.eye(2))
    artifact.save_rank_metrics(unit=unit, rank=1, fold=0,
                               metrics={"nr_corr": 0.2})
    plan = artifact.resume_plan(unit=unit, ranks=[1, 2], folds=[0],
                                require_target_factors=True)
    assert plan["completed_pairs"] == {(1, 0)}
    assert plan["ranks_by_fold"] == {0: [2]}


def test_atomic_manifest_contains_codec_and_size(tmp_path):
    artifact = SignificanceArtifact(results_root=tmp_path).begin(config={"x": 1})
    artifact.save_field("effective_config", {"x": 1})
    entry = next(iter(artifact.manifest()["entries"]["effective_config"].values()))
    assert entry["codec"] == "json"
    assert entry["size"] > 0
    assert "sha256" not in entry
    assert not list(tmp_path.rglob(".*.tmp"))


def test_native_codecs_and_readable_selector_paths(tmp_path):
    import pandas as pd

    artifact = RRRArtifact(results_root=tmp_path).begin(config={"rank_grid": [1]})
    unit = UnitSelector(method="rrr")
    artifact.save_rank_metrics(
        unit=unit, rank=1, fold=0, metrics={"corr": np.array([0.25])}
    )
    artifact.save_fold_results(
        unit=unit, results=pd.DataFrame({"fold": [0], "score": [0.25]})
    )
    artifact.save_field(
        "plots", b"png", selector={"subject": "UTS01", "name": "acf_diagnostic"}
    )

    files = {path.relative_to(artifact._variant_dir()).as_posix()
             for path in artifact._variant_dir().rglob("*") if path.is_file()}
    assert "rank_metrics/method-rrr/rank-1/fold-0/metrics.npz" in files
    assert "fold_results/method-rrr/fold_results.csv" in files
    assert "plots/subject-UTS01/acf_diagnostic.png" in files


def test_selector_order_is_owned_by_each_script(tmp_path):
    significance = SignificanceArtifact(results_root=tmp_path).begin(config={"x": 1})
    significance.save_field(
        "diagnostics", {"ok": True},
        selector={"z_custom": "last", "subject": "UTS01"},
    )
    significance_entry = next(iter(
        significance.manifest()["entries"]["diagnostics"].values()
    ))
    assert significance_entry["file"] == (
        "diagnostics/subject-UTS01/z_custom-last/diagnostics.json"
    )

    rrr = RRRArtifact(results_root=tmp_path).begin(config={"rank_grid": [1]})
    rrr.save_field(
        "rank_metrics", {"corr": np.array([0.25])},
        selector={"fold": 2, "method": "rrr", "rank": 64},
    )
    rrr_entry = next(iter(rrr.manifest()["entries"]["rank_metrics"].values()))
    assert rrr_entry["file"] == (
        "rank_metrics/method-rrr/rank-64/fold-2/metrics.npz"
    )
