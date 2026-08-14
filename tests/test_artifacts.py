import json

import numpy as np
import pytest

from src.core.artifacts import (
    ArtifactAmbiguityError,
    ArtifactError,
    ArtifactIncompleteError,
    normalize_dataset,
    normalize_model,
)
from scripts.sig_lags.run_significance import Artifact as SignificanceArtifact
from scripts.rrr.run_rrr import Artifact as RRRArtifact, UnitSelector


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


def test_selective_fields_partial_completion_and_checksums(tmp_path):
    artifact = SignificanceArtifact(results_root=tmp_path).begin(
        config={"subjects": [1], "refresh": False}
    )
    artifact.save_subject(1, encoding_results={"test_idxs": [[0]]},
                          correlations=np.ones((1, 1, 1)))
    loaded = SignificanceArtifact(results_root=tmp_path).load_artifacts(
        ["encoding_results"], selectors={"subjects": [1]}
    )
    assert loaded.correlations is None
    assert next(iter(loaded.encoding_results.values()))["test_idxs"] == [[0]]
    with pytest.raises(ArtifactIncompleteError):
        artifact.mark_complete({
            "encoding_results": [{"subject": 1}],
            "correlations": [{"subject": 1}],
        })

    manifest = artifact.manifest()
    entry = next(iter(manifest["entries"]["encoding_results"].values()))
    payload = artifact._variant_dir() / entry["file"]
    payload.write_bytes(payload.read_bytes() + b"corrupt")
    with pytest.raises(ArtifactError, match="Checksum mismatch"):
        artifact.load_field("encoding_results", selector={"subject": 1})


def test_variant_ambiguity_and_refresh(tmp_path):
    for value in (1, 2):
        artifact = SignificanceArtifact(results_root=tmp_path).begin(
            config={"scientific_parameter": value}
        )
        artifact.save_field("effective_config", {"value": value})
    with pytest.raises(ArtifactAmbiguityError):
        SignificanceArtifact(results_root=tmp_path).load_artifacts(["effective_config"])

    artifact = SignificanceArtifact(results_root=tmp_path).begin(
        config={"scientific_parameter": 1}, refresh=True
    )
    assert not artifact.has_field("effective_config")


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


def test_atomic_manifest_contains_codec_and_checksum(tmp_path):
    artifact = SignificanceArtifact(results_root=tmp_path).begin(config={"x": 1})
    artifact.save_field("effective_config", {"x": 1})
    entry = next(iter(artifact.manifest()["entries"]["effective_config"].values()))
    assert entry["codec"] == "json"
    assert len(entry["sha256"]) == 64
    assert not list(tmp_path.rglob(".*.tmp"))
