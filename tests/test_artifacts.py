import json

import numpy as np
import pandas as pd
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
from scripts.run_language_decoding import Artifact as DecodingArtifact
from scripts.run_language_reconstruction import Artifact as ReconstructionArtifact
from src.utils.rrr.significance import validate_cross_dataset_significance_alignment


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


def test_language_artifact_facades_expose_script_owned_contracts(tmp_path):
    decoding = DecodingArtifact(results_root=tmp_path).begin(config={"tasks": ["pos"]})
    reconstruction = ReconstructionArtifact(results_root=tmp_path).begin(
        config={"predictor_sets": ["tasks"]}
    )

    assert decoding.identity["script"] == "run_language_decoding"
    assert reconstruction.identity["script"] == "run_language_reconstruction"


def test_dataframe_codec_accepts_object_columns_containing_lists(tmp_path):
    artifact = DecodingArtifact(results_root=tmp_path).begin(config={"tasks": ["pos"]})
    metrics = pd.DataFrame({
        "task": ["pos"],
        "classifier_classes": [["NOUN", "VERB"]],
    })

    artifact.save_field(
        "fit_metrics", metrics,
        selector={"task": "pos", "subspace": "rrr", "rank": 1, "fold": 0},
    )

    loaded = artifact.load_field(
        "fit_metrics",
        selector={"task": "pos", "subspace": "rrr", "rank": 1, "fold": 0},
    )
    assert loaded.loc[0, "classifier_classes"] == "['NOUN', 'VERB']"


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


@pytest.mark.parametrize("legacy_upstreams", [None, "missing"])
def test_legacy_manifest_without_upstreams_resumes(tmp_path, legacy_upstreams):
    artifact = SignificanceArtifact(results_root=tmp_path).begin(
        config={"parameter": 1}
    )
    manifest_path = artifact._manifest_path()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if legacy_upstreams == "missing":
        manifest.pop("upstreams")
    else:
        manifest["upstreams"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    resumed = SignificanceArtifact(results_root=tmp_path).begin(
        config={"parameter": 1}, refresh=False
    )

    assert resumed.variant == "default"


def test_legacy_manifest_without_upstreams_rejects_expected_dependency(tmp_path):
    artifact = SignificanceArtifact(results_root=tmp_path).begin(
        config={"parameter": 1}
    )
    manifest_path = artifact._manifest_path()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("upstreams")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactAmbiguityError, match="variant_name"):
        SignificanceArtifact(results_root=tmp_path).begin(
            config={"parameter": 1},
            upstreams=[{"script": "run_significance", "model": "gpt2"}],
            refresh=False,
        )


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


def test_rrr_artifact_loads_laplacian_variant_from_source_parameters(tmp_path):
    selection = {
        "mode": "within-layer",
        "reference_artifact": {
            "mode": "sig_lags",
            "dataset": "brain-treebank-cars-2",
            "offset": 0,
            "model": "gpt2",
            "layer": 12,
            "variant": "rereference-laplacian",
        },
    }
    config = {
        "encoding": {
            "per_dataset_config": {
                "movie": "cars-2",
                "rereference": "laplacian",
            },
        },
        "significance_selection": selection,
    }
    producer = RRRArtifact(
        dataset="brain-treebank-cars-2",
        model="gpt2",
        layer=12,
        results_root=tmp_path,
    ).begin(
        config=config,
        variant_extras={"rereference": "laplacian"},
    )
    producer.save_field("effective_config", config)

    loaded = RRRArtifact(
        dataset="brain-treebank-cars-2",
        model="gpt2",
        layer=12,
        significance_selection=selection,
        rereference="laplacian",
        results_root=tmp_path,
    ).load_artifacts(["effective_config"])

    assert loaded.variant.startswith("rereference-laplacian/")
    assert loaded.effective_config == config


def test_subspace_aggregation_is_a_distinct_readable_variant(tmp_path):
    selection = {
        "mode": "within-layer",
        "reference_artifact": {
            "mode": "held_out_session",
            "dataset": "deep-fmri",
            "offset": 0,
            "model": "gpt2",
            "layer": 12,
        },
    }
    extrinsic_config = {
        "significance_selection": selection,
        "subspace_aggregation": "extrinsic",
    }
    factor_config = {
        "significance_selection": selection,
        "subspace_aggregation": "factor_weighted",
    }
    prediction_config = {
        "significance_selection": selection,
        "subspace_aggregation": "prediction_weighted",
    }
    unspecified_config = {"significance_selection": selection}

    extrinsic = RRRArtifact(results_root=tmp_path).begin(config=extrinsic_config)
    factor = RRRArtifact(results_root=tmp_path).begin(config=factor_config)
    prediction = RRRArtifact(results_root=tmp_path).begin(config=prediction_config)
    unspecified = RRRArtifact(results_root=tmp_path).begin(config=unspecified_config)

    assert extrinsic.variant.startswith("aggregation-extrinsic/significance-")
    assert factor.variant.startswith("aggregation-factor_weighted/significance-")
    assert prediction.variant.startswith(
        "aggregation-prediction_weighted/significance-"
    )
    assert unspecified.variant.startswith("significance-within-layer-")
    assert "aggregation-" not in unspecified.variant
    assert len({
        extrinsic.variant, factor.variant, prediction.variant, unspecified.variant,
    }) == 4

    loaded = RRRArtifact(
        results_root=tmp_path,
        significance_selection=selection,
        subspace_aggregation="factor_weighted",
    )
    assert loaded.manifest()["configuration"]["subspace_aggregation"] == (
        "factor_weighted"
    )


def test_source_variant_properties_propagate_into_downstream_variant():
    base_source = {
        "script": "run_rrr",
        "mode": "internal",
        "dataset": "deep-fmri",
        "offset": 0,
        "model": "gpt2",
        "layer": 12,
    }
    target_config = {
        "significance_selection": {
            "mode": "within-layer",
            "reference_artifact": {
                "dataset": "podcast",
                "model": "gpt2",
                "layer": 12,
            },
        }
    }

    extrinsic = readable_variant(
        target_config,
        [{
            **base_source,
            "variant": "aggregation-extrinsic/significance-within-layer-source",
        }],
    )
    factor = readable_variant(
        target_config,
        [
            {
                **base_source,
                "variant": (
                    "aggregation-factor_weighted/"
                    "significance-within-layer-source"
                ),
            }
        ],
    )
    prediction = readable_variant(
        target_config,
        [
            {
                **base_source,
                "variant": (
                    "aggregation-prediction_weighted/"
                    "significance-within-layer-source"
                ),
            }
        ],
    )
    laplacian = readable_variant(
        target_config,
        [{
            **base_source,
            "variant": (
                "rereference-laplacian/aggregation-extrinsic/"
                "significance-within-layer-source"
            ),
        }],
        {"rereference": "laplacian"},
    )

    assert "source-aggregation-extrinsic" in extrinsic
    assert "source-aggregation-factor_weighted" in factor
    assert "source-aggregation-prediction_weighted" in prediction
    assert "source-rereference-laplacian" in laplacian
    assert "/rereference-laplacian/" in f"/{laplacian}/"
    assert len({extrinsic, factor, prediction}) == 3


def test_rrr_artifact_rejects_variant_and_significance_selection(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        RRRArtifact(
            results_root=tmp_path,
            variant="manual",
            significance_selection={"reference_artifact": {"dataset": "podcast"}},
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        RRRArtifact(
            results_root=tmp_path,
            variant="manual",
            subspace_aggregation="factor_weighted",
        )


def test_cross_dataset_significance_alignment_allows_only_dataset_to_differ():
    target_identity = {
        "dataset": "brain-treebank-cars-2", "offset": 0,
        "model": "gpt-j-6b", "layer": 0,
    }
    source_identity = {
        "dataset": "deep-fmri", "offset": 0,
        "model": "gpt-j-6b", "layer": 0,
    }
    target = {
        "mode": "within-layer",
        "reference_artifact": {
            **target_identity, "mode": "sig_lags",
        },
    }
    source = {
        "mode": "within-layer",
        "reference_artifact": {
            **source_identity, "mode": "held_out_session",
        },
    }
    validate_cross_dataset_significance_alignment(
        target, target_identity, source, source_identity
    )

    source["reference_artifact"]["layer"] = 12
    with pytest.raises(ValueError, match="does not match its RRR Artifact identity"):
        validate_cross_dataset_significance_alignment(
            target, target_identity, source, source_identity
        )


def test_cross_dataset_significance_alignment_accepts_matching_cross_layer_unions():
    target_identity = {
        "dataset": "podcast", "offset": 0, "model": "gpt2", "layer": 0,
    }
    source_identity = {
        "dataset": "deep-fmri", "offset": 0, "model": "gpt2", "layer": 0,
    }
    artifacts = [
        {"offset": 0, "model": "gpt2", "layer": [0, 3, 6, 9, 12]},
        {"offset": 0, "model": "glove-300", "layer": None},
    ]
    target = {
        "mode": "cross-model-layer",
        "reference_artifact": {**target_identity, "mode": "sig_lags"},
        "artifacts": [
            {**item, "dataset": "podcast", "mode": "sig_lags"}
            for item in artifacts
        ],
    }
    source = {
        "mode": "cross-model-layer",
        "reference_artifact": {
            **source_identity, "mode": "held_out_session",
        },
        "artifacts": [
            {**item, "dataset": "deep-fmri", "mode": "held_out_session"}
            for item in artifacts
        ],
    }

    validate_cross_dataset_significance_alignment(
        target, target_identity, source, source_identity
    )

    source["artifacts"][0]["layer"] = [0, 6, 12]
    with pytest.raises(ValueError, match="same model/layer union"):
        validate_cross_dataset_significance_alignment(
            target, target_identity, source, source_identity
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


def test_rrr_compacts_data_and_keeps_readable_plot_paths(tmp_path):
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
    assert "artifact.h5" in files
    assert "plots/subject-UTS01/acf_diagnostic.png" in files
    assert not any(path.endswith((".csv", ".npy", ".npz", ".json"))
                   for path in files if not path.endswith("manifest.json"))
    assert np.array_equal(
        artifact.load_rank_metrics(unit=unit, rank=1, fold=0)["corr"],
        np.array([0.25]),
    )
    assert artifact.load_fold_results().to_dict("records") == [
        {"fold": 0, "score": 0.25, "method": "rrr"}
    ]


def test_npz_codec_round_trips_ragged_nested_significance_payload(tmp_path):
    artifact = SignificanceArtifact(results_root=tmp_path).begin(config={"x": 1})
    payload = {
        "max_lag_correlation_per_electrode": [
            np.ones((3, 2)),
            np.ones((3, 4)),
        ],
        "subjects": np.array([1, 2]),
        "electrodes": [["A1", "A2"], ["B1", "B2", "B3", "B4"]],
        "num_null_shifts": 3,
        "subject_shift_metadata": {
            1: {"shift_matrix": np.array([[1, 2], [3, 4]])},
            2: {"shift_matrix": np.array([[5, 6]])},
        },
    }

    artifact.save_subject_field("null_distribution", "all", payload)
    loaded = artifact.load_field(
        "null_distribution", selector={"subject": "all"}
    )

    assert loaded["electrodes"] == payload["electrodes"]
    assert loaded["num_null_shifts"] == 3
    assert np.array_equal(loaded["subjects"], payload["subjects"])
    for actual, expected in zip(
        loaded["max_lag_correlation_per_electrode"],
        payload["max_lag_correlation_per_electrode"],
    ):
        assert np.array_equal(actual, expected)
    assert np.array_equal(
        loaded["subject_shift_metadata"][1]["shift_matrix"],
        payload["subject_shift_metadata"][1]["shift_matrix"],
    )


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
    assert rrr_entry["container"] == "artifact.h5"
    assert rrr_entry["object"].startswith(
        "/entries/rank_metrics/method-rrr/rank-64/fold-2/metrics/versions/"
    )
