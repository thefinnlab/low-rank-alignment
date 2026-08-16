from pathlib import Path

import json
import numpy as np
import pandas as pd
import pytest

from scripts.run_rrr import Artifact as RRRArtifact
from scripts.utilities.migrate_results import (
    Mapping, _abstract_legacy_dependencies, _rrr_source_from_legacy_path,
    copy_mapping, discover_ambiguous, discover_mappings,
)


def mapping(source):
    return Mapping(
        source=str(source), producer="run_significance", mode="sig_lags",
        dataset="podcast", offset=0, model="gpt2-xl-batch", layer=24,
        variant=None,
    )


def test_dry_run_apply_and_idempotence(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "all_subject_sig.csv").write_text("subject,elec\n1,A1\n")
    results = tmp_path / "canonical"
    dry = copy_mapping(mapping(legacy), results, apply=False)
    assert dry["operations"][0]["action"] == "convert"
    assert not results.exists()
    copy_mapping(mapping(legacy), results, apply=True)
    second = copy_mapping(mapping(legacy), results, apply=True)
    assert second["operations"][0]["action"] == "skip-identical"


def test_conflict_refusal_preserves_destination(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    source = legacy / "all_subject_sig.csv"
    source.write_text("new")
    results = tmp_path / "canonical"
    item = mapping(legacy)
    copy_mapping(item, results, apply=True)
    source.write_text("different")
    with pytest.raises(FileExistsError, match="Differing-content"):
        copy_mapping(item, results, apply=True)


def test_discovery_excludes_new_canonical_tree(tmp_path):
    legacy = tmp_path / "results" / "old"
    legacy.mkdir(parents=True)
    (legacy / "grid_results.csv").write_text("k\n1\n")
    canonical = tmp_path / "results" / "run_rrr" / "cross_subject"
    canonical.mkdir(parents=True)
    (canonical / "grid_results.csv").write_text("k\n1\n")
    found = discover_ambiguous(tmp_path / "results", tmp_path / "results")
    assert [entry["source"] for entry in found] == [str(legacy)]


def test_discovery_infers_configured_identity_and_reports_shared_root(tmp_path):
    legacy = tmp_path / "results" / "gpt2-xl_layer24_offset0"
    legacy.mkdir(parents=True)
    (legacy / "config.yml").write_text(
        "encoding:\n  dataset: podcast\n  model: transformer\n"
        "  offset: 0\n  per_model_config:\n    model_dir: gpt2-xl\n    layer: 24\n"
    )
    (legacy / "all_subject_sig.csv").write_text("subject,elec\n1,A1\n")
    (legacy / "summary.csv").write_text("k\n1\n")
    inferred, ambiguous = discover_mappings(tmp_path / "results", tmp_path / "results")
    assert inferred == []
    assert any(item["reason"] == "shared or conflicting producer markers"
               for item in ambiguous)


def test_shared_mapping_converts_to_selected_producer(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "all_subject_sig.csv").write_text("subject,elec\n1,A1\n")
    (legacy / "summary.csv").write_text("k\n1\n")
    results = tmp_path / "canonical"
    report = copy_mapping(mapping(legacy), results, apply=True)
    assert report["artifact"]["script"] == "run_significance"
    assert all(item["action"] == "convert" for item in report["operations"])


def rrr_mapping(source, *, variant=None):
    return Mapping(
        source=str(source), producer="run_rrr", mode="cross_subject",
        dataset="podcast", offset=0, model="gpt2", layer=9,
        variant=variant,
    )


def write_rrr_config(path):
    path.write_text(
        "encoding:\n  dataset: podcast\n  model: transformer\n"
        "  offset: 0\n  per_model_config:\n    model_dir: gpt2\n    layer: 9\n"
        "mode: cross_subject\n"
    )


def test_discovery_finds_significance_config_in_child_directory(tmp_path):
    root = tmp_path / "results"
    source = root / "gpt2_layer9_offset0"
    (source / "significance").mkdir(parents=True)
    (source / "all_subject_sig.csv").write_text("subject,elec\n1,A1\n")
    write_rrr_config(source / "significance" / "config.yml")

    inferred, ambiguous = discover_mappings(root, root)

    assert len(inferred) == 1
    assert ambiguous == []
    assert inferred[0].producer == "run_significance"
    assert inferred[0].model == "gpt2"
    assert inferred[0].layer == 9


def test_discovery_excludes_deferred_workflows_and_configless_cross_dataset(tmp_path):
    root = tmp_path / "results"
    deferred = [
        root / "rank_saturation",
        root / "subspace_comparisons" / "comparison",
        root / "model_layer0_offset0" / "multi_view_rrr",
        root / "model_layer0_offset0" / "cross_dataset_rrr" / "orphan_layer24",
    ]
    for path in deferred:
        path.mkdir(parents=True)
        (path / "summary.csv").write_text("k\n1\n")

    inferred, ambiguous = discover_mappings(root, root)

    assert inferred == []
    assert ambiguous == []


def test_glove_identity_is_inferred_without_config(tmp_path):
    root = tmp_path / "results"
    source = root / "glove_dim300_offset0" / "rrr"
    source.mkdir(parents=True)
    (source / "summary.csv").write_text("k,nr_corr\n1,0.1\n")

    inferred, ambiguous = discover_mappings(root, root)

    assert ambiguous == []
    assert len(inferred) == 1
    assert inferred[0].model == "glove-300"
    assert inferred[0].layer is None


def test_configless_glove_rrr_and_pca_share_one_variant(tmp_path):
    root = tmp_path / "results"
    for method, value in (("pca", 1.0), ("rrr", 2.0)):
        source = root / "glove_dim300_offset0" / method
        (source / "fold_0").mkdir(parents=True)
        np.save(source / "fold_0" / "A.npy", np.array([[value]]))
        pd.DataFrame({"k": [1], "score": [value]}).to_csv(
            source / "summary.csv", index=False
        )
    inferred, ambiguous = discover_mappings(root, root)
    assert ambiguous == []

    canonical = tmp_path / "canonical"
    reports = [copy_mapping(item, canonical, apply=True) for item in inferred]

    assert len({report["artifact"]["variant"] for report in reports}) == 1
    artifact = RRRArtifact(
        mode="internal", dataset="podcast", offset=0, model="glove-300", layer=None,
        results_root=canonical,
    )
    artifact.load_artifacts(["input_factors"])
    assert len(artifact.input_factors) == 2


def test_rrr_and_pca_sources_keep_distinct_method_selectors(tmp_path):
    legacy = tmp_path / "legacy"
    for method, value in (("pca", 1.0), ("rrr", 2.0)):
        source = legacy / method
        (source / "fold_0").mkdir(parents=True)
        write_rrr_config(source / "config.yml")
        np.save(source / "fold_0" / "A.npy", np.array([[value]]))
        pd.DataFrame({"fold": [0], "k": [1], "score": [value]}).to_csv(
            source / "grid_results.csv", index=False
        )
        pd.DataFrame({"k": [1], "score": [value]}).to_csv(
            source / "summary.csv", index=False
        )

    results = tmp_path / "canonical"
    copy_mapping(rrr_mapping(legacy / "pca"), results, apply=True)
    report = copy_mapping(rrr_mapping(legacy / "rrr"), results, apply=True)
    artifact = RRRArtifact(
        mode="internal", dataset="podcast", offset=0, model="gpt2", layer=9,
        variant=report["artifact"]["variant"], results_root=results,
    )

    np.testing.assert_array_equal(
        artifact.load_field("input_factors", selector={"method": "pca", "fold": 0}),
        np.array([[1.0]]),
    )
    np.testing.assert_array_equal(
        artifact.load_field("input_factors", selector={"method": "rrr", "fold": 0}),
        np.array([[2.0]]),
    )
    assert len(artifact.load_collection("fold_results")) == 2
    assert len(artifact.load_collection("mean_results")) == 2


def test_nested_cross_dataset_pca_source_uses_pca_selector(tmp_path):
    source = tmp_path / "legacy" / "cross_dataset_pca" / "movie-cars-2"
    (source / "k1" / "fold_0").mkdir(parents=True)
    write_rrr_config(source / "config.yml")
    np.savez(source / "k1" / "fold_0" / "metrics.npz", score=np.array([0.5]))
    pd.DataFrame({"k": [1], "score": [0.5]}).to_csv(
        source / "summary.csv", index=False
    )

    results = tmp_path / "canonical"
    report = copy_mapping(rrr_mapping(source), results, apply=True)
    artifact = RRRArtifact(
        mode="internal", dataset="podcast", offset=0, model="gpt2", layer=9,
        variant=report["artifact"]["variant"], results_root=results,
    )

    assert artifact.has_field(
        "rank_metrics", selector={"method": "pca", "rank": 1, "fold": 0}
    )


def test_rrr_migrates_named_diagnostics_plots_and_unit_metadata(tmp_path):
    source = tmp_path / "legacy" / "rrr"
    diagnostic = source / "k4" / "fold_2" / "nr_alpha"
    diagnostic.mkdir(parents=True)
    write_rrr_config(source / "config.yml")
    (diagnostic / "alpha_diagnostics.json").write_text(json.dumps({"best": 3.0}))
    np.savez(diagnostic / "alpha_diagnostics.npz", scores=np.array([1.0, 2.0]))
    np.savez(diagnostic / "alpha_grid_cache.npz", grid=np.array([0.1, 1.0]))
    (source / "rank_analysis.png").write_bytes(b"png payload")
    (source / "subspace_aggregation.json").write_text(json.dumps({"kind": "mean"}))
    pd.DataFrame({"k": [4], "score": [0.5]}).to_csv(
        source / "summary.csv", index=False
    )

    results = tmp_path / "canonical"
    report = copy_mapping(rrr_mapping(source), results, apply=True)
    artifact = RRRArtifact(
        mode="internal", dataset="podcast", offset=0, model="gpt2", layer=9,
        variant=report["artifact"]["variant"], results_root=results,
    )

    diagnostics = artifact.load_collection("diagnostics", {"rank": 4, "fold": 2})
    assert len(diagnostics) == 3
    assert any(value == {"best": 3.0} for value in diagnostics.values())
    assert artifact.load_field(
        "plots", selector={"method": "rrr", "name": "rank_analysis"}
    ) == b"png payload"
    assert artifact.load_field(
        "unit_metadata", selector={"method": "rrr"}
    ) == {"kind": "mean"}


def test_fmri_source_factor_roles_are_preserved(tmp_path):
    source = tmp_path / "legacy" / "within_fmri_rrr"
    factors = source / "source_factors"
    (factors / "delay_1").mkdir(parents=True)
    (source / "rrr" / "k2").mkdir(parents=True)
    write_rrr_config(source / "config.yml")
    np.save(factors / "delay_1" / "A.npy", np.array([[1.0]]))
    np.save(factors / "full_fir_A.npy", np.array([[2.0]]))
    np.save(factors / "pre_fir_pca_A.npy", np.array([[3.0]]))
    np.save(source / "rrr" / "k2" / "A.npy", np.array([[4.0]]))
    pd.DataFrame({"method": ["rrr"], "rank": [2], "corr": [0.4]}).to_csv(
        source / "grid_results.csv", index=False
    )

    results = tmp_path / "canonical"
    item = Mapping(
        source=str(source), producer="run_rrr", mode="within_fmri",
        dataset="deep-fmri", offset=0, model="gpt2", layer=9,
    )
    report = copy_mapping(item, results, apply=True)
    artifact = RRRArtifact(
        mode="internal", dataset="deep-fmri", offset=0, model="gpt2", layer=9,
        variant=report["artifact"]["variant"], results_root=results,
    )

    expected = {
        ("delayed", 1): np.array([[1.0]]),
        ("full_fir", None): np.array([[2.0]]),
        ("pre_fir_pca", None): np.array([[3.0]]),
    }
    for (basis, delay), value in expected.items():
        selector = {"method": "rrr", "basis": basis}
        if delay is not None:
            selector["delay"] = delay
        np.testing.assert_array_equal(
            artifact.load_field("input_factors", selector=selector), value
        )
    assert artifact.load_field(
        "input_factors", selector={"method": "rrr", "rank": 2}
    ).item() == 4.0
    assert len(artifact.load_collection("mean_results")) == 1


def test_explicit_readable_variant_is_used(tmp_path):
    source = tmp_path / "legacy"
    source.mkdir()
    (source / "all_subject_sig.csv").write_text("subject,elec\n1,A1\n")
    item = mapping(source)
    item = Mapping(**{**item.__dict__, "variant": "named-migration"})

    report = copy_mapping(item, tmp_path / "canonical", apply=True)

    assert report["artifact"]["variant"] == "named-migration"


def test_significance_verifies_nested_object_null_distribution(tmp_path):
    source = tmp_path / "legacy"
    source.mkdir()
    payload = np.empty((), dtype=object)
    payload[()] = {1: {"shift_matrix": np.array([[1, 2], [3, 4]])}}
    np.savez(
        source / "cluster_null_distribution.npz",
        nested=payload,
        subjects=np.array([1]),
        electrodes=np.array(["A1"]),
    )
    (source / "all_subject_sig.csv").write_text("subject,elec\n1,A1\n")

    report = copy_mapping(mapping(source), tmp_path / "canonical", apply=True)

    assert report["artifact"]["script"] == "run_significance"


def test_unknown_legacy_file_refuses_partial_migration(tmp_path):
    source = tmp_path / "legacy" / "rrr"
    source.mkdir(parents=True)
    write_rrr_config(source / "config.yml")
    (source / "unrecognized_results.bin").write_bytes(b"important")
    pd.DataFrame({"k": [1]}).to_csv(source / "summary.csv", index=False)

    with pytest.raises(ValueError, match="refusing partial migration"):
        copy_mapping(rrr_mapping(source), tmp_path / "canonical", apply=False)


def test_significance_preserves_named_controls_diagnostics_and_plots(tmp_path):
    source = tmp_path / "legacy"
    subject = source / "subjects" / "sub_1"
    subject.mkdir(parents=True)
    (source / "all_subject_sig.csv").write_text("subject,elec\n1,A1\n")
    observed = np.array([[[0.1, 0.2]]], dtype=np.float32)
    arbitrary = np.array([[[0.01, 0.02]]], dtype=np.float32)
    np.save(subject / "corrs.npy", observed)
    np.save(subject / "arbitrary_corrs.npy", arbitrary)
    (subject / "alpha_diagnostics.json").write_text(json.dumps({"best": 1.0}))
    np.savez(subject / "alpha_diagnostics.npz", grid=np.array([0.1, 1.0]))
    (subject / "acf_diagnostic.png").write_bytes(b"acf")
    (subject / "corrs_plot.png").write_bytes(b"corrs")
    (subject / "alpha_diagnostics.png").write_bytes(b"alpha")

    results = tmp_path / "canonical"
    report = copy_mapping(mapping(source), results, apply=True)
    from scripts.run_significance import Artifact as SignificanceArtifact
    artifact = SignificanceArtifact(
        mode="sig_lags", dataset="podcast", offset=0,
        model="gpt2-xl-batch", layer=24,
        variant=report["artifact"]["variant"], results_root=results,
    )

    np.testing.assert_array_equal(
        artifact.load_field("correlations", selector={"subject": "1"}), observed
    )
    np.testing.assert_array_equal(
        artifact.load_field(
            "diagnostics",
            selector={"subject": "1", "name": "arbitrary_correlations"},
        ),
        arbitrary,
    )
    assert len(artifact.load_collection("diagnostics", {"subject": "1"})) == 3
    assert len(artifact.load_collection("plots", {"subject": "1"})) == 3


def test_significance_verifies_fitted_objects_in_encoding_results(tmp_path):
    from sklearn.preprocessing import StandardScaler

    source = tmp_path / "legacy"
    subject = source / "subjects" / "sub_1"
    subject.mkdir(parents=True)
    (source / "all_subject_sig.csv").write_text("subject,elec\n1,A1\n")
    scaler = StandardScaler().fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    import joblib
    joblib.dump({"scalers": [scaler]}, subject / "encoding_results.joblib")

    report = copy_mapping(mapping(source), tmp_path / "canonical", apply=True)

    assert report["artifact"]["script"] == "run_significance"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            "results/gpt2_layer9_offset0/rrr",
            {"mode": "internal", "dataset": "podcast", "offset": 0,
             "model": "gpt2", "layer": 9},
        ),
        (
            "results/gpt2_layer9_offset0/movie-cars-2/rrr",
            {"mode": "internal", "dataset": "brain-treebank-cars-2", "offset": 0,
             "model": "gpt2", "layer": 9},
        ),
        (
            "results/fmri/gpt2_layer9/within_fmri_rrr",
            {"mode": "internal", "dataset": "deep-fmri", "offset": 0,
             "model": "gpt2", "layer": 9},
        ),
        (
            "results/glove_dim300_offset0/rrr",
            {"mode": "internal", "dataset": "podcast", "offset": 0,
             "model": "glove-300", "layer": None},
        ),
    ],
)
def test_legacy_rrr_paths_become_abstract_identities(path, expected):
    assert _rrr_source_from_legacy_path(path) == expected


def test_dependency_path_is_removed_from_effective_config():
    original = {
        "source_rrr_unit_dir": "results/gpt2_layer9_offset0/movie-cars-2/rrr",
        "output_dir": "results/legacy-output",
        "rank_grid": [1, 2],
    }

    converted = _abstract_legacy_dependencies(original)

    assert "source_rrr_unit_dir" not in converted
    assert converted["rrr_source"] == {
        "mode": "internal", "dataset": "brain-treebank-cars-2", "offset": 0,
        "model": "gpt2", "layer": 9,
    }
    assert original["source_rrr_unit_dir"].endswith("movie-cars-2/rrr")
