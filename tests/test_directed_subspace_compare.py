import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import joblib
import pytest
from himalaya.backend import set_backend
from scipy.stats import spearmanr

from src.core.config import SubspaceComparisonConfig
import src.utils.subspace_comparison.directed as directed
from src.utils.subspace_comparison.directed import (
    ALL_METRICS,
    ComparisonSpace,
    EvaluationData,
    EvaluationView,
    align_views,
    factor_for,
    fold_indices,
    load_ieeg_heldout_ids,
    score_ordered_pair,
    shared_ranks,
    standardize_evaluation_values,
)
from scripts.rrr.compare_subspaces import (
    Artifact as ComparisonArtifact,
    ComparisonJob,
    _cached_metrics,
    _canonicalize_symmetric_metrics,
    _matrix,
    _write_matrices_and_plots,
    generate_comparison_jobs,
)


def test_comparison_artifact_collection_requires_datasets_and_filters_endpoints(tmp_path):
    producer = ComparisonArtifact(
        analysis_name="demo", evaluation_dataset="podcast",
        source_dataset="podcast", source_model="gpt2", source_layer=6,
        source_kind="rrr", variant="source-rrr", results_root=tmp_path,
    ).begin(config={"analysis_name": "demo"})
    rows = pd.DataFrame({
        "source_dataset": ["podcast", "podcast"],
        "source_offset": [0, 0], "source_model": ["gpt2", "gpt2"],
        "source_layer": [6, 6], "source_kind": ["rrr", "rrr"],
        "target_dataset": ["cars-2", "fmri"], "target_offset": [0, 0],
        "target_model": ["gpt2-xl", "gpt2-xl"], "target_layer": [24, 24],
        "target_kind": ["rrr", "rrr"], "ridge_r2": [.2, .3],
    })
    producer.save_field("fold_metrics", rows)
    producer.save_field("summary", rows)
    producer.save_field("effective_config", {"analysis_name": "demo"})
    producer.save_field("upstream_identities", [])
    producer.save_field("source_metadata", {"kind": "rrr"})
    producer.save_field("validation", {"ok": True})
    cell = {"target_dataset": "cars-2", "target_offset": 0,
            "target_model": "gpt2-xl", "target_layer": 24,
            "target_kind": "rrr", "comparison_mode": "equal_capacity",
            "geometry": "span", "source_rank": 1, "target_rank": 1, "fold": 0}
    producer.save_field("cell_metrics", {"metrics": {}}, selector=cell)
    producer.mark_complete({"cell_metrics": [cell]})

    selection = [
        {"model": "gpt2", "layers": [6], "datasets": ["podcast"]},
        {"model": "gpt2-xl", "layers": [24], "datasets": ["cars-2"]},
    ]
    loaded = ComparisonArtifact(
        analysis_name="demo", evaluation_dataset="podcast",
        models=selection, results_root=tmp_path,
    ).load_artifacts(["fold_metrics"])

    assert loaded.fold_metrics["target_dataset"].tolist() == ["cars-2"]
    assert len(loaded.available_comparisons) == 1
    with pytest.raises(ValueError, match="datasets"):
        ComparisonArtifact(
            analysis_name="demo", evaluation_dataset="podcast",
            models=[{"model": "gpt2", "layers": [6]}], results_root=tmp_path,
        )


@pytest.fixture(autouse=True)
def _restore_himalaya_backend():
    yield
    set_backend("numpy")


def _space(tmp_path, name, *, dataset="podcast", heldout=None, ranks=None):
    unit = tmp_path / name / "rrr"
    return ComparisonSpace(
        dataset=dataset,
        offset=0,
        model=name,
        layer=0,
        kind="rrr",
        result_root=unit.parent,
        unit_dir=unit,
        source_format="fold_factors",
        ranks=set(ranks or {1, 2, 4}),
        heldout_ids=heldout,
    )


def test_shared_ranks_uses_global_intersection_and_inclusive_cap(tmp_path):
    spaces = [
        _space(tmp_path, "a", ranks={1, 2, 4, 8}),
        _space(tmp_path, "b", ranks={1, 2, 8}),
    ]

    assert shared_ranks(spaces, 2) == [1, 2]


def test_comparison_modes_generate_explicit_source_and_target_ranks(tmp_path):
    spaces = [
        _space(tmp_path, "a", dataset="podcast"),
        _space(tmp_path, "b", dataset="fmri"),
    ]
    config = SimpleNamespace(
        comparison_modes=[
            "equal_capacity",
            "capacity_controlled",
            "full_neural_space",
        ],
        models={
            "a": {"saturation_ranks": {"rrr": {"podcast": {0: 4}}}},
            "b": {"saturation_ranks": {"rrr": {"fmri": {0: 8}}}},
        },
    )

    jobs = generate_comparison_jobs(config, spaces, [1, 2])

    assert len(jobs) == 8
    assert ComparisonJob("equal_capacity", 0, 1, 2, 2, 2) in jobs
    assert ComparisonJob("capacity_controlled", 0, 1, 4, 4) in jobs
    assert ComparisonJob("capacity_controlled", 1, 0, 4, 4) in jobs
    assert ComparisonJob("full_neural_space", 0, 1, 4, 8) in jobs
    assert ComparisonJob("full_neural_space", 1, 0, 8, 4) in jobs


def test_ieeg_folds_require_saved_word_provenance(tmp_path):
    path = tmp_path / "subjects" / "sub_1" / "encoding_results.joblib"
    path.parent.mkdir(parents=True)
    joblib.dump({"test_idxs": [np.array([0, 1])]}, path)

    with pytest.raises(ValueError, match="Missing analysis_selection"):
        load_ieeg_heldout_ids(tmp_path, [1])


def test_align_views_uses_canonical_ids_in_left_order():
    left = EvaluationView(
        ids=np.array([10, 11, 12, 13, 14]),
        values=np.arange(10).reshape(5, 2),
    )
    right = EvaluationView(
        ids=np.array([14, 12, 10, 11, 99]),
        values=np.array([[14], [12], [10], [11], [99]]),
    )

    ids, left_values, right_values = align_views(left, right)

    np.testing.assert_array_equal(ids, np.array(["10", "11", "12", "14"]))
    np.testing.assert_array_equal(left_values, left.values[[0, 1, 2, 4]])
    np.testing.assert_array_equal(right_values[:, 0], [10, 11, 12, 14])


def test_align_views_reuses_values_when_ids_are_already_identical():
    left = EvaluationView(ids=np.arange(5), values=np.ones((5, 2)))
    right = EvaluationView(ids=np.arange(5), values=np.zeros((5, 3)))

    ids, left_values, right_values = align_views(left, right)

    np.testing.assert_array_equal(ids, ["0", "1", "2", "3", "4"])
    assert left_values is left.values
    assert right_values is right.values


def test_fold_indices_intersects_both_spaces_heldout_ids(tmp_path):
    source = _space(tmp_path, "a", heldout=[np.array([2, 3, 4])])
    target = _space(tmp_path, "b", heldout=[np.array([3, 4, 5])])
    evaluation = EvaluationData(
        alias="podcast",
        fold_test_ids=[np.array([1, 2, 3, 4, 5])],
        load_view=lambda *_: None,
        metadata={},
    )

    train, test = fold_indices(
        np.array(["0", "1", "2", "3", "4", "5"]),
        source,
        target,
        evaluation,
        0,
    )

    np.testing.assert_array_equal(test, [3, 4])
    np.testing.assert_array_equal(train, [0, 1, 2, 5])


def test_factor_for_uses_fold_factor_within_and_extrinsic_mean_across(tmp_path):
    within = _space(tmp_path, "within")
    cross = _space(tmp_path, "cross", dataset="cars-2")
    for space in (within, cross):
        for fold, factor in enumerate((np.eye(3, 2), -np.eye(3, 2))):
            folder = space.unit_dir / f"fold_{fold}"
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / "A.npy", factor)

    weighted = factor_for(within, 2, 0, "podcast", "weighted")
    cross_span = factor_for(cross, 2, 0, "podcast", "span")

    np.testing.assert_allclose(weighted, np.eye(3, 2))
    np.testing.assert_allclose(cross_span @ cross_span.T, np.diag([1.0, 1.0, 0.0]))
    assert factor_for(cross, 2, 0, "podcast", "weighted") is None


def test_fold_matched_values_use_saved_input_scaler(tmp_path):
    space = _space(tmp_path, "within")

    class FakeArtifact:
        dataset = "podcast"

        def load_field(self, field, selector):
            assert field == "input_scalers"
            assert selector == {"method": "rrr", "fold": 1}
            return {
                "mean": np.array([10.0, -2.0]),
                "scale": np.array([2.0, 4.0]),
                "coordinates": "standardized-v1",
            }

    space.artifact = FakeArtifact()
    values = np.array([[8.0, 2.0], [12.0, 6.0], [100.0, 100.0]])

    standardized = standardize_evaluation_values(
        space,
        values,
        np.array([0, 1]),
        fold=1,
        evaluation_dataset="podcast",
    )

    np.testing.assert_allclose(
        standardized,
        np.array([[-1.0, 1.0], [1.0, 2.0], [45.0, 25.5]], dtype=np.float32),
    )


def test_cross_dataset_values_keep_evaluation_fold_standardization(tmp_path):
    space = _space(tmp_path, "cross", dataset="cars-2")

    class FakeArtifact:
        dataset = "brain-treebank-cars-2"

        def load_field(self, *_args, **_kwargs):
            raise AssertionError("cross-dataset aggregation has no single saved scaler")

    space.artifact = FakeArtifact()
    values = np.array([[1.0, 2.0], [3.0, 6.0], [9.0, 20.0]])

    standardized = standardize_evaluation_values(
        space,
        values,
        np.array([0, 1]),
        fold=0,
        evaluation_dataset="podcast",
    )

    np.testing.assert_allclose(
        standardized,
        np.array([[-1.0, -1.0], [1.0, 1.0], [7.0, 8.0]], dtype=np.float32),
    )


def test_score_ordered_pair_recovers_rotated_scaled_space():
    rng = np.random.default_rng(17)
    source = rng.standard_normal((80, 3))
    rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    target = 2.0 * source @ rotation + np.array([3.0, -2.0, 1.0])

    result = score_ordered_pair(
        source,
        target,
        np.arange(60),
        np.arange(60, 80),
        ridge_alphas=[1e-8, 1e-6],
        rsm_metric="cosine",
        device="cpu",
    )

    assert result["ridge_r2"] > 0.999
    np.testing.assert_allclose(
        result["ridge_normalized_error"], 1.0 - result["ridge_r2"]
    )
    assert result["procrustes_r2"] > 0.999
    assert result["direct_cka"] > 0.999
    assert not any(key.startswith("cca_") for key in result)


@pytest.mark.skipif(not directed.torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_score_matches_torch_cpu_score():
    rng = np.random.default_rng(23)
    source = rng.standard_normal((50, 4))
    target = source @ rng.standard_normal((4, 4)) + 0.01 * rng.standard_normal((50, 4))
    kwargs = {
        "train_idx": np.arange(40),
        "test_idx": np.arange(40, 50),
        "ridge_alphas": [1e-4, 1e-2, 1.0],
        "rsm_metric": "cosine",
    }

    cpu = score_ordered_pair(source, target, device="cpu", **kwargs)
    cuda = score_ordered_pair(source, target, device="cuda", **kwargs)

    assert set(cpu) == set(cuda) == set(ALL_METRICS)
    for metric in ALL_METRICS:
        np.testing.assert_allclose(cuda[metric], cpu[metric], rtol=1e-9, atol=1e-9)


def test_torch_spearman_handles_large_tie_groups_without_integer_overflow():
    left = np.repeat(np.arange(4, dtype=float), 30_000)
    right = np.roll(left, 17_000)
    device_name = "cuda" if directed.torch.cuda.is_available() else "cpu"
    device = directed.configure_metric_backend(device_name)

    actual = directed._spearman(
        directed._metric_tensor(left, device),
        directed._metric_tensor(right, device),
    )

    np.testing.assert_allclose(float(actual), spearmanr(left, right).statistic)


def test_cuda_metric_backend_fails_without_cuda(monkeypatch):
    monkeypatch.setattr(directed.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="require CUDA"):
        directed.configure_metric_backend("cuda")


def test_metric_cache_invalidates_legacy_payload(tmp_path):
    path = tmp_path / "cell.json"
    path.write_text('{"ridge_r2": 0.5, "cca_first_test_corr": 0.9}')
    replacement = {metric: float(index) for index, metric in enumerate(ALL_METRICS)}

    result = _cached_metrics(path, False, lambda: replacement)

    assert result == replacement
    saved = json.loads(path.read_text())
    assert saved["metrics"] == replacement
    assert "implementation" in saved


def test_symmetric_metric_canonicalization_keeps_unordered_pairs_distinct():
    rows = []
    pair_values = {
        ("a", "b"): 0.8,
        ("a", "c"): 0.2,
    }
    for (left, right), value in pair_values.items():
        for source, target in ((left, right), (right, left)):
            row = {
                "comparison_mode": "equal_capacity",
                "geometry": "span",
                "source_rank": 4,
                "target_rank": 4,
                "equal_capacity_rank": 4,
                "fold": 0,
                "source_label": source,
                "target_label": target,
            }
            row.update({metric: value for metric in directed.SYMMETRIC_METRICS})
            rows.append(row)

    result = _canonicalize_symmetric_metrics(pd.DataFrame(rows))

    for (left, right), expected in pair_values.items():
        selected = result[
            result.source_label.isin([left, right])
            & result.target_label.isin([left, right])
        ]
        for metric in directed.SYMMETRIC_METRICS:
            np.testing.assert_array_equal(selected[metric], [expected, expected])


def test_symmetric_metric_canonicalization_matches_swapped_unequal_ranks():
    rows = []
    for source, target, source_rank, target_rank, value in (
        ("a", "b", 4, 8, 0.7),
        ("b", "a", 8, 4, 0.9),
    ):
        row = {
            "comparison_mode": "full_neural_space",
            "geometry": "span",
            "source_rank": source_rank,
            "target_rank": target_rank,
            "equal_capacity_rank": None,
            "fold": 0,
            "source_label": source,
            "target_label": target,
        }
        row.update({metric: value for metric in directed.SYMMETRIC_METRICS})
        rows.append(row)

    result = _canonicalize_symmetric_metrics(pd.DataFrame(rows))

    for metric in directed.SYMMETRIC_METRICS:
        np.testing.assert_array_equal(result[metric], [0.8, 0.8])


def test_output_writer_removes_only_legacy_cca_metric_files(tmp_path, monkeypatch):
    matrix_dir = tmp_path / "matrices"
    figure_dir = tmp_path / "figures"
    matrix_dir.mkdir()
    figure_dir.mkdir()
    stale = matrix_dir / "cca_mean_all_test_corr__span__k1.csv"
    retained = matrix_dir / "notes.txt"
    stale.write_text("old")
    retained.write_text("keep")
    monkeypatch.setattr(
        "scripts.rrr.compare_subspaces.ALL_METRICS", ()
    )

    _write_matrices_and_plots(
        pd.DataFrame({"equal_capacity_rank": []}),
        [],
        "equal_capacity",
        type("Config", (), {"geometries": []})(),
        tmp_path,
    )

    assert not stale.exists()
    assert retained.read_text() == "keep"


def test_fmri_word_loader_uses_heldout_words_without_temporal_resampling(monkeypatch):
    monkeypatch.setattr(
        directed,
        "resolve_fmri_stories",
        lambda *_: (["train_story"], ["heldout_story"], {1: ["train_story"]}),
    )
    calls = []

    def fake_embeddings(root, model, stories, layer, pooling):
        calls.append((root, model, tuple(stories), layer, pooling))
        return {"heldout_story": np.arange(24, dtype=np.float32).reshape(6, 4)}

    monkeypatch.setattr(directed, "load_story_embeddings", fake_embeddings)
    config = SubspaceComparisonConfig(
        evaluation_dataset="fmri",
        datasets={
            "fmri": {
                "loader": "fmri_words",
                "results_root": "results",
                "fmri_repo_root": "fmri-repo",
                "embedding_root": "word-embeddings",
                "sessions": [1, 2],
                "num_folds": 3,
            }
        },
        models={
            "gpt2": {"offsets": [0], "layers": [6], "datasets": ["fmri"]}
        },
    )

    evaluation = directed.load_evaluation_data(config, [])
    view = evaluation.load_view("gpt2", 6, 0)

    assert calls == [("word-embeddings", "gpt2", ("heldout_story",), 6, "mean")]
    assert evaluation.metadata["fold_mode"] == "contiguous_blocks_within_heldout_story"
    assert [len(fold) for fold in evaluation.fold_test_ids] == [2, 2, 2]
    np.testing.assert_array_equal(
        view.ids,
        [f"heldout_story:{index}" for index in range(6)],
    )


def test_saved_matrix_places_source_on_rows_and_target_on_columns():
    summary = pd.DataFrame(
        [
            {
                "source_label": "source",
                "target_label": "target",
                "geometry": "span",
                "equal_capacity_rank": 4,
                "ridge_r2_mean": 0.75,
                "ridge_r2_sem": 0.1,
                "ridge_r2_n": 5,
            }
        ]
    )

    mean, sem_values, counts = _matrix(
        summary, ["source", "target"], "span", "ridge_r2", 4
    )

    assert mean[0, 1] == 0.75
    assert sem_values[0, 1] == 0.1
    assert counts[0, 1] == 5
    assert np.isnan(mean[1, 0])
