import json

import numpy as np
import pandas as pd
import joblib
import pytest
from himalaya.backend import set_backend
from scipy.stats import spearmanr

from src.core.config import SubspaceComparisonConfig
import src.utils.directed_subspace_compare as directed
from src.utils.directed_subspace_compare import (
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
)
from src.utils.subspace_compare import (
    projected_rsm,
    rsa_lower_triangle_correlations,
    score_procrustes_alignment,
)
from scripts.rrr.compare_subspaces import (
    _cached_metrics,
    _matrix,
    _write_matrices_and_plots,
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


def test_torch_procrustes_and_rsa_match_numpy_reference():
    rng = np.random.default_rng(31)
    source_train = rng.standard_normal((50, 4))
    target_train = rng.standard_normal((50, 4))
    source_test = rng.standard_normal((20, 4))
    target_test = rng.standard_normal((20, 4))

    torch_result = directed._score_procrustes(
        directed._metric_tensor(source_train, directed.configure_metric_backend("cpu")),
        directed._metric_tensor(target_train, directed.configure_metric_backend("cpu")),
        directed._metric_tensor(source_test, directed.configure_metric_backend("cpu")),
        directed._metric_tensor(target_test, directed.configure_metric_backend("cpu")),
    )
    numpy_result = score_procrustes_alignment(
        source_train, target_train, source_test, target_test
    )
    for torch_key, numpy_key in (
        ("test_r2", "test_r2"),
        ("test_normalized_mse", "test_normalized_mse"),
        ("test_rmse", "test_rmse"),
        ("scale", "scale"),
        ("test_mean_row_cosine", "test_mean_row_cosine"),
    ):
        np.testing.assert_allclose(torch_result[torch_key], numpy_result[numpy_key])

    torch_rsa = directed._score_rsa(
        directed._metric_tensor(source_test, directed.configure_metric_backend("cpu")),
        directed._metric_tensor(target_test, directed.configure_metric_backend("cpu")),
        "cosine",
    )
    numpy_rsa = rsa_lower_triangle_correlations(
        projected_rsm(source_test, metric="cosine"),
        projected_rsm(target_test, metric="cosine"),
    )
    np.testing.assert_allclose(torch_rsa["pearson"], numpy_rsa["pearson"])
    np.testing.assert_allclose(torch_rsa["spearman"], numpy_rsa["spearman"])


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
        pd.DataFrame(), [], [], type("Config", (), {"geometries": []})(), tmp_path
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
                "rank": 4,
                "ridge_r2_mean": 0.75,
                "ridge_r2_sem": 0.1,
                "ridge_r2_n": 5,
            }
        ]
    )

    mean, sem_values, counts = _matrix(
        summary, ["source", "target"], "span", 4, "ridge_r2"
    )

    assert mean[0, 1] == 0.75
    assert sem_values[0, 1] == 0.1
    assert counts[0, 1] == 5
    assert np.isnan(mean[1, 0])
