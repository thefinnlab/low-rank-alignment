import numpy as np
import pandas as pd
import joblib
import pytest

from src.core.config import SubspaceComparisonConfig
import src.utils.directed_subspace_compare as directed
from src.utils.directed_subspace_compare import (
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
from scripts.rrr.compare_subspaces import _matrix


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
        shared_corr_threshold=0.2,
    )

    assert result["ridge_r2"] > 0.999
    np.testing.assert_allclose(
        result["ridge_normalized_error"], 1.0 - result["ridge_r2"]
    )
    assert result["procrustes_r2"] > 0.999
    assert result["direct_cka"] > 0.999
    assert result["cca_mean_all_test_corr"] > 0.999


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
