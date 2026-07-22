import numpy as np
import pytest

from src.utils.subspace_compare import (
    aggregate_subspace_basis,
    cca_shared_dimensionality,
    default_cross_rank_text,
    discover_subspace_sources,
    filter_available_ranks,
    fit_regularized_maxvar_gcca,
    is_pca_source_kind,
    lower_triangle_values,
    maxvar_ambient_map,
    parse_rank_list,
    principal_angles,
    projected_rsm,
    rsa_for_projected_pair,
    rsa_lower_triangle_correlations,
    extrinsic_mean_basis,
    load_unit_subspace_matrices,
    principal_angles,
    principal_angles_from_bases,
    sample_random_principal_angles,
    score_cca_alignment,
    score_maxvar_leave_one_view_out,
    score_projected_cca_alignment,
    score_projected_rsa_alignment,
    score_procrustes_alignment,
    summarize_cca_fold_results,
    summarize_rsa_fold_results,
    summarize_procrustes_fold_results,
    select_maxvar_alpha,
    transform_maxvar_gcca_view,
    validate_aggregation_cache,
    validate_matching_fold_indices,
)


def test_default_cross_rank_text_includes_powers_of_two_and_max_rank():
    assert default_cross_rank_text() == "1,2,4,8,16,32,64,128,256,512,1024,1600"


def test_parse_rank_list_accepts_commas_ranges_and_deduplicates():
    assert parse_rank_list("1, 2, 4-6, 2") == [1, 2, 4, 5, 6]


def test_parse_rank_list_rejects_invalid_values():
    for text in ["", "0", "5-3", "1-", "abc"]:
        try:
            parse_rank_list(text)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected {text!r} to be rejected")


def test_filter_available_ranks_splits_unavailable_ranks():
    assert filter_available_ranks([1, 2, 8, 16], 8) == ([1, 2, 8], [16])


def test_is_pca_source_kind_marks_only_pca_baselines():
    assert is_pca_source_kind("pca")
    assert is_pca_source_kind("per_subject_pca")
    assert is_pca_source_kind("per_region_pca")
    assert is_pca_source_kind("loso_pca")
    assert not is_pca_source_kind("rrr")
    assert not is_pca_source_kind("per_subject_rrr")
    assert not is_pca_source_kind("per_region_rrr")
    assert not is_pca_source_kind("loso_rrr")


def test_discover_subspace_sources_finds_nested_rrr_and_pca(tmp_path):
    rrr_dir = tmp_path / "results" / "run" / "rrr" / "fold_0"
    pca_dir = (
        tmp_path
        / "results"
        / "run"
        / "per_subject_pca"
        / "subjects"
        / "sub_1"
        / "fold_1"
    )
    loso_pca_dir = tmp_path / "results" / "run" / "loso_pca" / "fold_2"
    rrr_dir.mkdir(parents=True)
    pca_dir.mkdir(parents=True)
    loso_pca_dir.mkdir(parents=True)
    np.save(rrr_dir / "A.npy", np.zeros((6, 3), dtype=np.float32))
    np.save(pca_dir / "A.npy", np.zeros((6, 4), dtype=np.float32))
    np.save(loso_pca_dir / "A.npy", np.zeros((6, 5), dtype=np.float32))

    sources = discover_subspace_sources([tmp_path / "results"])

    assert len(sources) == 3
    assert {s.kind for s in sources} == {"rrr", "per_subject_pca", "loso_pca"}
    assert {s.fold for s in sources} == {0, 1, 2}
    assert {s.max_rank for s in sources} == {3, 4, 5}
    assert all(s.ambient_dim == 6 for s in sources)


def test_principal_angles_identical_subspaces_are_zero():
    matrix = np.eye(5, 3)

    angles = principal_angles(matrix, matrix, rank_a=3, rank_b=3)

    np.testing.assert_allclose(angles, 0.0, atol=1e-12)


def test_principal_angles_orthogonal_subspaces_are_ninety_degrees():
    matrix_a = np.eye(4)[:, :2]
    matrix_b = np.eye(4)[:, 2:]

    angles = principal_angles(matrix_a, matrix_b, rank_a=2, rank_b=2)

    np.testing.assert_allclose(angles, np.pi / 2, atol=1e-12)


def test_random_principal_angles_are_reproducible():
    draws_a = sample_random_principal_angles(ambient_dim=8, rank_a=2, rank_b=3, n_draws=5, seed=7)
    draws_b = sample_random_principal_angles(ambient_dim=8, rank_a=2, rank_b=3, n_draws=5, seed=7)

    assert draws_a.shape == (5, 2)
    np.testing.assert_allclose(draws_a, draws_b)


def test_projected_rsm_cosine_and_euclidean():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    cosine = projected_rsm(vectors, metric="cosine")
    euclidean = projected_rsm(vectors, metric="euclidean")

    assert cosine.shape == (3, 3)
    np.testing.assert_allclose(np.diag(cosine), 0.0)
    np.testing.assert_allclose(cosine, cosine.T)
    np.testing.assert_allclose(cosine[1, 0], 1.0)
    np.testing.assert_allclose(euclidean[1, 0], np.sqrt(2.0))


def test_lower_triangle_values_excludes_diagonal():
    matrix = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    values = lower_triangle_values(matrix)

    np.testing.assert_array_equal(values, np.array([3.0, 6.0, 7.0]))


def test_rsa_lower_triangle_correlations_reports_pearson_and_spearman():
    rsm_a = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    rsm_b = np.array([[0.0, 2.0, 4.0], [2.0, 0.0, 6.0], [4.0, 6.0, 0.0]])

    result = rsa_lower_triangle_correlations(rsm_a, rsm_b)

    np.testing.assert_allclose(result["pearson"], 1.0)
    np.testing.assert_allclose(result["spearman"], 1.0)


def test_validate_matching_fold_indices_rejects_mismatches():
    validate_matching_fold_indices([np.array([0, 2]), np.array([1])], [np.array([0, 2]), np.array([1])])

    try:
        validate_matching_fold_indices([np.array([0])], [np.array([1])])
    except ValueError as exc:
        assert "fold 0 differs" in str(exc)
    else:
        raise AssertionError("Expected mismatched fold indices to raise")


def test_rsa_for_projected_pair_and_summary():
    embeddings = np.eye(4)
    test_idx = np.array([0, 1, 2])
    basis = np.eye(4, 2)

    fold_result = rsa_for_projected_pair(
        embeddings,
        test_idx,
        basis,
        basis,
        metric="euclidean",
    )
    summary = summarize_rsa_fold_results([fold_result, fold_result])

    np.testing.assert_allclose(fold_result["pearson"], 1.0)
    np.testing.assert_allclose(fold_result["spearman"], 1.0)
    np.testing.assert_allclose(summary["mean_pearson"], 1.0)
    np.testing.assert_allclose(summary["mean_spearman"], 1.0)


def test_procrustes_recovers_known_rotation_and_scale_on_held_out_rows():
    theta = np.pi / 5
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    source = np.array(
        [[-2.0, 0.5], [-1.0, -1.5], [0.0, 1.0], [1.5, -0.5], [2.0, 2.0]]
    )
    target = 2.5 * (source @ rotation) + np.array([10.0, -3.0])

    result = score_procrustes_alignment(
        source[:3],
        target[:3],
        source[3:],
        target[3:],
    )

    np.testing.assert_allclose(result["test_rmse"], 0.0, atol=1e-12)
    np.testing.assert_allclose(result["test_r2"], 1.0, atol=1e-12)
    np.testing.assert_allclose(result["test_mean_row_cosine"], 1.0, atol=1e-12)
    np.testing.assert_allclose(result["scale"], 2.5, atol=1e-12)


def test_procrustes_uses_training_means_for_test_scoring():
    source_train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    target_train = source_train + np.array([5.0, -2.0])
    source_test = np.array([[3.0, 4.0], [5.0, 6.0]])
    target_test = source_test + np.array([5.0, -2.0])

    result = score_procrustes_alignment(
        source_train,
        target_train,
        source_test,
        target_test,
    )

    np.testing.assert_allclose(
        result["target_centered_test"],
        target_test - target_train.mean(axis=0),
    )
    np.testing.assert_allclose(result["test_rmse"], 0.0, atol=1e-12)


def test_procrustes_rejects_mismatched_coordinate_dimensions():
    try:
        score_procrustes_alignment(
            np.ones((3, 2)),
            np.ones((3, 3)),
            np.ones((2, 2)),
            np.ones((2, 3)),
        )
    except ValueError as exc:
        assert "matching shapes" in str(exc)
    else:
        raise AssertionError("Expected mismatched Procrustes dimensions to raise")


def test_procrustes_rejects_insufficient_or_zero_variance_training_rows():
    for source_train in [np.ones((1, 2)), np.ones((3, 2))]:
        try:
            score_procrustes_alignment(
                source_train,
                np.ones_like(source_train),
                np.ones((2, 2)),
                np.ones((2, 2)),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Expected degenerate Procrustes training rows to raise")


def test_summarize_procrustes_fold_results_averages_metrics():
    summary = summarize_procrustes_fold_results(
        [
            {
                "test_rmse": 1.0,
                "test_normalized_mse": 0.5,
                "test_r2": 0.5,
                "test_mean_row_cosine": 0.25,
            },
            {
                "test_rmse": 3.0,
                "test_normalized_mse": 1.5,
                "test_r2": -0.5,
                "test_mean_row_cosine": 0.75,
            },
        ]
    )

    assert summary == {
        "mean_test_rmse": 2.0,
        "mean_test_normalized_mse": 1.0,
        "mean_test_r2": 0.0,
        "mean_test_mean_row_cosine": 0.5,
    }


def test_cca_recovers_known_shared_linear_structure_on_held_out_rows():
    rng = np.random.default_rng(7)
    shared = rng.standard_normal((80, 4))
    source = shared @ rng.standard_normal((4, 5)) + 0.01 * rng.standard_normal((80, 5))
    target = shared @ rng.standard_normal((4, 4)) + 0.01 * rng.standard_normal((80, 4))

    result = score_cca_alignment(
        source[:60],
        target[:60],
        source[60:],
        target[60:],
    )

    assert result["n_components"] == 4
    assert result["mean_top10_test_corr"] > 0.95
    assert result["mean_all_test_corr"] > 0.95
    assert result["test_shared_dim_pr"] > 3.9
    assert result["n_test_corr_ge_threshold"] == 4


def test_cca_allows_unequal_dimensions_and_caps_components_by_train_rows():
    rng = np.random.default_rng(13)
    source_train = rng.standard_normal((5, 7))
    target_train = rng.standard_normal((5, 3))
    source_test = rng.standard_normal((4, 7))
    target_test = rng.standard_normal((4, 3))

    result = score_cca_alignment(
        source_train,
        target_train,
        source_test,
        target_test,
    )

    assert result["n_components"] == 3
    assert result["train_corrs"].shape == (3,)
    assert result["test_corrs"].shape == (3,)


def test_cca_reports_train_and_test_correlations_separately():
    rng = np.random.default_rng(23)
    source_train = rng.standard_normal((40, 2))
    target_train = source_train @ np.array([[0.7, -0.4], [0.2, 0.9]])
    source_test = rng.standard_normal((20, 2))
    target_test = rng.standard_normal((20, 2))

    result = score_cca_alignment(
        source_train,
        target_train,
        source_test,
        target_test,
    )

    assert result["mean_top10_train_corr"] > 0.99
    assert result["mean_top10_test_corr"] < 0.9
    assert result["train_shared_dim_pr"] >= result["test_shared_dim_pr"]
    assert not np.allclose(result["train_corrs"], result["test_corrs"])


def test_cca_rejects_insufficient_rows_or_components():
    try:
        score_cca_alignment(
            np.ones((1, 2)),
            np.ones((1, 2)),
            np.ones((2, 2)),
            np.ones((2, 2)),
        )
    except ValueError as exc:
        assert "at least two training rows" in str(exc)
    else:
        raise AssertionError("Expected insufficient CCA training rows to raise")


def test_cca_shared_dimensionality_uses_positive_squared_spectrum():
    uniform = cca_shared_dimensionality(np.array([0.5, 0.5, 0.5]), threshold=0.2)
    concentrated = cca_shared_dimensionality(np.array([0.9, 0.0, -0.9]), threshold=0.2)
    empty = cca_shared_dimensionality(np.array([0.0, -0.5, np.nan]), threshold=0.2)

    np.testing.assert_allclose(uniform["shared_dim_pr"], 3.0)
    assert uniform["n_corr_ge_threshold"] == 3
    np.testing.assert_allclose(concentrated["shared_dim_pr"], 1.0)
    assert concentrated["n_corr_ge_threshold"] == 1
    assert empty["shared_dim_pr"] == 0.0
    assert empty["n_corr_ge_threshold"] == 0


def test_cca_shared_dimensionality_threshold_uses_signed_correlations():
    result = cca_shared_dimensionality(np.array([0.3, -0.4, 0.19]), threshold=0.2)

    assert result["n_corr_ge_threshold"] == 1


def test_summarize_cca_fold_results_averages_top10_all_and_shared_metrics():
    summary = summarize_cca_fold_results(
        [
            {
                "mean_top10_test_corr": 0.8,
                "mean_all_test_corr": 0.6,
                "first_test_corr": 0.9,
                "median_test_corr": 0.7,
                "mean_top10_train_corr": 0.95,
                "n_components": 4,
                "test_shared_dim_pr": 3.0,
                "train_shared_dim_pr": 3.5,
                "n_test_corr_ge_threshold": 3,
                "n_train_corr_ge_threshold": 4,
            },
            {
                "mean_top10_test_corr": 0.4,
                "mean_all_test_corr": 0.2,
                "first_test_corr": 0.5,
                "median_test_corr": 0.3,
                "mean_top10_train_corr": 0.85,
                "n_components": 2,
                "test_shared_dim_pr": 1.0,
                "train_shared_dim_pr": 1.5,
                "n_test_corr_ge_threshold": 1,
                "n_train_corr_ge_threshold": 2,
            },
        ]
    )

    assert summary == {
        "mean_top10_test_corr": 0.6000000000000001,
        "mean_all_test_corr": 0.4,
        "mean_first_test_corr": 0.7,
        "mean_median_test_corr": 0.5,
        "mean_top10_train_corr": 0.8999999999999999,
        "mean_n_components": 3.0,
        "mean_test_shared_dim_pr": 2.0,
        "mean_train_shared_dim_pr": 2.5,
        "mean_n_test_corr_ge_threshold": 2.0,
        "mean_n_train_corr_ge_threshold": 3.0,
    }


def test_score_projected_cca_alignment_saves_rotations_and_optional_ambient_axes():
    rng = np.random.default_rng(31)
    shared = rng.standard_normal((50, 2))
    embeddings_a = shared @ rng.standard_normal((2, 5))
    embeddings_b = shared @ rng.standard_normal((2, 4))
    basis_a = np.eye(5, 3)
    basis_b = np.eye(4, 2)

    metrics, artifact = score_projected_cca_alignment(
        embeddings_a,
        embeddings_b,
        np.arange(35),
        np.arange(35, 50),
        basis_a,
        basis_b,
        save_ambient_axes=True,
    )

    assert metrics["n_components"] == 2
    assert artifact["base_x_rotations"].shape == (3, 2)
    assert artifact["peer_y_rotations"].shape == (2, 2)
    assert artifact["base_ambient_axes"].shape == (5, 2)
    assert artifact["peer_ambient_axes"].shape == (4, 2)


def test_score_projected_rsa_alignment_optionally_returns_vectors():
    rng = np.random.default_rng(41)
    embeddings = rng.standard_normal((8, 5))
    basis = np.eye(5, 2)

    result = score_projected_rsa_alignment(
        embeddings,
        embeddings,
        np.arange(5),
        basis,
        basis,
        return_vectors=True,
    )

    assert result["n_test"] == 5
    np.testing.assert_allclose(result["pearson"], 1.0)
    np.testing.assert_allclose(result["spearman"], 1.0)
    assert result["rsm_a_vector"].shape == (10,)
    assert result["rsm_b_vector"].shape == (10,)


def test_regularized_maxvar_recovers_shared_held_out_scores():
    rng = np.random.default_rng(51)
    latent = rng.standard_normal((160, 3))
    views = {
        "a": latent @ rng.standard_normal((3, 6)) + 0.01 * rng.standard_normal((160, 6)),
        "b": latent @ rng.standard_normal((3, 5)) + 0.01 * rng.standard_normal((160, 5)),
        "c": latent @ rng.standard_normal((3, 4)) + 0.01 * rng.standard_normal((160, 4)),
    }
    train = {name: values[:120] for name, values in views.items()}
    test = {name: values[120:] for name, values in views.items()}

    result = fit_regularized_maxvar_gcca(train, 3, alpha=1e-4)
    score = score_maxvar_leave_one_view_out(result, train, test)

    assert result.shared_scores.shape == (120, 3)
    assert result.eigenvalues.shape == (3,)
    assert score["mean_loo_corr"] > 0.95


def test_maxvar_ambient_map_matches_project_then_transform():
    rng = np.random.default_rng(52)
    embeddings = rng.standard_normal((80, 7))
    basis_a, basis_b = np.eye(7, 4), np.eye(7)[:, 3:7]
    views = {"a": embeddings @ basis_a, "b": embeddings @ basis_b}
    result = fit_regularized_maxvar_gcca(views, 2, alpha=0.01)

    ambient, intercept = maxvar_ambient_map(result, "a", basis_a)
    direct = transform_maxvar_gcca_view(result, "a", views["a"])

    np.testing.assert_allclose(embeddings @ ambient + intercept, direct, atol=1e-10)


def test_maxvar_zero_ridge_handles_rank_deficient_views():
    x = np.linspace(-1, 1, 40)[:, None]
    views = {
        "a": np.hstack([x, x, x**2]),
        "b": np.hstack([2 * x, x**2, x**2]),
    }

    result = fit_regularized_maxvar_gcca(views, 2, alpha=0.0)

    assert result.effective_ranks == {"a": 2, "b": 2}
    assert np.all(np.isfinite(result.shared_scores))


def test_maxvar_solution_is_invariant_to_within_view_rotation():
    rng = np.random.default_rng(54)
    views = {
        "a": rng.standard_normal((70, 4)),
        "b": rng.standard_normal((70, 3)),
    }
    rotation, _ = np.linalg.qr(rng.standard_normal((4, 4)))

    original = fit_regularized_maxvar_gcca(views, 3, alpha=0.0, scale=False)
    rotated = fit_regularized_maxvar_gcca(
        {"a": views["a"] @ rotation, "b": views["b"]},
        3,
        alpha=0.0,
        scale=False,
    )

    np.testing.assert_allclose(original.eigenvalues, rotated.eigenvalues, atol=1e-10)
    np.testing.assert_allclose(
        original.shared_scores @ original.shared_scores.T,
        rotated.shared_scores @ rotated.shared_scores.T,
        atol=1e-10,
    )


def test_select_maxvar_alpha_returns_choice_per_dimension():
    rng = np.random.default_rng(53)
    latent = rng.standard_normal((90, 2))
    views = {
        "a": latent @ rng.standard_normal((2, 4)) + 0.05 * rng.standard_normal((90, 4)),
        "b": latent @ rng.standard_normal((2, 3)) + 0.05 * rng.standard_normal((90, 3)),
        "c": latent @ rng.standard_normal((2, 5)) + 0.05 * rng.standard_normal((90, 5)),
    }
    splits = [(np.arange(60), np.arange(60, 90)), (np.arange(30, 90), np.arange(30))]

    selected, rows = select_maxvar_alpha(views, splits, [1, 2], [0.0, 0.01])

    assert set(selected) == {1, 2}
    assert len(rows) == 4
    assert all(value in {0.0, 0.01} for value in selected.values())


def test_maxvar_partial_support_exceeds_smallest_view_rank_and_conserves_objective():
    rng = np.random.default_rng(155)
    views = {
        "rank2": rng.standard_normal((100, 2)),
        "rank4": rng.standard_normal((100, 4)),
        "rank6": rng.standard_normal((100, 6)),
    }
    result = fit_regularized_maxvar_gcca(views, 5, alpha=0.0)

    contributions = np.stack(list(result.objective_contributions.values()))
    shares = np.stack(list(result.contribution_shares.values()))
    np.testing.assert_allclose(contributions.sum(axis=0), result.eigenvalues)
    np.testing.assert_allclose(shares.sum(axis=0), 1.0)
    assert np.linalg.matrix_rank(transform_maxvar_gcca_view(result, "rank2", views["rank2"])) <= 2


def test_partial_support_selection_reaches_mass_and_always_keeps_two_views():
    rng = np.random.default_rng(156)
    latent = rng.standard_normal((80, 3))
    views = {
        "dominant": latent,
        "peer": latent[:, :2],
        "weak": 0.01 * rng.standard_normal((80, 1)),
    }
    result = fit_regularized_maxvar_gcca(views, 3, alpha=1e-3)
    from src.utils.subspace_compare import select_maxvar_support_views

    groups = select_maxvar_support_views(result, cumulative_mass=0.8)
    for component, group in enumerate(groups):
        assert len(group) >= 2
        assert sum(result.contribution_shares[name][component] for name in group) >= 0.8


def test_fold_alignment_recovers_permutation_signs_and_rotated_block():
    from src.utils.subspace_compare import (
        align_maxvar_folds,
        maxvar_block_overlap,
        maxvar_component_blocks,
    )

    rng = np.random.default_rng(157)
    reference, _ = np.linalg.qr(rng.standard_normal((120, 3)))
    candidate = reference[:, [1, 0, 2]] * np.array([-1.0, 1.0, -1.0])
    aligned = align_maxvar_folds(
        {0: reference, 1: candidate}, {0: np.array([2.0, 1.0, 0.99]), 1: np.array([2.0, 1.0, 0.99])}
    )
    np.testing.assert_array_equal(aligned["assignments"][1], [1, 0, 2])
    np.testing.assert_allclose(aligned["match_correlations"][1], 1.0)

    rotation = np.array([[np.cos(0.7), -np.sin(0.7)], [np.sin(0.7), np.cos(0.7)]])
    overlap = maxvar_block_overlap(reference[:, 1:3], reference[:, 1:3] @ rotation)
    np.testing.assert_allclose(overlap, 1.0)
    assert maxvar_component_blocks([True, False, False], [0.5, 0.01, 0.01]) == ((1, 2),)
def test_load_unit_subspace_matrices_loads_fold_a_files(tmp_path):
    unit_dir = tmp_path / "rrr"
    for fold in [1, 0]:
        fold_dir = unit_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        np.save(fold_dir / "A.npy", np.full((4, 2), fold, dtype=np.float32))

    matrices = load_unit_subspace_matrices(unit_dir)

    assert len(matrices) == 2
    np.testing.assert_array_equal(matrices[0], np.zeros((4, 2), dtype=np.float32))
    np.testing.assert_array_equal(matrices[1], np.ones((4, 2), dtype=np.float32))


def test_extrinsic_mean_basis_is_sign_invariant_for_identical_subspaces():
    basis = np.eye(5, 2)
    mean_basis = extrinsic_mean_basis([basis, -basis], rank=2)

    angles = principal_angles_from_bases(basis, mean_basis)
    np.testing.assert_allclose(angles, 0.0, atol=1e-6)


def test_extrinsic_mean_basis_rejects_mismatched_ambient_dimensions():
    with np.testing.assert_raises_regex(ValueError, "ambient dimension"):
        extrinsic_mean_basis([np.eye(5, 2), np.eye(4, 2)], rank=2)


def test_extrinsic_mean_basis_rejects_insufficient_source_rank():
    with np.testing.assert_raises_regex(ValueError, "exceeds source matrix width"):
        extrinsic_mean_basis([np.eye(5, 2)], rank=3)


def test_unnormalized_aggregation_preserves_factor_energy():
    e1 = np.array([[10.0], [0.0]])
    e2 = np.array([[0.0], [1.0]])
    matrices = [e1, e2, e2]

    extrinsic = aggregate_subspace_basis(matrices, rank=1, method="extrinsic")
    unnormalized = aggregate_subspace_basis(
        matrices, rank=1, method="unnormalized"
    )

    assert abs(float(extrinsic[:, 0] @ np.array([0.0, 1.0]))) > 0.999
    assert abs(float(unnormalized[:, 0] @ np.array([1.0, 0.0]))) > 0.999


def test_aggregate_subspace_basis_rejects_unknown_method():
    with np.testing.assert_raises_regex(ValueError, "method must"):
        aggregate_subspace_basis([np.eye(3, 1)], rank=1, method="weighted")


def test_aggregation_cache_rejects_incompatible_reuse(tmp_path):
    validate_aggregation_cache(tmp_path, "extrinsic")
    with pytest.raises(ValueError, match="use subspace aggregation"):
        validate_aggregation_cache(tmp_path, "unnormalized")


def test_aggregation_cache_treats_legacy_outputs_as_extrinsic(tmp_path):
    (tmp_path / "metrics.npz").write_bytes(b"legacy")
    with pytest.raises(ValueError, match="legacy results"):
        validate_aggregation_cache(tmp_path, "unnormalized")
    validate_aggregation_cache(tmp_path, "extrinsic")
