import numpy as np

from src.utils.subspace_compare import (
    discover_subspace_sources,
    extrinsic_mean_basis,
    load_unit_subspace_matrices,
    principal_angles,
    principal_angles_from_bases,
    sample_random_principal_angles,
)


def test_discover_subspace_sources_finds_nested_rrr_and_pca(tmp_path):
    rrr_dir = tmp_path / "results" / "run" / "rrr" / "fold_0"
    pca_dir = tmp_path / "results" / "run" / "per_subject_pca" / "subjects" / "sub_1" / "fold_1"
    rrr_dir.mkdir(parents=True)
    pca_dir.mkdir(parents=True)
    np.save(rrr_dir / "A.npy", np.zeros((6, 3), dtype=np.float32))
    np.save(pca_dir / "A.npy", np.zeros((6, 4), dtype=np.float32))

    sources = discover_subspace_sources([tmp_path / "results"])

    assert len(sources) == 2
    assert {s.kind for s in sources} == {"rrr", "per_subject_pca"}
    assert {s.fold for s in sources} == {0, 1}
    assert {s.max_rank for s in sources} == {3, 4}
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
