import numpy as np

from src.utils.subspace_compare import (
    discover_subspace_sources,
    principal_angles,
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
