import numpy as np

from src.utils.subspace_io import ambient_dim, load_fold_bases, max_available_rank, sanitize_name


def test_sanitize_name_replaces_path_unsafe_characters():
    assert sanitize_name("podcast/gpt2 xl:l24") == "podcast-gpt2-xl-l24"


def test_load_fold_bases_and_dimensions(tmp_path):
    for fold in range(2):
        fold_dir = tmp_path / "rrr" / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        np.save(fold_dir / "A.npy", np.eye(5, 3, dtype=np.float32))

    bases = load_fold_bases(tmp_path / "rrr", rank=2)

    assert sorted(bases) == [0, 1]
    assert bases[0].shape == (5, 2)
    assert max_available_rank(tmp_path / "rrr") == 3
    assert ambient_dim(tmp_path / "rrr") == 5
