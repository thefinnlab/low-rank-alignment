"""Tests for RRR sufficiency bootstrap utilities and synthetic script path."""

import os
import sys

import numpy as np

from src.core.config import RRRSufficiencyConfig
from src.utils.config_utils import dict_to_dataclass
from src.utils.rrr_sufficiency import (
    block_bootstrap_indices,
    corr_per_target,
    fisher_z,
    is_sufficient,
    paired_corr_delta_bootstrap,
)


def test_corr_per_target_uses_himalaya_and_filters_nan_samples():
    Y = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, np.nan],
            [3.0, 3.0, 1.0],
            [4.0, 4.0, 1.0],
        ]
    )
    Yhat = np.array(
        [
            [1.0, 4.0, 2.0],
            [2.0, 3.0, 2.0],
            [3.0, 2.0, 2.0],
            [4.0, 1.0, 2.0],
        ]
    )
    r = corr_per_target(Y, Yhat)
    assert np.isclose(r[0], 1.0)
    assert np.isclose(r[1], -1.0)
    assert np.isclose(r[2], 0.0)
    assert np.isfinite(fisher_z(r)).all()


def test_block_bootstrap_indices_length_bounds_reproducible():
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    idx1 = block_bootstrap_indices(17, 5, rng1)
    idx2 = block_bootstrap_indices(17, 5, rng2)
    assert idx1.shape == (17,)
    assert idx1.min() >= 0
    assert idx1.max() < 17
    np.testing.assert_array_equal(idx1, idx2)


def test_sufficiency_decision_uses_strict_upper_bound():
    assert is_sufficient(0.0049, 0.005)
    assert not is_sufficient(0.005, 0.005)
    assert not is_sufficient(0.006, 0.005)


def test_rrr_sufficiency_config_roundtrip():
    config = dict_to_dataclass(
        {
            "rank_grid": [2, 4],
            "n_bootstrap": 12,
            "block_size": 3,
            "epsilon_values": [0.02],
            "mode": "leave_one_subject_out",
            "subspace_types": ["rrr", "pca"],
            "encoding": {"model": "glove", "per_model_config": {"dim": 50}},
        },
        RRRSufficiencyConfig,
    )
    assert config.rank_grid == [2, 4]
    assert config.n_bootstrap == 12
    assert config.block_size == 3
    assert config.epsilon_values == [0.02]
    assert config.mode.value == "leave_one_subject_out"
    assert config.subspace_types == ["rrr", "pca"]
    assert config.encoding.model == "glove"
    assert config.encoding.per_model_config == {"dim": 50}


def test_validate_subspace_types_accepts_single_string():
    scripts_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "rrr")
    )
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from run_rrr_sufficiency import _validate_subspace_types

    assert _validate_subspace_types("pca") == ["pca"]


def test_paired_corr_delta_bootstrap_detects_rank_gap():
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((60, 3))
    Yhat_full = Y + 0.01 * rng.standard_normal(Y.shape)
    Yhat_rank = 0.3 * Y + 0.7 * rng.standard_normal(Y.shape)
    out = paired_corr_delta_bootstrap(
        Y,
        Yhat_full,
        Yhat_rank,
        n_bootstrap=20,
        block_size=6,
        rng=np.random.default_rng(1),
    )
    assert out["max_delta"].shape == (20,)
    assert out["target_delta"].shape == (20, 3)
    assert out["observed"]["max_delta"] > 0


def test_build_oof_predictions_synthetic(tmp_path):
    scripts_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "rrr")
    )
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from run_rrr_sufficiency import build_oof_predictions, run_bootstrap

    rng = np.random.default_rng(42)
    n, n_embed, n_targets = 36, 6, 4
    X = rng.standard_normal((n, n_embed)).astype(np.float32)
    A_true = rng.standard_normal((n_embed, 2)).astype(np.float32)
    B_true = rng.standard_normal((2, n_targets)).astype(np.float32)
    Y = (X @ A_true @ B_true + 0.05 * rng.standard_normal((n, n_targets))).astype(
        np.float32
    )
    test_idxs = [np.arange(0, 18), np.arange(18, 36)]

    for fold in range(2):
        fold_dir = tmp_path / "rrr" / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        np.save(fold_dir / "A.npy", A_true)

    config = RRRSufficiencyConfig(
        rank_grid=[1, 2],
        n_bootstrap=10,
        block_size=4,
        random_seed=7,
    )
    config.encoding.ridge_cv_folds = 2
    config.encoding.ridge_alphas = [0.1, 1.0]

    Y_oof, Yhat_full, Yhat_rank = build_oof_predictions(
        X,
        Y,
        test_idxs,
        str(tmp_path / "rrr"),
        config.rank_grid,
        config,
        out_dir=str(tmp_path / "suff"),
    )
    assert np.isfinite(Y_oof).all()
    assert np.isfinite(Yhat_full).all()
    assert np.isfinite(Yhat_rank[1]).all()
    assert np.isfinite(Yhat_rank[2]).all()

    summary, target_payload, boot_payload = run_bootstrap(
        Y_oof, Yhat_full, Yhat_rank, config
    )
    assert set(summary["k"]) == {1, 2}
    assert "sufficient_epsilon_0p005" in summary.columns
    assert "delta_k1" in target_payload
    assert "max_delta_k1" in boot_payload


def test_build_oof_predictions_pca_synthetic(tmp_path):
    scripts_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "rrr")
    )
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from run_rrr_sufficiency import build_oof_predictions

    rng = np.random.default_rng(123)
    n, n_embed, n_targets = 30, 5, 3
    X = rng.standard_normal((n, n_embed)).astype(np.float32)
    W = rng.standard_normal((n_embed, n_targets)).astype(np.float32)
    Y = (X @ W + 0.05 * rng.standard_normal((n, n_targets))).astype(np.float32)
    test_idxs = [np.arange(0, 15), np.arange(15, 30)]

    config = RRRSufficiencyConfig(rank_grid=[1, 3], n_bootstrap=5, block_size=3)
    config.encoding.ridge_cv_folds = 2
    config.encoding.ridge_alphas = [0.1, 1.0]

    Y_oof, Yhat_full, Yhat_rank = build_oof_predictions(
        X,
        Y,
        test_idxs,
        None,
        config.rank_grid,
        config,
        out_dir=str(tmp_path / "pca_suff"),
        subspace_type="pca",
    )
    assert np.isfinite(Y_oof).all()
    assert np.isfinite(Yhat_full).all()
    assert np.isfinite(Yhat_rank[1]).all()
    assert np.isfinite(Yhat_rank[3]).all()


def test_loso_sufficiency_writes_rrr_and_pca_units(tmp_path, monkeypatch):
    scripts_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "rrr")
    )
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import run_rrr_sufficiency

    rng = np.random.default_rng(5)
    n, n_embed, n_targets = 24, 5, 2
    X = rng.standard_normal((n, n_embed)).astype(np.float32)
    A = rng.standard_normal((n_embed, 2)).astype(np.float32)
    Y1 = (X @ A @ rng.standard_normal((2, n_targets))).astype(np.float32)
    Y2 = (X @ A @ rng.standard_normal((2, n_targets))).astype(np.float32)
    test_idxs = [np.arange(0, 12), np.arange(12, 24)]

    config = RRRSufficiencyConfig(
        rank_grid=[1],
        n_bootstrap=4,
        block_size=3,
        subspace_types=["rrr", "pca"],
        random_seed=11,
    )
    config.encoding.ridge_cv_folds = 2
    config.encoding.ridge_alphas = [0.1, 1.0]

    rrr_root = tmp_path / "loso_rrr"
    for held_out in [1, 2]:
        for unit in ["train_subjects", "baseline"]:
            for fold in range(2):
                fold_dir = rrr_root / f"held_out_sub_{held_out}" / unit / f"fold_{fold}"
                fold_dir.mkdir(parents=True)
                np.save(fold_dir / "A.npy", A)

    def fake_reference_test_idxs(_output_dir, _subject_ids):
        return test_idxs

    def fake_build_xy(subject_ids, _enc, _sig_electrodes, _aligned, _word_in_glove):
        if subject_ids == [1]:
            return X, Y1
        if subject_ids == [2]:
            return X, Y2
        return X, np.concatenate([Y1, Y2], axis=1)

    monkeypatch.setattr(
        run_rrr_sufficiency, "_reference_test_idxs", fake_reference_test_idxs
    )
    monkeypatch.setattr(run_rrr_sufficiency, "_build_xy", fake_build_xy)

    run_rrr_sufficiency._run_loso_sufficiency(
        config=config,
        enc=config.encoding,
        output_dir=str(tmp_path / "out"),
        rrr_source_dir=str(rrr_root),
        subspace_types=["rrr", "pca"],
        subject_ids=[1, 2],
        sig_electrodes={1: ["E1"], 2: ["E2"]},
        aligned_embeddings=X,
        word_in_glove=None,
        metadata_base={"config": {}, "mode": "leave_one_subject_out"},
    )

    expected = [
        rrr_root / "held_out_sub_1" / "train_subjects" / "sufficiency_bootstrap",
        rrr_root / "held_out_sub_1" / "transfer" / "sufficiency_bootstrap",
        rrr_root / "held_out_sub_1" / "baseline" / "sufficiency_bootstrap",
        tmp_path / "loso_pca" / "held_out_sub_1" / "train_subjects" / "sufficiency_bootstrap",
        tmp_path / "loso_pca" / "held_out_sub_1" / "transfer" / "sufficiency_bootstrap",
        tmp_path / "loso_pca" / "held_out_sub_1" / "baseline" / "sufficiency_bootstrap",
    ]
    for suff_dir in expected:
        assert (suff_dir / "sufficiency_summary.csv").exists()
        assert (suff_dir / "target_deltas.npz").exists()
        assert (suff_dir / "bootstrap_distribution.npz").exists()
        assert (suff_dir / "metadata.json").exists()
