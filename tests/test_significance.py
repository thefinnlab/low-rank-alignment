"""Tests for utils/significance.py."""

import numpy as np
import pandas as pd
import pytest
import torch

import src.utils.significance as significance_utils
from src.utils.significance import (
    apply_cluster_fdr,
    aggregate_global_null,
    batch_corrcoef,
    build_cluster_filter,
    compute_fold_block_correlations,
    compute_fold_correlations,
    compute_per_electrode_cluster_p_values,
    compute_sem_normalized_target_scores,
    derive_subject_null_seed,
    fold_local_circular_shift_null_scores,
    label_lag_clusters,
    mark_clusters_by_min_size,
    reshape_fold_predictions_and_targets,
    score_fold_lag_clusters,
    score_lag_clusters,
    significant_lags_all_electrodes_gpu,
    split_fold_into_blocks,
    validate_min_cluster_size,
    valid_fold_shifts,
)

# ---------------------------------------------------------------------------
# batch_corrcoef
# ---------------------------------------------------------------------------


class TestBatchCorrcoef:
    def _make_tensors(self, n_electrodes=4, n_lags=5, n_words=30, seed=0):
        rng = np.random.default_rng(seed)
        x = torch.tensor(
            rng.standard_normal((n_electrodes, n_lags, n_words)), dtype=torch.float32
        )
        return x

    def test_identical_inputs_corr_one(self):
        x = self._make_tensors()
        r = batch_corrcoef(x, x)
        assert r.shape == (x.shape[0], x.shape[1])
        np.testing.assert_allclose(r.numpy(), np.ones_like(r.numpy()), atol=1e-5)

    def test_negated_input_corr_minus_one(self):
        x = self._make_tensors()
        r = batch_corrcoef(x, -x)
        np.testing.assert_allclose(r.numpy(), -np.ones_like(r.numpy()), atol=1e-5)

    def test_orthogonal_inputs_corr_zero(self):
        """Construct x and y that are orthogonal along the words dimension."""
        n_electrodes, n_lags, n_words = 2, 3, 100
        rng = np.random.default_rng(7)
        # Build a signal and its residual after projection (guaranteed orthogonal)
        raw = rng.standard_normal((n_electrodes, n_lags, n_words))
        # Orthogonalize by centering then subtracting projection
        raw_c = raw - raw.mean(axis=-1, keepdims=True)
        # Create y by rotating 90 degrees via random mixing
        mix = rng.standard_normal((n_words, n_words))
        q, _ = np.linalg.qr(mix)
        # y = x @ q  (q is orthogonal), so corr(x, y) should be near 0 on average
        y = np.einsum("eln,nm->elm", raw_c, q)
        x_t = torch.tensor(raw_c, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        r = batch_corrcoef(x_t, y_t)
        # Correlations should be small (not necessarily exactly 0, but close)
        assert r.abs().max().item() < 0.3

    def test_output_shape(self):
        n_electrodes, n_lags, n_words = 7, 6, 50
        x = self._make_tensors(n_electrodes, n_lags, n_words)
        r = batch_corrcoef(x, x)
        assert r.shape == (n_electrodes, n_lags)

    def test_returns_tensor(self):
        x = self._make_tensors()
        r = batch_corrcoef(x, x)
        assert isinstance(r, torch.Tensor)


# ---------------------------------------------------------------------------
# Cluster-based lag significance helpers
# ---------------------------------------------------------------------------


class TestClusterLagHelpers:
    def test_split_fold_into_uneven_blocks(self):
        blocks = split_fold_into_blocks(fold_length=10, blocks_per_fold=3)
        assert [len(block) for block in blocks] == [4, 3, 3]
        np.testing.assert_array_equal(np.concatenate(blocks), np.arange(10))

    def test_split_fold_drops_empty_blocks(self):
        blocks = split_fold_into_blocks(fold_length=2, blocks_per_fold=5)
        assert [len(block) for block in blocks] == [1, 1]

    def test_compute_fold_block_correlations_handles_uneven_lengths(self):
        rng = np.random.default_rng(11)
        fold_predictions = [
            rng.standard_normal((5, 2, 3)),
            rng.standard_normal((4, 2, 3)),
        ]
        fold_targets = [
            rng.standard_normal((5, 2, 3)),
            rng.standard_normal((4, 2, 3)),
        ]
        block_corrs, block_rows = compute_fold_block_correlations(
            fold_predictions, fold_targets, blocks_per_fold=3, device="cpu"
        )
        assert block_corrs.shape == (6, 2, 3)
        assert len(block_rows) == 6
        assert {row["fold_idx"] for row in block_rows} == {0, 1}

    def test_reshape_fold_predictions_and_targets_returns_explicit_arrays(self):
        Y = np.arange(8 * 6).reshape(8, 6)
        fold_predictions = [
            np.full((4, 6), 1.0),
            np.full((4, 6), 2.0),
        ]
        test_idxs = [np.array([0, 2, 4, 6]), np.array([1, 3, 5, 7])]
        pred_3d, target_3d = reshape_fold_predictions_and_targets(
            fold_predictions, test_idxs, Y, n_electrodes=2, n_lags=3
        )
        assert len(pred_3d) == 2
        assert len(target_3d) == 2
        assert pred_3d[0].shape == (4, 2, 3)
        assert target_3d[1].shape == (4, 2, 3)
        np.testing.assert_array_equal(target_3d[0].reshape(4, 6), Y[test_idxs[0]])

    def test_compute_fold_correlations_uses_explicit_arrays(self):
        rng = np.random.default_rng(13)
        fold_predictions = [
            rng.standard_normal((5, 2, 3)),
            rng.standard_normal((4, 2, 3)),
        ]
        fold_targets = [arr.copy() for arr in fold_predictions]
        corr = compute_fold_correlations(fold_predictions, fold_targets, device="cpu")
        np.testing.assert_allclose(corr, np.ones((2, 3)), atol=1e-5)

    def test_score_fold_lag_clusters_uses_shared_scoring_path(self):
        rng = np.random.default_rng(14)
        fold_predictions = [
            rng.standard_normal((6, 2, 3)),
            rng.standard_normal((5, 2, 3)),
        ]
        fold_targets = [arr.copy() for arr in fold_predictions]
        result = score_fold_lag_clusters(
            fold_predictions,
            fold_targets,
            blocks_per_fold=2,
            max_matching_distance=1,
            positive_fraction=1.0,
            mean_corr_threshold=0.9,
            device="cpu",
        )
        assert result["corr"].shape == (2, 3)
        assert result["block_corrs"].shape == (4, 2, 3)
        assert result["cluster_labels"].shape == (2, 3)
        assert len(result["block_rows"]) == 4
        assert len(result["scored_clusters"]) == 2
        assert all("peak_corr" in cluster for cluster in result["scored_clusters"])
        assert all(
            "peak_target_score" in cluster for cluster in result["scored_clusters"]
        )

    def test_cluster_creation_with_max_matching_distance(self):
        mask = np.array(
            [
                [False, True, False, True, False, False, True],
                [True, False, False, True, False, True, False],
            ]
        )
        labels, clusters = label_lag_clusters(mask, max_matching_distance=1)
        assert len(clusters) == 6
        np.testing.assert_array_equal(clusters[0]["lag_indices"], np.array([1]))
        np.testing.assert_array_equal(clusters[1]["lag_indices"], np.array([3]))
        np.testing.assert_array_equal(clusters[2]["lag_indices"], np.array([6]))
        assert labels[0, 1] != labels[0, 3]
        assert labels[0, 6] != labels[0, 1]

        labels, clusters = label_lag_clusters(mask, max_matching_distance=2)
        assert len(clusters) == 4
        np.testing.assert_array_equal(clusters[0]["lag_indices"], np.array([1, 3]))
        np.testing.assert_array_equal(clusters[1]["lag_indices"], np.array([6]))
        assert labels[0, 1] == labels[0, 3]
        assert labels[0, 6] != labels[0, 1]

    def test_cluster_score_uses_sem_normalized_fisher_z(self):
        clusters = [
            {"cluster_id": 0, "electrode_idx": 0, "lag_indices": np.array([0, 1])},
            {"cluster_id": 1, "electrode_idx": 1, "lag_indices": np.array([0])},
        ]
        block_corrs = np.array(
            [
                [[0.50, 0.20], [0.10, 0.05]],
                [[0.40, 0.30], [0.20, 0.10]],
                [[0.60, 0.25], [0.15, 0.00]],
            ]
        )
        target_scores, mean_z, sem_z = compute_sem_normalized_target_scores(block_corrs)
        scored = score_lag_clusters(target_scores, clusters)
        expected = target_scores[0, 0] + target_scores[0, 1]
        assert scored[0]["cluster_score"] == pytest.approx(expected)
        assert "peak_corr" not in scored[0]
        assert scored[0]["peak_target_score"] == pytest.approx(
            target_scores[0, :2].max()
        )
        assert scored[1]["cluster_score"] == pytest.approx(target_scores[1, 0])
        np.testing.assert_allclose(target_scores, mean_z / sem_z)

    def test_build_cluster_filter_positive_fraction_and_mean_corr(self):
        block_corrs = np.array(
            [
                [[0.2, 0.3, -0.2]],
                [[0.1, -0.1, -0.1]],
                [[0.4, 0.2, 0.2]],
                [[0.3, -0.2, -0.3]],
            ]
        )
        mask, positive_fraction, mean_corr = build_cluster_filter(
            block_corrs, positive_fraction=0.75, mean_corr_threshold=0.0
        )
        np.testing.assert_array_equal(mask, np.array([[True, False, False]]))
        assert positive_fraction[0, 0] == pytest.approx(1.0)
        assert mean_corr.shape == (1, 3)
        assert mean_corr[0, 0] == pytest.approx(
            np.tanh(np.mean(np.arctanh(block_corrs[:, 0, 0])))
        )

    def test_valid_fold_shifts_and_short_fold_error(self):
        np.testing.assert_array_equal(
            valid_fold_shifts(10, minimal_shift=2, shift_step=2),
            np.array([2, 4, 6]),
        )
        with pytest.raises(ValueError, match="too short"):
            valid_fold_shifts(4, minimal_shift=2)

    def test_fold_local_shift_null_scores_shape_and_metadata(self):
        rng = np.random.default_rng(12)
        fold_predictions = []
        fold_targets = []
        for fold_idx, fold_len in enumerate([8, 9]):
            del fold_idx
            fold_predictions.append(rng.standard_normal((fold_len, 2, 3)))
            fold_targets.append(rng.standard_normal((fold_len, 2, 3)))

        max_per_electrode, meta = fold_local_circular_shift_null_scores(
            fold_predictions,
            fold_targets,
            blocks_per_fold=2,
            minimal_shift=2,
            shift_step=2,
            num_null_shifts=3,
            seed=123,
            positive_fraction=0.0,
            mean_corr_threshold=-1.0,
            device="cpu",
        )
        assert max_per_electrode.shape == (3, 2)
        assert meta["shift_matrix"].shape == (3, 2)
        np.testing.assert_array_equal(meta["fold_lengths"], np.array([8, 9]))
        assert meta["seed"] == 123
        assert meta["sampling_mode"] == "iid_with_replacement_per_fold"

    def test_fold_local_shift_sampling_is_reproducible_and_iid(self, monkeypatch):
        monkeypatch.setattr(
            significance_utils,
            "score_fold_lag_clusters",
            lambda *args, **kwargs: {"scored_clusters": []},
        )
        rng = np.random.default_rng(15)
        predictions = [rng.standard_normal((12, 1, 1)) for _ in range(3)]
        targets = [rng.standard_normal((12, 1, 1)) for _ in range(3)]
        kwargs = {
            "blocks_per_fold": 2,
            "minimal_shift": 2,
            "shift_step": 2,
            "num_null_shifts": 100,
            "device": "cpu",
        }

        _, first = fold_local_circular_shift_null_scores(
            predictions, targets, seed=10, **kwargs
        )
        _, repeated = fold_local_circular_shift_null_scores(
            predictions, targets, seed=10, **kwargs
        )
        _, changed = fold_local_circular_shift_null_scores(
            predictions, targets, seed=11, **kwargs
        )

        np.testing.assert_array_equal(first["shift_matrix"], repeated["shift_matrix"])
        assert not np.array_equal(first["shift_matrix"], changed["shift_matrix"])
        allowed = set(valid_fold_shifts(12, minimal_shift=2, shift_step=2))
        assert all(
            set(first["shift_matrix"][:, fold]).issubset(allowed) for fold in range(3)
        )
        assert not np.array_equal(
            first["shift_matrix"][:, 0], first["shift_matrix"][:, 1]
        )

    def test_subject_null_seed_is_stable_and_subject_specific(self):
        assert derive_subject_null_seed(42, 1) == derive_subject_null_seed(42, 1)
        assert derive_subject_null_seed(42, 1) != derive_subject_null_seed(42, 2)
        assert derive_subject_null_seed(42, 1) != derive_subject_null_seed(43, 1)

    def test_null_sampling_validates_count(self):
        arrays = [np.zeros((8, 1, 1))]
        with pytest.raises(ValueError, match="positive integer"):
            fold_local_circular_shift_null_scores(
                arrays,
                arrays,
                minimal_shift=2,
                num_null_shifts=0,
                device="cpu",
            )

    def test_global_null_aggregation_across_subjects_and_electrodes(self):
        sub_a = np.array([[1.0, 2.0], [3.0, 0.5], [4.0, 1.0]])
        sub_b = np.array([[0.0, 5.0, 1.0], [2.0, 1.0, 0.0], [6.0, 0.0, 1.0]])
        global_max = aggregate_global_null([sub_a, sub_b])
        np.testing.assert_array_equal(global_max, np.array([5.0, 3.0, 6.0]))

    def test_global_null_requires_matching_sample_counts(self):
        sub_a = np.array([[1.0, 2.0], [3.0, 0.5], [4.0, 1.0]])
        sub_b = np.array([[0.0, 5.0, 1.0], [2.0, 1.0, 0.0]])
        with pytest.raises(ValueError, match="same number of samples"):
            aggregate_global_null([sub_a, sub_b])

    def test_per_electrode_cluster_p_values_use_matching_electrode_null_columns(self):
        max_per_electrode = np.array(
            [
                [0.0, 10.0, 3.0],
                [1.0, 20.0, 4.0],
                [2.0, 30.0, 5.0],
            ]
        )
        records = [
            {"subject": 1, "subject_electrode_index": 0, "cluster_score": 1.5},
            {"subject": 1, "subject_electrode_index": 2, "cluster_score": 4.5},
        ]

        p_values = compute_per_electrode_cluster_p_values(records, max_per_electrode)

        np.testing.assert_allclose(p_values, np.array([2 / 4, 2 / 4]))

    def test_per_electrode_cluster_p_values_plus_one_correction(self):
        max_per_electrode = np.array([[0.1], [0.2], [0.3]])
        records = [{"subject": 1, "subject_electrode_index": 0, "cluster_score": 10.0}]

        p_values = compute_per_electrode_cluster_p_values(records, max_per_electrode)

        np.testing.assert_allclose(p_values, np.array([1 / 4]))

    def test_per_electrode_cluster_p_values_missing_null_error(self):
        records = [{"subject": 1, "subject_electrode_index": 7, "cluster_score": 1.0}]
        with pytest.raises(
            ValueError,
            match="Missing null distribution for subject=1, subject_electrode_index=7",
        ):
            compute_per_electrode_cluster_p_values(records, np.array([[1.0]]))

    def test_cluster_fdr_mode_preserves_all_cluster_hypotheses(self):
        records = [
            {
                "subject": 1,
                "electrode": "E1",
                "cluster_id": 0,
                "cluster_score": 4.0,
                "cluster_p_value": 0.01,
            },
            {
                "subject": 1,
                "electrode": "E1",
                "cluster_id": 1,
                "cluster_score": 3.0,
                "cluster_p_value": 0.04,
            },
            {
                "subject": 2,
                "electrode": "E2",
                "cluster_id": 0,
                "cluster_score": 2.0,
                "cluster_p_value": 0.2,
            },
        ]

        apply_cluster_fdr(records, fdr_q=0.05, fdr_mode="cluster")

        expected_reject, expected_q, _, _ = significance_utils.multipletests(
            [0.01, 0.04, 0.2], alpha=0.05, method="fdr_bh"
        )
        np.testing.assert_array_equal([r["sig"] for r in records], expected_reject)
        np.testing.assert_allclose([r["q_value"] for r in records], expected_q)
        np.testing.assert_allclose([r["cluster_q_value"] for r in records], expected_q)
        np.testing.assert_allclose([r["p_value"] for r in records], [0.01, 0.04, 0.2])
        assert all("electrode_p_value" in record for record in records)
        assert all("electrode_q_value" in record for record in records)

    def test_electrode_fdr_uses_one_deterministic_winner_per_electrode_for_sig_only(
        self,
    ):
        records = [
            {
                "subject": 1,
                "electrode": "E1",
                "cluster_id": 8,
                "cluster_score": 5.0,
                "cluster_p_value": 0.001,
            },
            {
                "subject": 1,
                "electrode": "E1",
                "cluster_id": 3,
                "cluster_score": 5.0,
                "cluster_p_value": 0.01,
            },
            {
                "subject": 1,
                "electrode": "E2",
                "cluster_id": 2,
                "cluster_score": 4.0,
                "cluster_p_value": 0.04,
            },
            {
                "subject": 2,
                "electrode": "E3",
                "cluster_id": 1,
                "cluster_score": 3.0,
                "cluster_p_value": 0.8,
            },
        ]

        apply_cluster_fdr(records, fdr_q=0.05, fdr_mode="electrode")

        expected_cluster_reject, expected_cluster_q, _, _ = (
            significance_utils.multipletests(
                [0.001, 0.01, 0.04, 0.8], alpha=0.05, method="fdr_bh"
            )
        )
        assert [r["is_electrode_best_cluster"] for r in records] == [
            False,
            True,
            True,
            True,
        ]
        expected_reject, expected_q, _, _ = significance_utils.multipletests(
            [0.01, 0.04, 0.8], alpha=0.05, method="fdr_bh"
        )
        assert sum(r["is_electrode_best_cluster"] for r in records) == 3
        np.testing.assert_allclose(
            [records[0]["electrode_p_value"], records[1]["electrode_p_value"]],
            [0.01, 0.01],
        )
        np.testing.assert_allclose(
            [record["p_value"] for record in records], [0.001, 0.01, 0.04, 0.8]
        )
        np.testing.assert_allclose(
            [record["q_value"] for record in records], expected_cluster_q
        )
        np.testing.assert_allclose(
            [record["cluster_q_value"] for record in records], expected_cluster_q
        )
        np.testing.assert_allclose(
            [
                records[1]["electrode_q_value"],
                records[2]["electrode_q_value"],
                records[3]["electrode_q_value"],
            ],
            expected_q,
        )
        assert records[0]["sig"] is False
        assert bool(expected_cluster_reject[0]) is True
        np.testing.assert_array_equal(
            [records[1]["sig"], records[2]["sig"], records[3]["sig"]],
            expected_reject,
        )

    def test_mark_clusters_by_min_size_filters_small_clusters(self):
        clusters = [
            {
                "cluster_id": 0,
                "electrode_idx": 0,
                "lag_indices": np.array([0, 1]),
                "cluster_size": 2,
            },
            {
                "cluster_id": 1,
                "electrode_idx": 0,
                "lag_indices": np.array([3]),
                "cluster_size": 1,
            },
        ]
        marked = mark_clusters_by_min_size(clusters, min_cluster_size=2)
        assert marked[0]["sig"] is True
        assert marked[1]["sig"] is False

    def test_mark_clusters_by_min_size_includes_equal_threshold(self):
        clusters = [
            {
                "cluster_id": 2,
                "electrode_idx": 1,
                "lag_indices": np.array([2, 3, 4]),
                "cluster_size": 3,
            }
        ]
        marked = mark_clusters_by_min_size(clusters, min_cluster_size=3)
        assert marked[0]["sig"] is True

    def test_mark_clusters_by_min_size_handles_clusters_independently(self):
        clusters = [
            {
                "cluster_id": 0,
                "electrode_idx": 0,
                "lag_indices": np.array([0, 1, 2]),
                "cluster_size": 3,
            },
            {
                "cluster_id": 1,
                "electrode_idx": 0,
                "lag_indices": np.array([5]),
                "cluster_size": 1,
            },
            {
                "cluster_id": 2,
                "electrode_idx": 1,
                "lag_indices": np.array([1, 2]),
                "cluster_size": 2,
            },
        ]
        marked = mark_clusters_by_min_size(clusters, min_cluster_size=2)
        assert [cluster["sig"] for cluster in marked] == [True, False, True]

    def test_min_cluster_size_requires_positive_integer(self):
        with pytest.raises(ValueError, match="positive integer"):
            validate_min_cluster_size(0)
        with pytest.raises(ValueError, match="positive integer"):
            validate_min_cluster_size(1.5)

    def test_detailed_csv_cluster_fields_and_legacy_sig_csv(self, tmp_path):
        detailed = pd.DataFrame(
            [
                {
                    "subject": 1,
                    "electrode": "E1",
                    "lag_idx": 0,
                    "lag_sec": 0.0,
                    "sig": True,
                    "p_value": 0.01,
                    "q_value": 0.02,
                    "obs_corr": 0.4,
                    "cluster_id": 7,
                    "cluster_score": 1.2,
                    "cluster_p_value": 0.01,
                    "cluster_q_value": 0.02,
                },
                {
                    "subject": 1,
                    "electrode": "E1",
                    "lag_idx": 1,
                    "lag_sec": 0.1,
                    "sig": False,
                    "p_value": 1.0,
                    "q_value": 1.0,
                    "obs_corr": 0.0,
                    "cluster_id": -1,
                    "cluster_score": np.nan,
                    "cluster_p_value": 1.0,
                    "cluster_q_value": 1.0,
                },
            ]
        )
        detailed_path = tmp_path / "significance_detailed.csv"
        detailed.to_csv(detailed_path, index=False)
        legacy = (
            detailed[detailed["sig"]][["subject", "electrode"]]
            .drop_duplicates()
            .rename(columns={"electrode": "elec"})
        )
        legacy_path = tmp_path / "all_subject_sig.csv"
        legacy.to_csv(legacy_path, index=False, columns=["subject", "elec"])

        loaded = pd.read_csv(detailed_path)
        assert {
            "cluster_id",
            "cluster_score",
            "cluster_p_value",
            "cluster_q_value",
        }.issubset(loaded.columns)
        legacy_loaded = pd.read_csv(legacy_path)
        assert list(legacy_loaded.columns) == ["subject", "elec"]
        assert legacy_loaded.to_dict("records") == [{"subject": 1, "elec": "E1"}]


# ---------------------------------------------------------------------------
# significant_lags_all_electrodes_gpu (CPU, small n_permutations)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSignificantLagsAllElectrodes:
    @pytest.fixture
    def cpu_device(self):
        return torch.device("cpu")

    def test_output_shapes(self, cpu_device):
        rng = np.random.default_rng(0)
        n_electrodes, n_lags, n_words = 3, 4, 20
        pred = rng.standard_normal((n_electrodes, n_lags, n_words)).astype(np.float32)
        actual = rng.standard_normal((n_electrodes, n_lags, n_words)).astype(np.float32)

        sig, p_values, q_values, obs_corrs = significant_lags_all_electrodes_gpu(
            pred, actual, device=cpu_device
        )
        for arr in [sig, p_values, q_values, obs_corrs]:
            assert arr.shape == (n_electrodes, n_lags)

    def test_obs_corrs_approx_one_when_identical(self, cpu_device):
        rng = np.random.default_rng(1)
        n_electrodes, n_lags, n_words = 2, 3, 30
        pred = rng.standard_normal((n_electrodes, n_lags, n_words)).astype(np.float32)

        sig, p_values, q_values, obs_corrs = significant_lags_all_electrodes_gpu(
            pred, pred, device=cpu_device
        )
        np.testing.assert_allclose(
            obs_corrs, np.ones((n_electrodes, n_lags)), atol=1e-5
        )

    def test_sig_dtype_is_bool(self, cpu_device):
        rng = np.random.default_rng(2)
        n_electrodes, n_lags, n_words = 2, 2, 15
        pred = rng.standard_normal((n_electrodes, n_lags, n_words)).astype(np.float32)
        actual = rng.standard_normal((n_electrodes, n_lags, n_words)).astype(np.float32)

        sig, _, _, _ = significant_lags_all_electrodes_gpu(
            pred, actual, device=cpu_device
        )
        assert sig.dtype == bool
