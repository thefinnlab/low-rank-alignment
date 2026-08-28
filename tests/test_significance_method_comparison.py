from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.utilities.compare_significance_methods import (
    METHODS,
    _observed_method_scores,
    apply_global_fdr,
    canonical_config_path,
    empirical_max_pvalues,
    load_comparison_config,
    mean_significant_correlations,
    membership_groups,
    print_report,
    significant_sets,
)
from src.utils.significance import ieeg


def test_fisher_mean_averages_foldwise_correlations():
    correlations = np.array([[[0.2, 0.5]], [[0.4, 0.7]]])
    expected = np.tanh(np.mean(np.arctanh(correlations), axis=0))

    result = ieeg.fisher_mean_correlations(correlations, axis=0)

    np.testing.assert_allclose(result, expected)


def test_observed_summary_fisher_averages_all_folds_and_lags_per_electrode():
    fold_corrs = np.array(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.2, 0.3], [0.4, 0.5]],
        ]
    )
    cluster_result = {
        "target_scores": np.ones((2, 2)),
        "scored_clusters": [],
    }

    observed = _observed_method_scores(cluster_result, fold_corrs)
    expected = np.tanh(np.mean(np.arctanh(fold_corrs), axis=(0, 2)))

    np.testing.assert_allclose(observed["mean_all_lags_corr"], expected)


def test_empirical_pvalues_use_electrode_specific_max_nulls_and_plus_one():
    observed = np.array([2.0, 5.0, np.nan])
    null = np.array([[1.0, 6.0, 0.0], [2.0, 4.0, 1.0], [3.0, 5.0, 2.0]])

    pvalues = empirical_max_pvalues(observed, null)

    np.testing.assert_allclose(pvalues, [0.75, 0.75, 1.0])


def test_matched_null_methods_receive_identical_shift_batches(monkeypatch):
    fold_predictions = [np.ones((8, 1, 2)), np.ones((8, 1, 2))]
    fold_targets = [np.ones((8, 1, 2)), np.ones((8, 1, 2))]
    shifts = np.array([[2, 3], [3, 4], [4, 5]])
    fold_seen = []
    block_seen = []

    def fake_fold(_predictions, _targets, batch_shift_matrix, device=None):
        fold_seen.append(batch_shift_matrix.copy())
        return np.zeros((len(batch_shift_matrix), 2, 1, 2))

    def fake_blocks(
        _predictions, _targets, blocks_per_fold, batch_shift_matrix, device=None
    ):
        block_seen.append(batch_shift_matrix.copy())
        return np.zeros((len(batch_shift_matrix), 4, 1, 2))

    monkeypatch.setattr(ieeg, "_compute_shifted_fold_correlations_batch", fake_fold)
    monkeypatch.setattr(ieeg, "_compute_shifted_block_correlations_batch", fake_blocks)

    maxima, metadata = ieeg.fold_local_comparison_null_scores(
        fold_predictions,
        fold_targets,
        blocks_per_fold=2,
        shift_matrix=shifts,
        null_batch_size=2,
    )

    assert set(maxima) == {"lag_corr", "lag_sem", "cluster"}
    np.testing.assert_array_equal(metadata["shift_matrix"], shifts)
    assert len(fold_seen) == len(block_seen) == 2
    for fold_batch, block_batch in zip(fold_seen, block_seen):
        np.testing.assert_array_equal(fold_batch, block_batch)


def test_config_resolution_selects_requested_layer_and_checks_source(tmp_path):
    path = canonical_config_path("podcast", "gpt2-xl")
    config, resolved = load_comparison_config("podcast", "gpt2-xl", 24)

    assert resolved == path
    assert config.encoding.per_model_config["layer"] == 24

    payload = yaml.safe_load(path.read_text())
    payload["ieeg"]["encoding"]["per_model_config"]["model_dir"] = "other-model"
    mismatch = tmp_path / "mismatch.yml"
    mismatch.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="does not match"):
        load_comparison_config("podcast", "gpt2-xl", 24, mismatch)


def test_config_resolution_rejects_unknown_dataset_and_model():
    with pytest.raises(ValueError, match="Unsupported"):
        canonical_config_path("unknown", "gpt2-xl")
    with pytest.raises(FileNotFoundError, match="No canonical"):
        canonical_config_path("podcast", "not-a-model")


def test_global_bh_and_overlap_summaries_cover_discordant_groups():
    frame = pd.DataFrame(
        {
            "subject": [1, 1, 2, 2],
            "electrode": ["A", "B", "C", "D"],
            "lag_corr_p": [0.001, 0.01, 0.9, 0.9],
            "lag_sem_p": [0.001, 0.9, 0.01, 0.9],
            "cluster_p": [0.001, 0.9, 0.9, 0.01],
        }
    )

    corrected = apply_global_fdr(frame, q=0.05)
    sets = significant_sets(corrected)
    groups = membership_groups(sets)

    assert all(f"{method}_q" in corrected for method in METHODS)
    assert sets == {
        "lag_corr": {"1:A", "1:B"},
        "lag_sem": {"1:A", "2:C"},
        "cluster": {"1:A", "2:D"},
    }
    assert groups["all three"] == {"1:A"}
    assert groups["lag correlation only"] == {"1:B"}
    assert groups["lag SEM only"] == {"2:C"}
    assert groups["cluster only"] == {"2:D"}


def test_empty_significant_sets_are_supported():
    frame = pd.DataFrame(
        {
            "subject": [1],
            "electrode": ["A"],
            **{f"{method}_significant": [False] for method in METHODS},
        }
    )
    sets = significant_sets(frame)

    assert sets == {method: set() for method in METHODS}
    assert all(not members for members in membership_groups(sets).values())


def test_mean_correlation_uses_common_all_lag_score_for_each_selected_set():
    frame = pd.DataFrame(
        {
            "mean_all_lags_corr": [0.1, 0.3, 0.5],
            "lag_corr_significant": [True, True, False],
            "lag_sem_significant": [False, True, True],
            "cluster_significant": [False, False, False],
        }
    )

    means = mean_significant_correlations(frame)

    assert means["lag_corr"] == pytest.approx(
        np.tanh(np.mean(np.arctanh([0.1, 0.3])))
    )
    assert means["lag_sem"] == pytest.approx(
        np.tanh(np.mean(np.arctanh([0.3, 0.5])))
    )
    assert np.isnan(means["cluster"])


def test_console_report_prints_counts_overlap_and_discordant_ids(capsys):
    frame = pd.DataFrame(
        {
            "subject": [1, 1],
            "electrode": ["A", "B"],
            "mean_all_lags_corr": [0.2, 0.4],
            "lag_corr_significant": [True, True],
            "lag_sem_significant": [True, False],
            "cluster_significant": [True, False],
        }
    )
    config = SimpleNamespace(
        num_null_shifts=100,
        fdr_q=0.05,
        blocks_per_fold=3,
        max_matching_distance=2,
        positive_fraction=0.8,
        mean_corr_threshold=0.0,
        encoding=SimpleNamespace(num_folds=5),
    )

    print_report(frame, config, Path("config.yml"), "podcast", "gpt2-xl", 24)
    output = capsys.readouterr().out

    assert "Significant electrode counts" in output
    assert "mean all-lag r (Fisher z→r)" in output
    assert "0.3033" in output
    assert "Pairwise overlap" in output
    assert "all three (1)" in output
    assert "lag correlation only (1)" in output
    assert "1:B" in output
