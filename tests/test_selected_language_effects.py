import numpy as np
import pandas as pd

from scripts.bootstrap_selected_language_effects import (
    _run_jobs,
    _standard_job,
    apply_dataset_rank_fdr,
    bootstrap_groups,
    smooth_preference,
)


def _pair(rrr_prediction, pca_prediction):
    truth = np.arange(12, dtype=float)
    return {
        "y_true": truth,
        "rrr": np.asarray(rrr_prediction, dtype=float),
        "pca": np.asarray(pca_prediction, dtype=float),
        "block_id": np.repeat(["s1", "s2", "s3"], 4),
    }


def test_smooth_preference_is_finite_near_zero_and_preserves_sign():
    rrr = np.array([0.0, 1e-100, -1e-100, -0.2, 0.4])
    pca = np.array([0.0, -1e-100, 1e-100, -0.1, 0.3])
    effect = smooth_preference(rrr, pca)
    assert np.isfinite(effect).all()
    np.testing.assert_array_equal(np.sign(effect), np.sign(rrr - pca))


def test_joint_bootstrap_is_deterministic_and_contrast_is_directly_paired():
    truth = np.arange(12, dtype=float)
    reverse = truth[::-1]
    groups = {
        "causal": {"a": [{"pair": _pair(truth, reverse)}]},
        "roberta": {"r": [{"pair": _pair(reverse, truth)}]},
    }
    first, _ = bootstrap_groups(groups, 30, np.random.default_rng(4))
    second, _ = bootstrap_groups(groups, 30, np.random.default_rng(4))
    np.testing.assert_allclose(first["causal"]["bootstrap"], second["causal"]["bootstrap"])
    contrast = first["causal"]["bootstrap"] - first["roberta"]["bootstrap"]
    assert np.all(contrast > 0)


def test_fdr_family_is_dataset_rank_specific_and_uses_available_rows():
    frame = pd.DataFrame({
        "dataset": ["podcast", "podcast", "podcast", "fmri"],
        "rank": [1, 1, 2, 1],
        "p_value": [0.01, 0.04, 0.04, 0.04],
        "smooth_effect": [1.0, 1.0, 1.0, -1.0],
    })
    result = apply_dataset_rank_fdr(frame)
    np.testing.assert_allclose(result.q_value, [0.02, 0.04, 0.04, 0.04])
    assert result.significant.tolist() == [True, True, True, False]


def test_multiple_workers_match_serial_results_and_print_progress(capsys):
    truth = np.arange(12, dtype=float)
    reverse = truth[::-1]
    group = {"gpt2-xl-batch": [{"pair": _pair(truth, reverse)}]}
    payloads = [
        (_standard_job, ("podcast", rank, "sentence_position", group, 5,
                         0.05, 0.05, rank, None, "config.yml"))
        for rank in (1, 2)
    ]
    serial = _run_jobs(payloads, 1)
    assert "2/2 cells" in capsys.readouterr().err
    parallel = _run_jobs(payloads, 2)
    assert "with 2 worker(s)" in capsys.readouterr().err
    assert serial == parallel
