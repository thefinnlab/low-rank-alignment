import numpy as np
import pandas as pd

from src.utils.rank_saturation import (
    electrode_units,
    fmri_spatial_blocks,
    normalized_log_auc,
    summarize_rank_curve,
)


def test_electrode_units_average_lags_in_fisher_space():
    manifest = pd.DataFrame({
        "subject": [1, 1, 1, 1],
        "electrode": ["A", "A", "B", "B"],
        "lag": [0, 1, 0, 1],
    })
    target_z = np.array([[1.0, 3.0, 2.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
    units, labels = electrode_units(target_z, manifest)
    np.testing.assert_allclose(units, [[2.0, 3.0], [3.0, 7.0]])
    assert labels.tolist() == ["1:A", "1:B"]


def test_median_loss_and_performance_rules_are_reported_separately():
    ranks = np.array([1, 2, 4, 8])
    target_z = np.array([
        [0.10, 0.11, 0.09, 0.10],
        [0.195, 0.245, 0.175, 0.215],
        [0.20, 0.26, 0.18, 0.22],
        [0.19, 0.25, 0.17, 0.21],
    ])
    result = summarize_rank_curve(
        ranks, target_z, np.arange(4), bootstrap_repeats=1000, seed=2
    )
    assert result.curve["peak_rank"].iloc[0] == 4
    assert result.curve["performance_saturation_rank"].iloc[0] == 2
    assert result.curve["saturation_rank"].iloc[0] == 4
    assert result.curve["performance_99pct_rank"].iloc[0] == 4
    assert result.curve["probability_worse_than_mean_best"].between(0, 1).all()
    assert result.retention["fraction_within_peak_sem"].between(0, 1).all()


def test_cluster_bootstrap_retains_target_weighted_mean():
    ranks = np.array([1, 2])
    target_z = np.array([[0.0, 0.0, 0.0, 1.0], [0.2, 0.2, 0.2, 1.0]])
    clusters = np.array(["large", "large", "large", "small"])
    result = summarize_rank_curve(
        ranks, target_z, clusters, bootstrap_repeats=200, seed=4
    )
    np.testing.assert_allclose(result.curve["mean_z"], [0.25, 0.4])


def test_normalized_log_auc_uses_monotonic_pearson_curve():
    ranks = np.array([1, 2, 4, 8])
    performance = np.array([0.2, 0.6, 0.5, 1.0])
    monotonic, normalized, auc = normalized_log_auc(ranks, performance)
    np.testing.assert_allclose(monotonic, [0.2, 0.6, 0.6, 1.0])
    np.testing.assert_allclose(normalized, [0.2, 0.6, 0.6, 1.0])
    assert np.isclose(auc, (0.4 + 0.6 + 0.8) / 3)


def test_fmri_blocks_recover_response_columns_from_thick_mask(tmp_path):
    import nibabel as nib

    mask = np.zeros((4, 2, 2), dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[1, 0, 0] = 1
    mask[3, 1, 1] = 1
    directory = (
        tmp_path / "data" / "ds003020" / "derivatives" / "pycortex-db"
        / "UTS01" / "transforms" / "UTS01_auto"
    )
    directory.mkdir(parents=True)
    nib.save(nib.Nifti1Image(mask, np.diag([2.0, 2.0, 2.0, 1.0])), directory / "mask_thick.nii.gz")
    manifest = pd.DataFrame({"subject": ["UTS01"] * 3, "voxel_index": [0, 1, 2]})
    blocks = fmri_spatial_blocks(manifest, tmp_path, block_mm=4.0)
    assert blocks.tolist() == ["UTS01:0:0:0", "UTS01:0:0:0", "UTS01:1:0:0"]
