import numpy as np
import pandas as pd
from pathlib import Path

from scripts.rrr.plot_rank_saturation import Analysis, _artifact_stem, output_parts
from src.utils.rank_saturation import (
    electrode_units,
    fmri_spatial_blocks,
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


def test_one_se_rule_selects_earliest_eligible_rank():
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
    assert result.curve["saturation_rank"].iloc[0] == 2
    assert result.retention["fraction_within_peak_sem"].between(0, 1).all()


def test_cluster_bootstrap_retains_target_weighted_mean():
    ranks = np.array([1, 2])
    target_z = np.array([[0.0, 0.0, 0.0, 1.0], [0.2, 0.2, 0.2, 1.0]])
    clusters = np.array(["large", "large", "large", "small"])
    result = summarize_rank_curve(
        ranks, target_z, clusters, bootstrap_repeats=200, seed=4
    )
    np.testing.assert_allclose(result.curve["mean_z"], [0.25, 0.4])


def test_equal_cluster_weight_gives_subjects_equal_influence():
    ranks = np.array([1, 2])
    target_z = np.array([[0.0, 0.0, 0.0, 1.0], [0.2, 0.2, 0.2, 1.0]])
    subjects = np.array(["many", "many", "many", "one"])
    result = summarize_rank_curve(
        ranks,
        target_z,
        subjects,
        bootstrap_repeats=200,
        seed=4,
        equal_cluster_weight=True,
    )
    np.testing.assert_allclose(result.curve["mean_z"], [0.5, 0.6])


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


def test_rank_saturation_output_parts_for_internal_movie_run():
    root = Path("results")
    analysis = Analysis(
        root / "gpt2-xl-batch_layer24_offset0" / "movie-cars-2",
        "ieeg",
        {},
    )
    assert output_parts(analysis, root) == (
        "cars-2", "internal_rrr", "gpt2-xl-batch", "layer24"
    )


def test_rank_saturation_output_parts_for_cross_dataset_fmri_run():
    root = Path("results")
    analysis = Analysis(
        root / "fmri" / "gpt2-xl_layer24" / "cross_dataset_rrr",
        "fmri",
        {},
    )
    assert output_parts(analysis, root) == (
        "fmri", "cross_dataset", "gpt2-xl", "layer24"
    )
    assert _artifact_stem(analysis, root) == "cross_dataset_rrr"
