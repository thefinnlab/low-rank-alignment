import numpy as np
import pandas as pd
from pathlib import Path

from scripts.rrr.plot_rank_saturation import (
    Analysis,
    _artifact_stem,
    _exact_family_signflip,
    _shared_saturation_max_exponent,
    output_parts,
)
from src.utils.rank_saturation import (
    _cluster_bootstrap_means,
    electrode_units,
    fmri_spatial_blocks,
    summarize_equivalence_curve,
    summarize_rank_curve,
)


def _serial_cluster_bootstrap_means(
    target_z, cluster_ids, repeats, seed, *, equal_cluster_weight=False
):
    codes, _ = pd.factorize(cluster_ids, sort=False)
    n_clusters = int(codes.max()) + 1
    sums = np.stack([
        np.nansum(target_z[:, codes == i], axis=1) for i in range(n_clusters)
    ])
    counts = np.stack([
        np.isfinite(target_z[:, codes == i]).sum(axis=1) for i in range(n_clusters)
    ])
    rng = np.random.default_rng(seed)
    output = np.empty((repeats, target_z.shape[0]), dtype=float)
    for repeat in range(repeats):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        if equal_cluster_weight:
            output[repeat] = np.nanmean(sums[chosen] / counts[chosen], axis=0)
        else:
            output[repeat] = sums[chosen].sum(axis=0) / counts[chosen].sum(axis=0)
    return output


def test_batched_bootstrap_matches_serial_sampling():
    target_z = np.array([
        [0.1, np.nan, 0.3, 0.4, 0.5],
        [0.2, 0.25, 0.35, np.nan, 0.55],
    ])
    clusters = np.array(["a", "a", "b", "c", "c"])
    for equal_weight in (False, True):
        expected = _serial_cluster_bootstrap_means(
            target_z, clusters, 137, 9, equal_cluster_weight=equal_weight
        )
        actual = _cluster_bootstrap_means(
            target_z,
            clusters,
            137,
            9,
            equal_cluster_weight=equal_weight,
            batch_size=17,
        )
        np.testing.assert_array_equal(actual, expected)


def test_summaries_accept_shared_bootstrap_means():
    ranks = np.array([1, 2, 4])
    target_z = np.array([
        [0.10, 0.12, 0.08],
        [0.18, 0.20, 0.16],
        [0.19, 0.21, 0.17],
    ])
    clusters = np.arange(3)
    boot = _cluster_bootstrap_means(target_z, clusters, 199, 3)
    curve = summarize_rank_curve(
        ranks, target_z, clusters, bootstrap_repeats=199, seed=3,
        bootstrap_means=boot,
    )
    equivalence = summarize_equivalence_curve(
        ranks, target_z, clusters, 0.02, bootstrap_repeats=199, seed=3,
        bootstrap_means=boot,
    )
    assert len(curve.bootstrap_saturation) == 199
    assert len(equivalence.curve) == len(ranks)


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


def test_fraction_of_best_rule_selects_earliest_eligible_rank():
    ranks = np.array([1, 2, 4, 8])
    target_z = np.array([
        [0.10, 0.11, 0.09, 0.10],
        [0.20, 0.255, 0.18, 0.215],
        [0.20, 0.26, 0.18, 0.22],
        [0.19, 0.25, 0.17, 0.21],
    ])
    result = summarize_rank_curve(
        ranks, target_z, np.arange(4), bootstrap_repeats=1000, seed=2
    )
    assert result.curve["peak_rank"].iloc[0] == 4
    assert result.curve["saturation_rank"].iloc[0] == 2
    assert result.curve["performance_fraction"].iloc[0] == 0.985
    assert result.curve["threshold_z"].iloc[0] == 0.985 * result.curve["mean_z"].max()
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


def test_equivalence_curve_selects_earliest_simultaneously_equivalent_rank():
    ranks = np.array([1, 2, 4])
    target_z = np.tile(np.array([[0.10], [0.195], [0.20]]), (1, 8))
    result = summarize_equivalence_curve(
        ranks,
        target_z,
        np.arange(8),
        delta_z=0.01,
        bootstrap_repeats=199,
        seed=3,
    )
    assert result.saturation_rank == 2
    assert result.curve["equivalent"].tolist() == [False, True, True]
    assert result.curve.loc[1, "max_statistic_adjusted_pvalue"] <= 0.05


def test_exact_family_signflip():
    assert _exact_family_signflip(np.array([1.0, 2.0])) == 0.25


def test_shared_saturation_ticks_start_at_one_and_cover_global_maximum():
    frame = pd.DataFrame({"saturation_rank": [1, 16, 65, np.nan]})
    exponent = _shared_saturation_max_exponent(frame)
    assert exponent == 7
    assert [2 ** value for value in range(0, exponent + 1)] == [
        1, 2, 4, 8, 16, 32, 64, 128
    ]


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


def test_rank_saturation_output_parts_for_from_dataset_fmri_run():
    root = Path("results")
    analysis = Analysis(
        root / "fmri" / "random-gpt2-xl_layer36" / "from_cars-2_rrr",
        "fmri",
        {},
    )
    assert output_parts(analysis, root) == (
        "fmri", "cross_dataset", "random-gpt2-xl", "layer36"
    )


def test_rank_saturation_output_parts_for_internal_fmri_run():
    root = Path("results")
    analysis = Analysis(
        root / "fmri" / "random-gpt2-xl_layer36" / "within_fmri_rrr",
        "fmri",
        {},
    )
    assert output_parts(analysis, root) == (
        "fmri", "internal_rrr", "random-gpt2-xl", "layer36"
    )
