"""Tests for utils/atlas.py."""

import numpy as np
import pytest

from src.utils.atlas import (
    _lookup_atlas_labels,
    _build_region_map_from_arrays,
    group_electrodes_by_region,
)


# ---------------------------------------------------------------------------
# _lookup_atlas_labels
# ---------------------------------------------------------------------------


class TestLookupAtlasLabels:
    """Tests for the pure numpy/scipy spatial lookup — no nilearn required."""

    def _make_atlas(self):
        """10x10x10 atlas with two labeled voxels; identity affine."""
        atlas_image = np.zeros((10, 10, 10), dtype=int)
        atlas_image[3, 5, 7] = (
            1  # region 1 at voxel (3,5,7) = MNI (3,5,7) with identity
        )
        atlas_image[8, 2, 4] = (
            2  # region 2 at voxel (8,2,4) = MNI (8,2,4) with identity
        )
        affine = np.eye(4)
        atlas_labels = ["Background", "Region1", "Region2"]
        return atlas_image, affine, atlas_labels

    def test_label_region1(self):
        atlas_image, affine, atlas_labels = self._make_atlas()
        coords = np.array([[3.0, 5.0, 7.0]])
        result = _lookup_atlas_labels(coords, atlas_image, affine, atlas_labels)
        assert result == ["Region1"]

    def test_label_region2(self):
        atlas_image, affine, atlas_labels = self._make_atlas()
        coords = np.array([[8.0, 2.0, 4.0]])
        result = _lookup_atlas_labels(coords, atlas_image, affine, atlas_labels)
        assert result == ["Region2"]

    def test_multiple_coordinates(self):
        atlas_image, affine, atlas_labels = self._make_atlas()
        coords = np.array([[3.0, 5.0, 7.0], [8.0, 2.0, 4.0]])
        result = _lookup_atlas_labels(coords, atlas_image, affine, atlas_labels)
        assert result == ["Region1", "Region2"]

    def test_outside_labeled_regions_maps_to_nearest(self):
        """Coordinates outside labeled voxels map to the nearest labeled voxel."""
        atlas_image, affine, atlas_labels = self._make_atlas()
        # (0,0,0) is background — KDTree only has labeled voxels, returns nearest
        coords = np.array([[0.0, 0.0, 0.0]])
        result = _lookup_atlas_labels(coords, atlas_image, affine, atlas_labels)
        # Should return a valid (non-background) label
        assert result[0] in ["Region1", "Region2"]

    def test_non_identity_affine(self):
        """With a scaled affine, voxel (1,0,0) maps to MNI (2,0,0)."""
        atlas_image = np.zeros((5, 5, 5), dtype=int)
        atlas_image[1, 0, 0] = 1
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        atlas_labels = ["Background", "RegionA"]
        coords = np.array([[2.0, 0.0, 0.0]])  # MNI coord = affine @ [1,0,0,1][:3]
        result = _lookup_atlas_labels(coords, atlas_image, affine, atlas_labels)
        assert result == ["RegionA"]


# ---------------------------------------------------------------------------
# group_electrodes_by_region
# ---------------------------------------------------------------------------


class TestGroupElectrodesByRegion:

    def test_basic_grouping(self):
        elec_names = ["e1", "e2", "e3"]
        atlas_labels = [
            "L G_temp_sup-Lateral",
            "R G_temp_sup-Lateral",
            "L G_front_middle",
        ]
        region_groups = {
            "temporal": ["L G_temp_sup-Lateral", "R G_temp_sup-Lateral"],
            "frontal": ["L G_front_middle"],
        }
        result = group_electrodes_by_region(elec_names, atlas_labels, region_groups)
        assert result["temporal"] == ["e1", "e2"]
        assert result["frontal"] == ["e3"]

    def test_left_hemisphere_only_matches_left_label(self):
        """Exact match: 'L G_temp_sup-Lateral' only matches when region_groups lists it."""
        elec_names = ["e1"]
        atlas_labels = ["L G_temp_sup-Lateral"]
        region_groups = {"temporal": ["L G_temp_sup-Lateral"]}
        result = group_electrodes_by_region(elec_names, atlas_labels, region_groups)
        assert result == {"temporal": ["e1"]}

    def test_left_label_does_not_match_without_hemisphere_prefix(self):
        """No match when region_groups only lists bare label without hemisphere prefix."""
        elec_names = ["e1"]
        atlas_labels = ["L G_temp_sup-Lateral"]
        region_groups = {"temporal": ["G_temp_sup-Lateral"]}  # missing "L "
        result = group_electrodes_by_region(elec_names, atlas_labels, region_groups)
        assert result == {}

    def test_right_hemisphere_does_not_match_left_entry(self):
        elec_names = ["e1"]
        atlas_labels = ["R G_temp_sup-Lateral"]
        region_groups = {"temporal": ["L G_temp_sup-Lateral"]}
        result = group_electrodes_by_region(elec_names, atlas_labels, region_groups)
        assert result == {}

    def test_both_hemispheres_grouped_together(self):
        elec_names = ["left_elec", "right_elec"]
        atlas_labels = ["L G_temp_sup-Lateral", "R G_temp_sup-Lateral"]
        region_groups = {
            "temporal": ["L G_temp_sup-Lateral", "R G_temp_sup-Lateral"],
        }
        result = group_electrodes_by_region(elec_names, atlas_labels, region_groups)
        assert set(result["temporal"]) == {"left_elec", "right_elec"}

    def test_unmatched_electrode_excluded(self):
        elec_names = ["e1", "e2"]
        atlas_labels = ["L G_temp_sup-Lateral", "L Medial_wall"]
        region_groups = {"temporal": ["L G_temp_sup-Lateral"]}
        result = group_electrodes_by_region(elec_names, atlas_labels, region_groups)
        assert result == {"temporal": ["e1"]}

    def test_region_with_no_matches_absent(self):
        elec_names = ["e1"]
        atlas_labels = ["L G_temp_sup-Lateral"]
        region_groups = {
            "temporal": ["L G_temp_sup-Lateral"],
            "frontal": ["L G_front_middle"],
        }
        result = group_electrodes_by_region(elec_names, atlas_labels, region_groups)
        assert "frontal" not in result
        assert result == {"temporal": ["e1"]}

    def test_empty_input_returns_empty_dict(self):
        result = group_electrodes_by_region(
            [], [], {"temporal": ["L G_temp_sup-Lateral"]}
        )
        assert result == {}


# ---------------------------------------------------------------------------
# _build_region_map_from_arrays
# ---------------------------------------------------------------------------


class TestBuildRegionMapFromArrays:
    """Tests for the pure-array core of build_electrode_region_map — no file I/O."""

    def _make_atlas(self):
        """10x10x10 atlas with two labeled regions; identity affine."""
        atlas_image = np.zeros((10, 10, 10), dtype=int)
        # Voxel (3,5,7) = DESTRIEUX index 34 = "L G_temp_sup-Lateral"
        atlas_image[3, 5, 7] = 34
        # Voxel (8,2,4) = DESTRIEUX index 14 = "L G_front_inf-Triangul"
        atlas_image[8, 2, 4] = 14
        affine = np.eye(4)
        return atlas_image, affine

    def test_single_subject_single_region(self):
        atlas_image, affine = self._make_atlas()
        per_subject_data = {
            1: (["e1"], np.array([[3.0, 5.0, 7.0]])),
        }
        region_groups = {"temporal": ["L G_temp_sup-Lateral"]}
        result = _build_region_map_from_arrays(
            per_subject_data, atlas_image, affine, region_groups
        )
        assert result == {"temporal": {1: ["e1"]}}

    def test_single_subject_multiple_regions(self):
        atlas_image, affine = self._make_atlas()
        per_subject_data = {
            1: (["e1", "e2"], np.array([[3.0, 5.0, 7.0], [8.0, 2.0, 4.0]])),
        }
        region_groups = {
            "temporal": ["L G_temp_sup-Lateral"],
            "frontal": ["L G_front_inf-Triangul"],
        }
        result = _build_region_map_from_arrays(
            per_subject_data, atlas_image, affine, region_groups
        )
        assert result["temporal"] == {1: ["e1"]}
        assert result["frontal"] == {1: ["e2"]}

    def test_multiple_subjects_same_region(self):
        atlas_image, affine = self._make_atlas()
        per_subject_data = {
            1: (["s1_e1"], np.array([[3.0, 5.0, 7.0]])),
            2: (["s2_e1"], np.array([[3.0, 5.0, 7.0]])),
        }
        region_groups = {"temporal": ["L G_temp_sup-Lateral"]}
        result = _build_region_map_from_arrays(
            per_subject_data, atlas_image, affine, region_groups
        )
        assert result["temporal"][1] == ["s1_e1"]
        assert result["temporal"][2] == ["s2_e1"]

    def test_multiple_subjects_different_regions(self):
        atlas_image, affine = self._make_atlas()
        per_subject_data = {
            1: (["s1_e1"], np.array([[3.0, 5.0, 7.0]])),
            2: (["s2_e1"], np.array([[8.0, 2.0, 4.0]])),
        }
        region_groups = {
            "temporal": ["L G_temp_sup-Lateral"],
            "frontal": ["L G_front_inf-Triangul"],
        }
        result = _build_region_map_from_arrays(
            per_subject_data, atlas_image, affine, region_groups
        )
        assert result["temporal"] == {1: ["s1_e1"]}
        assert result["frontal"] == {2: ["s2_e1"]}

    def test_electrode_in_unlisted_region_excluded(self):
        atlas_image, affine = self._make_atlas()
        per_subject_data = {
            1: (["e1", "e2"], np.array([[3.0, 5.0, 7.0], [8.0, 2.0, 4.0]])),
        }
        region_groups = {"temporal": ["L G_temp_sup-Lateral"]}  # frontal not listed
        result = _build_region_map_from_arrays(
            per_subject_data, atlas_image, affine, region_groups
        )
        assert result == {"temporal": {1: ["e1"]}}

    def test_subject_with_no_matching_electrodes_excluded(self):
        atlas_image, affine = self._make_atlas()
        per_subject_data = {
            1: (["e1"], np.array([[8.0, 2.0, 4.0]])),  # frontal only
        }
        region_groups = {"temporal": ["L G_temp_sup-Lateral"]}
        result = _build_region_map_from_arrays(
            per_subject_data, atlas_image, affine, region_groups
        )
        assert result == {}

    def test_empty_per_subject_data_returns_empty(self):
        atlas_image, affine = self._make_atlas()
        result = _build_region_map_from_arrays(
            {}, atlas_image, affine, {"temporal": ["L G_temp_sup-Lateral"]}
        )
        assert result == {}
