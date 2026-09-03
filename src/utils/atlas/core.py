"""Utilities used by retained script workflows."""

from __future__ import annotations
import numpy as np
from scipy.spatial import KDTree
DESTRIEUX_2009_LABELS: list[str] = [
    "Background",                       # 0
    "L G_and_S_frontomargin",          # 1
    "L G_and_S_occipital_inf",         # 2
    "L G_and_S_paracentral",           # 3
    "L G_and_S_subcentral",            # 4
    "L G_and_S_transv_frontopol",      # 5
    "L G_and_S_cingul-Ant",            # 6
    "L G_and_S_cingul-Mid-Ant",        # 7
    "L G_and_S_cingul-Mid-Post",       # 8
    "L G_cingul-Post-dorsal",          # 9
    "L G_cingul-Post-ventral",         # 10
    "L G_cuneus",                       # 11
    "L G_front_inf-Opercular",         # 12
    "L G_front_inf-Orbital",           # 13
    "L G_front_inf-Triangul",          # 14
    "L G_front_middle",                # 15
    "L G_front_sup",                   # 16
    "L G_Ins_lg_and_S_cent_ins",       # 17
    "L G_insular_short",               # 18
    "L G_occipital_middle",            # 19
    "L G_occipital_sup",               # 20
    "L G_oc-temp_lat-fusifor",         # 21
    "L G_oc-temp_med-Lingual",         # 22
    "L G_oc-temp_med-Parahip",         # 23
    "L G_orbital",                     # 24
    "L G_pariet_inf-Angular",          # 25
    "L G_pariet_inf-Supramar",         # 26
    "L G_parietal_sup",                # 27
    "L G_postcentral",                 # 28
    "L G_precentral",                  # 29
    "L G_precuneus",                   # 30
    "L G_rectus",                      # 31
    "L G_subcallosal",                 # 32
    "L G_temp_sup-G_T_transv",         # 33
    "L G_temp_sup-Lateral",            # 34
    "L G_temp_sup-Plan_polar",         # 35
    "L G_temp_sup-Plan_tempo",         # 36
    "L G_temporal_inf",                # 37
    "L G_temporal_middle",             # 38
    "L Lat_Fis-ant-Horizont",          # 39
    "L Lat_Fis-ant-Vertical",          # 40
    "L Lat_Fis-post",                  # 41
    "L Medial_wall",                   # 42
    "L Pole_occipital",                # 43
    "L Pole_temporal",                 # 44
    "L S_calcarine",                   # 45
    "L S_central",                     # 46
    "L S_cingul-Marginalis",           # 47
    "L S_circular_insula_ant",         # 48
    "L S_circular_insula_inf",         # 49
    "L S_circular_insula_sup",         # 50
    "L S_collat_transv_ant",           # 51
    "L S_collat_transv_post",          # 52
    "L S_front_inf",                   # 53
    "L S_front_middle",                # 54
    "L S_front_sup",                   # 55
    "L S_interm_prim-Jensen",          # 56
    "L S_intrapariet_and_P_trans",     # 57
    "L S_oc_middle_and_Lunatus",       # 58
    "L S_oc_sup_and_transversal",      # 59
    "L S_occipital_ant",               # 60
    "L S_oc-temp_lat",                 # 61
    "L S_oc-temp_med_and_Lingual",     # 62
    "L S_orbital_lateral",             # 63
    "L S_orbital_med-olfact",          # 64
    "L S_orbital-H_Shaped",            # 65
    "L S_parieto_occipital",           # 66
    "L S_pericallosal",                # 67
    "L S_postcentral",                 # 68
    "L S_precentral-inf-part",         # 69
    "L S_precentral-sup-part",         # 70
    "L S_suborbital",                  # 71
    "L S_subparietal",                 # 72
    "L S_temporal_inf",                # 73
    "L S_temporal_sup",                # 74
    "L S_temporal_transverse",         # 75
    "R G_and_S_frontomargin",          # 76
    "R G_and_S_occipital_inf",         # 77
    "R G_and_S_paracentral",           # 78
    "R G_and_S_subcentral",            # 79
    "R G_and_S_transv_frontopol",      # 80
    "R G_and_S_cingul-Ant",            # 81
    "R G_and_S_cingul-Mid-Ant",        # 82
    "R G_and_S_cingul-Mid-Post",       # 83
    "R G_cingul-Post-dorsal",          # 84
    "R G_cingul-Post-ventral",         # 85
    "R G_cuneus",                       # 86
    "R G_front_inf-Opercular",         # 87
    "R G_front_inf-Orbital",           # 88
    "R G_front_inf-Triangul",          # 89
    "R G_front_middle",                # 90
    "R G_front_sup",                   # 91
    "R G_Ins_lg_and_S_cent_ins",       # 92
    "R G_insular_short",               # 93
    "R G_occipital_middle",            # 94
    "R G_occipital_sup",               # 95
    "R G_oc-temp_lat-fusifor",         # 96
    "R G_oc-temp_med-Lingual",         # 97
    "R G_oc-temp_med-Parahip",         # 98
    "R G_orbital",                     # 99
    "R G_pariet_inf-Angular",          # 100
    "R G_pariet_inf-Supramar",         # 101
    "R G_parietal_sup",                # 102
    "R G_postcentral",                 # 103
    "R G_precentral",                  # 104
    "R G_precuneus",                   # 105
    "R G_rectus",                      # 106
    "R G_subcallosal",                 # 107
    "R G_temp_sup-G_T_transv",         # 108
    "R G_temp_sup-Lateral",            # 109
    "R G_temp_sup-Plan_polar",         # 110
    "R G_temp_sup-Plan_tempo",         # 111
    "R G_temporal_inf",                # 112
    "R G_temporal_middle",             # 113
    "R Lat_Fis-ant-Horizont",          # 114
    "R Lat_Fis-ant-Vertical",          # 115
    "R Lat_Fis-post",                  # 116
    "R Medial_wall",                   # 117
    "R Pole_occipital",                # 118
    "R Pole_temporal",                 # 119
    "R S_calcarine",                   # 120
    "R S_central",                     # 121
    "R S_cingul-Marginalis",           # 122
    "R S_circular_insula_ant",         # 123
    "R S_circular_insula_inf",         # 124
    "R S_circular_insula_sup",         # 125
    "R S_collat_transv_ant",           # 126
    "R S_collat_transv_post",          # 127
    "R S_front_inf",                   # 128
    "R S_front_middle",                # 129
    "R S_front_sup",                   # 130
    "R S_interm_prim-Jensen",          # 131
    "R S_intrapariet_and_P_trans",     # 132
    "R S_oc_middle_and_Lunatus",       # 133
    "R S_oc_sup_and_transversal",      # 134
    "R S_occipital_ant",               # 135
    "R S_oc-temp_lat",                 # 136
    "R S_oc-temp_med_and_Lingual",     # 137
    "R S_orbital_lateral",             # 138
    "R S_orbital_med-olfact",          # 139
    "R S_orbital-H_Shaped",            # 140
    "R S_parieto_occipital",           # 141
    "R S_pericallosal",                # 142
    "R S_postcentral",                 # 143
    "R S_precentral-inf-part",         # 144
    "R S_precentral-sup-part",         # 145
    "R S_suborbital",                  # 146
    "R S_subparietal",                 # 147
    "R S_temporal_inf",                # 148
    "R S_temporal_sup",                # 149
    "R S_temporal_transverse",         # 150
]

REGION_GROUPS: dict[str, list[str]] = {
    "EAC": [
        "L G_temp_sup-G_T_transv",
        "L G_temp_sup-Plan_polar",
        "L G_temp_sup-Lateral",
        "L S_temporal_sup",
        "L G_temp_sup-Plan_tempo",
        "L S_temporal_transverse",
    ],
    "MTG": [
        "L G_temporal_middle",
    ],
    "ITG": [
        "L G_oc-temp_lat-fusifor",
        "L S_oc-temp_lat",
        "L G_temporal_inf",
        "L S_temporal_inf",
    ],
    "TP": [
        "L Pole_temporal",
    ],
    "IFG": [
        "L G_front_inf-Triangul",
        "L G_front_inf-Opercular",
        "L G_front_inf-Orbital",
        "L S_front_inf",
        "L Lat_Fis-ant-Vertical",
        "L Lat_Fis-ant-Horizont",
    ],
    "TPJ": [
        "L G_pariet_inf-Supramar",
        "L G_pariet_inf-Angular",
        "L Lat_Fis-post",
        "L S_intrapariet_and_P_trans",
        "L S_interm_prim-Jensen",
    ],
    "PRC": [
        "L G_precentral",
        "L S_precentral-inf-part",
        "L S_precentral-sup-part",
    ],
    "PC": [
        "L G_postcentral",
        "L S_postcentral",
        "L G_and_S_subcentral",
        "L S_central",
        "L G_and_S_paracentral",
    ],
    "RIGHT": [
        "R G_postcentral",
        "R S_postcentral",
        "R S_interm_prim-Jensen",
        "R Lat_Fis-ant-Horizont",
        "R S_central",
        "R G_and_S_paracentral",
        "R G_and_S_subcentral",
        "R G_precentral",
        "R G_pariet_inf-Angular",
        "R S_intrapariet_and_P_trans",
        "R Lat_Fis-post",
        "R S_precentral-inf-part",
        "R S_precentral-sup-part",
        "R G_pariet_inf-Supramar",
        "R G_front_inf-Orbital",
        "R G_front_inf-Opercular",
        "R G_front_inf-Triangul",
        "R G_oc-temp_lat-fusifor",
        "R G_temporal_inf",
        "R Lat_Fis-ant-Vertical",
        "R S_front_inf",
        "R S_oc-temp_lat",
        "R G_temporal_middle",
        "R S_temporal_sup",
        "R G_temp_sup-Lateral",
        "R G_temp_sup-G_T_transv",
        "R G_temp_sup-Plan_polar",
        "R G_temp_sup-Plan_tempo",
        "R Pole_temporal",
        "R S_temporal_transverse",
        "R S_temporal_inf",
    ],
}


def _lookup_atlas_labels(
    coords: np.ndarray,
    atlas_image: np.ndarray,
    affine: np.ndarray,
    atlas_labels: list[str],
) -> list[str]:
    """Map MNI coordinates to atlas labels using nearest-neighbor lookup.

    Pure numpy/scipy implementation — no nilearn required. Suitable for unit tests.

    Args:
        coords: (n_elecs, 3) array of MNI coordinates in mm.
        atlas_image: 3-D integer array (voxel label indices).
        affine: (4, 4) affine matrix mapping voxel indices to MNI space.
        atlas_labels: List of label strings indexed by atlas integer value (0 = background).

    Returns:
        List of atlas label strings, one per electrode.
    """
    image_label_coords = np.nonzero(atlas_image)  # (i_arr, j_arr, k_arr) for non-zero voxels
    i_arr, j_arr, k_arr = image_label_coords
    n = len(i_arr)

    # Transform voxel indices to MNI space using affine (no nilearn needed)
    vox_homo = np.vstack([
        i_arr.astype(float),
        j_arr.astype(float),
        k_arr.astype(float),
        np.ones(n),
    ])  # (4, n)
    mni = (affine @ vox_homo)[:3].T  # (n, 3)

    tree = KDTree(mni)
    _, nearest_neighbor = tree.query(coords, k=1)

    x = image_label_coords[0][nearest_neighbor]
    y = image_label_coords[1][nearest_neighbor]
    z = image_label_coords[2][nearest_neighbor]

    elec_label_ids = atlas_image[x, y, z]
    return [atlas_labels[i] for i in elec_label_ids]


def group_electrodes_by_region(
    elec_names: list[str],
    atlas_labels: list[str],
    region_groups: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Group electrode names by brain region using exact atlas label matching.

    Pure function, no I/O. Labels must include hemisphere prefix (e.g.,
    "L G_temp_sup-Lateral"), as returned by `get_coord_atlas_labels`.

    Args:
        elec_names: Electrode names (one per electrode).
        atlas_labels: Atlas label per electrode, with hemisphere prefix.
        region_groups: Mapping of region_name -> list of atlas labels belonging to that region.

    Returns:
        Dict of {region_name: [elec_names]}, omitting regions with no matching electrodes.
        Electrodes that match no group are excluded.
    """
    label_to_region: dict[str, str] = {}
    for region_name, labels in region_groups.items():
        for label in labels:
            label_to_region[label] = region_name

    result: dict[str, list[str]] = {}
    for elec_name, label in zip(elec_names, atlas_labels):
        region = label_to_region.get(label)
        if region is not None:
            result.setdefault(region, []).append(elec_name)

    return result


def _build_region_map_from_arrays(
    per_subject_data: dict[int, tuple[list[str], np.ndarray]],
    atlas_image: np.ndarray,
    affine: np.ndarray,
    region_groups: dict[str, list[str]],
) -> dict[str, dict[int, list[str]]]:
    """Core region-map computation from pre-loaded arrays — no file I/O.

    Pure function, no I/O. Suitable for unit tests without mocks.

    Args:
        per_subject_data: {sub_id: (elec_names, coords)} where coords is
            (n_elecs, 3) MNI coordinates in mm.
        atlas_image: 3-D integer array (voxel label indices).
        affine: (4, 4) affine matrix mapping voxel indices to MNI space.
        region_groups: Mapping of region_name -> list of atlas labels (with hemisphere prefix).

    Returns:
        {region_name: {sub_id: [elec_names]}} — only subjects/regions with matching
        electrodes are included.
    """
    result: dict[str, dict[int, list[str]]] = {}

    for sub_id, (elec_names, coords) in per_subject_data.items():
        labels = _lookup_atlas_labels(coords, atlas_image, affine, DESTRIEUX_2009_LABELS)
        sub_region_map = group_electrodes_by_region(elec_names, labels, region_groups)

        for region_name, elecs in sub_region_map.items():
            result.setdefault(region_name, {})[sub_id] = elecs

    return result


def build_electrode_region_map(
    subject_ids: list[int],
    data_root: str,
    region_groups: dict[str, list[str]],
    atlas_path: str | None = None,
) -> dict[str, dict[int, list[str]]]:
    """Build a mapping of {region_name: {sub_id: [elec_names]}} for all subjects.

    For each subject, loads the raw BIDS file header (no waveform data) to extract
    electrode MNI coordinates, then assigns electrodes to regions via atlas lookup.

    Atlas loading:
        - atlas_path=None: downloads Destrieux 2009 via nilearn (cached after first use).
        - atlas_path provided: loads directly (no internet needed).

    Args:
        subject_ids: List of integer subject IDs.
        data_root: Root path of the BIDS dataset.
        region_groups: Mapping of region_name -> list of atlas labels (with hemisphere prefix).
        atlas_path: Optional path to Destrieux 2009 .nii.gz. None = download via nilearn.

    Returns:
        {region_name: {sub_id: [elec_names]}} — only subjects/regions with matching
        electrodes are included.
    """
    import os

    import mne
    from mne_bids import BIDSPath
    from nilearn import datasets, image as nli_image

    if atlas_path is None:
        atlas = datasets.fetch_atlas_destrieux_2009()
        atlas_path = atlas["maps"]

    img = nli_image.load_img(atlas_path)
    atlas_image = img.get_fdata().astype(int)
    affine = img.affine

    per_subject_data: dict[int, tuple[list[str], np.ndarray]] = {}
    for sub_id in subject_ids:
        file_path = BIDSPath(
            root=os.path.join(data_root, "derivatives/ecogprep"),
            subject=f"{sub_id:02}",
            task="podcast",
            datatype="ieeg",
            description="highgamma",
            suffix="ieeg",
            extension=".fif",
        )
        raw = mne.io.read_raw_fif(file_path, preload=False, verbose=False)
        coords = np.array([ch["loc"][:3] * 1000 for ch in raw.info["chs"]])
        per_subject_data[sub_id] = (raw.ch_names, coords)

    return _build_region_map_from_arrays(per_subject_data, atlas_image, affine, region_groups)
