import json

import numpy as np
import pandas as pd

from scripts.utilities.electrode_region_distribution import (
    ElectrodeRecord,
    _apply_atlas_labels,
    _btb_records,
    _discover_btb_subjects,
    _load_btb_label_map,
    _load_btb_localization,
    _load_btb_surface_coords,
    _normalize_label_payload,
    _normalize_btb_region,
    plot_distribution,
    plot_brain_locations,
    plot_brain_locations_2d,
    plot_brain_locations_nilearn,
    summarize_records,
)


def test_normalize_brain_treebank_label_extracts_region_and_coords():
    label = _normalize_label_payload(
        4,
        {
            "name": "LH1",
            "destrieux_label": "L G_temp_sup-Lateral",
            "mni": [-52.0, -20.0, 8.0],
            "hemisphere": "L",
        },
    )

    assert label["electrode"] == "LH1"
    assert label["region"] == "L G_temp_sup-Lateral"
    assert label["atlas_label"] == "L G_temp_sup-Lateral"
    assert label["hemisphere"] == "L"
    assert label["x"] == -52.0
    assert label["y"] == -20.0
    assert label["z"] == 8.0


def test_load_brain_treebank_label_map_accepts_dict_keys(tmp_path):
    path = tmp_path / "electrode_labels.json"
    path.write_text(
        json.dumps(
            {
                "electrode_2": {
                    "label": "A2",
                    "region": "superior temporal",
                    "coords": {"x": 1, "y": 2, "z": 3},
                },
                "4": "B4",
            }
        )
    )

    labels = _load_btb_label_map(str(path))

    assert labels[2]["electrode"] == "A2"
    assert labels[2]["region"] == "superior temporal"
    assert labels[2]["x"] == 1.0
    assert labels[4]["electrode"] == "B4"


def test_load_brain_treebank_localization_normalizes_destrieux(tmp_path):
    path = tmp_path / "depth-wm.csv"
    pd.DataFrame(
        [
            {
                "Electrode": "A1",
                "L": 10,
                "I": 20,
                "P": 30,
                "DesikanKilliany": "ctx-lh-superiortemporal",
                "Destrieux": "ctx_lh_G_temp_sup-Lateral",
                "DKT": "ctx-lh-superiortemporal",
                "ShiftDist": 0,
                "ConfType": 0,
            }
        ]
    ).to_csv(path, index=False)

    localization = _load_btb_localization(str(path))

    assert _normalize_btb_region("ctx_rh_G_front_middle") == "R G_front_middle"
    assert localization["A1"]["region"] == "L G_temp_sup-Lateral"
    assert localization["A1"]["atlas_label"] == "L G_temp_sup-Lateral"
    assert localization["A1"]["hemisphere"] == "L"


def test_brain_treebank_records_can_load_without_movie_or_hdf5(tmp_path):
    labels_path = tmp_path / "electrode_labels" / "sub_3" / "electrode_labels.json"
    labels_path.parent.mkdir(parents=True)
    labels_path.write_text(json.dumps(["A1", "A2"]))

    loc_path = tmp_path / "localization" / "sub_3" / "depth-wm.csv"
    loc_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"Electrode": "A1", "Destrieux": "ctx_lh_G_front_middle"},
            {"Electrode": "A2", "Destrieux": "ctx_lh_S_front_middle"},
        ]
    ).to_csv(loc_path, index=False)
    surface_path = tmp_path / "localization" / "elec_coords_full.csv"
    pd.DataFrame(
        [
            {
                "Subject": "sub_3",
                "Electrode": "A1",
                "X": 10,
                "Y": 20,
                "Z": 30,
                "Hemisphere": 1,
            }
        ]
    ).to_csv(surface_path, index=False)

    records = _btb_records([3], str(tmp_path), movie=None)

    assert [record.electrode for record in records] == ["A1", "A2"]
    assert [record.region for record in records] == [
        "L G_front_middle",
        "L S_front_middle",
    ]
    assert {record.region_source for record in records} == {"localization"}
    assert records[0].surface_x == 10
    assert records[0].surface_hemi == "L"
    assert records[1].surface_x is None


def test_load_brain_treebank_surface_coords(tmp_path):
    path = tmp_path / "elec_coords_full.csv"
    pd.DataFrame(
        [
            {
                "Subject": "sub_4",
                "Electrode": "E1",
                "X": 1,
                "Y": 2,
                "Z": 3,
                "Hemisphere": 0,
            }
        ]
    ).to_csv(path, index=False)

    coords = _load_btb_surface_coords(str(path))

    assert coords[(4, "E1")]["surface_x"] == 1
    assert coords[(4, "E1")]["surface_hemi"] == "R"


def test_discover_brain_treebank_subjects_defaults_to_all_movies(tmp_path):
    metadata_dir = tmp_path / "subject_metadata"
    metadata_dir.mkdir()
    (metadata_dir / "sub_1_trial000_metadata.json").write_text(
        json.dumps({"filename": "movie-a"})
    )
    (metadata_dir / "sub_2_trial000_metadata.json").write_text(
        json.dumps({"filename": "movie-b"})
    )

    assert _discover_btb_subjects(str(tmp_path), movie=None) == [1, 2]
    assert _discover_btb_subjects(str(tmp_path), movie="movie-a") == [1]


def test_summarize_records_counts_per_subject_and_significance():
    records = [
        ElectrodeRecord("podcast", 1, "A1", "temporal", "metadata", significant=True),
        ElectrodeRecord("podcast", 1, "A2", "temporal", "metadata", significant=False),
        ElectrodeRecord("podcast", 1, "B1", "frontal", "metadata", significant=True),
        ElectrodeRecord("podcast", 2, "C1", "temporal", "metadata", significant=False),
    ]

    summary = summarize_records(records)
    temporal_s1 = summary[(summary.subject == 1) & (summary.region == "temporal")].iloc[0]

    assert temporal_s1.n_electrodes == 2
    assert temporal_s1.subject_total == 3
    assert temporal_s1.n_significant == 1
    assert temporal_s1.subject_significant_total == 2


def test_auto_atlas_labels_only_fill_unknown_regions(monkeypatch):
    records = [
        ElectrodeRecord("brain_treebank", 1, "A1", "metadata region", "metadata", x=0, y=0, z=0),
        ElectrodeRecord("brain_treebank", 1, "A2", "Unknown", "missing", x=1, y=1, z=1),
    ]

    def fake_load_atlas(_atlas_path):
        atlas_image = np.zeros((2, 2, 2), dtype=int)
        atlas_image[1, 1, 1] = 34
        return atlas_image, np.eye(4)

    monkeypatch.setattr(
        "scripts.utilities.electrode_region_distribution._load_atlas",
        fake_load_atlas,
    )

    _apply_atlas_labels(records, atlas_path=None, only_missing=True)

    assert records[0].region == "metadata region"
    assert records[1].region == "L G_temp_sup-Lateral"


def test_plot_distribution_writes_file(tmp_path):
    records = [
        ElectrodeRecord("brain_treebank", 1, "A1", "R1", "localization"),
        ElectrodeRecord("brain_treebank", 1, "A2", "R2", "localization"),
        ElectrodeRecord("brain_treebank", 2, "B1", "R1", "localization"),
    ]
    summary = summarize_records(records)
    plot_path = tmp_path / "plot.png"

    plot_distribution(summary, str(plot_path), max_regions=2)

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_plot_brain_locations_writes_subplot_figure(tmp_path):
    detail_df = pd.DataFrame(
        [
            {
                "dataset": "brain_treebank",
                "subject": 1,
                "electrode": "A1",
                "region": "R1",
                "region_source": "localization",
                "hemisphere": "L",
                "surface_hemi": "L",
                "surface_x": 10,
                "surface_y": 20,
                "surface_z": 30,
            },
            {
                "dataset": "brain_treebank",
                "subject": 2,
                "electrode": "B1",
                "region": "R2",
                "region_source": "localization",
                "hemisphere": "R",
                "surface_hemi": "R",
                "surface_x": 15,
                "surface_y": 25,
                "surface_z": 35,
            },
        ]
    )
    plot_path = tmp_path / "brain_locations.png"

    plot_brain_locations(detail_df, str(plot_path))

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_plot_brain_locations_2d_writes_brain_image_figure(tmp_path):
    detail_df = pd.DataFrame(
        [
            {
                "dataset": "brain_treebank",
                "subject": 1,
                "electrode": "A1",
                "region": "R1",
                "region_source": "localization",
                "hemisphere": "L",
                "surface_hemi": "L",
                "surface_x": 10,
                "surface_y": 20,
                "surface_z": 30,
            },
            {
                "dataset": "brain_treebank",
                "subject": 1,
                "electrode": "A2",
                "region": "R2",
                "region_source": "localization",
                "hemisphere": "R",
                "surface_hemi": "R",
                "surface_x": 15,
                "surface_y": 25,
                "surface_z": 35,
            },
        ]
    )
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = np.ones((8, 12, 3), dtype=float)
    import matplotlib.pyplot as plt

    plt.imsave(image_dir / "left_hem_clean.png", image)
    plt.imsave(image_dir / "right_hem_clean.png", image)
    plot_path = tmp_path / "brain_locations_2d.png"

    plot_brain_locations_2d(detail_df, str(plot_path), image_dir=str(image_dir))

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_plot_brain_locations_nilearn_writes_glass_brain(tmp_path):
    detail_df = pd.DataFrame(
        [
            {
                "dataset": "brain_treebank",
                "subject": 1,
                "electrode": "A1",
                "region": "R1",
                "region_source": "localization",
                "surface_hemi": "L",
                "surface_x": 10,
                "surface_y": 20,
                "surface_z": 30,
            },
            {
                "dataset": "brain_treebank",
                "subject": 1,
                "electrode": "A2",
                "region": "R2",
                "region_source": "localization",
                "surface_hemi": "R",
                "surface_x": 15,
                "surface_y": -25,
                "surface_z": 5,
            },
        ]
    )
    plot_path = tmp_path / "nilearn_locations.png"

    plot_brain_locations_nilearn(detail_df, str(plot_path), display_mode="ortho")

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
