#!/usr/bin/env python
"""Summarize electrode region distributions per subject.

Supports both the podcast BIDS/FIF layout and Brain Treebank HDF5 metadata.
The detailed CSV has one row per electrode; the summary CSV/console table counts
electrodes per subject and region.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.registry import resolve_dataset_context
from src.utils.atlas import DESTRIEUX_2009_LABELS, _lookup_atlas_labels


EXPLICIT_REGION_KEYS = (
    "region",
    "atlas_region",
    "destrieux_region",
    "destrieux_label",
    "anatomy",
    "anatomical_label",
    "fs_label",
    "location",
)
ATLAS_LABEL_KEYS = (
    "atlas_label",
    "destrieux_label",
    "destrieux_region",
    "aparc_a2009s",
    "fs_label",
)
NAME_KEYS = ("name", "label", "electrode", "electrode_label", "channel", "channel_name")
HEMISPHERE_KEYS = ("hemisphere", "hemi", "side")


@dataclass
class ElectrodeRecord:
    dataset: str
    subject: int
    electrode: str
    region: str
    region_source: str
    atlas_label: str | None = None
    hemisphere: str | None = None
    x: float | None = None
    y: float | None = None
    z: float | None = None
    loc_l: float | None = None
    loc_i: float | None = None
    loc_p: float | None = None
    surface_x: float | None = None
    surface_y: float | None = None
    surface_z: float | None = None
    surface_hemi: str | None = None
    significant: bool | None = None


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    lowered = {str(k).lower(): v for k, v in mapping.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value not in (None, ""):
            return value
    return None


def _coerce_float_triplet(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, dict):
        values = [_first_present(value, keys) for keys in (("x", "mni_x"), ("y", "mni_y"), ("z", "mni_z"))]
    elif isinstance(value, (list, tuple)) and len(value) >= 3:
        values = value[:3]
    else:
        return None

    try:
        coords = tuple(float(v) for v in values)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(coords)):
        return None
    return coords


def _extract_coords(mapping: dict[str, Any]) -> tuple[float, float, float] | None:
    for keys in (
        ("mni_x", "mni_y", "mni_z"),
        ("x", "y", "z"),
        ("MNI_X", "MNI_Y", "MNI_Z"),
    ):
        if all(key in mapping for key in keys):
            coords = _coerce_float_triplet([mapping[key] for key in keys])
            if coords is not None:
                return coords

    for key in ("mni", "MNI", "mni_coord", "mni_coords", "coordinates", "coords", "coord", "xyz"):
        if key in mapping:
            coords = _coerce_float_triplet(mapping[key])
            if coords is not None:
                return coords
    return None


def _normalize_label_payload(electrode_num: int, payload: Any) -> dict[str, Any]:
    if isinstance(payload, str):
        return {"electrode": payload}
    if not isinstance(payload, dict):
        return {"electrode": f"electrode_{electrode_num}"}

    name = _first_present(payload, NAME_KEYS) or f"electrode_{electrode_num}"
    region = _first_present(payload, EXPLICIT_REGION_KEYS)
    atlas_label = _first_present(payload, ATLAS_LABEL_KEYS)
    hemisphere = _first_present(payload, HEMISPHERE_KEYS)
    coords = _extract_coords(payload)
    result = {
        "electrode": str(name),
        "region": str(region) if region is not None else None,
        "atlas_label": str(atlas_label) if atlas_label is not None else None,
        "hemisphere": str(hemisphere) if hemisphere is not None else None,
    }
    if coords is not None:
        result.update({"x": coords[0], "y": coords[1], "z": coords[2]})
    return result


def _load_btb_label_map(path: str) -> dict[int, dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        labels = json.load(f)

    if isinstance(labels, list):
        return {idx: _normalize_label_payload(idx, value) for idx, value in enumerate(labels)}
    if isinstance(labels, dict):
        parsed = {}
        for key, value in labels.items():
            try:
                electrode_num = int(key)
            except (TypeError, ValueError):
                match = re.search(r"(\d+)$", str(key))
                if not match:
                    continue
                electrode_num = int(match.group(1))
            parsed[electrode_num] = _normalize_label_payload(electrode_num, value)
        return parsed
    return {}


def _normalize_btb_region(value: Any) -> str | None:
    if value in (None, "", "Unknown"):
        return None
    label = str(value)
    for prefix, hemi in (("ctx_lh_", "L "), ("ctx-rh-", "R "), ("ctx_rh_", "R "), ("ctx-lh-", "L ")):
        if label.startswith(prefix):
            return hemi + label[len(prefix):]
    return label


def _load_btb_localization(path: str) -> dict[str, dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if "Electrode" not in df.columns:
        return {}

    result = {}
    for row in df.to_dict("records"):
        electrode = str(row["Electrode"])
        destrieux = _normalize_btb_region(row.get("Destrieux"))
        desikan = _normalize_btb_region(row.get("DesikanKilliany"))
        region = destrieux or desikan
        if region is None:
            continue
        hemisphere = None
        if region.startswith(("L ", "R ")):
            hemisphere = region[0]
        result[electrode] = {
            "region": region,
            "atlas_label": destrieux,
            "hemisphere": hemisphere,
            "loc_l": row.get("L"),
            "loc_i": row.get("I"),
            "loc_p": row.get("P"),
        }
    return result


def _load_btb_surface_coords(path: str) -> dict[tuple[int, str], dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    required = {"Subject", "Electrode", "X", "Y", "Z", "Hemisphere"}
    if not required <= set(df.columns):
        return {}

    result = {}
    for row in df.to_dict("records"):
        match = re.search(r"(\d+)$", str(row["Subject"]))
        if not match:
            continue
        hemi = "L" if int(row["Hemisphere"]) == 1 else "R"
        result[(int(match.group(1)), str(row["Electrode"]))] = {
            "surface_x": row["X"],
            "surface_y": row["Y"],
            "surface_z": row["Z"],
            "surface_hemi": hemi,
        }
    return result


def _load_atlas(atlas_path: str | None):
    from nilearn import datasets, image as nli_image

    if atlas_path is None:
        atlas_path = datasets.fetch_atlas_destrieux_2009()["maps"]
    img = nli_image.load_img(atlas_path)
    return img.get_fdata().astype(int), img.affine


def _apply_atlas_labels(
    records: list[ElectrodeRecord],
    atlas_path: str | None,
    only_missing: bool = False,
) -> None:
    indexed = [
        (idx, (record.x, record.y, record.z))
        for idx, record in enumerate(records)
        if not only_missing or record.region == "Unknown"
        if record.x is not None and record.y is not None and record.z is not None
    ]
    if not indexed:
        return
    atlas_image, affine = _load_atlas(atlas_path)
    labels = _lookup_atlas_labels(
        np.asarray([coords for _, coords in indexed], dtype=float),
        atlas_image,
        affine,
        DESTRIEUX_2009_LABELS,
    )
    for (idx, _), label in zip(indexed, labels):
        records[idx].atlas_label = label
        records[idx].region = label
        records[idx].region_source = "atlas"
        if label.startswith(("L ", "R ")):
            records[idx].hemisphere = label[0]


def _load_significance(path: str | None) -> set[tuple[int, str]] | None:
    if path is None:
        return None
    sig_csv = path
    if os.path.isdir(path):
        sig_csv = os.path.join(path, "all_subject_sig.csv")
    if not os.path.exists(sig_csv):
        raise FileNotFoundError(f"Significance CSV not found: {sig_csv}")
    sig_df = pd.read_csv(sig_csv)
    if not {"subject", "elec"} <= set(sig_df.columns):
        raise ValueError(f"Expected columns subject, elec in {sig_csv}")
    return {(int(row.subject), str(row.elec)) for row in sig_df.itertuples()}


def _discover_podcast_subjects(data_root: str) -> list[int]:
    pattern = os.path.join(data_root, "derivatives", "ecogprep", "sub-*", "ieeg")
    subjects = []
    for path in sorted(glob.glob(pattern)):
        match = re.search(r"sub-(\d+)", path)
        if match:
            subjects.append(int(match.group(1)))
    return subjects or list(range(1, 10))


def _discover_btb_subjects(data_root: str, movie: str | None) -> list[int]:
    pattern = os.path.join(data_root, "subject_metadata", "sub_*_trial*_metadata.json")
    subjects = set()
    for path in glob.glob(pattern):
        with open(path) as f:
            metadata = json.load(f)
        if movie is not None and metadata.get("filename") != movie:
            continue
        match = re.search(r"sub_(\d+)_trial", os.path.basename(path))
        if match:
            subjects.add(int(match.group(1)))
    return sorted(subjects)


def _podcast_records(subject_ids: list[int], data_root: str) -> list[ElectrodeRecord]:
    import mne
    from mne_bids import BIDSPath

    records = []
    for subject_id in subject_ids:
        file_path = BIDSPath(
            root=os.path.join(data_root, "derivatives", "ecogprep"),
            subject=f"{subject_id:02}",
            task="podcast",
            datatype="ieeg",
            description="highgamma",
            suffix="ieeg",
            extension=".fif",
        )
        if not os.path.exists(file_path):
            fallback = file_path.copy()
            fallback.update(description=None)
            file_path = fallback
        if not os.path.exists(file_path):
            print(f"Subject {subject_id}: recording not found, skipping")
            continue

        raw = mne.io.read_raw_fif(file_path, preload=False, verbose=False)
        for ch in raw.info["chs"]:
            coords = np.asarray(ch["loc"][:3], dtype=float) * 1000
            if not np.all(np.isfinite(coords)):
                coords = np.array([np.nan, np.nan, np.nan])
            records.append(
                ElectrodeRecord(
                    dataset="podcast",
                    subject=subject_id,
                    electrode=ch["ch_name"],
                    region="Unknown",
                    region_source="missing",
                    x=None if np.isnan(coords[0]) else float(coords[0]),
                    y=None if np.isnan(coords[1]) else float(coords[1]),
                    z=None if np.isnan(coords[2]) else float(coords[2]),
                )
            )
    return records


def _btb_records(
    subject_ids: list[int],
    data_root: str,
    movie: str | None = None,
) -> list[ElectrodeRecord]:
    context = None
    if movie is not None:
        context = resolve_dataset_context(
            dataset="brain_treebank",
            subject_ids=subject_ids,
            per_dataset_config={"movie": movie},
            data_root=data_root,
        )

    surface_coords = _load_btb_surface_coords(
        os.path.join(data_root, "localization", "elec_coords_full.csv")
    )
    records = []
    for subject_id in subject_ids:
        if context is not None:
            metadata = context.subject_metadata[subject_id]
            recording_path = metadata["recording_path"]
            electrode_labels_path = metadata["electrode_labels_path"]
        else:
            recording_path = None
            electrode_labels_path = os.path.join(
                data_root, "electrode_labels", f"sub_{subject_id}", "electrode_labels.json"
            )

        label_map = _load_btb_label_map(electrode_labels_path)
        localization_path = os.path.join(
            data_root, "localization", f"sub_{subject_id}", "depth-wm.csv"
        )
        localization = _load_btb_localization(localization_path)

        if recording_path is not None and os.path.exists(recording_path):
            import h5py

            with h5py.File(recording_path, "r") as h5f:
                electrode_nums = []
                for key in h5f["data"].keys():
                    match = re.fullmatch(r"electrode_(\d+)", key)
                    if match:
                        electrode_nums.append(int(match.group(1)))
        else:
            electrode_nums = sorted(label_map.keys())

        for electrode_num in sorted(electrode_nums):
            label = label_map.get(electrode_num, {})
            electrode = label.get("electrode") or f"electrode_{electrode_num}"
            loc = localization.get(str(electrode), {})
            region = label.get("region") or label.get("atlas_label") or loc.get("region") or "Unknown"
            region_source = "metadata" if label.get("region") or label.get("atlas_label") else (
                "localization" if loc.get("region") else "missing"
            )
            surface = surface_coords.get((subject_id, str(electrode)), {})
            records.append(
                ElectrodeRecord(
                    dataset="brain_treebank",
                    subject=subject_id,
                    electrode=str(electrode),
                    region=str(region),
                    region_source=region_source,
                    atlas_label=label.get("atlas_label") or loc.get("atlas_label"),
                    hemisphere=label.get("hemisphere") or loc.get("hemisphere"),
                    x=label.get("x"),
                    y=label.get("y"),
                    z=label.get("z"),
                    loc_l=loc.get("loc_l"),
                    loc_i=loc.get("loc_i"),
                    loc_p=loc.get("loc_p"),
                    surface_x=surface.get("surface_x"),
                    surface_y=surface.get("surface_y"),
                    surface_z=surface.get("surface_z"),
                    surface_hemi=surface.get("surface_hemi"),
                )
            )
    return records


def _coerce_plot_coords(detail_df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str, str], tuple[str, str, str]]:
    if {"surface_x", "surface_y", "surface_z"} <= set(detail_df.columns):
        coord_cols = ("surface_x", "surface_y", "surface_z")
        labels = ("Surface x", "Surface y", "Surface z")
        df = detail_df.copy()
        for col in coord_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[list(coord_cols)].notna().all(axis=1).any():
            return df, coord_cols, labels

    if {"loc_l", "loc_i", "loc_p"} <= set(detail_df.columns):
        coord_cols = ("loc_l", "loc_i", "loc_p")
        labels = ("L", "I", "P")
        df = detail_df.copy()
        for col in coord_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[list(coord_cols)].notna().all(axis=1).any():
            return df, coord_cols, labels

    coord_cols = ("x", "y", "z")
    labels = ("MNI x", "MNI y", "MNI z")
    df = detail_df.copy()
    for col in coord_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, coord_cols, labels


def plot_brain_locations(
    detail_df: pd.DataFrame,
    plot_path: str,
    max_subjects: int | None = None,
) -> None:
    if detail_df.empty:
        return

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    plot_df, coord_cols, axis_labels = _coerce_plot_coords(detail_df)
    plot_df = plot_df.dropna(subset=list(coord_cols))
    if plot_df.empty:
        print("No electrode coordinates available; skipping brain-location plot")
        return

    subjects = sorted(plot_df["subject"].unique())
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
    n_subjects = len(subjects)
    n_cols = min(5, n_subjects)
    n_rows = int(np.ceil(n_subjects / n_cols))
    fig = plt.figure(figsize=(3.2 * n_cols, 3.0 * n_rows))

    all_coords = plot_df[list(coord_cols)].to_numpy(dtype=float)
    center = np.nanmean(all_coords, axis=0)
    radius = np.nanmax(np.ptp(all_coords, axis=0)) / 2
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0

    u = np.linspace(0, 2 * np.pi, 32)
    v = np.linspace(0, np.pi, 16)
    ellipsoid_x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    ellipsoid_y = center[1] + 0.75 * radius * np.outer(np.sin(u), np.sin(v))
    ellipsoid_z = center[2] + 0.9 * radius * np.outer(np.ones_like(u), np.cos(v))

    cmap = plt.get_cmap("coolwarm")
    for idx, subject in enumerate(subjects, start=1):
        ax = fig.add_subplot(n_rows, n_cols, idx, projection="3d")
        sub_df = plot_df[plot_df["subject"] == subject]
        coords = sub_df[list(coord_cols)].to_numpy(dtype=float)
        hemi_series = sub_df.get("surface_hemi", sub_df.get("hemisphere"))
        colors = [
            cmap(0.15) if hemi == "L" else cmap(0.85) if hemi == "R" else (0.35, 0.35, 0.35, 1.0)
            for hemi in hemi_series
        ]

        ax.plot_surface(
            ellipsoid_x,
            ellipsoid_y,
            ellipsoid_z,
            color="#d9d9d9",
            alpha=0.08,
            linewidth=0,
            shade=False,
        )
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            s=12,
            c=list(colors),
            alpha=0.85,
            depthshade=False,
        )
        ax.set_title(f"Subject {subject} ({len(coords)})", fontsize=10)
        ax.set_xlabel(axis_labels[0], fontsize=8)
        ax.set_ylabel(axis_labels[1], fontsize=8)
        ax.set_zlabel(axis_labels[2], fontsize=8)
        ax.tick_params(labelsize=7, pad=0)
        ax.view_init(elev=18, azim=-65)
        ax.set_box_aspect((1, 0.75, 0.9))

    for idx in range(n_subjects + 1, n_rows * n_cols + 1):
        ax = fig.add_subplot(n_rows, n_cols, idx, projection="3d")
        ax.set_axis_off()

    fig.suptitle("Electrode Locations by Subject", y=0.995)
    fig.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def _normalize_values(values: pd.Series, vmin: float, vmax: float) -> pd.Series:
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return pd.Series(np.full(len(values), 0.5), index=values.index)
    return (values - vmin) / (vmax - vmin)


def plot_brain_locations_2d(
    detail_df: pd.DataFrame,
    plot_path: str,
    image_dir: str | None = None,
    max_subjects: int | None = None,
) -> None:
    if detail_df.empty:
        return

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    plot_df, coord_cols, _ = _coerce_plot_coords(detail_df)
    plot_df = plot_df.dropna(subset=list(coord_cols)).copy()
    if plot_df.empty:
        print("No electrode coordinates available; skipping 2D brain-location plot")
        return

    hemi_col = "surface_hemi" if "surface_hemi" in plot_df.columns else "hemisphere"
    plot_df["plot_hemi"] = plot_df[hemi_col].fillna(plot_df.get("hemisphere", ""))
    plot_df["plot_hemi"] = plot_df["plot_hemi"].where(plot_df["plot_hemi"].isin(["L", "R"]), "L")

    horizontal_col = coord_cols[1]
    vertical_col = coord_cols[2]
    plot_df["x2d"] = _normalize_values(
        plot_df[horizontal_col], plot_df[horizontal_col].min(), plot_df[horizontal_col].max()
    )
    plot_df["y2d"] = _normalize_values(
        plot_df[vertical_col], plot_df[vertical_col].min(), plot_df[vertical_col].max()
    )

    left_img = right_img = None
    if image_dir is not None:
        left_path = os.path.join(image_dir, "left_hem_clean.png")
        right_path = os.path.join(image_dir, "right_hem_clean.png")
        if os.path.exists(left_path):
            left_img = mpimg.imread(left_path)
        if os.path.exists(right_path):
            right_img = mpimg.imread(right_path)

    subjects = sorted(plot_df["subject"].unique())
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
    n_subjects = len(subjects)
    n_cols = min(5, n_subjects)
    n_rows = int(np.ceil(n_subjects / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 2.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    colors = {"L": "#2f6fb0", "R": "#c9473d"}
    for ax, subject in zip(axes, subjects):
        sub_df = plot_df[plot_df["subject"] == subject]
        if left_img is not None:
            ax.imshow(left_img, extent=(0, 1, 0, 1), aspect="auto", zorder=0)
        else:
            ax.add_patch(plt.Circle((0.5, 0.5), 0.43, color="#d6d6d6", alpha=0.7))
        if right_img is not None:
            ax.imshow(right_img, extent=(1.08, 2.08, 0, 1), aspect="auto", zorder=0)
        else:
            ax.add_patch(plt.Circle((1.58, 0.5), 0.43, color="#d6d6d6", alpha=0.7))

        for hemi, offset in (("L", 0.0), ("R", 1.08)):
            hemi_df = sub_df[sub_df["plot_hemi"] == hemi]
            if hemi_df.empty:
                continue
            ax.scatter(
                offset + hemi_df["x2d"],
                hemi_df["y2d"],
                s=12,
                c=colors[hemi],
                alpha=0.85,
                edgecolors="white",
                linewidths=0.25,
                zorder=2,
            )

        ax.text(0.5, -0.06, "L", ha="center", va="top", fontsize=8)
        ax.text(1.58, -0.06, "R", ha="center", va="top", fontsize=8)
        ax.set_title(f"Subject {subject} ({len(sub_df)})", fontsize=10)
        ax.set_xlim(0, 2.08)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes[n_subjects:]:
        ax.axis("off")

    fig.suptitle("Electrode Locations on Lateral Brain Views by Subject", y=0.995)
    fig.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def _coerce_nilearn_coords(detail_df: pd.DataFrame) -> pd.DataFrame:
    df = detail_df.copy()
    if {"surface_x", "surface_y", "surface_z", "surface_hemi"} <= set(df.columns):
        for col in ("surface_x", "surface_y", "surface_z"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["surface_x", "surface_y", "surface_z"]).copy()
        if not df.empty:
            hemi = df["surface_hemi"].fillna(df.get("hemisphere", ""))
            df["plot_x"] = np.where(hemi == "L", -np.abs(df["surface_x"]), np.abs(df["surface_x"]))
            df["plot_y"] = df["surface_y"]
            df["plot_z"] = df["surface_z"]
            df["plot_hemi"] = hemi.where(hemi.isin(["L", "R"]), "R")
            return df

    for col in ("x", "y", "z"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["x", "y", "z"]).copy()
    df["plot_x"] = df["x"]
    df["plot_y"] = df["y"]
    df["plot_z"] = df["z"]
    df["plot_hemi"] = np.where(df["plot_x"] < 0, "L", "R")
    return df


def plot_brain_locations_nilearn(
    detail_df: pd.DataFrame,
    plot_path: str,
    max_subjects: int | None = None,
    display_mode: str = "lyrz",
) -> None:
    if detail_df.empty:
        return

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        from nilearn import plotting
    except ImportError as exc:
        raise ImportError(
            "nilearn is required for --nilearn_plot_path. Install it in the active "
            "environment, e.g. `python -m pip install nilearn`."
        ) from exc
    import matplotlib.pyplot as plt

    plot_df = _coerce_nilearn_coords(detail_df)
    if plot_df.empty:
        print("No electrode coordinates available; skipping nilearn brain-location plot")
        return

    subjects = sorted(plot_df["subject"].unique())
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
    n_subjects = len(subjects)
    n_cols = min(2, n_subjects)
    n_rows = int(np.ceil(n_subjects / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.2 * n_cols, 2.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, subject in zip(axes, subjects):
        sub_df = plot_df[plot_df["subject"] == subject]
        coords = sub_df[["plot_x", "plot_y", "plot_z"]].to_numpy(dtype=float)
        colors = [
            "#2f6fb0" if hemi == "L" else "#c9473d"
            for hemi in sub_df["plot_hemi"]
        ]
        adjacency = np.zeros((len(coords), len(coords)), dtype=float)
        plotting.plot_connectome(
            adjacency,
            coords,
            node_color=colors,
            node_size=18,
            display_mode=display_mode,
            figure=fig,
            axes=ax,
            title=f"Subject {subject} ({len(coords)})",
            annotate=False,
            colorbar=False,
            black_bg=False,
            alpha=0.85,
        )

    for ax in axes[n_subjects:]:
        ax.axis("off")

    fig.suptitle("Nilearn Glass-Brain Electrode Projections by Subject", y=0.995)
    fig.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def plot_distribution(
    summary_df: pd.DataFrame,
    plot_path: str,
    max_regions: int = 20,
    value_col: str = "fraction",
) -> None:
    if summary_df.empty:
        return

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    totals = summary_df.groupby("region")["n_electrodes"].sum().sort_values(ascending=False)
    top_regions = totals.head(max_regions).index.tolist()
    plot_df = summary_df.copy()
    plot_df["plot_region"] = np.where(
        plot_df["region"].isin(top_regions), plot_df["region"], "Other"
    )
    plot_df = (
        plot_df.groupby(["subject", "plot_region"], as_index=False)
        .agg({"n_electrodes": "sum", "subject_total": "first"})
    )
    plot_df["fraction"] = plot_df["n_electrodes"] / plot_df["subject_total"]

    region_order = [r for r in top_regions if r in set(plot_df["plot_region"])]
    if "Other" in set(plot_df["plot_region"]):
        region_order.append("Other")
    subjects = sorted(plot_df["subject"].unique())
    pivot = (
        plot_df.pivot(index="subject", columns="plot_region", values=value_col)
        .reindex(index=subjects, columns=region_order)
        .fillna(0.0)
    )

    height = max(4.5, 0.45 * len(subjects) + 1.8)
    width = 12
    fig, ax = plt.subplots(figsize=(width, height))
    left = np.zeros(len(pivot), dtype=float)
    cmap = plt.get_cmap("tab20")
    for idx, region in enumerate(region_order):
        values = pivot[region].to_numpy()
        ax.barh(
            [str(s) for s in pivot.index],
            values,
            left=left,
            label=region,
            color=cmap(idx % cmap.N),
            edgecolor="white",
            linewidth=0.4,
        )
        left += values

    ax.set_xlabel("Fraction of subject electrodes" if value_col == "fraction" else "Electrode count")
    ax.set_ylabel("Subject")
    ax.set_title("Electrode Region Distribution by Subject")
    ax.set_xlim(0, 1 if value_col == "fraction" else max(left.max() * 1.05, 1))
    ax.grid(axis="x", alpha=0.25)
    ax.legend(
        title=f"Top {min(max_regions, len(totals))} regions",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize="small",
    )
    fig.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def summarize_records(records: list[ElectrodeRecord]) -> pd.DataFrame:
    rows = []
    counts = Counter((record.subject, record.region) for record in records)
    totals = Counter(record.subject for record in records)
    sig_counts = Counter(
        (record.subject, record.region)
        for record in records
        if record.significant is True
    )
    sig_totals = Counter(record.subject for record in records if record.significant is True)

    for (subject, region), count in sorted(counts.items()):
        row = {
            "subject": subject,
            "region": region,
            "n_electrodes": count,
            "subject_total": totals[subject],
            "fraction": count / totals[subject],
        }
        if any(record.significant is not None for record in records):
            row["n_significant"] = sig_counts[(subject, region)]
            row["subject_significant_total"] = sig_totals[subject]
        rows.append(row)
    return pd.DataFrame(rows)


def _load_config_defaults(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize per-subject electrode region distributions."
    )
    parser.add_argument("--config", default=None, help="Optional analysis YAML config")
    parser.add_argument("--dataset", choices=["podcast", "brain_treebank"], default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--movie", default=None, help="Brain Treebank movie id")
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    parser.add_argument("--atlas_path", default=None, help="Destrieux atlas .nii.gz")
    parser.add_argument(
        "--region_source",
        choices=["auto", "metadata", "atlas"],
        default="auto",
        help="auto uses metadata labels when present, then atlas coordinates.",
    )
    parser.add_argument(
        "--significance",
        default=None,
        help="all_subject_sig.csv or directory containing it; adds significant counts.",
    )
    parser.add_argument("--out_csv", default=None, help="Write per-electrode rows")
    parser.add_argument("--summary_csv", default=None, help="Write region-count rows")
    parser.add_argument(
        "--plot_path",
        default="electrode_region_distribution.png",
        help="Write stacked per-subject region distribution plot",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--brain_plot_path",
        default="electrode_brain_locations_3d.png",
        help="Write 3D one-figure subplot view of electrode coordinates per subject",
    )
    parser.add_argument(
        "--brain_2d_plot_path",
        default="electrode_brain_locations_2d.png",
        help="Write 2D lateral brain-image subplot view of electrode coordinates per subject",
    )
    parser.add_argument(
        "--nilearn_plot_path",
        default="electrode_brain_locations_nilearn.png",
        help="Write nilearn glass-brain projection subplot view per subject",
    )
    parser.add_argument(
        "--no_brain_plot",
        action="store_true",
        help="Skip brain-location plot generation",
    )
    parser.add_argument(
        "--max_brain_plot_subjects",
        type=int,
        default=None,
        help="Maximum number of subject subplots in the brain-location figure",
    )
    parser.add_argument(
        "--max_plot_regions",
        type=int,
        default=20,
        help="Maximum named regions in plot; remaining regions are grouped as Other",
    )
    parser.add_argument("--top", type=int, default=None, help="Limit printed rows")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    defaults = _load_config_defaults(args.config)
    dataset = args.dataset or defaults.get("dataset", "podcast")
    data_root = args.data_root or defaults.get("data_root", "data/")
    movie = args.movie
    subject_ids = args.subjects

    if subject_ids is None:
        if dataset == "brain_treebank":
            subject_ids = _discover_btb_subjects(data_root, movie)
        else:
            subject_ids = _discover_podcast_subjects(data_root)
    subject_ids = [int(s) for s in subject_ids]

    if dataset == "brain_treebank":
        records = _btb_records(subject_ids, data_root, movie)
    else:
        records = _podcast_records(subject_ids, data_root)

    needs_atlas = args.region_source == "atlas" or (
        args.region_source == "auto" and any(record.region == "Unknown" for record in records)
    )
    if needs_atlas:
        _apply_atlas_labels(
            records,
            args.atlas_path,
            only_missing=args.region_source == "auto",
        )

    sig_set = _load_significance(args.significance)
    if sig_set is not None:
        for record in records:
            record.significant = (record.subject, record.electrode) in sig_set

    detail_df = pd.DataFrame([asdict(record) for record in records])
    summary_df = summarize_records(records)

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        detail_df.to_csv(args.out_csv, index=False)
        print(f"Wrote per-electrode rows: {args.out_csv}")
    if args.summary_csv:
        Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.summary_csv, index=False)
        print(f"Wrote summary rows: {args.summary_csv}")
    if not args.no_plot:
        plot_distribution(summary_df, args.plot_path, max_regions=args.max_plot_regions)
        print(f"Wrote plot: {args.plot_path}")
    if not args.no_brain_plot:
        plot_brain_locations(
            detail_df,
            args.brain_plot_path,
            max_subjects=args.max_brain_plot_subjects,
        )
        print(f"Wrote 3D brain-location plot: {args.brain_plot_path}")
        plot_brain_locations_2d(
            detail_df,
            args.brain_2d_plot_path,
            image_dir=os.path.join(data_root, "localization"),
            max_subjects=args.max_brain_plot_subjects,
        )
        print(f"Wrote 2D brain-location plot: {args.brain_2d_plot_path}")
        plot_brain_locations_nilearn(
            detail_df,
            args.nilearn_plot_path,
            max_subjects=args.max_brain_plot_subjects,
        )
        print(f"Wrote nilearn brain-location plot: {args.nilearn_plot_path}")

    print(f"Dataset: {dataset}")
    if movie:
        print(f"Movie: {movie}")
    print(f"Subjects: {subject_ids}")
    print(f"Electrodes: {len(records)}")
    if detail_df.empty:
        return

    print()
    display = summary_df.sort_values(
        ["subject", "n_electrodes", "region"], ascending=[True, False, True]
    )
    if args.top is not None:
        display = display.groupby("subject", group_keys=False).head(args.top)
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(display.to_string(index=False))


if __name__ == "__main__":
    main()
