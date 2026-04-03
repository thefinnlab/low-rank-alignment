"""
print_atlas_regions.py — Print all Destrieux atlas regions present in the data.

For each subject, loads the BIDS file header (no waveform data) to extract
electrode MNI coordinates, looks them up in the Destrieux 2009 atlas, and
prints a sorted summary of region -> electrode count across all subjects.

Optionally cross-references with significance results to show how many
electrodes per region have at least one significant lag.

Usage:
    python scripts/print_atlas_regions.py --data_root data/
    python scripts/print_atlas_regions.py --data_root data/ --subjects 1 2 3
    python scripts/print_atlas_regions.py --data_root data/ --output_dir results/gpt2xl_layer24_offset0
    python scripts/print_atlas_regions.py --data_root data/ --atlas_path /path/to/destrieux.nii.gz
"""

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd

from src.utils.atlas import DESTRIEUX_2009_LABELS, _lookup_atlas_labels


def main():
    parser = argparse.ArgumentParser(
        description="Print atlas regions present in the data"
    )
    parser.add_argument(
        "--data_root", required=True, help="Root path of the BIDS dataset"
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=None,
        help="Subject IDs to include (default: 1-9)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Results dir containing all_subject_sig.csv; enables significance column",
    )
    parser.add_argument(
        "--atlas_path",
        default=None,
        help="Path to Destrieux 2009 .nii.gz; omit to download via nilearn",
    )
    args = parser.parse_args()

    import mne
    from mne_bids import BIDSPath
    from nilearn import datasets, image as nli_image

    subject_ids = args.subjects or list(range(1, 10))

    # -- Load significance set if output_dir provided --
    sig_set: set[tuple[int, str]] | None = None
    if args.output_dir:
        sig_csv = os.path.join(args.output_dir, "all_subject_sig.csv")
        if os.path.exists(sig_csv):
            sig_df = pd.read_csv(sig_csv)
            sig_set = set(zip(sig_df["subject"], sig_df["elec"]))
            print(
                f"Loaded significance: {len(sig_set)} significant electrodes from {sig_csv}"
            )
        else:
            print(
                f"Warning: all_subject_sig.csv not found at {sig_csv}, skipping significance column"
            )

    # -- Load atlas --
    if args.atlas_path:
        atlas_path = args.atlas_path
    else:
        print("Fetching Destrieux 2009 atlas via nilearn (cached after first use)...")
        atlas_path = datasets.fetch_atlas_destrieux_2009()["maps"]

    img = nli_image.load_img(atlas_path)
    atlas_image = img.get_fdata().astype(int)
    affine = img.affine

    counts: Counter = Counter()
    sig_counts: Counter = Counter()

    for sub_id in subject_ids:
        file_path = BIDSPath(
            root=os.path.join(args.data_root, "derivatives/ecogprep"),
            subject=f"{sub_id:02}",
            task="podcast",
            datatype="ieeg",
            description="highgamma",
            suffix="ieeg",
            extension=".fif",
        )
        if not os.path.exists(file_path):
            print(f"  Subject {sub_id}: file not found, skipping")
            continue

        raw = mne.io.read_raw_fif(file_path, preload=False, verbose=False)
        coords = np.array([ch["loc"][:3] * 1000 for ch in raw.info["chs"]])
        labels = _lookup_atlas_labels(
            coords, atlas_image, affine, DESTRIEUX_2009_LABELS
        )

        for elec_name, label in zip(raw.ch_names, labels):
            counts[label] += 1
            if sig_set is not None and (sub_id, elec_name) in sig_set:
                sig_counts[label] += 1

        print(f"  Subject {sub_id}: {len(raw.ch_names)} electrodes")

    # -- Print results --
    print()
    if sig_set is not None:
        print(f"{'Region':<40} {'Electrodes':>10} {'Significant':>12}")
        print("-" * 64)
        for label, count in counts.most_common():
            sig = sig_counts.get(label, 0)
            print(f"{label:<40} {count:>10} {sig:>12}")
        print("-" * 64)
        print(
            f"{'Total':<40} {sum(counts.values()):>10} {sum(sig_counts.values()):>12}"
        )
    else:
        print(f"{'Region':<40} {'Electrodes':>10}")
        print("-" * 52)
        for label, count in counts.most_common():
            print(f"{label:<40} {count:>10}")
        print("-" * 52)
        print(f"{'Total':<40} {sum(counts.values()):>10}")


if __name__ == "__main__":
    main()
