"""Backfill correlation plots for existing run directories.

Usage:
    python scripts/generate_plots.py --run-dir results/gpt2xl_layer24_offset0
    python scripts/generate_plots.py --run-dir results/... --subjects 1 3 5
    python scripts/generate_plots.py --run-dir results/... --force
"""

import argparse
import glob
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.encoding import load_encoding_results, load_all_subject_correlations
from utils.plotting import (
    add_corrs,
    plot_corrs_vs_arbitrary,
    plot_corrs_by_significance,
)


def _subject_dirs(run_dir, subjects=None):
    """Return sorted list of (sub_id, sub_dir) tuples, optionally filtered by subjects."""
    pattern = os.path.join(run_dir, "subjects", "sub_*", "encoding_results.joblib")
    paths = sorted(glob.glob(pattern))
    if subjects is not None:
        subjects_set = set(subjects)
        paths = [
            p
            for p in paths
            if int(os.path.basename(os.path.dirname(p)).split("_")[1]) in subjects_set
        ]
    return [
        (int(os.path.basename(os.path.dirname(p)).split("_")[1]), os.path.dirname(p))
        for p in paths
    ]


def _load_sig_mask(run_dir, elec_names, lags):
    """Build a (n_electrodes, n_lags) boolean sig_mask from significance_detailed.csv.

    Rows are aligned to elec_names order; columns are aligned to lags order.
    Returns None if the CSV is not present.
    """
    csv_path = os.path.join(run_dir, "significance_detailed.csv")
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    n_elecs = len(elec_names)
    n_lags = len(lags)
    mask = np.zeros((n_elecs, n_lags), dtype=bool)

    elec_index = {name: i for i, name in enumerate(elec_names)}
    lag_index = {round(float(l), 8): j for j, l in enumerate(lags)}

    for _, row in df.iterrows():
        ei = elec_index.get(row["electrode"])
        li = lag_index.get(round(float(row["lag_sec"]), 8))
        if ei is not None and li is not None:
            mask[ei, li] = bool(row["sig"])

    return mask


def generate_per_subject_plots(run_dir, subjects=None, force=False):
    dirs = _subject_dirs(run_dir, subjects)
    if not dirs:
        print(f"No subject results found under {run_dir}/subjects/")
        return

    for sub_id, sub_dir in dirs:
        save_path = os.path.join(sub_dir, "corrs_plot.png")
        results = load_encoding_results(
            os.path.join(sub_dir, "encoding_results.joblib")
        )
        corrs = results["corrs"]
        lags = results["lags"]
        arb_corrs = results.get("arb_corrs")
        if os.path.exists(save_path) and not force:
            print(f"  sub_{sub_id}: skip (already exists, use --force to overwrite)")
        else:
            if arb_corrs is not None:
                plot_corrs_vs_arbitrary(corrs, arb_corrs, lags, save_path=save_path)
            else:
                fig, ax = plt.subplots(figsize=(8, 4))
                add_corrs(corrs, lags, label="Real", color="blue", ax=ax)
                ax.axhline(0, color="black", linewidth=0.5)
                ax.set_xlabel("Lag (s)")
                ax.set_ylabel("Correlation")
                ax.legend()
                fig.tight_layout()
                fig.savefig(save_path, dpi=150)
                plt.close(fig)

            print(f"  sub_{sub_id}: wrote {save_path}")

        # Significance split plot
        sig_save_path = os.path.join(sub_dir, "sig_corrs_plot.png")
        if os.path.exists(sig_save_path) and not force:
            print(f"  sub_{sub_id}: skip sig plot (already exists)")
        else:
            sig_mask = _load_sig_mask(run_dir, results["elec_names"], lags)
            if sig_mask is not None:
                plot_corrs_by_significance(
                    corrs, sig_mask, lags, save_path=sig_save_path
                )
                print(f"  sub_{sub_id}: wrote {sig_save_path}")
            else:
                print(f"  sub_{sub_id}: skip sig plot (no significance_detailed.csv)")


def generate_cross_subject_plot(run_dir, subjects=None, force=False):
    save_path = os.path.join(run_dir, "corrs_plot.png")
    if os.path.exists(save_path) and not force:
        print(f"Cross-subject: skip (already exists, use --force to overwrite)")
        return

    corrs_all, _ = load_all_subject_correlations(run_dir, subjects)

    # Get lags from first subject
    dirs = _subject_dirs(run_dir, subjects)
    first_results = load_encoding_results(
        os.path.join(dirs[0][1], "encoding_results.joblib")
    )
    lags = first_results["lags"]

    # Load per-subject arbitrary_corrs.npy if available
    arb_list = []
    for sub_id, sub_dir in dirs:
        arb_path = os.path.join(sub_dir, "arbitrary_corrs.npy")
        if os.path.exists(arb_path):
            arb_list.append(np.load(arb_path))
        else:
            arb_list.append(None)

    n_with_arb = sum(1 for a in arb_list if a is not None)

    if n_with_arb == len(dirs):
        arb_corrs_all = np.concatenate(arb_list, axis=1)
        plot_corrs_vs_arbitrary(corrs_all, arb_corrs_all, lags, save_path=save_path)
    else:
        if 0 < n_with_arb < len(dirs):
            warnings.warn(
                f"Only {n_with_arb}/{len(dirs)} subjects have arbitrary_corrs.npy; "
                "plotting real correlations only."
            )
        fig, ax = plt.subplots(figsize=(8, 4))
        add_corrs(corrs_all, lags, label="Real", color="blue", ax=ax)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Lag (s)")
        ax.set_ylabel("Correlation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    print(f"Cross-subject: wrote {save_path}")

    # Cross-subject significance split plot
    sig_save_path = os.path.join(run_dir, "sig_corrs_plot.png")
    if os.path.exists(sig_save_path) and not force:
        print("Cross-subject: skip sig plot (already exists)")
    else:
        # Collect all electrode names in the same order as corrs_all
        all_elec_names = []
        for _, sub_dir in dirs:
            r = load_encoding_results(os.path.join(sub_dir, "encoding_results.joblib"))
            all_elec_names.extend(r["elec_names"])
        sig_mask = _load_sig_mask(run_dir, all_elec_names, lags)
        if sig_mask is not None:
            plot_corrs_by_significance(
                corrs_all, sig_mask, lags, save_path=sig_save_path
            )
            print(f"Cross-subject: wrote {sig_save_path}")
        else:
            print("Cross-subject: skip sig plot (no significance_detailed.csv)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate correlation plots for a run directory."
    )
    parser.add_argument(
        "--run-dir", required=True, help="Path to run output directory."
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=None,
        help="Subject IDs to include (default: all found).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing plots."
    )
    args = parser.parse_args()

    print("Per-subject plots:")
    generate_per_subject_plots(args.run_dir, subjects=args.subjects, force=args.force)

    print("Cross-subject plot:")
    generate_cross_subject_plot(args.run_dir, subjects=args.subjects, force=args.force)


if __name__ == "__main__":
    main()
