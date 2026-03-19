"""Backfill correlation plots for existing run directories.

Usage:
    python scripts/generate_plots.py --run-dir results/gpt2xl_layer24_offset0
    python scripts/generate_plots.py --run-dir results/... --subjects 1 3 5
    python scripts/generate_plots.py --run-dir results/... --force
    python scripts/generate_plots.py --run-dir results/... --threshold 0.90
    python scripts/generate_plots.py --run-dir results/... --svd-modes cross_subject per_region
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
    plot_svd_truncation_results,
    plot_svd_corrs,
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


def _load_truncation_results(npz_path):
    """Load truncation_results.npz and return (results_dict, best_k) or (None, None)."""
    if not os.path.exists(npz_path):
        return None, None
    data = np.load(npz_path)
    k_values = data["k_values"]
    best_k = int(data["best_k"])
    truncated_corrs = np.stack([data[f"trunc_corrs_k{k}"] for k in k_values])
    truncated_corrs_per_fold = [data[f"trunc_corrs_per_fold_k{k}"] for k in k_values]
    results = {
        "k_values": k_values,
        "full_corrs": data["full_corrs"],
        "full_corrs_per_fold": data["full_corrs_per_fold"],
        "sig_mask": data["sig_mask"],
        "truncated_corrs": truncated_corrs,
        "truncated_corrs_per_fold": truncated_corrs_per_fold,
        "best_k": best_k,
    }
    return results, best_k


def _detect_svd_modes(run_dir):
    """Return list of SVD mode strings whose output directories exist under run_dir."""
    modes = []
    if os.path.isdir(os.path.join(run_dir, "svd")):
        modes.append("cross_subject")
    if os.path.isdir(os.path.join(run_dir, "per_subject_svd")):
        modes.append("per_subject")
    if os.path.isdir(os.path.join(run_dir, "per_region_svd")):
        modes.append("per_region")
    return modes


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


def generate_cross_subject_svd_plots(run_dir, subjects=None, threshold=0.95, force=False):
    svd_dir = os.path.join(run_dir, "svd")
    npz_path = os.path.join(svd_dir, "truncation_results.npz")
    results, best_k = _load_truncation_results(npz_path)
    if results is None:
        print(f"  cross_subject SVD: no truncation_results.npz at {npz_path}, skipping")
        return

    dirs = _subject_dirs(run_dir, subjects)
    if not dirs:
        print(f"  cross_subject SVD: no subject encoding results found, skipping")
        return
    lags = load_encoding_results(os.path.join(dirs[0][1], "encoding_results.joblib"))["lags"]

    # Truncation plot
    trunc_plot_path = os.path.join(svd_dir, "truncation_plot.png")
    if os.path.exists(trunc_plot_path) and not force:
        print(f"  cross_subject SVD: skip truncation_plot.png")
    else:
        plot_svd_truncation_results(results, threshold=threshold, save_path=trunc_plot_path)
        print(f"  cross_subject SVD: wrote {trunc_plot_path}")

    # Per-subject plots
    for sub_dir in sorted(glob.glob(os.path.join(svd_dir, "subjects", "sub_*"))):
        sub_id = int(os.path.basename(sub_dir).split("_")[1])
        if subjects is not None and sub_id not in set(subjects):
            continue

        corrs_npy = os.path.join(sub_dir, f"corrs_k{best_k}.npy")
        if not os.path.exists(corrs_npy):
            print(f"  cross_subject SVD sub_{sub_id}: no corrs_k{best_k}.npy, skipping")
            continue
        corrs_trunc = np.load(corrs_npy)

        main_sub_enc = load_encoding_results(
            os.path.join(run_dir, "subjects", f"sub_{sub_id}", "encoding_results.joblib")
        )
        full_corrs = main_sub_enc["corrs"]
        elec_names = main_sub_enc["elec_names"]

        corrs_plot_path = os.path.join(sub_dir, f"corrs_plot_k{best_k}.png")
        if os.path.exists(corrs_plot_path) and not force:
            print(f"  cross_subject SVD sub_{sub_id}: skip corrs_plot_k{best_k}.png")
        else:
            plot_svd_corrs(
                corrs_trunc, full_corrs, lags,
                title=f"Subject {sub_id}",
                save_path=corrs_plot_path,
                best_k=best_k,
            )
            print(f"  cross_subject SVD sub_{sub_id}: wrote {corrs_plot_path}")

        sig_corrs_path = os.path.join(sub_dir, f"sig_corrs_plot_k{best_k}.png")
        if os.path.exists(sig_corrs_path) and not force:
            print(f"  cross_subject SVD sub_{sub_id}: skip sig_corrs_plot_k{best_k}.png")
        else:
            sig_mask = _load_sig_mask(run_dir, elec_names, lags)
            if sig_mask is not None:
                plot_corrs_by_significance(
                    corrs_trunc, sig_mask, lags, save_path=sig_corrs_path
                )
                print(f"  cross_subject SVD sub_{sub_id}: wrote {sig_corrs_path}")
            else:
                print(
                    f"  cross_subject SVD sub_{sub_id}: skip sig plot "
                    "(no significance_detailed.csv)"
                )

    # Cross-subject aggregate
    data = np.load(npz_path)
    trunc_per_fold = data[f"trunc_corrs_per_fold_k{best_k}"]
    sig_mask_global = results["sig_mask"]

    cross_corrs_path = os.path.join(svd_dir, f"cross_subject_corrs_plot_k{best_k}.png")
    if os.path.exists(cross_corrs_path) and not force:
        print(f"  cross_subject SVD: skip cross_subject_corrs_plot_k{best_k}.png")
    else:
        corrs_full, _ = load_all_subject_correlations(run_dir, subjects)
        plot_svd_corrs(
            trunc_per_fold, corrs_full, lags,
            title="Cross-subject",
            save_path=cross_corrs_path,
            best_k=best_k,
        )
        print(f"  cross_subject SVD: wrote {cross_corrs_path}")

    cross_sig_path = os.path.join(svd_dir, f"cross_subject_sig_corrs_plot_k{best_k}.png")
    if os.path.exists(cross_sig_path) and not force:
        print(f"  cross_subject SVD: skip cross_subject_sig_corrs_plot_k{best_k}.png")
    else:
        plot_corrs_by_significance(
            trunc_per_fold, sig_mask_global, lags, save_path=cross_sig_path
        )
        print(f"  cross_subject SVD: wrote {cross_sig_path}")


def generate_per_subject_svd_plots(run_dir, subjects=None, threshold=0.95, force=False):
    ps_svd_dir = os.path.join(run_dir, "per_subject_svd")
    sub_dirs = sorted(glob.glob(os.path.join(ps_svd_dir, "subjects", "sub_*")))
    if not sub_dirs:
        print(f"  per_subject SVD: no subject dirs found under {ps_svd_dir}/subjects/")
        return

    for sub_dir in sub_dirs:
        sub_id = int(os.path.basename(sub_dir).split("_")[1])
        if subjects is not None and sub_id not in set(subjects):
            continue

        npz_path = os.path.join(sub_dir, "truncation_results.npz")
        results, best_k = _load_truncation_results(npz_path)
        if results is None:
            print(f"  per_subject SVD sub_{sub_id}: no truncation_results.npz, skipping")
            continue

        enc_path = os.path.join(run_dir, "subjects", f"sub_{sub_id}", "encoding_results.joblib")
        if not os.path.exists(enc_path):
            print(f"  per_subject SVD sub_{sub_id}: no encoding_results.joblib, skipping")
            continue
        lags = load_encoding_results(enc_path)["lags"]

        trunc_plot_path = os.path.join(sub_dir, "truncation_plot.png")
        if os.path.exists(trunc_plot_path) and not force:
            print(f"  per_subject SVD sub_{sub_id}: skip truncation_plot.png")
        else:
            plot_svd_truncation_results(results, threshold=threshold, save_path=trunc_plot_path)
            print(f"  per_subject SVD sub_{sub_id}: wrote {trunc_plot_path}")

        data = np.load(npz_path)
        corrs_trunc = data[f"trunc_corrs_per_fold_k{best_k}"]
        full_corrs = results["full_corrs_per_fold"]
        sig_mask = results["sig_mask"]

        corrs_plot_path = os.path.join(sub_dir, f"corrs_plot_k{best_k}.png")
        if os.path.exists(corrs_plot_path) and not force:
            print(f"  per_subject SVD sub_{sub_id}: skip corrs_plot_k{best_k}.png")
        else:
            plot_svd_corrs(
                corrs_trunc, full_corrs, lags,
                title=f"Subject {sub_id}",
                save_path=corrs_plot_path,
                best_k=best_k,
            )
            print(f"  per_subject SVD sub_{sub_id}: wrote {corrs_plot_path}")

        sig_corrs_path = os.path.join(sub_dir, f"sig_corrs_plot_k{best_k}.png")
        if os.path.exists(sig_corrs_path) and not force:
            print(f"  per_subject SVD sub_{sub_id}: skip sig_corrs_plot_k{best_k}.png")
        else:
            plot_corrs_by_significance(
                corrs_trunc, sig_mask, lags, save_path=sig_corrs_path
            )
            print(f"  per_subject SVD sub_{sub_id}: wrote {sig_corrs_path}")


def generate_per_region_svd_plots(run_dir, subjects=None, threshold=0.95, force=False):
    pr_svd_dir = os.path.join(run_dir, "per_region_svd")
    regions_dir = os.path.join(pr_svd_dir, "regions")
    if not os.path.isdir(regions_dir):
        print(f"  per_region SVD: no regions dir found at {regions_dir}")
        return

    dirs = _subject_dirs(run_dir, subjects)
    if not dirs:
        print(f"  per_region SVD: no subject encoding results found for lags, skipping")
        return
    lags = load_encoding_results(os.path.join(dirs[0][1], "encoding_results.joblib"))["lags"]

    for region_dir in sorted(glob.glob(os.path.join(regions_dir, "*"))):
        if not os.path.isdir(region_dir):
            continue
        region_name = os.path.basename(region_dir)

        npz_path = os.path.join(region_dir, "truncation_results.npz")
        results, best_k = _load_truncation_results(npz_path)
        if results is None:
            print(f"  per_region SVD {region_name}: no truncation_results.npz, skipping")
            continue

        data = np.load(npz_path)
        trunc_per_fold = data[f"trunc_corrs_per_fold_k{best_k}"]
        full_corrs_per_fold = results["full_corrs_per_fold"]
        sig_mask = results["sig_mask"]

        trunc_plot_path = os.path.join(region_dir, "truncation_plot.png")
        if os.path.exists(trunc_plot_path) and not force:
            print(f"  per_region SVD {region_name}: skip truncation_plot.png")
        else:
            plot_svd_truncation_results(results, threshold=threshold, save_path=trunc_plot_path)
            print(f"  per_region SVD {region_name}: wrote {trunc_plot_path}")

        region_corrs_path = os.path.join(region_dir, f"region_corrs_plot_k{best_k}.png")
        if os.path.exists(region_corrs_path) and not force:
            print(f"  per_region SVD {region_name}: skip region_corrs_plot_k{best_k}.png")
        else:
            plot_svd_corrs(
                trunc_per_fold, full_corrs_per_fold, lags,
                title=f"Region: {region_name}",
                save_path=region_corrs_path,
                best_k=best_k,
            )
            print(f"  per_region SVD {region_name}: wrote {region_corrs_path}")

        region_sig_path = os.path.join(region_dir, f"region_sig_corrs_plot_k{best_k}.png")
        if os.path.exists(region_sig_path) and not force:
            print(f"  per_region SVD {region_name}: skip region_sig_corrs_plot_k{best_k}.png")
        else:
            plot_corrs_by_significance(
                trunc_per_fold, sig_mask, lags, save_path=region_sig_path
            )
            print(f"  per_region SVD {region_name}: wrote {region_sig_path}")

        # Per-subject plots — iterate in sorted order to track electrode offset
        offset = 0
        for sub_dir in sorted(glob.glob(os.path.join(region_dir, "subjects", "sub_*"))):
            sub_id = int(os.path.basename(sub_dir).split("_")[1])

            corrs_npy = os.path.join(sub_dir, f"corrs_k{best_k}.npy")
            if not os.path.exists(corrs_npy):
                print(
                    f"  per_region SVD {region_name} sub_{sub_id}: "
                    f"no corrs_k{best_k}.npy, stopping iteration"
                )
                break
            corrs_trunc = np.load(corrs_npy)
            n_sub_elecs = corrs_trunc.shape[1]

            if subjects is None or sub_id in set(subjects):
                full_corrs_sub = full_corrs_per_fold[:, offset:offset + n_sub_elecs, :]
                sig_mask_sub = sig_mask[offset:offset + n_sub_elecs]

                corrs_plot_path = os.path.join(sub_dir, f"corrs_plot_k{best_k}.png")
                if os.path.exists(corrs_plot_path) and not force:
                    print(
                        f"  per_region SVD {region_name} sub_{sub_id}: "
                        f"skip corrs_plot_k{best_k}.png"
                    )
                else:
                    plot_svd_corrs(
                        corrs_trunc, full_corrs_sub, lags,
                        title=f"Region {region_name} - Subject {sub_id}",
                        save_path=corrs_plot_path,
                        best_k=best_k,
                    )
                    print(
                        f"  per_region SVD {region_name} sub_{sub_id}: wrote {corrs_plot_path}"
                    )

                sig_corrs_path = os.path.join(sub_dir, f"sig_corrs_plot_k{best_k}.png")
                if os.path.exists(sig_corrs_path) and not force:
                    print(
                        f"  per_region SVD {region_name} sub_{sub_id}: "
                        f"skip sig_corrs_plot_k{best_k}.png"
                    )
                else:
                    plot_corrs_by_significance(
                        corrs_trunc, sig_mask_sub, lags, save_path=sig_corrs_path
                    )
                    print(
                        f"  per_region SVD {region_name} sub_{sub_id}: wrote {sig_corrs_path}"
                    )

            offset += n_sub_elecs


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
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="SVD performance retention threshold for truncation plots (default: 0.95).",
    )
    parser.add_argument(
        "--svd-modes",
        nargs="+",
        choices=["cross_subject", "per_subject", "per_region"],
        default=None,
        help=(
            "SVD modes to backfill (default: auto-detect from run dir). "
            "Example: --svd-modes cross_subject per_region"
        ),
    )
    args = parser.parse_args()

    print("Per-subject plots:")
    generate_per_subject_plots(args.run_dir, subjects=args.subjects, force=args.force)

    print("Cross-subject plot:")
    generate_cross_subject_plot(args.run_dir, subjects=args.subjects, force=args.force)

    modes = args.svd_modes or _detect_svd_modes(args.run_dir)
    if modes:
        print(f"SVD modes detected: {modes}")
        svd_kwargs = dict(subjects=args.subjects, threshold=args.threshold, force=args.force)
        if "cross_subject" in modes:
            print("Cross-subject SVD plots:")
            generate_cross_subject_svd_plots(args.run_dir, **svd_kwargs)
        if "per_subject" in modes:
            print("Per-subject SVD plots:")
            generate_per_subject_svd_plots(args.run_dir, **svd_kwargs)
        if "per_region" in modes:
            print("Per-region SVD plots:")
            generate_per_region_svd_plots(args.run_dir, **svd_kwargs)


if __name__ == "__main__":
    main()
