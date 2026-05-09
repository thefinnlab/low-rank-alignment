import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import os
    import re

    import matplotlib.pyplot as plt
    import scipy.linalg as la
    import numpy as np
    from src.utils.plot_helpers import (
        apply_publication_style,
        make_viridis_cmap,
        plot_cross_k_panel,
        plot_lag_panel,
    )
    from src.utils import load_precomputed_epochs

    return (
        apply_publication_style,
        make_viridis_cmap,
        mo,
        np,
        os,
        plot_cross_k_panel,
        plot_lag_panel,
        plt,
        re,
    )


@app.cell
def _(np, os, re):
    def load_fold_data(result_dir, item_prefix, num_folds):
        k_dirs = np.array([dir for dir in os.listdir(result_dir) if re.match("k(\d+)", dir)])
        ks = np.array([int(re.match("k(\d+)", dir).group(1)) for dir in k_dirs])
        sorted_order = np.argsort(ks)
        ks = ks[sorted_order]
        k_dirs = k_dirs[sorted_order]
        corr_results = np.zeros((len(ks), num_folds))
        per_target_results = None
        for i, k_dir in enumerate(k_dirs):
            curr_dir = os.path.join(result_dir, k_dir)
            for fold_dir in os.listdir(curr_dir):
                fold = re.match("fold_(\d+)", fold_dir).group(1)
                metrics = np.load(os.path.join(curr_dir, fold_dir, "metrics.npz"))
                curr_corr = metrics[f"{item_prefix}_corr"]
                curr_per_target_corr = metrics[f"{item_prefix}_corr_per_target"]
                if per_target_results is None:
                    per_target_results = np.zeros((len(ks), num_folds, *curr_per_target_corr.shape))
                per_target_results[i, int(fold)] = curr_per_target_corr
                corr_results[i, int(fold)] = curr_corr
        return {"k": ks, "corr": corr_results, "per_target_corr": per_target_results}

    def load_As(result_dir):
        fold_dirs = np.array([dir for dir in os.listdir(result_dir) if re.match("fold_(\d+)", dir)])
        folds = np.array([int(re.match("fold_(\d+)", dir).group(1)) for dir in fold_dirs])
        sorted_order = np.argsort(folds)
        folds = folds[sorted_order]
        fold_dirs = fold_dirs[sorted_order]
        As = None
        for i, fold_dir in enumerate(fold_dirs):
            A_path = os.path.join(result_dir, fold_dir, "A.npy")
            A = np.load(A_path)
            if As is None:
                As = np.zeros((len(folds), *A.shape))
            As[i] = A
        return {"folds": folds, "As": As}

    return load_As, load_fold_data


@app.cell
def _(
    apply_publication_style,
    load_fold_data,
    make_viridis_cmap,
    np,
    plot_cross_k_panel,
    plot_lag_panel,
    plt,
):
    # Colors for the two conditions
    _BASELINE_COLOR = "#d62728"   # red  — within-subject baseline
    _TRANSFER_COLOR = "#2166ac"   # blue — cross-subject transfer

    for _sub in range(1, 10):
        _base = load_fold_data(
            f"results/gpt2xl_layer24_offset0/loso_rrr/held_out_sub_{_sub}/baseline",
            "nr", 5,
        )
        _xfer = load_fold_data(
            f"results/gpt2xl_layer24_offset0/loso_rrr/held_out_sub_{_sub}/transfer",
            "transfer", 5,
        )

        _ks = _base["k"]

        # per_target_corr is stored flat: (n_ks, n_folds, n_elecs * n_lags).
        # Reshape to (n_ks, n_folds, n_elecs, n_lags) using the epoch cache.
        # _epoch_cache = load_precomputed_epochs(_sub, "data/")
        _n_lags = 128 # _epoch_cache["n_lags"]
        _lags = np.linspace(-2.0, 2.0, _n_lags, endpoint=False)

        def _reshape(pt):
            return pt.reshape((*pt.shape[:2], -1, _n_lags))

        _base_pt = _reshape(_base["per_target_corr"])   # (n_ks, n_folds, n_elecs, n_lags)
        _xfer_pt = _reshape(_xfer["per_target_corr"])

        apply_publication_style()
        _CMAP = make_viridis_cmap()

        _fig, (_ax_a, _ax_b) = plt.subplots(
            1, 2, figsize=(10, 3.5),
            gridspec_kw={"wspace": 0.40},
        )
        # Get best performing k for baseline
        _mean_k = _base["corr"].mean(1)
        _dk = abs(_mean_k[1:] - np.max(_mean_k))
        _best_base_k = sorted(np.argwhere(_dk < 0.001)[0])[0] + 1
        _fig.suptitle(f"Held-out Subject {_sub} (n = {_base_pt.shape[2]}, k={_ks[_best_base_k]})", fontsize=11, fontweight="bold")
        _fig.subplots_adjust(left=0.08, right=0.92, bottom=0.15, top=0.85)

        # ── Panel (a): Cross-k ─────────────────────────────────────────────────
        plot_cross_k_panel(
            _ax_a, _ks,
            mean_corrs=_base["corr"].mean(1),
            fold_corrs=_base["corr"],
            label="Baseline (within-subject)",
            color=_BASELINE_COLOR,
        )
        plot_cross_k_panel(
            _ax_a, _ks,
            mean_corrs=_xfer["corr"].mean(1),
            fold_corrs=_xfer["corr"],
            label="Transfer (cross-subject)",
            color=_TRANSFER_COLOR,
        )
        _ax_a.set_ylim(-0.01, 0.06)
        _ax_a.set_title("Mean Correlation vs. Subspace Rank", fontsize=9, pad=8)
        _ax_a.text(-0.12, 1.05, "a", transform=_ax_a.transAxes,
                   fontsize=12, fontweight="bold", va="bottom")

        # ── Panel (b): Lag-specific (transfer vs baseline mean) ────────────────
        # Baseline reference: mean across k, folds, and electrodes → (n_lags,).

        _base_lag_mean = _base_pt[_best_base_k].mean((0, 1))
        plot_lag_panel(
            _ax_b, _lags,
            data=_xfer_pt,
            ks=_ks,
            base_corrs_mean=_base_lag_mean,
            sig_mask=None,
            saturation_k=20,
            cmap=_CMAP,
            fig=_fig,
            y_min=-0.01,
            y_max=0.1,
            base_label="Baseline (within-subject)",
        )
        _ax_b.set_ylim(-0.01, 0.1)
        _ax_b.text(-0.12, 1.05, "b", transform=_ax_b.transAxes,
                   fontsize=12, fontweight="bold", va="bottom")

        plt.show()
    return


@app.cell
def _(
    apply_publication_style,
    load_fold_data,
    make_viridis_cmap,
    np,
    plot_cross_k_panel,
    plot_lag_panel,
    plt,
):
    # Colors for the two conditions
    _BASELINE_COLOR = "#d62728"   # red  — within-subject baseline
    _TRANSFER_COLOR = "#2166ac"   # blue — cross-subject transfer

    _bases = []
    _xfers = []
    for _sub in range(1, 10):
        _base = load_fold_data(
            f"results/gpt2xl_layer24_offset0/loso_rrr/held_out_sub_{_sub}/baseline",
            "nr", 5,
        )
        _xfer = load_fold_data(
            f"results/gpt2xl_layer24_offset0/loso_rrr/held_out_sub_{_sub}/transfer",
            "transfer", 5,
        )
        _bases.append(_base)
        _xfers.append(_xfer)

        _ks = _base["k"]

    # per_target_corr is stored flat: (n_ks, n_folds, n_elecs * n_lags).
    # Reshape to (n_ks, n_folds, n_elecs, n_lags) using the epoch cache.
    # _epoch_cache = load_precomputed_epochs(_sub, "data/")
    _n_lags = 128 # _epoch_cache["n_lags"]
    _lags = np.linspace(-2.0, 2.0, _n_lags, endpoint=False)

    def _reshape(pt):
        return pt.reshape((*pt.shape[:2], -1, _n_lags))

    _total_elecs = np.sum([_x["per_target_corr"].shape[2] for _x in _bases]) // _n_lags
    _base_pt = np.stack([_reshape(_x["per_target_corr"]).mean(2) for _x in _bases])  # (n_subs, n_ks, n_folds, n_lags)
    _xfer_pt = np.stack([_reshape(_x["per_target_corr"]).mean(2) for _x in _xfers])

    apply_publication_style()
    _CMAP = make_viridis_cmap()

    _fig, (_ax_a, _ax_b) = plt.subplots(
        1, 2, figsize=(10, 3.5),
        gridspec_kw={"wspace": 0.40},
    )
    # Get best performing k for baseline
    _mean_k = _base_pt.mean((0, 2))
    _dk = abs(_mean_k[1:] - np.max(_mean_k))
    _best_base_k = sorted(np.argwhere(_dk < 0.001)[0])[0] + 1
    _fig.suptitle(f"Leave-One-Subject-Out vs Within Subject (n = {_total_elecs}, k={_ks[_best_base_k]})", fontsize=11, fontweight="bold")
    _fig.subplots_adjust(left=0.08, right=0.92, bottom=0.15, top=0.85)

    # ── Panel (a): Cross-k ─────────────────────────────────────────────────
    plot_cross_k_panel(
        _ax_a, _ks,
        mean_corrs=_base_pt.mean((0, 2, 3)),
        fold_corrs=_base_pt.mean((2, 3)).transpose((1, 0)),
        label="Baseline (within-subject)",
        color=_BASELINE_COLOR,
    )
    plot_cross_k_panel(
        _ax_a, _ks,
        mean_corrs=_xfer_pt.mean((0, 2, 3)),
        fold_corrs=_xfer_pt.mean((2, 3)).transpose((1, 0)),
        label="Transfer (cross-subject)",
        color=_TRANSFER_COLOR,
    )
    _ax_a.set_ylim(-0.01, 0.06)
    _ax_a.set_title("Mean Correlation vs. Subspace Rank", fontsize=9, pad=8)
    _ax_a.text(-0.12, 1.05, "a", transform=_ax_a.transAxes,
               fontsize=12, fontweight="bold", va="bottom")

    # ── Panel (b): Lag-specific (transfer vs baseline mean) ────────────────
    # Baseline reference: mean across k, folds, and electrodes → (n_lags,).

    _base_lag_mean = _base_pt[:, _best_base_k].mean((0, 1))
    plot_lag_panel(
        _ax_b, _lags,
        data=_xfer_pt.mean((0, 2)),
        ks=_ks,
        base_corrs_mean=_base_lag_mean,
        sig_mask=None,
        saturation_k=20,
        cmap=_CMAP,
        fig=_fig,
        # y_min=-0.01,
        # y_max=0.1,
        base_label="Baseline (within-subject)",
    )
    _ax_b.set_ylim(-0.01, 0.1)
    _ax_b.text(-0.12, 1.05, "b", transform=_ax_b.transAxes,
               fontsize=12, fontweight="bold", va="bottom")
    plt.savefig("subject_splits.png")
    plt.show()
    return


@app.cell
def _(
    apply_publication_style,
    load_fold_data,
    make_viridis_cmap,
    np,
    plot_cross_k_panel,
    plot_lag_panel,
    plt,
):
    # Colors for the two conditions
    _BASELINE_COLOR = "#d62728"   # red  — within-subject baseline
    _TRANSFER_COLOR = "#2166ac"   # blue — cross-subject transfer

    _bases = []
    _xfers = []
    for _sub in range(1, 10):
        _base = load_fold_data(
            f"results/gpt2xl_layer24_offset0/loso_rrr/held_out_sub_{_sub}/baseline",
            "nr", 5,
        )
        _xfer = load_fold_data(
            f"results/gpt2xl_layer24_offset0/loso_rrr/held_out_sub_{_sub}/transfer",
            "transfer", 5,
        )
        _bases.append(_base)
        _xfers.append(_xfer)

        _ks = _base["k"]

    # per_target_corr is stored flat: (n_ks, n_folds, n_elecs * n_lags).
    # Reshape to (n_ks, n_folds, n_elecs, n_lags) using the epoch cache.
    # _epoch_cache = load_precomputed_epochs(_sub, "data/")
    _n_lags = 128 # _epoch_cache["n_lags"]
    _lags = np.linspace(-2.0, 2.0, _n_lags, endpoint=False)

    def _reshape(pt):
        return pt.reshape((*pt.shape[:2], -1, _n_lags))

    _elecs_per_subject = [_x["per_target_corr"].shape[2] // _n_lags for _x in _bases]
    _total_elecs = np.sum(_elecs_per_subject)
    _base_pt = np.stack([_reshape(_x["per_target_corr"]).mean(2) for _x in _bases])  # (n_subs, n_ks, n_folds, n_lags)
    _xfer_pt = np.stack([_reshape(_x["per_target_corr"]).mean(2) for _x in _xfers])

    apply_publication_style()
    _CMAP = make_viridis_cmap()

    _fig, (_ax_a, _ax_b) = plt.subplots(
        1, 2, figsize=(10, 3.5),
        gridspec_kw={"wspace": 0.40},
    )
    # Get best performing k for baseline
    _mean_k = _base_pt.mean((0, 2))
    _dk = abs(_mean_k[1:] - np.max(_mean_k))
    _best_base_k = sorted(np.argwhere(_dk < 0.001)[0])[0] + 1
    _fig.suptitle(f"Leave-One-Subject-Out vs Within Subject (n = {_total_elecs}, k={_ks[_best_base_k]})", fontsize=11, fontweight="bold")
    _fig.subplots_adjust(left=0.08, right=0.92, bottom=0.15, top=0.85)

    # ── Panel (a): Cross-k ─────────────────────────────────────────────────
    plot_cross_k_panel(
        _ax_a, _ks,
        mean_corrs=np.average(_base_pt.mean((2, 3)), axis=0, weights=_elecs_per_subject),
        fold_corrs=_base_pt.mean((2, 3)).transpose((1, 0)),
        label="Baseline (within-subject)",
        color=_BASELINE_COLOR,
    )
    plot_cross_k_panel(
        _ax_a, _ks,
        mean_corrs=np.average(_xfer_pt.mean((2, 3)), axis=0, weights=_elecs_per_subject),
        fold_corrs=_xfer_pt.mean((2, 3)).transpose((1, 0)),
        label="Transfer (cross-subject)",
        color=_TRANSFER_COLOR,
    )
    _ax_a.set_ylim(-0.01, 0.06)
    _ax_a.set_title("Mean Correlation vs. Subspace Rank", fontsize=9, pad=8)
    _ax_a.text(-0.12, 1.05, "a", transform=_ax_a.transAxes,
               fontsize=12, fontweight="bold", va="bottom")

    # ── Panel (b): Lag-specific (transfer vs baseline mean) ────────────────
    # Baseline reference: mean across k, folds, and electrodes → (n_lags,).

    _base_lag_mean = np.average(_base_pt[:, _best_base_k].mean((1)), axis=0, weights=_elecs_per_subject)
    plot_lag_panel(
        _ax_b, _lags,
        data=np.average(_xfer_pt.mean((2)), axis=0, weights=_elecs_per_subject),
        ks=_ks,
        base_corrs_mean=_base_lag_mean,
        sig_mask=None,
        saturation_k=20,
        cmap=_CMAP,
        fig=_fig,
        # y_min=-0.01,
        # y_max=0.1,
        base_label="Baseline (within-subject)",
    )
    _ax_b.set_ylim(-0.01, 0.1)
    _ax_b.text(-0.12, 1.05, "b", transform=_ax_b.transAxes,
               fontsize=12, fontweight="bold", va="bottom")
    plt.savefig("subject_splits.png")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ### A Analysis
    """)
    return


@app.cell
def _(np, plt):
    def orthonormalize(A1, A2):
        # Step 1: Orthonormalize via QR decomposition
        Q1, _ = np.linalg.qr(A1)
        Q2, _ = np.linalg.qr(A2)
        return Q1, Q2

    def run_analysis(Q1, Q2):
        # 1. Principal Angles (via SVD of the overlap matrix)
        # Cosines are the singular values of Q1.T @ Q2
        C = Q1.T @ Q2
        singular_values = np.linalg.svd(C, compute_uv=False)
        # Clip for numerical stability
        cos_angles = np.clip(singular_values, 0, 1)
        angles_rad = np.arccos(cos_angles)

        # 2. Global Metrics
        grassmann_dist = np.sqrt(np.sum(angles_rad**2))
        # Subspace Overlap (fraction of variance of Q1 captured by Q2)
        overlap = np.trace(Q1.T @ Q2 @ Q2.T @ Q1) / Q1.shape[1]

        # 3. Axis-wise Presence
        # How much of each column in A1 (not Q1) is in the space spanned by Q2?
        # We use Q2 @ Q2.T as the projection operator onto Subspace 2
        presence = []
        for i in range(Q1.shape[1]):
            q_vec = Q1[:, i]
            projection = Q2 @ (Q2.T @ q_vec)
            presence.append(np.linalg.norm(projection))

        return {
            "angles": angles_rad,
            "cosines": cos_angles,
            "grassmann": grassmann_dist,
            "overlap": overlap,
            "presence": np.array(presence)
        }

    def plot_results(results, K):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Principal Angles
        ax1.plot(range(1, K + 1), np.degrees(results["angles"]), marker='o', color='teal')
        ax1.axhline(90, color='red', linestyle='--', alpha=0.5)
        ax1.axhline(45, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title("Principal Angles (Degrees)")
        ax1.set_xlabel("Rank Index")
        ax1.set_ylabel("Angle (°)")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 95)

        # Plot 2: Axis-wise Presence
        ax2.bar(range(1, K + 1), results["presence"], color='salmon', alpha=0.8)
        ax2.set_title("Presence of A1 Axes in Subspace 2")
        ax2.set_xlabel("Axis Index (A1)")
        ax2.set_ylabel("Norm of Projection")
        ax2.set_ylim(0, 1.1)
        ax2.axhline(1.0, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()
        return fig

    def display_metrics(results):
        print(
            f"""
            ### Global Similarity Summary
            - **Subspace Overlap (0-1):** {results['overlap']:.4f}
            - **Grassmann Distance:** {results['grassmann']:.4f}
            - **Mean Principal Angle:** {np.degrees(results['angles']).mean():.2f}°
            """
        )

    return display_metrics, orthonormalize, plot_results, run_analysis


@app.cell
def _(np):
    list(np.load("results/gpt2xl_layer24_offset0/loso_rrr/held_out_sub_1/transfer/k1/fold_0/metrics.npz").keys())
    return


@app.cell
def _(
    display_metrics,
    load_As,
    orthonormalize,
    plot_results,
    plt,
    run_analysis,
):
    transfer_A = load_As("results/gpt2xl_layer24_offset0/loso_rrr/held_out_sub_1/train_subjects/")["As"][:, :, :20]
    baseline_A = load_As("results/gpt2xl_layer24_offset0/loso_rrr/held_out_sub_1/baseline/")["As"][:, :, :20]

    for fold in range(5):
        print(f"Fold {fold}:")
        Q_transfer, Q_baseline = orthonormalize(transfer_A[fold], baseline_A[fold])
        analysis_results = run_analysis(Q_transfer, Q_baseline)
        display_metrics(analysis_results)
        fig = plot_results(analysis_results, K=Q_transfer.shape[1])
        plt.show()
    return (baseline_A,)


@app.cell
def _(baseline_A):
    print(baseline_A.shape)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
