import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(np):
    def sample_gaussian_ellipsoid(n_points=100, center=None, scales=(1.0, 1.0, 1.0), rotation=None, seed=None):
        """Sample points from a Gaussian ellipsoid in the dimension given by scales."""
        _rng = np.random.default_rng(seed)
        _scales = np.asarray(scales, dtype=float)
        if _scales.ndim != 1 or _scales.size < 2:
            raise ValueError("scales must contain at least two values")
        _n_dimensions = _scales.size
        _center = np.zeros(_n_dimensions) if center is None else np.asarray(center, dtype=float)
        if _center.shape != (_n_dimensions,):
            raise ValueError("center must have the same length as scales")
        if np.any(_scales < 0):
            raise ValueError("scales must be non-negative")
        _points = _rng.normal(size=(n_points, _n_dimensions)) * _scales
        if rotation is not None:
            _rotation = np.asarray(rotation, dtype=float)
            if _rotation.shape != (_n_dimensions, _n_dimensions):
                raise ValueError("rotation must be a square matrix matching the scale dimension")
            _points = _points @ _rotation.T
        return _points + _center


    return (sample_gaussian_ellipsoid,)


@app.cell
def _(mo, np, plt, sample_gaussian_ellipsoid):
    import io as _io


    def plot_gaussian_ellipsoid(
        *,
        scales=(1.35, 0.48, 0.20),
        angles=(-12, -24, 28),
        center=None,
        color="#3778A8",
        n_points=420,
        seed=12,
    ):
        """Render a 2D Gaussian ellipse or 3D Gaussian ellipsoid as an SVG.

        The length of ``scales`` sets the dimensionality. ``angles`` contains one
        counterclockwise angle in 2D, or z/y/x Euler angles in 3D.
        """
        _scales = np.asarray(scales, dtype=float)
        if _scales.ndim != 1 or _scales.size not in (2, 3):
            raise ValueError("scales must contain either two or three values")
        _n_dimensions = _scales.size
        _center = np.zeros(_n_dimensions) if center is None else np.asarray(center, dtype=float)
        if _center.shape != (_n_dimensions,):
            raise ValueError("center must have the same length as scales")

        _angles = np.asarray(angles, dtype=float).reshape(-1)
        _n_angles = _n_dimensions * (_n_dimensions - 1) // 2
        if _angles.size != _n_angles:
            raise ValueError(f"angles must contain {_n_angles} value(s) for {_n_dimensions}D")

        if _n_dimensions == 2:
            _angle = np.deg2rad(_angles[0])
            _rotation = np.array([
                [np.cos(_angle), -np.sin(_angle)],
                [np.sin(_angle), np.cos(_angle)],
            ])
        else:
            _angle_z, _angle_y, _angle_x = np.deg2rad(_angles)
            _rotation_z = np.array([
                [np.cos(_angle_z), -np.sin(_angle_z), 0.0],
                [np.sin(_angle_z), np.cos(_angle_z), 0.0],
                [0.0, 0.0, 1.0],
            ])
            _rotation_y = np.array([
                [np.cos(_angle_y), 0.0, np.sin(_angle_y)],
                [0.0, 1.0, 0.0],
                [-np.sin(_angle_y), 0.0, np.cos(_angle_y)],
            ])
            _rotation_x = np.array([
                [1.0, 0.0, 0.0],
                [0.0, np.cos(_angle_x), -np.sin(_angle_x)],
                [0.0, np.sin(_angle_x), np.cos(_angle_x)],
            ])
            _rotation = _rotation_z @ _rotation_y @ _rotation_x

        _points = sample_gaussian_ellipsoid(
            n_points=n_points,
            center=_center,
            scales=_scales,
            rotation=_rotation,
            seed=seed,
        )
        _fig = plt.figure(figsize=(4.1, 3.25), facecolor="white")
        if _n_dimensions == 2:
            _ax = _fig.add_subplot(111)
            _ax.set_facecolor("white")
            _ax.scatter(
                _points[:, 0], _points[:, 1], color=color, s=11, alpha=0.84,
                linewidths=0,
            )
            _axis_spread = np.sqrt(np.sum((_rotation * _scales[None, :]) ** 2, axis=1))
            _ax.set_xlim(_center[0] - 2.8 * _axis_spread[0], _center[0] + 2.8 * _axis_spread[0])
            _ax.set_ylim(_center[1] - 2.8 * _axis_spread[1], _center[1] + 2.8 * _axis_spread[1])
            _ax.set_aspect("equal", adjustable="box")
            _ax.set_axis_off()
        else:
            _ax = _fig.add_subplot(111, projection="3d")
            _ax.set_facecolor("white")
            _ax.scatter(
                _points[:, 0], _points[:, 1], _points[:, 2],
                color=color, s=11, alpha=0.84,
                linewidths=0, depthshade=True,
            )
            _axis_spread = np.sqrt(np.sum((_rotation * _scales[None, :]) ** 2, axis=1))
            for _set_limit, _midpoint, _spread in zip(
                (_ax.set_xlim, _ax.set_ylim, _ax.set_zlim), _center, _axis_spread
            ):
                _set_limit(_midpoint - 2.8 * _spread, _midpoint + 2.8 * _spread)
            _ax.set_box_aspect(_axis_spread)
            _ax.view_init(elev=23, azim=-55)
            _ax.set_axis_off()

        _svg = _io.StringIO()
        _fig.savefig(
            _svg, format="svg", bbox_inches="tight", facecolor="white",
            transparent=False,
        )
        plt.savefig(f"{color}.svg")
        plt.close(_fig)
        return mo.Html(_svg.getvalue())


    _first_ellipsoid = plot_gaussian_ellipsoid(
        scales=(1.35, 0.48, 0.20),
        angles=(-12, -24, 28),
        seed=12,
        color="#B8B8B8"
    )
    # _second_ellipsoid = plot_gaussian_ellipsoid(
    #     scales=(0.95, 0.62, 0.28),
    #     angles=(34, -18, 42),
    #     color="#B8B8B8",
    #     seed=24,
    # )
    # mo.hstack([_first_ellipsoid, _second_ellipsoid])

    _flat_ellipsoid = plot_gaussian_ellipsoid(
        scales=(0.5, 0.30),
        angles=(0),
        seed=12,
        color="#B8B8B8"
    )

    _flat_ellipsoid
    return


@app.cell
def rrr_artifact_imports():
    from pathlib import Path
    import json
    import sys
    import pandas as pd

    REPO_ROOT = Path("/home/handrail6120/Documents/projects/alignment-attribution")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from scripts.run_rrr import Artifact
    from scripts.run_language_decoding import Artifact as LanguageDecodingArtifact

    return Artifact, LanguageDecodingArtifact, Path, REPO_ROOT, json, pd


@app.cell(hide_code=True)
def best_rank_heading(
    find_cross_model_layers,
    find_loso_model_layers,
    find_rrr_models,
    mo,
):
    _model_options = sorted({
        model
        for _dataset in ("podcast", "cars-2", "fmri")
        for model in find_rrr_models(dataset=_dataset)
    } | {
        model
        for _dataset in ("podcast", "cars-2", "fmri")
        for model, _layer in find_loso_model_layers(dataset=_dataset)
    } | {
        model
        for _dataset in ("podcast", "cars-2", "fmri")
        for model, _layer in find_cross_model_layers(dataset=_dataset)
    })
    plot_model_filter = mo.ui.multiselect(
        options=_model_options,
        value=_model_options,
        label="Models included in RRR plots",
    )
    mo.vstack([
        mo.md("## Performance relative to each target’s best rank"),
        plot_model_filter,
    ])
    return (plot_model_filter,)


@app.cell(hide_code=True)
def best_rank_controls(mo, model_layer_options):
    best_rank_model_layer_selector = mo.ui.dropdown(
        options=model_layer_options,
        value=next(iter(model_layer_options)),
        label="Model layer",
    )
    best_rank_model_layer_selector
    return (best_rank_model_layer_selector,)


@app.cell(hide_code=True)
def best_rank_helpers(plt):
    def summarize_target_best_rank(performances):
        """Average folds first, then compare each target with its best rank for the supplied method."""
        target_means = performances.groupby(
            ["target", "rank"], as_index=False
        )["correlation"].mean()
        target_means["best_correlation"] = target_means.groupby("target")[
            "correlation"
        ].transform("max")
        target_means["difference"] = (
            target_means["correlation"] - target_means["best_correlation"]
        )
        summary = target_means.groupby("rank")["difference"].agg(
            mean="mean", sd="std", targets="count"
        ).reset_index()
        return target_means, summary


    def plot_target_best_rank(performances, mean_results, dataset_label, model, layer):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        for method, color, label in (("rrr", "#6ab2e2", "RRR"), ("pca", "#63bc51", "PCA")):
            method_results = mean_results.loc[mean_results["method"] == method].sort_values("rank")
            if method_results.empty:
                axes[0].text(0.02, 0.95, f"{label} results unavailable", transform=axes[0].transAxes)
                continue
            metric = "nr_orth_corr" if method == "rrr" and "nr_orth_corr" in method_results else "corr"
            axes[0].plot(method_results["rank"], method_results[metric], "o-", color=color, label=label)
        if "full_corr" in mean_results:
            full_correlations = mean_results["full_corr"].dropna()
            if not full_correlations.empty:
                axes[0].axhline(
                    float(full_correlations.iloc[0]),
                    color="black",
                    linewidth=1.2,
                    linestyle="--",
                    label="Full encoding model",
                )
        axes[0].set(ylabel="Mean correlation", title="RRR versus PCA (saved results)")
        axes[0].legend()
        target_counts = []
        for method, color, label in (("rrr", "#6ab2e2", "RRR"), ("pca", "#63bc51", "PCA")):
            if method not in performances:
                continue
            target_means, differences = summarize_target_best_rank(performances[method])
            target_counts.append(f"{label}: {target_means.target.nunique():,}")
            ranks = differences["rank"].to_numpy()
            mean = differences["mean"].to_numpy()
            sd = differences["sd"].to_numpy()
            axes[1].plot(ranks, mean, "o-", color=color, label=label)
            axes[1].fill_between(ranks, mean - sd, mean + sd, color=color, alpha=0.2)
        axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].set(ylabel="Correlation − target’s best correlation", title="Difference from each target’s best rank (±1 SD)")
        axes[1].legend()
        target_label = "voxels" if dataset_label == "fMRI" else "electrodes"
        axes[1].text(0.98, 0.03, f"{'; '.join(target_counts)} {target_label}; folds averaged first",
                     transform=axes[1].transAxes, ha="right", fontsize=8)
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Rank $k$")
            ax.grid(axis="y", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{dataset_label}: {model} / {'layer ' + str(layer) if layer is not None else 'layerless'}")
        fig.tight_layout()
        return fig, axes


    return (plot_target_best_rank,)


@app.cell(hide_code=True)
def best_rank_plots(
    REPO_ROOT,
    best_rank_model_layer_selector,
    internal_rrr_artifact,
    load_rrr_target_performances,
    mo,
    plot_target_best_rank,
):
    _best_model, _best_layer = best_rank_model_layer_selector.value
    best_rank_figures = []
    for _dataset, _label in (("podcast", "Podcast"), ("cars-2", "Cars 2"), ("fmri", "fMRI")):
        _mean_results = internal_rrr_artifact(
            dataset=_dataset, model=_best_model, layer=_best_layer,
            results_root=REPO_ROOT / "results",
        ).load_mean_results()
        _performances = {
            _method: load_rrr_target_performances(
                dataset=_dataset, model=_best_model, layer=_best_layer, method=_method,
            )
            for _method in ("rrr", "pca")
            if _method in _mean_results["method"].values
        }
        best_rank_figures.append(plot_target_best_rank(
            _performances, _mean_results, _label, _best_model, _best_layer,
        )[0])
    mo.vstack([
        mo.md("Each target’s reference is its highest fold-averaged correlation across tested ranks of the same method (RRR or PCA). iEEG lags are averaged within electrodes in Fisher-z space. Shading shows ±1 SD across targets, not a confidence interval."),
        *best_rank_figures,
    ])
    return


@app.cell(hide_code=True)
def best_rrr_model_layer_bar_helpers(
    REPO_ROOT,
    find_rrr_layers,
    find_rrr_models,
    internal_rrr_artifact,
    np,
    pd,
    plt,
):
    def load_best_rrr_model_layer_performance(
        *, models=None, dataset, results_root=REPO_ROOT / "results"
    ):
        """Load each model/layer's best saved mean RRR correlation across ranks."""
        rows = []
        _models = find_rrr_models(dataset=dataset) if models is None else models
        for model in _models:
            for layer in find_rrr_layers(dataset=dataset, model=model):
                try:
                    results = internal_rrr_artifact(
                        dataset=dataset,
                        model=model,
                        layer=layer,
                        results_root=results_root,
                    ).load_mean_results()
                except Exception as error:
                    print(f"Skipping {dataset}/{model}/layer-{layer}: {error}")
                    continue
                rrr = results[results["method"] == "rrr"]
                metric = "nr_orth_corr" if "nr_orth_corr" in rrr else "corr"
                if rrr.empty or metric not in rrr:
                    print(f"Skipping {dataset}/{model}/layer-{layer}: no RRR correlation summary")
                    continue
                best = rrr[metric].dropna().max()
                if pd.notna(best):
                    rows.append({
                        "dataset": dataset,
                        "model": model,
                        "layer": layer,
                        "best_rrr_correlation": float(best),
                    })
        if not rows:
            raise ValueError(f"No best-RRR model/layer summaries found for {dataset}")
        return pd.DataFrame(rows)


    def plot_best_rrr_model_layers(performance, dataset_label, model_colors):
        """Plot layer-ordered best RRR bars in contiguous model groups."""
        fig, ax = plt.subplots(figsize=(13, 5.2))
        positions = []
        tick_labels = []
        next_position = 0.0
        for model in sorted(performance["model"].unique()):
            model_rows = performance[performance["model"] == model].copy()
            model_rows["_layer_order"] = model_rows["layer"].map(
                lambda layer: (layer is None or pd.isna(layer), -1 if pd.isna(layer) else layer)
            )
            model_rows = model_rows.sort_values("_layer_order")
            model_positions = np.arange(len(model_rows), dtype=float) + next_position
            ax.bar(
                model_positions,
                model_rows["best_rrr_correlation"],
                width=0.82,
                color=model_colors[model],
                label=model,
            )
            positions.extend(model_positions)
            tick_labels.extend(
                "layerless" if pd.isna(layer) else str(int(layer))
                for layer in model_rows["layer"]
            )
            group_center = float(model_positions.mean())
            ax.text(
                group_center,
                -0.16,
                model,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
            )
            next_position = float(model_positions[-1]) + 1.8

        ax.set_xticks(positions, tick_labels, rotation=45, ha="right")
        ax.set(
            ylabel="Best mean RRR correlation",
            xlabel="Layer (grouped by model)",
            title=f"{dataset_label}: best RRR performance by model and layer",
        )
        ax.axhline(0, color="#6b7280", linewidth=0.8)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(title="Model", frameon=False, ncol=min(4, len(model_colors)))
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        return fig, ax

    return load_best_rrr_model_layer_performance, plot_best_rrr_model_layers


@app.cell(hide_code=True)
def best_rrr_model_layer_bar_plots(
    find_rrr_models,
    load_best_rrr_model_layer_performance,
    mo,
    plot_best_rrr_model_layers,
    plot_model_filter,
    plt,
):
    _dataset_labels = {
        "podcast": "Podcast",
        "cars-2": "Cars 2",
        "fmri": "fMRI",
    }
    _model_names = sorted({
        model
        for dataset in _dataset_labels
        for model in find_rrr_models(dataset=dataset)
        if model in plot_model_filter.value
    })
    _colormap = plt.get_cmap("tab20", max(len(_model_names), 1))
    _model_colors = {
        model: _colormap(index) for index, model in enumerate(_model_names)
    }
    best_rrr_model_layer_plots = []
    for _dataset, _label in _dataset_labels.items():
        _performance = load_best_rrr_model_layer_performance(dataset=_dataset, models=_model_names)
        best_rrr_model_layer_plots.append(
            plot_best_rrr_model_layers(_performance, _label, _model_colors)[0]
        )
    mo.vstack([
        mo.md(
            "Each bar is the highest saved mean RRR correlation across tested ranks. "
            "Bars are ordered by layer within model; model colors are shared across datasets."
        ),
        *best_rrr_model_layer_plots,
    ])
    return


@app.cell
def _():
    from functools import lru_cache


    def select_model_layer(frame, model, layer):
        """Select a model layer, including layerless embeddings, from loaded data."""
        layer_mask = frame["layer"].isna() if layer is None else frame["layer"].eq(layer)
        return frame[frame["model"].eq(model) & layer_mask].copy()


    return lru_cache, select_model_layer


@app.cell(hide_code=True)
def load_rrr_correlations(Artifact, Path, REPO_ROOT, np, pd):
    DATASET_ROOTS = {
        "podcast": "podcast",
        "cars-2": "brain-treebank-cars-2",
        "fmri": "deep-fmri",
    }

    FMRI_INPUT_SCALING_VARIANT = "input_scaling-pooled-training-stories-v1"


    def internal_rrr_artifact(*, dataset, model, layer, results_root):
        """Select the corrected pooled-training-story variant for internal fMRI."""
        dataset_root = DATASET_ROOTS.get(dataset, dataset)
        artifact = Artifact(
            mode="internal",
            dataset=dataset_root,
            model=model,
            layer=layer,
            results_root=results_root,
        )
        if dataset_root != "deep-fmri":
            return artifact

        index = artifact._read_json(artifact._leaf() / "manifest.json", default={})
        variants = [
            variant
            for variant in sorted(index.get("variants", {}))
            if FMRI_INPUT_SCALING_VARIANT in variant.split("/")
        ]
        if len(variants) != 1:
            raise ValueError(
                f"Expected exactly one {FMRI_INPUT_SCALING_VARIANT!r} internal fMRI "
                f"variant for {model}/layer-{layer}, found {len(variants)}: {variants}"
            )
        artifact.variant = variants[0]
        return artifact

    def find_rrr_models(
        *,
        results_root=REPO_ROOT / "results",
        dataset="podcast",
    ):
        """Find models with at least one internal RRR layer directory."""
        models_root = (
            Path(results_root) / "run_rrr" / "internal"
            / DATASET_ROOTS[dataset] / "offset-0"
        )
        return sorted(
            path.name
            for path in models_root.iterdir()
            if path.is_dir()
            and any(
                child.is_dir()
                and child.name.removeprefix("layer-") == "none"
                or child.is_dir()
                and child.name.removeprefix("layer-").isdigit()
                for child in path.glob("layer-*")
            )
        )


    def find_rrr_layers(
        *,
        results_root=REPO_ROOT / "results",
        dataset="podcast",
        model="gpt2",
    ):
        """Find internal RRR layers with saved Artifact manifests."""
        model_root = (
            Path(results_root) / "run_rrr" / "internal"
            / DATASET_ROOTS[dataset] / "offset-0" / model
        )
        layers = []
        for path in model_root.glob("layer-*"):
            if not path.is_dir():
                continue
            suffix = path.name.removeprefix("layer-")
            if suffix == "none":
                layers.append(None)
            elif suffix.isdigit():
                layers.append(int(suffix))
        return sorted(layers, key=lambda layer: (layer is not None, layer or -1))


    def load_rrr_correlations(
        *,
        results_root=REPO_ROOT / "results",
        dataset="podcast",
        model="gpt2",
        layers,
    ):
        """Load compact per-rank Fisher-z means for one or more layers."""
        rows = []
        columns = ["dataset", "model", "layer", "method", "rank", "mean_fisher_z"]
        metric_names = (
            {"rrr": "corr_per_target", "pca": "corr_per_target"}
            if dataset == "fmri"
            else {"rrr": "nr_orth_corr_per_target", "pca": "pca_corr_per_target"}
        )
        for layer in layers:
            print(f"Loading {model}, layer {layer}...")
            # Load one method at a time and reduce its target arrays immediately.
            # This avoids expanding hundreds of MB of arrays into Python rows.
            for method, metric_name in metric_names.items():
                artifact = internal_rrr_artifact(
                    dataset=dataset,
                    model=model,
                    layer=layer,
                    results_root=results_root,
                ).load_artifacts(["rank_metrics"], selectors={"method": method})
                for selector, metrics in artifact.rank_metrics.items():
                    selector = dict(selector)
                    if metric_name not in metrics:
                        continue
                    values = np.asarray(metrics[metric_name], dtype=float)
                    fisher_z = np.arctanh(np.clip(values, -1 + 1e-7, 1 - 1e-7))
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "layer": layer,
                            "method": method,
                            "rank": int(selector["rank"]),
                            "mean_fisher_z": float(np.nanmean(fisher_z)),
                        }
                    )
                del artifact
        return pd.DataFrame(rows, columns=columns)

    return (
        DATASET_ROOTS,
        FMRI_INPUT_SCALING_VARIANT,
        find_rrr_layers,
        find_rrr_models,
        internal_rrr_artifact,
        load_rrr_correlations,
    )


@app.cell
def _(mo):
    mo.md("""
    ## RRR Aggregate Figures
    """)
    return


@app.cell(hide_code=True)
def rrr_aggregate_figure_helpers(
    normalized_log_rank_auc,
    normalized_performance_curve,
    np,
    paired_permutation_tests,
    pd,
    plot_rank_summary_distributions,
    plt,
):
    def plot_rrr_aggregate_figure(
        sample_frames, convergence_frames, distribution_frames,
        distribution_methods, distribution_colors, distribution_tests, *, distribution_grid=(1, 3),
        percentage=100,
    ):
        """Draw dataset rows at left and column-organized saturation plots at right."""
        dataset_labels = {"podcast": "Podcast", "cars-2": "Cars 2", "fmri": "fMRI"}
        fig = plt.figure(figsize=(19, 7.4))
        grid = fig.add_gridspec(3, 5, width_ratios=[1.25, 1.25, 1.0, 1.0, 1.0])
        method_colors = {
            "rrr": "#6ab2e2", "pca": "#63bc51",
            "RRR baseline": "#6ab2e2", "RRR transfer": "#4b98cc", "PCA baseline": "#63bc51",
            "Within RRR": "#6ab2e2", "Within PCA": "#63bc51",
            "Transfer RRR": "#4b98cc", "Transfer PCA": "#45963a",
        }
        shared_sample_ax = None
        shared_convergence_ax = None
        sample_legend_items = {}
        convergence_legend_items = {}
        for row, dataset in enumerate(("podcast", "cars-2", "fmri")):
            sample_ax = fig.add_subplot(grid[row, 0], sharex=shared_sample_ax)
            convergence_ax = fig.add_subplot(grid[row, 1], sharex=shared_convergence_ax)
            if shared_sample_ax is None:
                shared_sample_ax = sample_ax
            if shared_convergence_ax is None:
                shared_convergence_ax = convergence_ax
            sample = sample_frames[dataset]
            if not sample.empty:
                for method in sample["method"].dropna().unique():
                    method_frame = sample[sample["method"] == method]
                    scores = method_frame.groupby("rank", as_index=False)["correlation"].mean().sort_values("rank")
                    color = "#dc2626" if method == "Full model baseline" else method_colors.get(method, "#6b7280")
                    sample_ax.plot(
                        scores["rank"], scores["correlation"], "o",
                        color=color,
                        linestyle="--" if method == "Full model baseline" or "transfer" in method.lower() else "-",
                        linewidth=1.7 if method == "Full model baseline" else 1.35,
                        markersize=2.2, label=str(method),
                    )
                sample_ax.set_xscale("log", base=2)
                for _handle, _label in zip(*sample_ax.get_legend_handles_labels()):
                    _display_label = {"rrr": "RRR", "pca": "PCA"}.get(_label, _label)
                    sample_legend_items.setdefault(_display_label, _handle)
                sample_ax.grid(axis="y", alpha=.2)
            sample_ax.set_title(dataset_labels[dataset], fontsize=14)
            sample_ax.set_ylabel("Correlation" if row == 1 else "", fontsize=13)
            sample_ax.set_xlabel("")
            sample_ax.tick_params(axis="both", labelsize=11)
            sample_ax.spines[["top", "right"]].set_visible(False)

            curves = convergence_frames[dataset]
            if not curves.empty:
                grouped = ["model", "layer", "method"]
                if "source_dataset" in curves:
                    grouped.insert(2, "source_dataset")
                grid_x = np.concatenate(([0.0], np.geomspace(1e-4, 1.0, 199)))
                for method in curves["method"].dropna().unique():
                    method_curves = []
                    selected = curves[curves["method"] == method]
                    for _, curve in selected.groupby(grouped, dropna=False, sort=True):
                        if "mean_fisher_z" in curve:
                            corr = np.tanh(curve.groupby("rank")["mean_fisher_z"].mean())
                            ranks, values = corr.index.to_numpy(), corr.to_numpy()
                        else:
                            scores = curve.groupby("rank", as_index=False)["correlation"].mean()
                            ranks, values = scores["rank"].to_numpy(), scores["correlation"].to_numpy()
                        x, y = normalized_performance_curve(ranks, values)
                        if x.size:
                            method_curves.append((x, y))
                            convergence_ax.plot(
                                x, y, color=method_colors.get(method, "#6b7280"),
                                alpha=.10, linewidth=.65, zorder=1,
                                linestyle="--" if "transfer" in method.lower() else "-",
                            )
                    if method_curves:
                        aligned = np.asarray([np.interp(grid_x, x, y) for x, y in method_curves])
                        convergence_ax.plot(grid_x, aligned.mean(axis=0), linewidth=1.7,
                                            color=method_colors.get(method, "#6b7280"),
                                                linestyle="--" if "transfer" in method.lower() else "-",
                                                label=str(method), zorder=3)
                convergence_ax.set_xscale("symlog", linthresh=.01)
                convergence_ax.set_xticks([0, .001, .01, .1, 1], ["0", ".1", "1", "10", "100%"])
                convergence_ax.set(xlim=(0, 1), ylim=(-.02, 1.02))
                for _handle, _label in zip(*convergence_ax.get_legend_handles_labels()):
                    _display_label = {"rrr": "RRR", "pca": "PCA"}.get(_label, _label)
                    convergence_legend_items.setdefault(_display_label, _handle)
                convergence_ax.grid(alpha=.2)
            convergence_ax.set_title(dataset_labels[dataset], fontsize=14)
            convergence_ax.set_ylabel("Normalized performance" if row == 1 else "", fontsize=13)
            convergence_ax.set_xlabel("")
            convergence_ax.tick_params(axis="both", labelsize=11)
            convergence_ax.spines[["top", "right"]].set_visible(False)

        if sample_legend_items:
            fig.legend(
                list(sample_legend_items.values()), list(sample_legend_items),
                loc="lower center", bbox_to_anchor=(.137, .84),
                bbox_transform=fig.transFigure, frameon=False, fontsize=9,
                ncol=2, handlelength=1.2, columnspacing=1.0,
            )
        if convergence_legend_items:
            fig.legend(
                list(convergence_legend_items.values()), list(convergence_legend_items),
                loc="lower center", bbox_to_anchor=(.353, .84),
                bbox_transform=fig.transFigure, frameon=False, fontsize=9,
                ncol=2, handlelength=1.2, columnspacing=1.0,
            )

        distribution_rows, distribution_cols = distribution_grid
        distribution_grid_spec = grid[:, 2:5].subgridspec(
            distribution_rows, distribution_cols, wspace=.24, hspace=.36
        )
        shared_y_ax = None
        for index, (label, values) in enumerate(distribution_frames):
            row = index % distribution_rows
            col = index // distribution_rows
            ax = fig.add_subplot(
                distribution_grid_spec[row, col],
                sharey=shared_y_ax if shared_y_ax is not None else None,
            )
            if shared_y_ax is None:
                shared_y_ax = ax
            _, ax = plot_rank_summary_distributions(
                values, label, distribution_methods, distribution_colors,
                metric="rank", percentage=percentage, tests=distribution_tests, ax=ax,
                compact_annotations=True,
            )
            ax.set_title(label, fontsize=13)
            ax.set_ylabel("Saturation rank $k$" if col == 0 and row == 0 else "", fontsize=13)
            ax.tick_params(axis="y", labelleft=(col == 0 and row == 0), labelsize=10)
            ax.tick_params(axis="x", labelsize=10, labelrotation=35)
            for text in ax.texts:
                text.set_fontsize(max(text.get_fontsize(), 10))
            ax.grid(axis="y", alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)

        # Shared column labels keep the three dataset rows compact.
        fig.text(.137, .965, "A. Sample Rank Curves", ha="center", va="bottom", fontsize=15, weight="bold")
        fig.text(.353, .965, "B. Mean Convergence", ha="center", va="bottom", fontsize=15, weight="bold")
        fig.text(.715, .965, f"C. Saturation Distributions ({percentage:g}% Peak Performance)", ha="center", va="bottom", fontsize=15, weight="bold")
        fig.text(.137, .025, "Rank $k$", ha="center", fontsize=14)
        fig.text(.353, .025, "Relative rank", ha="center", fontsize=14)
        fig.subplots_adjust(left=.055, right=.985, top=.78, bottom=.12, hspace=.48, wspace=.36)
        return fig


    def plot_cross_dataset_aggregate_figure(directed_curves_by_target):
        """Show directed convergence and three paired normalized AUC comparisons."""
        dataset_labels = {"podcast": "Podcast", "cars-2": "Cars 2", "fmri": "fMRI"}
        dataset_order = ("podcast", "cars-2", "fmri")
        method_styles = {
            "Within RRR": ("#6ab2e2", "Within RRR"),
            "Within PCA": ("#63bc51", "Within PCA"),
            "Transfer RRR": ("#4b98cc", "RRR transfer"),
            "Transfer PCA": ("#45963a", "PCA transfer"),
        }
        fig = plt.figure(figsize=(21, 10.5))
        grid = fig.add_gridspec(3, 6)
        directed_grid = grid[:, :3].subgridspec(2, 3, hspace=.48, wspace=.3)
        difference_grid = grid[:, 3:6].subgridspec(3, 1, hspace=.30)
        fig.text(.28, .965, "A. Directed Transfer Curves", ha="center", va="bottom", fontsize=15, weight="bold")
        fig.text(.75, .965, "B. Normalized AUC Differences", ha="center", va="bottom", fontsize=15, weight="bold")
        internal_transfer_ax = fig.add_subplot(difference_grid[0, 0])
        saturation_ax = fig.add_subplot(difference_grid[1, 0], sharex=internal_transfer_ax)
        pca_internal_transfer_ax = fig.add_subplot(difference_grid[2, 0], sharex=internal_transfer_ax)
        relative_rank_grid = np.concatenate(([0.0], np.geomspace(1e-4, 1.0, 199)))

        directed_ax0 = None
        direction_index = 0
        for target in dataset_order:
            for source in dataset_order:
                if source == target:
                    continue
                row, col = direction_index % 2, direction_index // 2
                ax = fig.add_subplot(
                    directed_grid[row, col],
                    sharex=directed_ax0 if directed_ax0 is not None else None,
                )
                if directed_ax0 is None:
                    directed_ax0 = ax
                panel = directed_curves_by_target[target]
                panel = panel[
                    (panel["target_dataset"] == target)
                    & (panel["source_dataset"] == source)
                ]
                for method, (color, label) in method_styles.items():
                    method_curves = []
                    method_frame = panel[panel["method"] == method]
                    for _, curve in method_frame.groupby(["model", "layer"], dropna=False, sort=True):
                        x, y = normalized_performance_curve(curve["rank"], curve["correlation"])
                        if x.size:
                            method_curves.append((x, y))
                            ax.plot(x, y, color=color, alpha=.10, linewidth=.65, zorder=1,
                                    linestyle="--" if method.startswith("Transfer ") else "-")
                    if method_curves:
                        aligned = np.asarray([
                            np.interp(relative_rank_grid, x, y) for x, y in method_curves
                        ])
                        ax.plot(relative_rank_grid, aligned.mean(axis=0), color=color,
                                linewidth=1.6, label=label, zorder=3,
                                linestyle="--" if method.startswith("Transfer ") else "-")
                ax.set_xscale("symlog", linthresh=.01)
                ax.set_xticks([0, .001, .01, .1, 1], ["0", ".1", "1", "10", "100%"])
                ax.set(xlim=(0, 1), ylim=(-.02, 1.02))
                ax.set_title(f"{dataset_labels[source]} → {dataset_labels[target]}", fontsize=13)
                ax.tick_params(axis="both", labelsize=10)
                ax.grid(alpha=.2)
                ax.spines[["top", "right"]].set_visible(False)
                ax.legend(frameon=False, fontsize=8, ncol=2, handlelength=1.2)
                direction_index += 1

        directed_labels = []
        difference_groups = []
        internal_transfer_groups = []
        pca_internal_transfer_groups = []
        colors = plt.get_cmap("tab10").colors
        for target in dataset_order:
            for source in dataset_order:
                if source == target:
                    continue
                label = f"{dataset_labels[source]} → {dataset_labels[target]}"
                frame = directed_curves_by_target[target][
                    (directed_curves_by_target[target]["target_dataset"] == target)
                    & (directed_curves_by_target[target]["source_dataset"] == source)
                ]
                paired_aucs = {}
                for (model, layer, method), curve in frame.groupby(
                    ["model", "layer", "method"], dropna=False, sort=True
                ):
                    auc = normalized_log_rank_auc(curve["rank"], curve["correlation"])
                    if np.isfinite(auc):
                        paired_aucs.setdefault((model, None if pd.isna(layer) else layer), {})[method] = float(auc)
                differences = np.asarray([
                    methods["Transfer RRR"] - methods["Transfer PCA"]
                    for methods in paired_aucs.values()
                    if "Transfer RRR" in methods and "Transfer PCA" in methods
                ], dtype=float)
                internal_transfer_differences = np.asarray([
                    methods["Within RRR"] - methods["Transfer RRR"]
                    for methods in paired_aucs.values()
                    if "Within RRR" in methods and "Transfer RRR" in methods
                ], dtype=float)
                pca_internal_transfer_differences = np.asarray([
                    methods["Within PCA"] - methods["Transfer PCA"]
                    for methods in paired_aucs.values()
                    if "Within PCA" in methods and "Transfer PCA" in methods
                ], dtype=float)
                directed_labels.append(label)
                difference_groups.append(differences)
                internal_transfer_groups.append(internal_transfer_differences)
                pca_internal_transfer_groups.append(pca_internal_transfer_differences)

        positions = np.arange(1, len(directed_labels) + 1, dtype=float)
        for position, (label, values, color) in enumerate(
            zip(directed_labels, difference_groups, colors), start=1
        ):
            if not len(values):
                continue
            violin = saturation_ax.violinplot(
                [values], positions=[position], widths=.68, showextrema=False
            )
            body = violin["bodies"][0]
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(.24)
            body.set_linewidth(1.0)
            boxes = saturation_ax.boxplot(
                [values], positions=[position], widths=.16, patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.1},
                whiskerprops={"color": "#4b5563", "linewidth": .8},
                capprops={"color": "#4b5563", "linewidth": .8},
                boxprops={"color": "#4b5563", "linewidth": .8},
            )
            boxes["boxes"][0].set_facecolor(color)
            boxes["boxes"][0].set_alpha(.2)
            offsets = np.linspace(-.18, .18, len(values)) if len(values) > 1 else np.zeros(1)
            saturation_ax.scatter(
                position + offsets, values, color=color, s=20,
                edgecolors="white", linewidths=.45, alpha=.78, zorder=3,
            )
            mean_value = float(np.mean(values))
            saturation_ax.scatter(position, mean_value, marker="D", s=34,
                                  color="black", zorder=4)
            results = paired_permutation_tests(
                {"Transfer RRR AUC − Transfer PCA AUC": values, "Zero": np.zeros_like(values)},
                [{"first": "Transfer RRR AUC − Transfer PCA AUC", "second": "Zero", "alternative": "greater"}],
            )
            result = results.iloc[0]
            p_value = result["p_value_holm"]
            p_text = "p<.001" if p_value < .001 else f"p={p_value:.3f}"
            mark = "*" if result["significant"] else "n.s."
            saturation_ax.text(
                position, .97, f"{mark} {p_text}",
                transform=saturation_ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=11, color="#374151",
            )

        saturation_ax.axhline(0, color="#111827", linewidth=1.2, linestyle="--", zorder=1)
        _all_differences = np.concatenate([values for values in difference_groups if len(values)])
        _difference_min = float(np.nanmin(_all_differences))
        _difference_max = float(np.nanmax(_all_differences))
        _difference_span = max(_difference_max - _difference_min, 1.0)
        saturation_ax.set_ylim(
            min(_difference_min, 0.0) - .12 * _difference_span,
            max(_difference_max, 0.0) + .24 * _difference_span,
        )
        saturation_ax.set_xticks(positions, directed_labels, rotation=28, ha="right", fontsize=10)
        saturation_ax.set_xlim(.45, len(directed_labels) + .55)
        _auc_span = max(abs(_difference_min), abs(_difference_max), 0.01)
        saturation_ax.set_ylim(-1.12 * _auc_span, 1.24 * _auc_span)
        saturation_ax.set_xlabel("Directed source → target comparison", fontsize=13)
        saturation_ax.set_ylabel("AUC difference", fontsize=12)
        saturation_ax.set_title(
            "Cross-dataset transfer AUC: RRR − PCA", fontsize=14
        )
        saturation_ax.tick_params(axis="y", labelsize=11)
        saturation_ax.grid(axis="y", alpha=.2)
        saturation_ax.spines[["top", "right"]].set_visible(False)

        for position, (label, values, color) in enumerate(
            zip(directed_labels, internal_transfer_groups, colors), start=1
        ):
            if not len(values):
                continue
            violin = internal_transfer_ax.violinplot(
                [values], positions=[position], widths=.68, showextrema=False
            )
            body = violin["bodies"][0]
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(.24)
            body.set_linewidth(1.0)
            boxes = internal_transfer_ax.boxplot(
                [values], positions=[position], widths=.16, patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.1},
                whiskerprops={"color": "#4b5563", "linewidth": .8},
                capprops={"color": "#4b5563", "linewidth": .8},
                boxprops={"color": "#4b5563", "linewidth": .8},
            )
            boxes["boxes"][0].set_facecolor(color)
            boxes["boxes"][0].set_alpha(.2)
            offsets = np.linspace(-.18, .18, len(values)) if len(values) > 1 else np.zeros(1)
            internal_transfer_ax.scatter(
                position + offsets, values, color=color, s=20,
                edgecolors="white", linewidths=.45, alpha=.78, zorder=3,
            )
            mean_value = float(np.mean(values))
            internal_transfer_ax.scatter(position, mean_value, marker="D", s=34,
                                  color="black", zorder=4)
            results = paired_permutation_tests(
                {"Transfer RRR AUC − Transfer PCA AUC": values, "Zero": np.zeros_like(values)},
                [{"first": "Transfer RRR AUC − Transfer PCA AUC", "second": "Zero", "alternative": "greater"}],
            )
            result = results.iloc[0]
            p_value = result["p_value_holm"]
            p_text = "p<.001" if p_value < .001 else f"p={p_value:.3f}"
            mark = "*" if result["significant"] else "n.s."
            internal_transfer_ax.text(
                position, .97, f"{mark} {p_text}",
                transform=internal_transfer_ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=11, color="#374151",
            )

        internal_transfer_ax.axhline(0, color="#111827", linewidth=1.2, linestyle="--", zorder=1)
        _all_differences = np.concatenate([values for values in internal_transfer_groups if len(values)])
        _difference_min = float(np.nanmin(_all_differences))
        _difference_max = float(np.nanmax(_all_differences))
        _difference_span = max(_difference_max - _difference_min, 1.0)
        internal_transfer_ax.set_ylim(
            min(_difference_min, 0.0) - .12 * _difference_span,
            max(_difference_max, 0.0) + .24 * _difference_span,
        )
        internal_transfer_ax.set_xticks(positions, directed_labels, rotation=28, ha="right", fontsize=10)
        internal_transfer_ax.set_xlim(.45, len(directed_labels) + .55)
        _auc_span = max(abs(_difference_min), abs(_difference_max), 0.01)
        internal_transfer_ax.set_ylim(-1.12 * _auc_span, 1.24 * _auc_span)
        internal_transfer_ax.set_ylabel("")
        internal_transfer_ax.set_title(
            "Internal RRR − transfer RRR normalized AUC", fontsize=14
        )
        internal_transfer_ax.tick_params(axis="y", labelsize=11)
        internal_transfer_ax.grid(axis="y", alpha=.2)
        internal_transfer_ax.spines[["top", "right"]].set_visible(False)

        for position, (label, values, color) in enumerate(
            zip(directed_labels, pca_internal_transfer_groups, colors), start=1
        ):
            if not len(values):
                continue
            violin = pca_internal_transfer_ax.violinplot(
                [values], positions=[position], widths=.68, showextrema=False
            )
            body = violin["bodies"][0]
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(.24)
            body.set_linewidth(1.0)
            boxes = pca_internal_transfer_ax.boxplot(
                [values], positions=[position], widths=.16, patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.1},
                whiskerprops={"color": "#4b5563", "linewidth": .8},
                capprops={"color": "#4b5563", "linewidth": .8},
                boxprops={"color": "#4b5563", "linewidth": .8},
            )
            boxes["boxes"][0].set_facecolor(color)
            boxes["boxes"][0].set_alpha(.2)
            offsets = np.linspace(-.18, .18, len(values)) if len(values) > 1 else np.zeros(1)
            pca_internal_transfer_ax.scatter(
                position + offsets, values, color=color, s=20,
                edgecolors="white", linewidths=.45, alpha=.78, zorder=3,
            )
            mean_value = float(np.mean(values))
            pca_internal_transfer_ax.scatter(position, mean_value, marker="D", s=34,
                                  color="black", zorder=4)
            results = paired_permutation_tests(
                {"Within PCA AUC − Transfer PCA AUC": values, "Zero": np.zeros_like(values)},
                [{"first": "Within PCA AUC − Transfer PCA AUC", "second": "Zero", "alternative": "greater"}],
            )
            result = results.iloc[0]
            p_value = result["p_value_holm"]
            p_text = "p<.001" if p_value < .001 else f"p={p_value:.3f}"
            mark = "*" if result["significant"] else "n.s."
            pca_internal_transfer_ax.text(
                position, .97, f"{mark} {p_text}",
                transform=pca_internal_transfer_ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=11, color="#374151",
            )

        pca_internal_transfer_ax.axhline(0, color="#111827", linewidth=1.2, linestyle="--", zorder=1)
        _all_differences = np.concatenate([values for values in pca_internal_transfer_groups if len(values)])
        _difference_min = float(np.nanmin(_all_differences))
        _difference_max = float(np.nanmax(_all_differences))
        _difference_span = max(_difference_max - _difference_min, 1.0)
        pca_internal_transfer_ax.set_ylim(
            min(_difference_min, 0.0) - .12 * _difference_span,
            max(_difference_max, 0.0) + .24 * _difference_span,
        )
        pca_internal_transfer_ax.set_xticks(positions, directed_labels, rotation=28, ha="right", fontsize=10)
        pca_internal_transfer_ax.set_xlim(.45, len(directed_labels) + .55)
        _auc_span = max(abs(_difference_min), abs(_difference_max), 0.01)
        pca_internal_transfer_ax.set_ylim(-1.12 * _auc_span, 1.24 * _auc_span)
        pca_internal_transfer_ax.set_ylabel("")
        pca_internal_transfer_ax.set_title(
            "Internal PCA − transfer PCA normalized AUC", fontsize=14
        )
        pca_internal_transfer_ax.tick_params(axis="y", labelsize=11)
        pca_internal_transfer_ax.grid(axis="y", alpha=.2)
        pca_internal_transfer_ax.spines[["top", "right"]].set_visible(False)

        internal_transfer_ax.tick_params(axis="x", labelbottom=False)
        internal_transfer_ax.set_xlabel("")
        saturation_ax.tick_params(axis="x", labelbottom=False)
        saturation_ax.set_xlabel("")
        pca_internal_transfer_ax.set_xlabel("Directed source → target comparison", fontsize=13)
        fig.text(.28, .075, "Relative rank", ha="center", fontsize=13)
        fig.subplots_adjust(left=.045, right=.99, top=.88, bottom=.15, hspace=.48, wspace=.42)
        return fig


    def plot_cross_dataset_auc_gap_correlation(directed_curves_by_target):
        """Correlate matched within-minus-transfer PCA and RRR normalized AUC gaps."""
        dataset_labels = {"podcast": "Podcast", "cars-2": "Cars 2", "fmri": "fMRI"}
        dataset_order = ("podcast", "cars-2", "fmri")
        fig, ax = plt.subplots(figsize=(10.5, 7.0))
        all_points = []
        colors = plt.get_cmap("tab10").colors
        direction_index = 0
        for target in dataset_order:
            for source in dataset_order:
                if source == target:
                    continue
                frame = directed_curves_by_target[target]
                frame = frame[
                    (frame["target_dataset"] == target)
                    & (frame["source_dataset"] == source)
                ]
                paired_aucs = {}
                for (model, layer, method), curve in frame.groupby(
                    ["model", "layer", "method"], dropna=False, sort=True
                ):
                    auc = normalized_log_rank_auc(curve["rank"], curve["correlation"])
                    if np.isfinite(auc):
                        paired_aucs.setdefault((model, None if pd.isna(layer) else layer), {})[method] = float(auc)
                points = np.asarray([
                    (
                        methods["Within PCA"] - methods["Transfer PCA"],
                        methods["Within RRR"] - methods["Transfer RRR"],
                    )
                    for methods in paired_aucs.values()
                    if all(method in methods for method in (
                        "Within PCA", "Transfer PCA", "Within RRR", "Transfer RRR"
                    ))
                ], dtype=float).reshape(-1, 2)
                if len(points):
                    all_points.append(points)
                    ax.scatter(
                        points[:, 0], points[:, 1],
                        color=colors[direction_index], s=35, alpha=.72,
                        edgecolors="white", linewidths=.4,
                        label=f"{dataset_labels[source]} → {dataset_labels[target]} (n={len(points)})",
                    )
                direction_index += 1

        if all_points:
            points = np.vstack(all_points)
            lower = min(0.0, float(points.min()))
            upper = max(0.0, float(points.max()))
            padding = max((upper - lower) * .06, .01)
            limits = (lower - padding, upper + padding)
            ax.plot(limits, limits, color="#94a3b8", linestyle="--", linewidth=1.2,
                    label="Equal gaps", zorder=0)
            if len(points) >= 2 and np.unique(points[:, 0]).size >= 2:
                slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
                line_x = np.linspace(points[:, 0].min(), points[:, 0].max(), 100)
                ax.plot(line_x, slope * line_x + intercept, color="#111827",
                        linewidth=2, label="Pooled linear fit")
            from scipy.stats import pearsonr as _pearsonr
            if len(points) >= 2 and np.unique(points[:, 0]).size > 1 and np.unique(points[:, 1]).size > 1:
                _correlation_result = _pearsonr(points[:, 0], points[:, 1])
                r = _correlation_result.statistic
                _p_value = _correlation_result.pvalue
            else:
                r = pd.Series(points[:, 0]).corr(pd.Series(points[:, 1]))
                _p_value = np.nan
            _p_text = (
                "p<.001" if np.isfinite(_p_value) and _p_value < .001
                else f"p={_p_value:.3f}" if np.isfinite(_p_value)
                else "p=NA"
            )
            ax.set_title(
                f"Internal − transfer AUC gaps across directed datasets\n"
                f"Pearson r = {r:.2f}, {_p_text}, n = {len(points)}",
                fontsize=14,
            )
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            ax.legend(frameon=False, fontsize=10, loc="center left", bbox_to_anchor=(1.02, .5))
        else:
            ax.set_title("Internal − transfer AUC gaps across directed datasets", fontsize=14)
            ax.text(.5, .5, "No matched model-layer results", transform=ax.transAxes,
                    ha="center", va="center")
        ax.axhline(0, color="#64748b", linewidth=.8, alpha=.65)
        ax.axvline(0, color="#64748b", linewidth=.8, alpha=.65)
        ax.set_xlabel("Within PCA − transfer PCA normalized AUC", fontsize=12)
        ax.set_ylabel("Within RRR − transfer RRR normalized AUC", fontsize=12)
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        fig.subplots_adjust(left=.13, right=.72, top=.85, bottom=.14)
        return fig


    def cross_dataset_gap_regression_data(directed_curves_by_target, feature_auc):
        """Match directed normalized AUC gaps to offset-zero feature agreement."""
        labels = {"podcast": "Podcast", "cars-2": "Cars 2", "fmri": "fMRI"}
        curves = pd.concat(directed_curves_by_target.values(), ignore_index=True)
        feature_groups = {
            (model, layer): group
            for (model, layer), group in feature_auc[
                feature_auc["label_offset"] == 0
            ].groupby(["model", "Layer"], dropna=False)
        }
        aucs = {}
        for (source, target, model, layer, method), curve in curves.groupby(
            ["source_dataset", "target_dataset", "model", "layer", "method"],
            dropna=False, sort=True,
        ):
            auc = normalized_log_rank_auc(curve["rank"], curve["correlation"])
            if np.isfinite(auc):
                key = (source, target, model, None if pd.isna(layer) else int(layer))
                aucs.setdefault(key, {})[method] = float(auc)
        rows = []
        required = {"Within PCA", "Transfer PCA", "Within RRR", "Transfer RRR"}
        for (source, target, model, layer), methods in aucs.items():
            if not required.issubset(methods):
                continue
            feature_layer = "layerless" if layer is None else layer
            feature_rows = feature_groups.get((model, feature_layer))
            if feature_rows is None:
                continue
            paired_features = feature_rows[[labels[source], labels[target]]].dropna()
            feature_r = paired_features.iloc[:, 0].corr(paired_features.iloc[:, 1])
            if not np.isfinite(feature_r):
                continue
            rows.append({
                "source_dataset": source, "target_dataset": target,
                "pair": f"{labels[source]} → {labels[target]}",
                "model": model, "layer": feature_layer,
                "pca_gap": methods["Within PCA"] - methods["Transfer PCA"],
                "rrr_gap": methods["Within RRR"] - methods["Transfer RRR"],
                "feature_r": float(feature_r),
                "n_features": len(paired_features),
            })
        return pd.DataFrame.from_records(rows)


    def plot_cross_dataset_gap_regression(gaps):
        """Compare PCA-only and PCA-plus-feature linear fits for the RRR gap."""
        fig, (prediction_ax, variance_ax) = plt.subplots(
            1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.2, 1]}
        )
        if gaps.empty:
            for ax in (prediction_ax, variance_ax):
                ax.text(.5, .5, "No matched results", transform=ax.transAxes,
                        ha="center", va="center")
            return fig

        y = gaps["rrr_gap"].to_numpy(dtype=float)
        x_pca = gaps[["pca_gap"]].to_numpy(dtype=float)
        x_full = gaps[["pca_gap", "feature_r"]].to_numpy(dtype=float)
        groups = list(zip(gaps["model"], gaps["layer"]))

        def fit_and_predict(x):
            design = np.column_stack([np.ones(len(x)), x])
            coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
            fitted = design @ coefficients
            held_out = np.empty(len(y), dtype=float)
            for group in set(groups):
                test = np.asarray([item == group for item in groups])
                train_coefficients = np.linalg.lstsq(design[~test], y[~test], rcond=None)[0]
                held_out[test] = design[test] @ train_coefficients
            return fitted, held_out, coefficients

        pca_fit, pca_cv, _ = fit_and_predict(x_pca)
        full_fit, full_cv, coefficients = fit_and_predict(x_full)
        total_variance = np.sum((y - y.mean()) ** 2)
        def r_squared(prediction):
            return 1 - np.sum((y - prediction) ** 2) / total_variance
        pca_r2 = [r_squared(pca_fit), r_squared(pca_cv)]
        full_r2 = [r_squared(full_fit), r_squared(full_cv)]

        pair_order = [
            f"{source} → {target}"
            for target in ("Podcast", "Cars 2", "fMRI")
            for source in ("Podcast", "Cars 2", "fMRI") if source != target
        ]
        colors = plt.get_cmap("tab10").colors
        for index, pair in enumerate(pair_order):
            selected = gaps["pair"] == pair
            if selected.any():
                prediction_ax.scatter(
                    y[selected], full_cv[selected], color=colors[index],
                    s=34, alpha=.72, edgecolors="white", linewidths=.35,
                    label=f"{pair} (n={selected.sum()})",
                )
        bound_min = min(float(y.min()), float(full_cv.min()), 0.0)
        bound_max = max(float(y.max()), float(full_cv.max()), 0.0)
        pad = max((bound_max - bound_min) * .06, .01)
        limits = (bound_min - pad, bound_max + pad)
        prediction_ax.plot(limits, limits, color="#94a3b8", linestyle="--", linewidth=1.2)
        prediction_ax.set(xlim=limits, ylim=limits)
        prediction_ax.set_title(f"Held-out RRR gap predictions\nR² = {full_r2[1]:.2f}, n = {len(gaps)}", fontsize=13)
        prediction_ax.set_xlabel("Observed within RRR − transfer RRR AUC", fontsize=11)
        prediction_ax.set_ylabel("Predicted within RRR − transfer RRR AUC", fontsize=11)
        prediction_ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")

        positions = np.arange(2)
        width = .33
        variance_ax.bar(positions - width / 2, pca_r2, width, color="#63bc51", label="PCA gap only")
        variance_ax.bar(positions + width / 2, full_r2, width, color="#6ab2e2", label="PCA gap + feature r")
        for x_values, scores in ((positions - width / 2, pca_r2), (positions + width / 2, full_r2)):
            for x, score in zip(x_values, scores):
                variance_ax.text(x, score + (.012 if score >= 0 else -.012), f"{score:.2f}",
                                 ha="center", va="bottom" if score >= 0 else "top", fontsize=10)
        variance_ax.axhline(0, color="#111827", linewidth=.8)
        variance_ax.set_xticks(positions, ["Fitted", "Held out by model/layer"])
        variance_ax.set_ylabel("Variance explained (R²)", fontsize=11)
        variance_ax.set_title("How much does feature agreement add?", fontsize=13)
        variance_ax.legend(frameon=False, fontsize=10)
        variance_ax.set_ylim(
            min(-.1, min(pca_r2 + full_r2) - .12),
            max(.1, max(pca_r2 + full_r2) + .12),
        )
        for ax in (prediction_ax, variance_ax):
            ax.grid(axis="y", alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(
            "RRR gap ~ PCA gap + cross-dataset feature-decoding correlation",
            fontsize=15,
        )
        fig.subplots_adjust(left=.07, right=.98, top=.83, bottom=.15, wspace=.28)
        return fig

    return (
        cross_dataset_gap_regression_data,
        plot_cross_dataset_aggregate_figure,
        plot_cross_dataset_auc_gap_correlation,
        plot_cross_dataset_gap_regression,
        plot_rrr_aggregate_figure,
    )


@app.cell(hide_code=True)
def rrr_aggregate_figures(
    METHOD_COLORS,
    QUANTILE_PEAK_FLOOR,
    REPO_ROOT,
    cross_curve_depth,
    cross_curve_mode,
    cross_curve_model,
    cross_dataset_gap_regression_data,
    cross_dataset_normalized_curves,
    filter_normalized_curves,
    internal_curve_depth,
    internal_curve_mode,
    internal_curve_model,
    internal_quantile_saturation,
    internal_rrr_artifact,
    internal_saturation_percent,
    internal_summary_curves,
    language_decoding_internal_consensus_auc,
    loso_auc_rrr_baseline_source,
    loso_curve_depth,
    loso_curve_mode,
    loso_curve_model,
    loso_model_layer_selector,
    loso_quantile_saturation,
    loso_rank_curves_by_dataset,
    loso_saturation_percent,
    loso_summary_curves_by_dataset,
    mo,
    model_layer_selector,
    np,
    pd,
    plot_cross_dataset_aggregate_figure,
    plot_cross_dataset_auc_gap_correlation,
    plot_cross_dataset_gap_regression,
    plot_rrr_aggregate_figure,
):
    _datasets = ("podcast", "cars-2", "fmri")
    _labels = {"podcast": "Podcast", "cars-2": "Cars 2", "fmri": "fMRI"}

    def _select_pair(frame, pair):
        _model, _layer = pair
        _selected = frame[frame["model"] == _model]
        _selected = _selected[_selected["layer"].isna()] if _layer is None else _selected[_selected["layer"] == _layer]
        return _selected.copy()


    def _internal_full_rank_baseline(dataset, model, layer):
        artifact = internal_rrr_artifact(
            dataset=dataset, model=model, layer=layer, results_root=REPO_ROOT / "results"
        ).load_artifacts(["full_metrics"])
        if dataset == "fmri":
            entries = artifact.full_metrics.items()
            metric_name = "corr_per_target"
        else:
            entries = artifact.load_collection("full_metrics", {"method": "rrr"}).items()
            metric_name = "full_corr_per_target"
        fold_means = []
        for _, metrics in entries:
            if metric_name not in metrics:
                continue
            values = np.asarray(metrics[metric_name], dtype=float).reshape(-1)
            fisher_z = np.arctanh(np.clip(values, -1 + 1e-7, 1 - 1e-7))
            if fisher_z.size and np.isfinite(fisher_z).any():
                fold_means.append(float(np.nanmean(fisher_z)))
        if not fold_means:
            raise ValueError(f"No full-rank baseline metrics for {dataset}/{model}/layer-{layer}")
        return float(np.tanh(np.mean(fold_means)))

    _internal_pair = model_layer_selector.value
    _internal_sample, _internal_convergence, _internal_distributions = {}, {}, []
    _internal_pct = internal_saturation_percent.value
    for _dataset in _datasets:
        _frame = internal_summary_curves[_dataset]
        _internal_sample[_dataset] = _select_pair(_frame, _internal_pair).assign(
            correlation=lambda _x: np.tanh(_x["mean_fisher_z"])
        )
        _full_rank_value = _internal_full_rank_baseline(
            _dataset, _internal_pair[0], _internal_pair[1]
        )
        _baseline_ranks = sorted(_internal_sample[_dataset]["rank"].unique())
        _internal_sample[_dataset] = pd.concat([
            _internal_sample[_dataset],
            pd.DataFrame({
                "method": "Full model baseline",
                "rank": _baseline_ranks,
                "correlation": _full_rank_value,
            }),
        ], ignore_index=True, sort=False)
        _filtered, _ = filter_normalized_curves(
            _frame, internal_curve_mode.value, internal_curve_model.value, internal_curve_depth.value
        )
        _internal_convergence[_dataset] = _filtered.assign(
            correlation=lambda _x: np.tanh(_x["mean_fisher_z"])
        )
        _internal_sat = internal_quantile_saturation(
            _dataset, _filtered, _internal_pct / 100,
        )
        _internal_distributions.append((
            _labels[_dataset], _internal_sat
        ))
    _internal_figure = plot_rrr_aggregate_figure(
        _internal_sample, _internal_convergence, _internal_distributions,
        {"RRR": "rrr", "PCA": "pca"}, METHOD_COLORS,
        [{"first": "RRR", "second": "PCA", "alternative": "less"}],
        percentage=_internal_pct,
    )

    _loso_pair = loso_model_layer_selector.value
    _loso_sample, _loso_convergence, _loso_distributions = {}, {}, []
    _loso_pct = loso_saturation_percent.value
    for _dataset in _datasets:
        _loso_sample[_dataset] = _select_pair(loso_rank_curves_by_dataset[_dataset], _loso_pair)
        _full_embedding_value = _internal_full_rank_baseline(
            _dataset, _loso_pair[0], _loso_pair[1]
        )
        _loso_baseline_ranks = sorted(_loso_sample[_dataset]["rank"].unique())
        _loso_sample[_dataset] = pd.concat([
            _loso_sample[_dataset],
            pd.DataFrame({
                "method": "Full model baseline",
                "rank": _loso_baseline_ranks,
                "correlation": _full_embedding_value,
            }),
        ], ignore_index=True, sort=False)
        _filtered, _ = filter_normalized_curves(
            loso_summary_curves_by_dataset[_dataset], loso_curve_mode.value,
            loso_curve_model.value, loso_curve_depth.value,
        )
        _loso_convergence[_dataset] = _filtered
        _loso_sat = loso_quantile_saturation(
            _dataset, _filtered, _loso_pct / 100,
            baseline_source=loso_auc_rrr_baseline_source.value,
        )
        _loso_distributions.append((
            _labels[_dataset], _loso_sat
        ))
    _loso_figure = plot_rrr_aggregate_figure(
        _loso_sample, _loso_convergence, _loso_distributions,
        {"RRR baseline": "RRR baseline", "RRR transfer": "RRR transfer", "PCA baseline": "PCA baseline"},
        {"RRR baseline": "#6ab2e2", "RRR transfer": "#4b98cc", "PCA baseline": "#63bc51"},
        [
            {"first": "RRR transfer", "second": "PCA baseline", "alternative": "less"},
            {"first": "RRR baseline", "second": "RRR transfer", "alternative": "less"},
        ],
        percentage=_loso_pct,
    )

    _directed_curves_by_target = {}
    _selected_cross, _ = filter_normalized_curves(
        cross_dataset_normalized_curves, cross_curve_mode.value,
        cross_curve_model.value, cross_curve_depth.value,
    )
    for _dataset in _datasets:
        _directed_curves_by_target[_dataset] = _selected_cross[
            _selected_cross["target_dataset"] == _dataset
        ]
    _cross_figure = plot_cross_dataset_aggregate_figure(_directed_curves_by_target)
    _cross_gap_figure = plot_cross_dataset_auc_gap_correlation(_directed_curves_by_target)
    _gap_regression_data = cross_dataset_gap_regression_data(
        _directed_curves_by_target, language_decoding_internal_consensus_auc
    )
    _gap_regression_figure = plot_cross_dataset_gap_regression(_gap_regression_data)
    for _figure in (
        _internal_figure, _loso_figure, _cross_figure,
        _cross_gap_figure, _gap_regression_figure,
    ):
        for _artist in _figure.findobj():
            if hasattr(_artist, "get_fontsize") and hasattr(_artist, "set_fontsize"):
                _artist.set_fontsize(_artist.get_fontsize() * 1.2)

    mo.vstack([
        mo.md(f"Sample rank curves show mean correlation at each rank; saturation requires every target quantile peaking above {QUANTILE_PEAK_FLOOR:g} to reach the selected percentage of its own peak."),
        _internal_figure, _loso_figure, _cross_figure,
        mo.md("### Cross-dataset AUC gap correlation"),
        _cross_gap_figure,
        mo.md("Each dot is one matched model/layer × directed source–target dataset pair. "
              "The figure uses the cross-dataset model and depth filters above."),
        mo.md("### Predicting the RRR transfer gap"),
        _gap_regression_figure,
        mo.md("The feature predictor is the Pearson correlation across shared decoded targets of "
              "offset-0, within-dataset RRR–PCA relative AUC for each model/layer and dataset pair. "
              "Held-out R² leaves out all directed pairs for one model/layer at a time."),
    ])
    return


@app.cell(hide_code=True)
def language_decoding_aggregate_figure(
    language_decoding_internal_consensus_auc,
    language_decoding_pooled_curves,
    language_decoding_pooled_depth_filter,
    language_decoding_pooled_layer_depth,
    language_decoding_pooled_model_filter,
    mo,
    np,
    plt,
):
    from scipy.stats import pearsonr as _pearsonr
    _aggregate_datasets = ("Cars 2", "Podcast", "fMRI")
    _aggregate_line_color = "#4b5563"
    _aggregate_linestyles = {"Cars 2": "-", "Podcast": "--", "fMRI": ":"}
    _aggregate_examples = (
        ("concreteness", "Concreteness"),
        ("sentence_position", "Sentence position"),
        ("standardized_entropy", "Entropy"),
    )
    _aggregate_pairs = (
        ("Cars 2", "Podcast"),
        ("Podcast", "fMRI"),
        ("Cars 2", "fMRI"),
    )
    _aggregate_curves = language_decoding_pooled_curves[
        language_decoding_pooled_curves["label_offset"] == 0
    ].copy()
    _aggregate_points = language_decoding_internal_consensus_auc[
        (language_decoding_internal_consensus_auc["label_offset"] == 0)
        & language_decoding_internal_consensus_auc["model"].isin(
            language_decoding_pooled_model_filter.value
        )
    ].copy()
    _aggregate_points["Depth"] = [
        language_decoding_pooled_layer_depth[(_model, _layer)]
        for _model, _layer in zip(_aggregate_points["model"], _aggregate_points["Layer"])
    ]
    _aggregate_points = _aggregate_points[
        _aggregate_points["Depth"].isin(language_decoding_pooled_depth_filter.value)
    ].copy()
    _fig = plt.figure(figsize=(20, 7.2))
    _grid = _fig.add_gridspec(3, 4, width_ratios=[1.25, 1, 1, 1], hspace=.42, wspace=.45)
    _right_values = _aggregate_points[list(_aggregate_datasets)].to_numpy(dtype=float)
    _right_bound = max(float(np.nanmax(np.abs(_right_values))), .05) if _right_values.size else .05
    _right_limits = (-1.05 * _right_bound, 1.05 * _right_bound)
    for _row, ((_feature, _title), (_x_name, _y_name)) in enumerate(
        zip(_aggregate_examples, _aggregate_pairs)
    ):
        _left = _fig.add_subplot(_grid[_row, 0])
        _example = _aggregate_curves[_aggregate_curves["task"] == _feature]
        for _dataset in _aggregate_datasets:
            _curve = _example[_example["Dataset"] == _dataset].sort_values("k")
            if not _curve.empty:
                _left.plot(
                    _curve["k"], _curve["Average relative difference"],
                    color=_aggregate_line_color,
                    linestyle=_aggregate_linestyles[_dataset], linewidth=2.1,
                    marker="o", markersize=3.5, label=_dataset,
                )
        _left.axhline(0, color="#111827", linewidth=.8, linestyle="--")
        _left.set_xscale("log", base=2)
        _ranks = sorted(_example["k"].astype(int).unique())
        if _ranks:
            _left.set_xticks(_ranks, [str(_rank) for _rank in _ranks])
        _left.set_title(_title, fontsize=12)
        _left.set_ylabel("Mean RRR–PCA difference" if _row == 1 else "", fontsize=9)
        _left.tick_params(axis="both", labelsize=8)
        if _row < 2:
            _left.tick_params(axis="x", labelbottom=False)
        if _row == 2:
            _left.set_xlabel("Decoding rank", fontsize=10)
        if _row == 0:
            _handles, _labels = _left.get_legend_handles_labels()
            _fig.legend(_handles, _labels, frameon=False, ncol=3, fontsize=9,
                        loc="upper center", bbox_to_anchor=(.19, .925))
        if _example.empty:
            _left.text(.5, .5, "No selected model/layer results", ha="center", va="center", transform=_left.transAxes)
        _left_initial_ylim = _left.get_ylim()
        _left_bound = max(abs(_left_initial_ylim[0]), abs(_left_initial_ylim[1]), 1e-12)
        _left.set_ylim(-_left_bound, _left_bound)
        _left_position = _left.get_position()
        _left.set_position([
            _left_position.x0, _left_position.y0,
            _left_position.width * .80, _left_position.height,
        ])
        _left_xlim, _left_ylim = _left.get_xlim(), _left.get_ylim()
        _gradient_scale = max(abs(_left_ylim[0]), abs(_left_ylim[1]), 1e-12)
        _gradient_values = np.linspace(*_left_ylim, 256) / _gradient_scale
        _gradient_white = np.ones(3)
        _gradient_green = np.array([99, 188, 81]) / 255
        _gradient_blue = np.array([106, 178, 226]) / 255
        _gradient_rgb = np.where(
            _gradient_values[:, None] < 0,
            _gradient_white - _gradient_values[:, None] * (_gradient_green - _gradient_white),
            _gradient_white + _gradient_values[:, None] * (_gradient_blue - _gradient_white),
        )
        _gradient_image = np.repeat(_gradient_rgb[:, None, :], 2, axis=1)
        _left.imshow(
            _gradient_image,
            extent=(*_left_xlim, *_left_ylim),
            origin="lower",
            aspect="auto",
            alpha=.25,
            interpolation="bilinear",
            zorder=-10,
        )
        _left.set_xlim(_left_xlim)
        _left.set_ylim(_left_ylim)
        _left.annotate(
            "", xy=(1.03, .76), xytext=(1.03, .55),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#6ab2e2", "linewidth": 1.5},
            annotation_clip=False, clip_on=False,
        )
        if _row == 1:
            _left.text(
                1.07, .83, "RRR better", transform=_left.transAxes,
                ha="center", va="center", rotation=90,
                color="#6ab2e2", fontsize=7, clip_on=False,
            )
        _left.annotate(
            "", xy=(1.03, .24), xytext=(1.03, .45),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#63bc51", "linewidth": 1.5},
            annotation_clip=False, clip_on=False,
        )
        if _row == 1:
            _left.text(
                1.07, .17, "PCA better", transform=_left.transAxes,
                ha="center", va="center", rotation=90,
                color="#63bc51", fontsize=7, clip_on=False,
            )
        _left.grid(axis="y", alpha=.2)
        _left.spines[["top", "right"]].set_visible(False)

        _right = _fig.add_subplot(_grid[:, _row + 1])
        _right.set_box_aspect(1)
        _pair_points = _aggregate_points[[_x_name, _y_name]].dropna()
        if not _pair_points.empty:
            _right.scatter(
                _pair_points[_x_name], _pair_points[_y_name],
                color="#2563eb", s=14, alpha=.38,
                edgecolors="white", linewidths=.25,
            )
            _right.plot(_right_limits, _right_limits, color="#94a3b8", linewidth=1, linestyle="--")
            if len(_pair_points) >= 2 and _pair_points[_x_name].nunique() >= 2:
                _slope, _intercept = np.polyfit(_pair_points[_x_name], _pair_points[_y_name], 1)
                _line_x = np.linspace(_pair_points[_x_name].min(), _pair_points[_x_name].max(), 100)
                _right.plot(_line_x, _slope * _line_x + _intercept, color="#111827", linewidth=2)
            if (
                len(_pair_points) >= 2
                and _pair_points[_x_name].nunique() > 1
                and _pair_points[_y_name].nunique() > 1
            ):
                _correlation_result = _pearsonr(
                    _pair_points[_x_name], _pair_points[_y_name]
                )
                _r = _correlation_result.statistic
                _p_value = _correlation_result.pvalue
            else:
                _r = _pair_points[_x_name].corr(_pair_points[_y_name])
                _p_value = np.nan
            _p_text = (
                "p<.001" if np.isfinite(_p_value) and _p_value < .001
                else f"p={_p_value:.3f}" if np.isfinite(_p_value)
                else "p=NA"
            )
            _right.set_title(
                f"{_x_name} vs {_y_name}\nPearson r = {_r:.2f}, {_p_text}\n"
                f"n = {len(_pair_points)}",
                fontsize=11,
            )
        else:
            _right.set_title(f"{_x_name} vs {_y_name}", fontsize=12)
            _right.text(.5, .5, "No selected model/layer pairs", ha="center", va="center", transform=_right.transAxes)
        _right.set_xlim(_right_limits)
        _right.set_ylim(_right_limits)
        _right.set_xlabel(f"{_x_name} relative AUC", fontsize=9)
        _right.set_ylabel(f"{_y_name} relative AUC", fontsize=9)
        _right.tick_params(axis="both", labelsize=8)
        _right.grid(alpha=.2)
        _right.spines[["top", "right"]].set_visible(False)
    _fig.text(
        .19, .95, "A. Relative-Difference Curves",
        ha="center", va="bottom", fontsize=15, weight="bold",
    )
    _fig.text(
        .72, .95, "B. Cross-Dataset AUC Correlations",
        ha="center", va="bottom", fontsize=15, weight="bold",
    )
    _fig.subplots_adjust(left=.07, right=.99, top=.82, bottom=.12)
    for _artist in _fig.findobj():
        if hasattr(_artist, "get_fontsize") and hasattr(_artist, "set_fontsize"):
            _artist.set_fontsize(_artist.get_fontsize() * 1.45)
    mo.vstack([
        mo.md("## Language decoding aggregate figure"),
        _fig,
        mo.md(
            "Offset 0 only. Left: internal dataset mean relative-difference curves for selected model/layers. "
            "Right: each point is one matched model/layer × feature; correlations use the same selected "
            "models and depths and require AUCs in all three datasets."
        ),
    ])
    return


@app.cell(hide_code=True)
def normalized_performance_curve():
    from src.utils.rrr.plot_metrics import (
        monotonic_normalized_curve as normalized_performance_curve,
    )

    return (normalized_performance_curve,)


@app.cell(hide_code=True)
def normalized_log_rank_auc():
    from src.utils.rrr.plot_metrics import normalized_log_rank_auc

    return (normalized_log_rank_auc,)


@app.cell(hide_code=True)
def _(interpolated_saturation_rank, normalized_log_rank_auc, np, pd):
    def fisher_mean(values):
        """Average correlations in Fisher-z space and return a correlation."""
        values = np.asarray(values, dtype=float)
        return float(np.tanh(np.nanmean(np.arctanh(np.clip(values, -1 + 1e-7, 1 - 1e-7)))))


    def mean_rank_correlations(frame):
        """Average fold/target summaries at each rank in Fisher-z space."""
        if "mean_fisher_z" in frame:
            scores = frame.groupby("rank", as_index=False)["mean_fisher_z"].mean()
            scores["correlation"] = np.tanh(scores["mean_fisher_z"])
        else:
            scores = frame.groupby("rank", as_index=False)["correlation"].agg(fisher_mean)
        return scores.sort_values("rank")


    def saturation_rank(ranks, correlations, fraction=1.0):
        """First interpolated rank reaching a fraction of peak correlation."""
        scores = mean_rank_correlations(pd.DataFrame({"rank": ranks, "correlation": correlations}))
        return interpolated_saturation_rank(
            scores["rank"], scores["correlation"], fraction
        )


    def summarize_rank_curves(curves, keys, *, metric, fraction=1.0):
        """Summarize each curve before pairing or averaging summary statistics."""
        rows = []
        for identity, frame in curves.groupby(keys, dropna=False, sort=True):
            scores = mean_rank_correlations(frame)
            value = (saturation_rank(scores["rank"], scores["correlation"], fraction)
                     if metric == "rank" else normalized_log_rank_auc(scores["rank"], scores["correlation"]))
            if np.isfinite(value):
                rows.append(dict(zip(keys, identity)) | {metric: value})
        return pd.DataFrame(rows, columns=[*keys, metric])

    return fisher_mean, mean_rank_correlations, summarize_rank_curves


@app.cell(hide_code=True)
def saturation_interpolation(np):
    def interpolated_saturation_rank(ranks, correlations, fraction):
        """Linearly interpolate the first threshold crossing in rank coordinates."""
        if not 0 < fraction <= 1:
            raise ValueError("Saturation fraction must be in (0, 1].")
        ranks = np.asarray(ranks, dtype=float)
        correlations = np.asarray(correlations, dtype=float)
        finite = np.isfinite(ranks) & np.isfinite(correlations)
        ranks, correlations = ranks[finite], correlations[finite]
        if not len(ranks):
            return np.nan
        order = np.argsort(ranks)
        ranks, correlations = ranks[order], correlations[order]
        threshold = fraction * correlations.max()
        reached = np.flatnonzero(correlations >= threshold)
        if not len(reached):
            return np.nan
        index = int(reached[0])
        if index == 0:
            return float(ranks[0])
        left, right = correlations[index - 1], correlations[index]
        if right == left:
            return float(ranks[index])
        weight = (threshold - left) / (right - left)
        return float(ranks[index - 1] + weight * (ranks[index] - ranks[index - 1]))


    return (interpolated_saturation_rank,)


@app.cell(hide_code=True)
def quantile_saturation_helpers(
    Artifact,
    DATASET_ROOTS,
    FMRI_INPUT_SCALING_VARIANT,
    REPO_ROOT,
    internal_rrr_artifact,
    load_rrr_target_performances,
    lru_cache,
    np,
    pd,
):
    QUANTILE_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    QUANTILE_PEAK_FLOOR = 0.005


    def quantile_saturation_rank(rank_quantiles, fraction, min_peak=QUANTILE_PEAK_FLOOR):
        """First rank where all quantiles above the peak floor crossed their threshold."""
        from src.utils.rrr.plot_metrics import quantile_saturation_rank as calculate

        return calculate(
            rank_quantiles.index.to_numpy(dtype=float),
            rank_quantiles.to_numpy(dtype=float), fraction, min_peak,
        )


    def mean_target_correlations(frame, keys=("rank", "target")):
        """Average repeated target observations in Fisher-z space."""
        values = frame[[*keys, "correlation"]].copy()
        values["fisher_z"] = np.arctanh(np.clip(
            values.pop("correlation").to_numpy(dtype=float), -1 + 1e-7, 1 - 1e-7,
        ))
        result = values.groupby(list(keys), dropna=False, sort=False, as_index=False)[
            "fisher_z"
        ].mean()
        result["correlation"] = np.tanh(result.pop("fisher_z"))
        return result


    def calculate_quantile_saturation_ranks(correlations, *, keys, fraction=1.0, quantiles=QUANTILE_LEVELS):
        """Average each target over folds, then require every eligible quantile to saturate.

        ``keys`` identify independent curves. A target is one electrode or voxel,
        with LOSO subjects kept separate when ``subject`` is included in ``keys``.
        """
        if not quantiles or any(not 0 < q < 1 for q in quantiles):
            raise ValueError("Quantiles must be a non-empty sequence in (0, 1).")
        required = set(keys) | {"rank", "target", "correlation"}
        missing = required.difference(correlations.columns)
        if missing:
            raise ValueError(f"Missing target correlation columns: {sorted(missing)}")
        if correlations.empty:
            return pd.DataFrame(columns=[*keys, "rank"])
        target_means = mean_target_correlations(correlations, [*keys, "rank", "target"])
        rows = []
        groups = target_means.groupby(keys, dropna=False, sort=True) if keys else [((), target_means)]
        for identity, frame in groups:
            by_rank = frame.groupby("rank")["correlation"].quantile(quantiles).unstack()
            rank = quantile_saturation_rank(by_rank, fraction)
            if np.isfinite(rank):
                identity = identity if isinstance(identity, tuple) else (identity,)
                rows.append(dict(zip(keys, identity)) | {"rank": rank})
        return pd.DataFrame(rows, columns=[*keys, "rank"])


    def target_metric_frame(artifact, *, method, metric_name, dataset):
        """Read one artifact method as electrode or voxel correlations by rank and fold."""
        manifest = next(
            (value for selector, value in artifact.target_manifest.items()
             if dict(selector).get("method") == method),
            artifact.target_manifest.get(()),
        )
        if manifest is None:
            raise ValueError(f"Missing target manifest for {dataset}/{method}")
        if dataset == "fmri":
            groups = None
            target_ids = np.arange(len(manifest))
        else:
            groups = manifest.groupby(["subject", "electrode"], sort=False).ngroup().to_numpy()
            target_ids = np.arange(int(groups.max()) + 1)
            counts = np.bincount(groups, minlength=len(target_ids))
        frames = []
        for selector, metrics in artifact.rank_metrics.items():
            selector = dict(selector)
            if selector.get("method") != method or "rank" not in selector or metric_name not in metrics:
                continue
            values = np.asarray(metrics[metric_name], dtype=float).reshape(-1)
            if len(values) != len(manifest):
                raise ValueError(f"Target metric and manifest length differ for {dataset}/{method}")
            if groups is not None:
                z = np.arctanh(np.clip(values, -1 + 1e-7, 1 - 1e-7))
                values = np.tanh(np.bincount(groups, weights=z, minlength=len(target_ids)) / counts)
            frames.append(pd.DataFrame({
                "rank": int(selector["rank"]), "fold": selector.get("fold", 0),
                "target": target_ids, "correlation": values,
            }))
        if not frames:
            raise ValueError(f"No target correlations for {dataset}/{method}")
        return pd.concat(frames, ignore_index=True)


    def rank_quantiles_from_targets(frame, quantiles=QUANTILE_LEVELS):
        """Compact target observations into one quantile curve per rank."""
        means = mean_target_correlations(frame)
        return means.groupby("rank")["correlation"].quantile(quantiles).unstack()


    @lru_cache(maxsize=None)
    def internal_rank_quantiles(dataset, model, layer, method):
        frame = load_rrr_target_performances(
            dataset=dataset, model=model, layer=layer, method=method,
        )
        return rank_quantiles_from_targets(frame)


    def internal_quantile_saturation(dataset, selected_curves, fraction):
        """Calculate quantile ranks for selected internal model/layer curves."""
        rows = []
        pairs = selected_curves[["model", "layer", "method"]].drop_duplicates()
        for item in pairs.to_dict("records"):
            model, layer, method = item["model"], item["layer"], item["method"]
            layer = None if pd.isna(layer) else layer
            if method not in ("rrr", "pca"):
                continue
            rank = quantile_saturation_rank(
                internal_rank_quantiles(dataset, model, layer, method), fraction,
            )
            if np.isfinite(rank):
                rows.append({"model": model, "layer": layer, "method": method,
                             "rank": rank})
        return pd.DataFrame(rows, columns=["model", "layer", "method", "rank"])


    @lru_cache(maxsize=None)
    def loso_rank_quantiles(dataset, model, layer, baseline_source):
        """Reduce each LOSO artifact directly to subject/method quantile curves."""
        artifact = Artifact(
            mode="leave_one_subject_out", dataset=DATASET_ROOTS[dataset],
            model=model, layer=layer, results_root=REPO_ROOT / "results",
        ).load_artifacts(["rank_metrics", "target_manifest"])
        labels = {
            ("rrr", "baseline"): "RRR baseline",
            ("rrr", "transfer"): "RRR transfer",
            ("pca", "baseline"): "PCA baseline",
        }
        metrics = {
            ("rrr", "baseline"): "corr_per_target" if dataset == "fmri" else "nr_orth_corr_per_target",
            ("rrr", "transfer"): "corr_per_target" if dataset == "fmri" else "transfer_orth_corr_per_target",
            ("pca", "baseline"): "corr_per_target" if dataset == "fmri" else "pca_corr_per_target",
        }
        manifests = {
            (str(dict(selector).get("held_out_subject")),
             dict(selector).get("method"), dict(selector).get("loso_phase")): value
            for selector, value in artifact.target_manifest.items()
        }
        grouped_lags = {}
        folded = {}
        for selector, values in artifact.rank_metrics.items():
            selector = dict(selector)
            phase = (selector.get("method"), selector.get("loso_phase"))
            if phase not in labels or "rank" not in selector:
                continue
            if baseline_source == "full" and phase == ("rrr", "baseline"):
                continue
            metric = metrics[phase]
            if metric not in values:
                continue
            subject = str(selector["held_out_subject"])
            manifest_key = (subject, *phase)
            manifest = manifests.get(manifest_key)
            if manifest is None:
                raise ValueError(f"Missing LOSO target manifest for {dataset}/{model}/{layer}/{manifest_key}")
            raw = np.asarray(values[metric], dtype=float).reshape(-1)
            if len(raw) != len(manifest):
                raise ValueError(f"LOSO target metric and manifest length differ for {dataset}/{model}/{layer}")
            z = np.arctanh(np.clip(raw, -1 + 1e-7, 1 - 1e-7))
            if "electrode" in manifest.columns:
                if manifest_key not in grouped_lags:
                    groups = manifest.groupby(["subject", "electrode"], sort=False).ngroup().to_numpy()
                    counts = np.bincount(groups)
                    grouped_lags[manifest_key] = (groups, counts)
                groups, counts = grouped_lags[manifest_key]
                z = np.bincount(groups, weights=z, minlength=len(counts)) / counts
            folded.setdefault((subject, labels[phase], int(selector["rank"])), []).append(z)

        rows = {}
        for (subject, method, rank), fold_values in folded.items():
            mean_z = np.nanmean(np.stack(fold_values), axis=0)
            correlations = np.tanh(mean_z)
            if np.isfinite(correlations).any():
                rows.setdefault((subject, method), []).append(
                    (rank, *np.nanquantile(correlations, QUANTILE_LEVELS))
                )

        if baseline_source == "full":
            full = internal_rrr_artifact(
                dataset=dataset, model=model, layer=layer,
                results_root=REPO_ROOT / "results",
            ).load_artifacts(["rank_metrics", "target_manifest"])
            manifest = next(
                (value for selector, value in full.target_manifest.items()
                 if dict(selector).get("method") == "rrr"),
                full.target_manifest.get(()),
            )
            if manifest is None:
                raise ValueError(f"Missing internal RRR target manifest for {dataset}/{model}/{layer}")
            if dataset == "fmri":
                subjects = manifest["subject"].astype(str).to_numpy()
                lag_groups = None
            else:
                lag_groups = manifest.groupby(["subject", "electrode"], sort=False).ngroup().to_numpy()
                counts = np.bincount(lag_groups)
                subjects = manifest.drop_duplicates(["subject", "electrode"])["subject"].astype(str).to_numpy()
            metric = "corr_per_target" if dataset == "fmri" else "nr_orth_corr_per_target"
            full_folds = {}
            for selector, values in full.rank_metrics.items():
                selector = dict(selector)
                if selector.get("method") != "rrr" or "rank" not in selector or metric not in values:
                    continue
                raw = np.asarray(values[metric], dtype=float).reshape(-1)
                if len(raw) != len(manifest):
                    raise ValueError(f"Internal target metric and manifest length differ for {dataset}/{model}/{layer}")
                z = np.arctanh(np.clip(raw, -1 + 1e-7, 1 - 1e-7))
                if lag_groups is not None:
                    z = np.bincount(lag_groups, weights=z, minlength=len(counts)) / counts
                full_folds.setdefault(int(selector["rank"]), []).append(z)
            for rank, fold_values in full_folds.items():
                correlations = np.tanh(np.nanmean(np.stack(fold_values), axis=0))
                for subject in np.unique(subjects):
                    selected = correlations[subjects == subject]
                    if np.isfinite(selected).any():
                        rows.setdefault((subject, "RRR baseline"), []).append(
                            (rank, *np.nanquantile(selected, QUANTILE_LEVELS))
                        )

        return {
            key: pd.DataFrame(values, columns=["rank", *QUANTILE_LEVELS]).set_index("rank").sort_index()
            for key, values in rows.items()
        }


    def loso_quantile_saturation(dataset, selected_curves, fraction, baseline_source="loso"):
        """Calculate one saturation rank per held-out subject and method."""
        rows = []
        for (model, layer), _ in selected_curves.groupby(["model", "layer"], dropna=False):
            layer = None if pd.isna(layer) else layer
            for (subject, method), curve in loso_rank_quantiles(
                dataset, model, layer, baseline_source,
            ).items():
                rank = quantile_saturation_rank(curve, fraction)
                if np.isfinite(rank):
                    rows.append({"model": model, "layer": layer, "subject": subject,
                                 "method": method, "rank": rank})
        if not rows:
            return pd.DataFrame(columns=["model", "layer", "subject", "method", "rank"])
        values = pd.DataFrame(rows)
        count = values.groupby(["model", "layer", "subject"], dropna=False)["method"].transform("nunique")
        return values[count == 3].reset_index(drop=True)


    def cross_transfer_target_frame(*, target_dataset, source_dataset, model, layer, method):
        """Read target-level transfer correlations for one directed pair and method."""
        base = Artifact(
            mode="cross_dataset", dataset=DATASET_ROOTS[target_dataset],
            model=model, layer=layer, results_root=REPO_ROOT / "results",
        )
        index = base._read_json(base._leaf() / "manifest.json", default={})
        variants = []
        for variant in sorted(index.get("variants", {})):
            parts = variant.split("/")
            if target_dataset == "fmri" and FMRI_INPUT_SCALING_VARIANT not in parts:
                continue
            if len(parts) > 2 and parts[0] == "source" and parts[2] == DATASET_ROOTS[source_dataset]:
                variants.append(variant)
        if not variants:
            raise ValueError(f"No {source_dataset} → {target_dataset} transfer variant for {model}/{layer}")
        frames = []
        for variant in variants:
            artifact = Artifact(
                mode="cross_dataset", dataset=DATASET_ROOTS[target_dataset],
                model=model, layer=layer, variant=variant, results_root=REPO_ROOT / "results",
            ).load_artifacts(["rank_metrics", "target_manifest"])
            if target_dataset == "fmri":
                metric_name = "corr_per_target"
            elif method == "pca":
                metric_name = "pca_corr_per_target"
            else:
                available = {name for _, metrics in artifact.rank_metrics.items() for name in metrics}
                metric_name = ("nr_orth_corr_per_target" if "nr_orth_corr_per_target" in available
                               else "nr_corr_per_target")
            frames.append(target_metric_frame(
                artifact, method=method, metric_name=metric_name, dataset=target_dataset,
            ))
        return pd.concat(frames, ignore_index=True)


    @lru_cache(maxsize=None)
    def cross_rank_quantiles(target, source, model, layer, display_method):
        method = "rrr" if display_method.endswith("RRR") else "pca"
        if display_method.startswith("Within"):
            return internal_rank_quantiles(target, model, layer, method)
        frame = cross_transfer_target_frame(
            target_dataset=target, source_dataset=source,
            model=model, layer=layer, method=method,
        )
        return rank_quantiles_from_targets(frame)


    def cross_quantile_saturation(selected_curves, fraction):
        """Calculate target-level quantile ranks for all selected transfer directions."""
        rows = []
        identities = selected_curves[[
            "target_dataset", "source_dataset", "model", "layer", "method"
        ]].drop_duplicates()
        for item in identities.to_dict("records"):
            target, source = item["target_dataset"], item["source_dataset"]
            model, layer, display_method = item["model"], item["layer"], item["method"]
            layer = None if pd.isna(layer) else layer
            if display_method not in ("Within RRR", "Within PCA", "Transfer RRR", "Transfer PCA"):
                continue
            rank = quantile_saturation_rank(
                cross_rank_quantiles(target, source, model, layer, display_method), fraction,
            )
            if np.isfinite(rank):
                rows.append({
                    "target_dataset": target, "source_dataset": source,
                    "model": model, "layer": layer, "method": display_method,
                    "rank": rank,
                })
        return pd.DataFrame(rows, columns=[
            "target_dataset", "source_dataset", "model", "layer", "method", "rank",
        ])


    return (
        QUANTILE_LEVELS,
        QUANTILE_PEAK_FLOOR,
        cross_quantile_saturation,
        internal_quantile_saturation,
        internal_rank_quantiles,
        loso_quantile_saturation,
        mean_target_correlations,
        quantile_saturation_rank,
    )


@app.cell(hide_code=True)
def color_controls(mo):
    color_by = mo.ui.radio(
        options={
            "Same color": "none",
            "Model type": "model",
            "Layer depth": "layer",
        },
        value="Same color",
        label="Color points by",
    )
    color_by
    return


@app.cell(hide_code=True)
def plot_saturation_distributions(np, plot_paired_distributions):
    METHOD_COLORS = {"RRR": "#6ab2e2", "PCA": "#63bc51"}


    def plot_saturation_distributions(saturation_ranks, dataset_label):
        """Plot paired RRR/PCA saturation-rank distributions."""
        paired = (
            saturation_ranks.pivot_table(
                index=["model", "layer"],
                columns="method",
                values="rank",
                aggfunc="first",
            )
            .dropna(subset=["rrr", "pca"])
            .reset_index()
        )
        if paired.empty:
            raise ValueError("no paired RRR/PCA saturation ranks")

        methods = {"RRR": "rrr", "PCA": "pca"}
        distributions = {
            label: np.log2(paired[method].to_numpy(dtype=float))
            for label, method in methods.items()
        }
        fig, ax = plot_paired_distributions(
            distributions,
            colors=METHOD_COLORS,
            figsize=(4.2, 4.4),
        )

        positions = np.arange(1, len(methods) + 1, dtype=float)
        for position, method in zip(positions, methods.values()):
            mean_rank = float(paired[method].mean())
            ax.scatter(
                position,
                np.log2(mean_rank),
                marker="D",
                s=28,
                color="black",
                zorder=5,
            )
            ax.annotate(
                f"mean = {mean_rank:.1f}",
                (position, np.log2(mean_rank)),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        max_exponent = max(
            0,
            int(np.ceil(np.log2(paired[list(methods.values())].to_numpy().max()))),
        )
        integer_ticks = np.arange(max_exponent + 1)
        ax.set_yticks(integer_ticks, [str(2**value) for value in integer_ticks])
        ax.set(
            ylabel="Saturation rank $k$",
            title=(
                f"{dataset_label}: RRR vs PCA saturation rank\n"
                f"({paired['model'].nunique()} models, {len(paired)} paired layers)"
            ),
            ylim=(-0.35, max_exponent + 0.35),
        )
        fig.tight_layout()
        return fig, ax


    return (METHOD_COLORS,)


@app.cell(hide_code=True)
def _(
    annotate_pairwise_tests,
    np,
    paired_permutation_tests,
    plot_paired_distributions,
):
    def plot_rank_summary_distributions(values, dataset_label, methods, colors, *, metric, percentage=None, tests=None, ax=None, compact_annotations=False):
        """Pair model layers, plot summaries, and annotate requested tests."""
        frame = values.copy()
        frame["layer_key"] = frame["layer"].astype(object).where(frame["layer"].notna(), "none")
        paired = frame.pivot_table(
            index=["model", "layer_key"], columns="method", values=metric, aggfunc="mean"
        ).reindex(columns=list(methods.values())).dropna()
        if paired.empty:
            raise ValueError(f"No complete paired {metric} values for {dataset_label}")
        distributions = {
            label: (np.log2(paired[method].to_numpy(dtype=float)) if metric == "rank"
                    else paired[method].to_numpy(dtype=float))
            for label, method in methods.items()
        }
        _provided_ax = ax is not None
        fig, ax = plot_paired_distributions(distributions, colors=colors, ax=ax)
        for position, method in enumerate(methods.values(), start=1):
            mean_value = float(paired[method].mean())
            height = np.log2(mean_value) if metric == "rank" else mean_value
            ax.scatter(position, height, marker="D", s=28, color="black", zorder=5)
            ax.annotate(
                rf"$\mu$ = {mean_value:.2f}",
                (position, height),
                xytext=(0, 8 + 12 * (position % 2)),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        if metric == "rank":
            maximum = int(np.ceil(np.log2(paired.to_numpy().max())))
            ticks = np.arange(maximum + 1)
            ax.set_yticks(ticks, [str(2**value) for value in ticks])
            if compact_annotations and _provided_ax:
                _current_upper = ax.get_ylim()[1]
                _annotation_headroom = 2.0 if tests and len(tests) > 1 else 1.0
                ax.set_ylim(-0.35, max(_current_upper, maximum + _annotation_headroom))
            else:
                ax.set_ylim(-0.35, maximum + 0.5)
            summary_label = f"{percentage:g}% saturation rank"
            ax.set_ylabel("Saturation rank $k$")
        else:
            summary_label = "normalized curve AUC"
            ax.set_ylabel("Normalized curve AUC")
            ax.margins(y=0.15)
        if tests:
            results = paired_permutation_tests(distributions, tests)
            annotate_pairwise_tests(ax, results, list(distributions), compact=compact_annotations)
        ax.set_title(
            f"{dataset_label}: {summary_label} "
            f"({paired.index.get_level_values('model').nunique()} models, {len(paired)} paired layers)"
        )
        if not _provided_ax:
            fig.tight_layout()
        return fig, ax



    return (plot_rank_summary_distributions,)


@app.cell(hide_code=True)
def plot_average_normalized_curves(
    mean_rank_correlations,
    normalized_performance_curve,
    np,
    plt,
):
    def plot_average_normalized_curves(
        correlations, dataset_label, aggregation_label="model layers", grid_size=200
    ):
        """Plot model/layer curves and their equal-weighted interpolated means."""
        method_styles = {
            "rrr": ("#6ab2e2", "RRR"),
            "pca": ("#63bc51", "PCA"),
        }
        curves = {}
        for (model, layer, method), frame in correlations.groupby(
            ["model", "layer", "method"], dropna=False, sort=True
        ):
            scores = mean_rank_correlations(frame)
            x, y = normalized_performance_curve(
                scores["rank"], scores["correlation"]
            )
            if x.size:
                curves[(model, layer, method)] = (x, y)
        if not curves:
            raise ValueError(f"No usable normalized curves for {dataset_label}")

        linthresh = 0.01
        positive_grid = np.geomspace(linthresh / 100, 1.0, grid_size - 1)
        grid = np.concatenate(([0.0], positive_grid))
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        contributor_counts = []
        for method, (color, label) in method_styles.items():
            method_curves = [
                curve for key, curve in curves.items() if key[2] == method
            ]
            if not method_curves:
                continue
            interpolated = np.empty((len(method_curves), grid.size))
            for index, (x, y) in enumerate(method_curves):
                ax.plot(x, y, color=color, alpha=0.10, linewidth=0.8, zorder=1)
                interpolated[index] = np.interp(grid, x, y)
            mean_curve = interpolated.mean(axis=0)
            ax.plot(grid, mean_curve, color=color, linewidth=2.6, label=label, zorder=3)
            contributor_counts.append(f"{label}: {len(method_curves)}")

        ax.set_xscale("symlog", linthresh=linthresh, linscale=1.0)
        percentage_ticks = [0.0, 0.001, 0.01, 0.1, 1.0]
        ax.set_xticks(
            percentage_ticks,
            ["0%", "0.1%", "1%", "10%", "100%"],
        )
        ax.set(
            xlim=(0.0, 1.0),
            ylim=(-0.02, 1.02),
            xlabel="Relative rank",
            ylabel="Normalized monotonic performance",
            title=f"{dataset_label}: mean convergence across {aggregation_label}",
        )
        ax.grid(axis="both", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        ax.text(
            0.02, 0.03,
            "Curves contributing throughout: " + "; ".join(contributor_counts),
            transform=ax.transAxes, fontsize=8, color="#4b5563",
        )
        fig.tight_layout()
        return fig, ax

    return (plot_average_normalized_curves,)


@app.cell(hide_code=True)
def rrr_pca_gap_helpers(np, pd, plt):
    def plot_rrr_pca_gap_groups(
        frame,
        method_pairs,
        grouping,
        dataset_label,
        *,
        title_suffix="",
    ):
        """Plot member and grouped mean RRR-minus-PCA rank curves."""
        _architectural_max_layers = {
            "gpt2": 12,
            "random-gpt2": 12,
            "gpt2-xl-batch": 48,
            "gpt-j-6b": 28,
            "llama3-8b": 32,
            "roberta-large": 24,
        }
        _work = frame.copy()
        if "mean_fisher_z" in _work:
            _work = (
                _work.groupby(
                    ["model", "layer", "method", "rank"],
                    as_index=False,
                    dropna=False,
                )["mean_fisher_z"]
                .mean()
            )
            _work["correlation"] = np.tanh(_work["mean_fisher_z"])
        else:
            _work = (
                _work.groupby(
                    ["model", "layer", "method", "rank"],
                    as_index=False,
                    dropna=False,
                )["correlation"]
                .mean()
            )

        _group_curves = {}
        for _rrr_method, _pca_method, _pair_label in method_pairs:
            _paired = _work[_work["method"].isin([_rrr_method, _pca_method])].pivot_table(
                index=["model", "layer", "rank"],
                columns="method",
                values="correlation",
                aggfunc="mean",
            ).dropna(subset=[_rrr_method, _pca_method])
            if _paired.empty:
                continue
            _paired["gap"] = _paired[_rrr_method] - _paired[_pca_method]
            _paired = _paired.reset_index()
            for (_model, _layer), _member in _paired.groupby(
                ["model", "layer"], dropna=False, sort=True
            ):
                if grouping == "depth":
                    if pd.isna(_layer):
                        continue
                    _max_layer = _architectural_max_layers.get(_model)
                    if _max_layer is None:
                        _max_layer = _paired.loc[
                            _paired["model"] == _model, "layer"
                        ].max()
                    _group = f"{round(100 * _layer / _max_layer):g}% depth"
                elif grouping == "model":
                    _group = str(_model)
                else:
                    raise ValueError(f"Unknown RRR/PCA gap grouping: {grouping}")
                _group_curves.setdefault((_group, _pair_label), []).append(
                    _member[["rank", "gap"]].sort_values("rank")
                )

        if not _group_curves:
            raise ValueError(f"No paired RRR/PCA curves available for {dataset_label}")

        _groups = sorted({group for group, _ in _group_curves})
        _group_sort = {
            group: int(group.removesuffix("% depth"))
            if group.endswith("% depth")
            else group
            for group in _groups
        }
        _groups = sorted(_groups, key=lambda group: _group_sort[group])
        _colors = plt.get_cmap(
            "viridis" if grouping == "depth" else "tab20",
            max(len(_groups), 1),
        )
        _color_by_group = {
            group: _colors(index) for index, group in enumerate(_groups)
        }
        _linestyles = ["-", "--", ":"]
        _pair_labels = [pair[2] for pair in method_pairs]
        _linestyle_by_pair = {
            label: _linestyles[index % len(_linestyles)]
            for index, label in enumerate(_pair_labels)
        }

        _fig, _ax = plt.subplots(figsize=(9.2, 5.4))
        _legend_handles = []
        _legend_labels = []
        for (_group, _pair_label), _members in sorted(
            _group_curves.items(),
            key=lambda item: (_groups.index(item[0][0]), _pair_labels.index(item[0][1])),
        ):
            _color = _color_by_group[_group]
            _style = _linestyle_by_pair[_pair_label]
            for _member in _members:
                _ax.plot(
                    _member["rank"],
                    _member["gap"],
                    color=_color,
                    linestyle=_style,
                    alpha=0.16,
                    linewidth=1.0,
                    zorder=1,
                )
            _long = pd.concat(
                [
                    _member.assign(member_index=_index)
                    for _index, _member in enumerate(_members)
                ],
                ignore_index=True,
            )
            _mean = _long.groupby("rank", as_index=False)["gap"].mean()
            _line = _ax.plot(
                _mean["rank"],
                _mean["gap"],
                color=_color,
                linestyle=_style,
                linewidth=2.6,
                label=f"{_group} - {_pair_label}",
                zorder=3,
            )[0]
            _legend_handles.append(_line)
            _legend_labels.append(f"{_group} - {_pair_label}")

        _ax.axhline(0, color="#111827", linewidth=0.9, linestyle="--", zorder=2)
        _ax.set(
            xlabel="Rank $k$",
            ylabel="Mean RRR - PCA correlation",
            title=f"{dataset_label}: RRR - PCA by {('model depth' if grouping == 'depth' else 'model')}{title_suffix}",
            xscale="log",
        )
        _ax.grid(axis="y", alpha=0.25)
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.legend(
            _legend_handles,
            _legend_labels,
            frameon=False,
            fontsize=8,
            ncol=2 if len(_legend_labels) > 5 else 1,
        )
        _fig.tight_layout()
        return _fig, _ax


    return (plot_rrr_pca_gap_groups,)


@app.cell(hide_code=True)
def normalized_curve_filter_helpers():
    def filter_normalized_curves(curves, mode, model, depth):
        """Select the members averaged by a normalized-curve plot."""
        if mode == "all":
            return curves, "model layers"
        if mode == "model":
            selected = curves[curves["model"] == model]
            label = f"layers of {model}"
        elif mode == "depth":
            _architectural_max_layers = {
                "gpt2": 12,
                "random-gpt2": 12,
                "gpt2-xl-batch": 48,
                "gpt-j-6b": 28,
                "llama3-8b": 32,
                "roberta-large": 24,
            }
            _observed_max_layers = curves.groupby("model")["layer"].transform("max")
            max_layers = curves["model"].map(_architectural_max_layers).fillna(
                _observed_max_layers
            )
            percentage_depth = (100 * curves["layer"] / max_layers).round()
            selected = curves[percentage_depth.eq(depth)]
            label = f"models at {depth}% depth"
        else:
            raise ValueError(f"Unknown normalized-curve filter mode: {mode}")
        if selected.empty:
            raise ValueError(f"No normalized curves match {label}")
        return selected, label


    def normalized_curve_filter_controls(mo, model_layers, label):
        models = sorted({model for model, _ in model_layers})
        _architectural_max_layers = {
            "gpt2": 12,
            "random-gpt2": 12,
            "gpt2-xl-batch": 48,
            "gpt-j-6b": 28,
            "llama3-8b": 32,
            "roberta-large": 24,
        }
        max_layer_by_model = {
            model: _architectural_max_layers.get(
                model,
                max(
                    layer
                    for candidate_model, layer in model_layers
                    if candidate_model == model and layer is not None
                ),
            )
            for model in models
            if any(
                candidate_model == model and layer is not None and layer > 0
                for candidate_model, layer in model_layers
            )
        }
        depths = sorted({
            round(100 * layer / max_layer_by_model[model])
            for model, layer in model_layers
            if model in max_layer_by_model and layer is not None
        })
        if not models or not depths:
            raise ValueError(f"No model/depth choices are available for {label}")
        mode = mo.ui.radio(
            options={
                "All model layers": "all",
                "One model (across layers)": "model",
                "One depth (across models)": "depth",
            },
            value="All model layers",
            label=f"{label} averaging mode",
        )
        model = mo.ui.dropdown(models, value=models[0], label="Model")
        depth_options = {f"{depth}% depth": depth for depth in depths}
        depth = mo.ui.dropdown(
            depth_options, value=next(iter(depth_options)), label="Model depth"
        )
        return mode, model, depth, mo.hstack([mode, model, depth], wrap=True)

    return filter_normalized_curves, normalized_curve_filter_controls


@app.cell(hide_code=True)
def _(mo):
    internal_saturation_percent = mo.ui.slider(
        start=1, stop=100, step=1, value=99, show_value=True,
        label="Internal: percent of maximum correlation", debounce=True,
    )
    internal_saturation_percent
    return (internal_saturation_percent,)


@app.cell(hide_code=True)
def rrr_saturation_plot(
    METHOD_COLORS,
    QUANTILE_PEAK_FLOOR,
    filter_normalized_curves,
    internal_curve_depth,
    internal_curve_mode,
    internal_curve_model,
    internal_quantile_saturation,
    internal_saturation_percent,
    internal_summary_curves,
    mo,
    plot_rank_summary_distributions,
):
    def load_dataset_saturation(dataset, fraction=1.0):
        """Summarize target quantiles for the selected internal curves."""
        selected, _ = filter_normalized_curves(
            internal_summary_curves[dataset], internal_curve_mode.value,
            internal_curve_model.value, internal_curve_depth.value,
        )
        return internal_quantile_saturation(dataset, selected, fraction)

    internal_saturation_by_dataset = {}
    plots = []
    for _dataset, _label in (("podcast", "Podcast"), ("cars-2", "Cars 2"), ("fmri", "fMRI")):
        _values = load_dataset_saturation(_dataset, internal_saturation_percent.value / 100)
        internal_saturation_by_dataset[_dataset] = _values
        plots.append(plot_rank_summary_distributions(
            _values, _label + ": internal", {"RRR": "rrr", "PCA": "pca"}, METHOD_COLORS,
            metric="rank", percentage=internal_saturation_percent.value,
            tests=[{"first": "RRR", "second": "PCA", "alternative": "less"}],
        ))
    mo.vstack([mo.md(f"All target quantiles peaking above {QUANTILE_PEAK_FLOOR:g} must reach the selected percentage of their own maximum; the displayed rank is the latest interpolated crossing."), *[_plot[0] for _plot in plots]])
    return


@app.cell(hide_code=True)
def _(find_rrr_layers, find_rrr_models, load_rrr_correlations, pd):
    def load_dataset_normalized_curves(dataset):
        """Load all available non-random model/layer curves for one dataset."""
        frames = []
        for model in find_rrr_models(dataset=dataset):
            if model == "random-gpt2":
                continue
            for layer in find_rrr_layers(dataset=dataset, model=model):
                try:
                    frame = load_rrr_correlations(
                        dataset=dataset, model=model, layers=[layer]
                    )
                except Exception as error:
                    print(f"Skipping {dataset}/{model}/layer-{layer}: {error}")
                    continue
                if not frame.empty:
                    frames.append(frame)
        if not frames:
            raise ValueError(f"No usable RRR/PCA curves for {dataset}")
        return pd.concat(frames, ignore_index=True).groupby(
            ["dataset", "model", "layer", "method", "rank"], as_index=False, dropna=False
        )["mean_fisher_z"].mean()


    return (load_dataset_normalized_curves,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### In-memory plot data

    Internal summaries retain mean Fisher-z scores per model/layer/method/rank.
    LOSO retains subject-level mean correlations; cross-dataset retains fold-level
    mean correlations. Saturation helpers cache compact target-quantile curves by
    model/layer/method, while target arrays are discarded after aggregation.
    Convergence filters, model/layer selectors in LOSO and cross-dataset plots,
    saturation sliders, and the LOSO baseline selector operate on these loaded tables.

    To pick up newly written results, clear `load_loso_rank_correlations.cache_clear()`,
    `load_cross_dataset_rank_correlations.cache_clear()`, and
    `load_internal_rrr_subject_correlations.cache_clear()`, plus
    `internal_rank_quantiles.cache_clear()`, `loso_rank_quantiles.cache_clear()`, and
    `cross_rank_quantiles.cache_clear()`, then rerun the data-loading cells (or restart
    the notebook). Target-level distribution plots still load the selected artifact
    because their violins require individual target values.
    """)
    return


@app.cell(hide_code=True)
def _(load_dataset_normalized_curves, plot_model_filter):
    internal_summary_curves = {
        _dataset: load_dataset_normalized_curves(_dataset)
        for _dataset in ("podcast", "cars-2", "fmri")
    }
    internal_summary_curves = {
        _dataset: _frame[_frame["model"].isin(plot_model_filter.value)].copy()
        for _dataset, _frame in internal_summary_curves.items()
    }
    return (internal_summary_curves,)


@app.cell(hide_code=True)
def _(
    METHOD_COLORS,
    filter_normalized_curves,
    internal_curve_depth,
    internal_curve_mode,
    internal_curve_model,
    internal_summary_curves,
    mean_rank_correlations,
    mo,
    normalized_log_rank_auc,
    np,
    pd,
    plot_rank_summary_distributions,
):
    internal_auc_by_dataset = {}
    internal_auc_plots = []
    for _dataset, _label in (("podcast", "Podcast"), ("cars-2", "Cars 2"), ("fmri", "fMRI")):
        _rows = []
        for (_model, _layer, _method), _frame in internal_summary_curves[_dataset].groupby(
            ["model", "layer", "method"], dropna=False
        ):
            _scores = mean_rank_correlations(_frame)
            _auc = normalized_log_rank_auc(_scores["rank"], _scores["correlation"])
            if np.isfinite(_auc):
                _rows.append({"model": _model, "layer": _layer, "method": _method, "auc": _auc})
        _values = pd.DataFrame(_rows)
        _values, _ = filter_normalized_curves(
            _values, internal_curve_mode.value, internal_curve_model.value, internal_curve_depth.value
        )
        internal_auc_by_dataset[_dataset] = _values
        internal_auc_plots.append(plot_rank_summary_distributions(
            internal_auc_by_dataset[_dataset], _label + ": internal", {"RRR": "rrr", "PCA": "pca"},
            METHOD_COLORS, metric="auc",
            tests=[{"first": "RRR", "second": "PCA", "alternative": "greater"}],
        ))
    mo.vstack([
        mo.md("### Internal RRR/PCA AUC"),
        *[_plot[0] for _plot in internal_auc_plots],
    ])
    return


@app.cell(hide_code=True)
def internal_normalized_curve_controls(
    find_rrr_layers,
    find_rrr_models,
    mo,
    normalized_curve_filter_controls,
):
    _datasets = ("podcast", "cars-2", "fmri")
    _internal_sets = [
        {
            (model, layer)
            for model in find_rrr_models(dataset=dataset)
            if model != "random-gpt2"
            for layer in find_rrr_layers(dataset=dataset, model=model)
        }
        for dataset in _datasets
    ]
    _internal_pairs = set.intersection(*_internal_sets)
    (
        internal_curve_mode,
        internal_curve_model,
        internal_curve_depth,
        _internal_controls,
    ) = normalized_curve_filter_controls(mo, _internal_pairs, "Internal RRR")
    _internal_controls
    return internal_curve_depth, internal_curve_mode, internal_curve_model


@app.cell(hide_code=True)
def internal_gap_controls(mo):
    internal_gap_grouping = mo.ui.radio(
        options={
            "Model depth (one line per depth)": "depth",
            "Model (average across layers)": "model",
        },
        value="Model depth (one line per depth)",
        label="RRR - PCA grouping",
        inline=True,
    )
    internal_gap_grouping
    return (internal_gap_grouping,)


@app.cell(hide_code=True)
def internal_rrr_pca_gap_plot(
    internal_gap_grouping,
    internal_summary_curves,
    mo,
    plot_rrr_pca_gap_groups,
):
    internal_rrr_pca_gap_plots = []
    for _dataset, _label in (
        ("podcast", "Podcast"),
        ("cars-2", "Cars 2"),
        ("fmri", "fMRI"),
    ):
        internal_rrr_pca_gap_plots.append(
            plot_rrr_pca_gap_groups(
                internal_summary_curves[_dataset],
                [("rrr", "pca", "RRR - PCA")],
                internal_gap_grouping.value,
                _label,
            )
        )
    mo.vstack([
        mo.md(
            "Faint lines are individual model-layer RRR - PCA curves; thick lines "
            "are means within each depth or model group."
        ),
        *[plot[0] for plot in internal_rrr_pca_gap_plots],
    ])
    return


@app.cell(hide_code=True)
def rrr_pca_gap_auc_helpers(np, pd, plt):
    def calculate_rrr_pca_gap_auc_values(frame, rrr_method="rrr", pca_method="pca"):
        """Calculate signed RRR-minus-PCA AUC for each model layer."""
        _architectural_max_layers = {
            "gpt2": 12,
            "random-gpt2": 12,
            "gpt2-xl-batch": 48,
            "gpt-j-6b": 28,
            "llama3-8b": 32,
            "roberta-large": 24,
        }
        _work = frame.copy()
        if "mean_fisher_z" in _work:
            _work = (
                _work.groupby(
                    ["model", "layer", "method", "rank"],
                    as_index=False,
                    dropna=False,
                )["mean_fisher_z"]
                .mean()
            )
            _work["correlation"] = np.tanh(_work["mean_fisher_z"])
        else:
            _work = (
                _work.groupby(
                    ["model", "layer", "method", "rank"],
                    as_index=False,
                    dropna=False,
                )["correlation"]
                .mean()
            )

        _paired = _work[_work["method"].isin([rrr_method, pca_method])].pivot_table(
            index=["model", "layer", "rank"],
            columns="method",
            values="correlation",
            aggfunc="mean",
        ).dropna(subset=[rrr_method, pca_method]).reset_index()
        rows = []
        for (_model, _layer), _member in _paired.groupby(
            ["model", "layer"], dropna=False, sort=True
        ):
            _member = _member.sort_values("rank")
            _ranks = _member["rank"].to_numpy(dtype=float)
            if len(_ranks) < 2 or _ranks[0] != 1 or _ranks[-1] <= 1:
                continue
            _x = np.log(_ranks)
            _x = _x / _x[-1]
            _gap = (
                _member[rrr_method].to_numpy(dtype=float)
                - _member[pca_method].to_numpy(dtype=float)
            )
            _max_layer = _architectural_max_layers.get(_model)
            if _max_layer is None:
                _max_layer = _paired.loc[
                    _paired["model"] == _model, "layer"
                ].dropna().max()
            _depth = (
                round(100 * _layer / _max_layer)
                if pd.notna(_layer) and pd.notna(_max_layer)
                else np.nan
            )
            _auc = float(np.trapezoid(_gap, x=_x))
            rows.append({
                "model": _model,
                "layer": _layer,
                "depth": _depth,
                "auc": _auc,
            })
        return pd.DataFrame(rows)


    def plot_rrr_pca_gap_auc_distributions(values, grouping, dataset_label):
        """Plot signed RRR-minus-PCA AUC distributions with model-layer points."""
        if grouping == "depth":
            _group_column = "depth"
            _labels = sorted(values[_group_column].dropna().unique())
            _labels = [f"{int(label)}% depth" for label in _labels]
            _keys = sorted(values[_group_column].dropna().unique())
            _label_values = {f"{int(key)}% depth": key for key in _keys}
        elif grouping == "model":
            _group_column = "model"
            _labels = sorted(values[_group_column].dropna().unique())
            _label_values = {label: label for label in _labels}
        else:
            raise ValueError(f"Unknown RRR/PCA gap grouping: {grouping}")

        _distributions = [
            values[values[_group_column] == _label_values[label]]["auc"].dropna().to_numpy()
            for label in _labels
        ]
        if any(len(_distribution) == 0 for _distribution in _distributions):
            raise ValueError(f"Empty RRR/PCA AUC distribution for {dataset_label}")

        _colors = plt.get_cmap(
            "viridis" if grouping == "depth" else "tab20",
            max(len(_labels), 1),
        )
        _fig, _ax = plt.subplots(
            figsize=(max(7.0, 1.25 * len(_labels)), 4.8)
        )
        _positions = np.arange(1, len(_labels) + 1, dtype=float)
        _violin = _ax.violinplot(
            _distributions,
            positions=_positions,
            widths=0.72,
            showextrema=False,
        )
        for _index, _body in enumerate(_violin["bodies"]):
            _color = _colors(_index)
            _body.set_facecolor(_color)
            _body.set_edgecolor(_color)
            _body.set_alpha(0.22)
        _boxes = _ax.boxplot(
            _distributions,
            positions=_positions,
            widths=0.20,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"color": "#4b5563", "linewidth": 0.8},
            capprops={"color": "#4b5563", "linewidth": 0.8},
            boxprops={"color": "#4b5563", "linewidth": 0.8},
        )
        for _index, _box in enumerate(_boxes["boxes"]):
            _box.set_facecolor(_colors(_index))
            _box.set_alpha(0.18)
        _position_by_label = {
            _label: _position
            for _label, _position in zip(_labels, _positions)
        }

        def _plot_contiguous_pairs(_x, _y):
            _x = np.asarray(_x, dtype=float)
            _y = np.asarray(_y, dtype=float)
            if len(_x) < 2:
                return
            _breaks = np.flatnonzero(np.diff(_x) > 1) + 1
            for _segment in np.split(np.arange(len(_x)), _breaks):
                if len(_segment) > 1:
                    _ax.plot(
                        _x[_segment],
                        _y[_segment],
                        color="#9ca3af",
                        linewidth=0.75,
                        alpha=0.55,
                        zorder=2,
                    )

        if grouping == "depth":
            _pair_groups = values.groupby("model", dropna=False)
            for _model, _member in _pair_groups:
                _member = _member.dropna(subset=["depth", "auc"]).sort_values("depth")
                _x = [
                    _position_by_label[f"{int(_depth)}% depth"]
                    for _depth in _member["depth"]
                ]
                _plot_contiguous_pairs(_x, _member["auc"])
        else:
            _pair_groups = values.groupby("layer", dropna=False)
            for _layer, _member in _pair_groups:
                _member = _member.dropna(subset=["auc"]).sort_values("model")
                _x = [_position_by_label[_model] for _model in _member["model"]]
                _plot_contiguous_pairs(_x, _member["auc"])

        for _index, (_position, _distribution) in enumerate(
            zip(_positions, _distributions)
        ):
            _offsets = (
                np.linspace(-0.14, 0.14, len(_distribution))
                if len(_distribution) > 1
                else np.zeros(1)
            )
            _ax.scatter(
                _position + _offsets,
                _distribution,
                color=_colors(_index),
                s=28,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
        _ax.axhline(0, color="#111827", linewidth=0.9, linestyle="--")
        _ax.set(
            xticks=_positions,
            xticklabels=_labels,
            ylabel="Signed AUC of RRR - PCA gap",
            title=f"{dataset_label}: RRR - PCA gap AUC by {('depth' if grouping == 'depth' else 'model')}",
        )
        _ax.grid(axis="y", alpha=0.25)
        _ax.spines[["top", "right"]].set_visible(False)
        _fig.tight_layout()
        return _fig, _ax


    return calculate_rrr_pca_gap_auc_values, plot_rrr_pca_gap_auc_distributions


@app.cell(hide_code=True)
def internal_rrr_pca_gap_auc_plot(
    calculate_rrr_pca_gap_auc_values,
    internal_gap_grouping,
    internal_summary_curves,
    mo,
    plot_rrr_pca_gap_auc_distributions,
):
    internal_rrr_pca_gap_auc_plots = []
    for _dataset, _label in (
        ("podcast", "Podcast"),
        ("cars-2", "Cars 2"),
        ("fmri", "fMRI"),
    ):
        _auc_values = calculate_rrr_pca_gap_auc_values(
            internal_summary_curves[_dataset]
        )
        internal_rrr_pca_gap_auc_plots.append(
            plot_rrr_pca_gap_auc_distributions(
                _auc_values,
                internal_gap_grouping.value,
                _label + " - internal RRR",
            )
        )
    mo.vstack([
        mo.md(
            "Each point is one model-layer signed AUC of the RRR - PCA gap "
            "over normalized log-rank. Positive values favor RRR."
        ),
        *[plot[0] for plot in internal_rrr_pca_gap_auc_plots],
    ])
    return


@app.cell(hide_code=True)
def average_normalized_curve_plot(
    filter_normalized_curves,
    internal_curve_depth,
    internal_curve_mode,
    internal_curve_model,
    internal_summary_curves,
    mo,
    plot_average_normalized_curves,
):
    normalized_curve_plots = []
    for _dataset, _label in (
        ("podcast", "Podcast"),
        ("cars-2", "Cars 2"),
        ("fmri", "fMRI"),
    ):
        _curves = internal_summary_curves[_dataset]
        _curves, _aggregation_label = filter_normalized_curves(
            _curves,
            internal_curve_mode.value,
            internal_curve_model.value,
            internal_curve_depth.value,
        )
        normalized_curve_plots.append(
            plot_average_normalized_curves(
                _curves, _label, aggregation_label=_aggregation_label
            )
        )
    mo.vstack([
        mo.md(
            "Each faint line is one model layer. Thick lines are equally weighted "
            "means after interpolation in log normalized-rank space; curves are not "
            "extrapolated below their first tested rank."
        ),
        *[plot[0] for plot in normalized_curve_plots],
    ])
    return


@app.cell(hide_code=True)
def load_rrr_target_differences(REPO_ROOT, internal_rrr_artifact, np, pd):
    def load_rrr_target_differences(
        *,
        results_root=REPO_ROOT / "results",
        dataset="podcast",
        model="gpt2",
        layer=None,
    ):
        """Load per-electrode/voxel rank-k minus full-model RRR correlations."""
        artifact = internal_rrr_artifact(
            dataset=dataset,
            model=model,
            layer=layer,
            results_root=results_root,
        ).load_artifacts(["full_metrics", "rank_metrics", "target_manifest"])
        is_fmri = dataset == "fmri"
        rank_metric_name = "corr_per_target" if is_fmri else "nr_orth_corr_per_target"
        full_metric_name = "corr_per_target" if is_fmri else "full_corr_per_target"

        if is_fmri:
            full_entries = artifact.full_metrics.items()
        else:
            full_entries = artifact.load_collection("full_metrics", {"method": "rrr"}).items()

        full_by_fold = {}
        for selector, metrics in full_entries:
            selector = dict(selector)
            if full_metric_name in metrics:
                full_by_fold[selector.get("fold")] = np.asarray(
                    metrics[full_metric_name], dtype=float
                ).reshape(-1)

        electrode_groups = None
        if not is_fmri:
            manifest_entries = artifact.target_manifest.items()
            target_manifest = next(
                value
                for selector, value in manifest_entries
                if dict(selector).get("method") == "rrr"
            )
            target_manifest = target_manifest.reset_index(drop=True)
            electrode_groups = target_manifest.groupby(
                ["subject", "electrode"], sort=False
            ).ngroup().to_numpy()
            electrode_count = int(electrode_groups.max()) + 1
            electrode_sizes = np.bincount(electrode_groups, minlength=electrode_count)

        rank_entries = artifact.rank_metrics.items()
        rows = []
        for selector, metrics in rank_entries:
            selector = dict(selector)
            if rank_metric_name not in metrics or "rank" not in selector:
                continue
            if selector.get("method") not in (None, "rrr"):
                continue
            fold = selector.get("fold")
            full = full_by_fold.get(fold, full_by_fold.get(None))
            if full is None:
                continue
            rank = np.asarray(metrics[rank_metric_name], dtype=float).reshape(-1)
            if rank.shape != full.shape:
                raise ValueError(
                    f"{dataset}/{model}/layer-{layer}: rank {selector['rank']} "
                    f"has {rank.size} targets but full model has {full.size}"
                )

            if is_fmri:
                rank_per_target = rank
                full_per_target = full
            else:
                if rank.size != electrode_groups.size:
                    raise ValueError(
                        f"{dataset}/{model}/layer-{layer}: target manifest has "
                        f"{electrode_groups.size} lag targets but metrics have {rank.size}"
                    )
                rank_z = np.arctanh(np.clip(rank, -1 + 1e-7, 1 - 1e-7))
                full_z = np.arctanh(np.clip(full, -1 + 1e-7, 1 - 1e-7))
                rank_z = np.bincount(
                    electrode_groups, weights=rank_z, minlength=electrode_count
                ) / electrode_sizes
                full_z = np.bincount(
                    electrode_groups, weights=full_z, minlength=electrode_count
                ) / electrode_sizes
                rank_per_target = np.tanh(rank_z)
                full_per_target = np.tanh(full_z)

            rows.append(
                pd.DataFrame(
                    {
                        "rank": int(selector["rank"]),
                        "fold": fold,
                        "target": np.arange(rank_per_target.size),
                        "correlation_delta": rank_per_target - full_per_target,
                    }
                )
            )
        if not rows:
            raise ValueError(
                f"No paired per-target RRR/full correlations for {dataset}/{model}/layer-{layer}"
            )
        return pd.concat(rows, ignore_index=True)


    return (load_rrr_target_differences,)


@app.cell
def _(mo):
    mo.md("""
    ## Performance Distributions
    """)
    return


@app.cell(hide_code=True)
def target_difference_controls(
    find_cross_model_layers,
    find_loso_model_layers,
    find_rrr_layers,
    find_rrr_models,
    mo,
    plot_model_filter,
):
    _common_model_layers = None
    for _dataset in ("podcast", "cars-2", "fmri"):
        _pairs = {
            (model, layer)
            for model in find_rrr_models(dataset=_dataset)
            if model != "random-gpt2" and model in plot_model_filter.value
            for layer in find_rrr_layers(dataset=_dataset, model=model)
        }
        _common_model_layers = _pairs if _common_model_layers is None else _common_model_layers & _pairs

    model_layer_pairs = sorted(
        _common_model_layers or set(),
        key=lambda item: (item[0], item[1] is not None, item[1] if item[1] is not None else -1),
    )
    model_layer_options = {
        f"{model} / {('layer ' + str(layer)) if layer is not None else 'layerless'}": (model, layer)
        for model, layer in model_layer_pairs
    }
    if not model_layer_options:
        raise ValueError("No model/layer pair is available in all three datasets")
    model_layer_selector = mo.ui.dropdown(
        options=model_layer_options,
        value=next(iter(model_layer_options)),
        label="Model layer",
    )
    model_layer_selector


    _loso_model_layer_sets = [
        set(find_loso_model_layers(dataset=_dataset))
        for _dataset in ("podcast", "cars-2", "fmri")
    ]
    _loso_model_layer_pairs = sorted(
        set.intersection(*_loso_model_layer_sets) if all(_loso_model_layer_sets) else set(),
        key=lambda item: (item[0], item[1] is not None, item[1] if item[1] is not None else -1),
    )
    _loso_model_layer_options = {
        f"{model} / {('layer ' + str(layer)) if layer is not None else 'layerless'}": (model, layer)
        for model, layer in _loso_model_layer_pairs
    }
    if not _loso_model_layer_options:
        raise ValueError("No model/layer pair is available in all three LOSO datasets")
    loso_model_layer_selector = mo.ui.dropdown(
        options=_loso_model_layer_options,
        value=next(iter(_loso_model_layer_options)),
        label="LOSO model layer",
    )
    _cross_model_layer_sets = [
        set(find_cross_model_layers(dataset=_dataset))
        for _dataset in ("podcast", "cars-2", "fmri")
    ]
    _cross_model_layer_pairs = sorted(
        set.intersection(*_cross_model_layer_sets) if all(_cross_model_layer_sets) else set(),
        key=lambda item: (item[0], item[1] is not None, item[1] if item[1] is not None else -1),
    )
    _cross_model_layer_options = {
        f"{model} / {('layer ' + str(layer)) if layer is not None else 'layerless'}": (model, layer)
        for model, layer in _cross_model_layer_pairs
    }
    if not _cross_model_layer_options:
        raise ValueError("No model/layer pair is available in all three cross-dataset targets")
    cross_model_layer_selector = mo.ui.dropdown(
        options=_cross_model_layer_options,
        value=next(iter(_cross_model_layer_options)),
        label="Cross-dataset model layer",
    )
    mo.vstack([model_layer_selector, loso_model_layer_selector, cross_model_layer_selector])
    return (
        cross_model_layer_selector,
        loso_model_layer_selector,
        model_layer_options,
        model_layer_pairs,
        model_layer_selector,
    )


@app.cell(hide_code=True)
def plot_rank_full_target_differences(np, plt):
    def plot_rank_full_target_differences(differences, dataset_label, model, layer):
        """Plot target-level rank-k minus full-model correlation distributions."""
        ranks = sorted(differences["rank"].unique())
        values = [
            differences.loc[differences["rank"] == rank, "correlation_delta"]
            .dropna()
            .to_numpy()
            for rank in ranks
        ]
        nonempty = [(rank, value) for rank, value in zip(ranks, values) if value.size]
        if not nonempty:
            raise ValueError(f"No non-empty rank distributions for {dataset_label}")
        ranks, values = zip(*nonempty)
        positions = np.arange(1, len(ranks) + 1, dtype=float)

        fig, ax = plt.subplots(figsize=(max(6.0, 0.48 * len(ranks)), 4.2))
        violin = ax.violinplot(
            values,
            positions=positions,
            widths=0.82,
            showextrema=False,
            showmeans=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor("#6ab2e2")
            body.set_edgecolor("#4b98cc")
            body.set_alpha(0.28)
            body.set_linewidth(0.8)

        boxes = ax.boxplot(
            values,
            positions=positions,
            widths=0.14,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.1},
            whiskerprops={"color": "#4b5563", "linewidth": 0.7},
            capprops={"color": "#4b5563", "linewidth": 0.7},
            boxprops={"color": "#4b5563", "linewidth": 0.7},
        )
        for patch in boxes["boxes"]:
            patch.set_facecolor("white")
            patch.set_alpha(0.8)

        means = [float(np.nanmean(value)) for value in values]
        ax.scatter(positions, means, marker="D", s=22, color="black", zorder=4)
        ax.axhline(0, color="#6b7280", linewidth=0.8, linestyle="--", zorder=1)
        ax.set(
            xticks=positions,
            xticklabels=[str(rank) for rank in ranks],
            xlabel="RRR rank $k$",
            ylabel="Per-target correlation: rank $k$ − full rank",
            title=(
                f"{dataset_label}: rank-vs-full RRR target differences\n"
                f"{model} / {'layer '+str(layer) if layer is not None else 'layerless'}"
            ),
        )
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.99,
            0.98,
            f"{len(differences):,} paired target/fold values",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#4b5563",
        )
        fig.tight_layout()
        return fig, ax


    return (plot_rank_full_target_differences,)


@app.cell(hide_code=True)
def rrr_target_difference_plot(
    QUANTILE_LEVELS,
    QUANTILE_PEAK_FLOOR,
    internal_rank_quantiles,
    internal_saturation_percent,
    load_rrr_target_differences,
    load_rrr_target_performances,
    mean_target_correlations,
    mo,
    model_layer_pairs,
    model_layer_selector,
    np,
    pd,
    plot_rank_full_target_differences,
    plot_rank_target_performances,
    plt,
    quantile_saturation_rank,
):
    selected_model, selected_layer = model_layer_selector.value
    selected_performance_plots = []
    selected_difference_plots = []
    _selected_performances_by_dataset = {}
    _selected_pca_performances_by_dataset = {}
    for _dataset, _label in (("podcast", "Podcast"), ("cars-2", "Cars 2"), ("fmri", "fMRI")):
        _performances = load_rrr_target_performances(
            dataset=_dataset,
            model=selected_model,
            layer=selected_layer,
        )
        _selected_performances_by_dataset[_dataset] = mean_target_correlations(_performances)
        _selected_pca_performances_by_dataset[_dataset] = mean_target_correlations(
            load_rrr_target_performances(
                dataset=_dataset, model=selected_model, layer=selected_layer, method="pca",
            )
        )
        selected_performance_plots.append(
            plot_rank_target_performances(
                _performances,
                _label,
                selected_model,
                selected_layer,
            )
        )
        _differences = load_rrr_target_differences(
            dataset=_dataset,
            model=selected_model,
            layer=selected_layer,
        )
        selected_difference_plots.append(
            plot_rank_full_target_differences(
                _differences,
                _label,
                selected_model,
                selected_layer,
            )
        )
    _quantile_levels = QUANTILE_LEVELS
    _quantile_colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(_quantile_levels)))
    _quantile_figure, _quantile_axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    for _ax, (_dataset, _label) in zip(
        _quantile_axes,
        (("podcast", "Podcast"), ("cars-2", "Cars 2"), ("fmri", "fMRI")),
    ):
        _performances = _selected_performances_by_dataset[_dataset]
        _rank_quantiles = _performances.groupby("rank")["correlation"].quantile(
            _quantile_levels
        ).unstack()
        _ranks = _rank_quantiles.index.to_numpy(dtype=int)
        for _quantile, _color in zip(_quantile_levels, _quantile_colors):
            _ax.plot(
                _ranks,
                _rank_quantiles[_quantile].to_numpy(),
                marker="o",
                linewidth=1.8,
                markersize=3,
                color=_color,
                label=f"{_quantile:.0%}",
            )
        _ax.set_xscale("log", base=2)
        _ax.set_xticks(_ranks, [str(_rank) for _rank in _ranks])
        _ax.set_title(_label)
        _ax.set_ylabel("Target correlation")
        _ax.grid(axis="y", alpha=.22)
        _ax.spines[["top", "right"]].set_visible(False)
    _quantile_figure.supxlabel("RRR rank $k$")
    _quantile_figure.suptitle(
        f"Target-performance quantiles by RRR rank\n"
        f"{selected_model} / {'layer ' + str(selected_layer) if selected_layer is not None else 'layerless'}"
    )
    _quantile_figure.legend(
        title="Target percentile", frameon=False, ncol=7,
        loc="upper center", bbox_to_anchor=(.5, .90),
    )
    _quantile_figure.tight_layout(rect=(.04, .08, .99, .80))

    _positive_fraction_figure, _positive_fraction_axes = plt.subplots(
        1, 3, figsize=(15, 4.2), sharey=False
    )
    for _ax, (_dataset, _label) in zip(
        _positive_fraction_axes,
        (("podcast", "Podcast"), ("cars-2", "Cars 2"), ("fmri", "fMRI")),
    ):
        _positive_fraction = (
            _selected_performances_by_dataset[_dataset]
            .groupby("rank")["correlation"]
            .apply(lambda _values: (_values.dropna() > 0).mean())
        )
        _ranks = _positive_fraction.index.to_numpy(dtype=int)
        _ax.plot(
            _ranks, _positive_fraction.to_numpy(),
            color="#4b5563", marker="o", linewidth=1.8, markersize=3,
        )
        _ax.set_xscale("log", base=2)
        _ax.set_xticks(_ranks, [str(_rank) for _rank in _ranks])
        _ax.set_ylim(0, 1)
        _ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
        _ax.set_title(_label)
        _ax.set_xlabel("RRR rank $k$")
        _ax.set_ylabel("Targets with positive correlation")
        _ax.grid(axis="y", alpha=.22)
        _ax.spines[["top", "right"]].set_visible(False)
    _positive_fraction_figure.suptitle(
        f"Fraction of targets above zero by RRR rank\n"
        f"{selected_model} / {'layer ' + str(selected_layer) if selected_layer is not None else 'layerless'}"
    )
    _positive_fraction_figure.tight_layout(rect=(.02, .04, .99, .88))

    _pca_quantile_figure, _pca_quantile_axes = plt.subplots(
        1, 3, figsize=(15, 4.8), sharey=False
    )
    for _ax, (_dataset, _label) in zip(
        _pca_quantile_axes,
        (("podcast", "Podcast"), ("cars-2", "Cars 2"), ("fmri", "fMRI")),
    ):
        _performances = _selected_pca_performances_by_dataset[_dataset]
        _rank_quantiles = _performances.groupby("rank")["correlation"].quantile(
            _quantile_levels
        ).unstack()
        _ranks = _rank_quantiles.index.to_numpy(dtype=int)
        for _quantile, _color in zip(_quantile_levels, _quantile_colors):
            _ax.plot(
                _ranks, _rank_quantiles[_quantile].to_numpy(),
                marker="o", linewidth=1.8, markersize=3,
                color=_color, label=f"{_quantile:.0%}",
            )
        _ax.set_xscale("log", base=2)
        _ax.set_xticks(_ranks, [str(_rank) for _rank in _ranks])
        _ax.set_title(_label)
        _ax.set_ylabel("Target correlation")
        _ax.grid(axis="y", alpha=.22)
        _ax.spines[["top", "right"]].set_visible(False)
    _pca_quantile_figure.supxlabel("PCA rank $k$")
    _pca_quantile_figure.suptitle(
        f"Target-performance quantiles by PCA rank\n"
        f"{selected_model} / {'layer ' + str(selected_layer) if selected_layer is not None else 'layerless'}"
    )
    _pca_quantile_figure.legend(
        title="Target percentile", frameon=False, ncol=7,
        loc="upper center", bbox_to_anchor=(.5, .90),
    )
    _pca_quantile_figure.tight_layout(rect=(.04, .08, .99, .80))

    _pca_positive_fraction_figure, _pca_positive_fraction_axes = plt.subplots(
        1, 3, figsize=(15, 4.2), sharey=False
    )
    for _ax, (_dataset, _label) in zip(
        _pca_positive_fraction_axes,
        (("podcast", "Podcast"), ("cars-2", "Cars 2"), ("fmri", "fMRI")),
    ):
        _positive_fraction = (
            _selected_pca_performances_by_dataset[_dataset]
            .groupby("rank")["correlation"]
            .apply(lambda _values: (_values.dropna() > 0).mean())
        )
        _ranks = _positive_fraction.index.to_numpy(dtype=int)
        _ax.plot(
            _ranks, _positive_fraction.to_numpy(),
            color="#63bc51", marker="o", linewidth=1.8, markersize=3,
        )
        _ax.set_xscale("log", base=2)
        _ax.set_xticks(_ranks, [str(_rank) for _rank in _ranks])
        _ax.set_ylim(0, 1)
        _ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
        _ax.set_title(_label)
        _ax.set_xlabel("PCA rank $k$")
        _ax.set_ylabel("Targets with positive correlation")
        _ax.grid(axis="y", alpha=.22)
        _ax.spines[["top", "right"]].set_visible(False)
    _pca_positive_fraction_figure.suptitle(
        f"Fraction of targets above zero by PCA rank\n"
        f"{selected_model} / {'layer ' + str(selected_layer) if selected_layer is not None else 'layerless'}"
    )
    _pca_positive_fraction_figure.tight_layout(rect=(.02, .04, .99, .88))

    _quantile_saturation_fraction = internal_saturation_percent.value / 100
    _quantile_saturation_figure, _quantile_saturation_axes = plt.subplots(
        2, 3, figsize=(16, 8), sharey=True
    )
    _quantile_saturation_handles = {}
    for _method_row, (_method_label, _method_data) in enumerate((
        ("RRR", _selected_performances_by_dataset),
        ("PCA", _selected_pca_performances_by_dataset),
    )):
        for _column, (_dataset, _dataset_label) in enumerate((
            ("podcast", "Podcast"), ("cars-2", "Cars 2"), ("fmri", "fMRI")
        )):
            _ax = _quantile_saturation_axes[_method_row, _column]
            _performances = _method_data[_dataset]
            _rank_quantiles = _performances.groupby("rank")["correlation"].quantile(
                _quantile_levels
            ).unstack()
            _all_ranks = _rank_quantiles.index.to_numpy(dtype=float)
            _all_hit_rank = quantile_saturation_rank(
                _rank_quantiles, _quantile_saturation_fraction,
            )
            for _quantile, _color in zip(_quantile_levels, _quantile_colors):
                _values = _rank_quantiles[_quantile].to_numpy(dtype=float)
                _valid = np.isfinite(_all_ranks) & np.isfinite(_values)
                _ranks, _values = _all_ranks[_valid], _values[_valid]
                if not len(_values) or float(np.max(_values)) <= QUANTILE_PEAK_FLOOR:
                    continue
                _monotonic_values = np.maximum.accumulate(np.maximum(_values, 0.0))
                _normalized_values = _monotonic_values / float(np.max(_monotonic_values))
                _line, = _ax.plot(
                    _ranks, _normalized_values,
                    color=_color, linewidth=1.7, marker="o", markersize=2.8,
                    label=f"{_quantile:.0%}",
                )
                _quantile_saturation_handles.setdefault(f"{_quantile:.0%}", _line)

            _ax.axhline(
                _quantile_saturation_fraction, color="#374151", linestyle="--",
                linewidth=1.0, label=f"{internal_saturation_percent.value:g}% threshold",
            )
            if np.isfinite(_all_hit_rank):
                _ax.axvline(
                    _all_hit_rank, color="#111827", linestyle=":", linewidth=1.2,
                    label="All-hit rank",
                )
                _panel_note = f"all-hit rank = {_all_hit_rank:g}"
            else:
                _panel_note = f"no quantile peaks above {QUANTILE_PEAK_FLOOR:g}"
            _ax.set_xscale("log", base=2)
            _ranks = _rank_quantiles.index.to_numpy(dtype=int)
            _ax.set_xticks(_ranks, [str(_rank) for _rank in _ranks])
            _ax.set_ylim(0, 1.05)
            _ax.set_title(f"{_method_label} · {_dataset_label}\n{_panel_note}")
            _ax.set_xlabel(f"{_method_label} rank $k$")
            _ax.grid(axis="both", alpha=.2)
            _ax.spines[["top", "right"]].set_visible(False)

    _quantile_saturation_figure.supylabel("Normalized quantile performance")
    _quantile_saturation_figure.suptitle(
        f"Quantile convergence at {internal_saturation_percent.value:g}% of peak\n"
        f"{selected_model} / {'layer ' + str(selected_layer) if selected_layer is not None else 'layerless'}"
    )
    _quantile_saturation_figure.legend(
        list(_quantile_saturation_handles.values()),
        list(_quantile_saturation_handles),
        title="Quantile", frameon=False, ncol=7,
        loc="upper center", bbox_to_anchor=(.5, .91),
    )
    _quantile_saturation_figure.tight_layout(rect=(.04, .05, .99, .80))
    _quantile_saturation_figure.text(
        .5, .01,
        f"Only quantiles peaking above {QUANTILE_PEAK_FLOOR:g} are included; all-hit rank is the latest first crossing among them.",
        ha="center", fontsize=9,
    )



    _quantile_summary_rows = []
    _quantile_summary_missing = []
    for _dataset, _dataset_label in (("podcast", "Podcast"), ("cars-2", "Cars 2"), ("fmri", "fMRI")):
        for _pair_index, (_model, _layer) in enumerate(model_layer_pairs):
            _pair_label = f"{_model} / {('layer ' + str(_layer)) if _layer is not None else 'layerless'}"
            for _method_label, _method in (("RRR", "rrr"), ("PCA", "pca")):
                try:
                    _by_rank = internal_rank_quantiles(_dataset, _model, _layer, _method)
                except (FileNotFoundError, ValueError, KeyError) as _error:
                    _quantile_summary_missing.append((_dataset_label, _pair_label, _method_label, str(_error)))
                    continue
                _all_hit_rank = quantile_saturation_rank(
                    _by_rank, _quantile_saturation_fraction,
                )
                if np.isfinite(_all_hit_rank):
                    _quantile_summary_rows.append({
                        "dataset": _dataset_label, "model": _model, "layer": _layer,
                        "pair": _pair_label, "pair_index": _pair_index,
                        "method": _method_label, "all_hit_rank": _all_hit_rank,
                    })
    _quantile_summary_data = pd.DataFrame(_quantile_summary_rows)
    _quantile_summary_figure, _quantile_summary_axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for _ax, (_dataset, _dataset_label) in zip(
        _quantile_summary_axes,
        (("podcast", "Podcast"), ("cars-2", "Cars 2"), ("fmri", "fMRI")),
    ):
        _subset = _quantile_summary_data[_quantile_summary_data["dataset"] == _dataset_label]
        _method_values = [_subset.loc[_subset["method"] == _method, "all_hit_rank"].to_numpy(dtype=float)
                          for _method in ("RRR", "PCA")]
        if any(len(_values) for _values in _method_values):
            _violins = _ax.violinplot(_method_values, positions=[0, 1], widths=.72, showextrema=False, showmeans=False)
            for _body, _color in zip(_violins["bodies"], ("#6ab2e2", "#63bc51")):
                _body.set_facecolor(_color); _body.set_edgecolor(_color); _body.set_alpha(.22)
            _boxes = _ax.boxplot(
                _method_values, positions=[0, 1], widths=.12, patch_artist=True, showfliers=False,
                medianprops={"color": "black", "linewidth": 1},
                whiskerprops={"color": "#4b5563", "linewidth": .7},
                capprops={"color": "#4b5563", "linewidth": .7},
                boxprops={"color": "#4b5563", "linewidth": .7},
            )
            for _patch, _color in zip(_boxes["boxes"], ("#6ab2e2", "#63bc51")):
                _patch.set_facecolor("white"); _patch.set_alpha(.85)
        _rrr = _subset[_subset["method"] == "RRR"].set_index("pair")["all_hit_rank"]
        _pca = _subset[_subset["method"] == "PCA"].set_index("pair")["all_hit_rank"]
        for _pair in _rrr.index.intersection(_pca.index):
            _i = int(_subset.loc[_subset["pair"] == _pair, "pair_index"].iloc[0])
            _jitter = ((_i % 7) - 3) * .018
            _y_rrr, _y_pca = float(_rrr[_pair]), float(_pca[_pair])
            _ax.plot([-0.12 + _jitter, 1.12 + _jitter], [_y_rrr, _y_pca],
                     color="#9ca3af", alpha=.55, linewidth=.7, zorder=2)
            _ax.scatter(0 + _jitter, _y_rrr, s=24, color="#4b98cc", edgecolor="white", linewidth=.4, zorder=3)
            _ax.scatter(1 + _jitter, _y_pca, s=24, color="#45963a", edgecolor="white", linewidth=.4, zorder=3)
        _ax.set_xticks([0, 1], ["RRR", "PCA"])
        _ax.set_title(_dataset_label)
        _ax.set_ylabel("All-hit saturation rank")
        _ax.set_yscale("log")
        _ax.grid(axis="y", alpha=.22)
        _ax.spines[["top", "right"]].set_visible(False)
    _quantile_summary_figure.suptitle(f"Model-layer saturation rank comparison at {internal_saturation_percent.value:g}% of peak | {selected_model}")
    _quantile_summary_figure.tight_layout(rect=(.03, .04, .99, .91))
    mo.vstack([
        mo.md("### Target-performance quantiles by rank"),
        _quantile_figure,
        mo.md("### PCA target-performance quantiles by rank"),
        _pca_quantile_figure,
        mo.md("### Fraction of target observations with positive correlation"),
        _positive_fraction_figure,
        mo.md("### PCA fraction of target observations with positive correlation"),
        _pca_positive_fraction_figure,
        mo.md("### Quantile saturation summary (all model layers)"),
        _quantile_summary_figure,
        mo.md("### Normalized quantile saturation convergence"),
        _quantile_saturation_figure,
        *[plot[0] for plot in selected_performance_plots],
        *[plot[0] for plot in selected_difference_plots],
    ])
    return


@app.cell(hide_code=True)
def load_rrr_target_performances(REPO_ROOT, internal_rrr_artifact, np, pd):
    def load_rrr_target_performances(
        *,
        results_root=REPO_ROOT / "results",
        dataset="podcast",
        model="gpt2",
        layer=None,
        method="rrr",
    ):
        """Load electrode-level iEEG or voxel-level fMRI correlations by rank."""
        artifact = internal_rrr_artifact(
            dataset=dataset,
            model=model,
            layer=layer,
            results_root=results_root,
        ).load_artifacts(["rank_metrics", "target_manifest"], selectors={"method": method})
        is_fmri = dataset == "fmri"
        metric_name = "corr_per_target" if is_fmri else ("nr_orth_corr_per_target" if method == "rrr" else "pca_corr_per_target")

        electrode_groups = None
        if not is_fmri:
            target_manifest = next(
                value
                for selector, value in artifact.target_manifest.items()
                if dict(selector).get("method") == method
            ).reset_index(drop=True)
            electrode_groups = target_manifest.groupby(
                ["subject", "electrode"], sort=False
            ).ngroup().to_numpy()
            electrode_count = int(electrode_groups.max()) + 1
            electrode_sizes = np.bincount(electrode_groups, minlength=electrode_count)

        rows = []
        for selector, metrics in artifact.rank_metrics.items():
            selector = dict(selector)
            if metric_name not in metrics or "rank" not in selector:
                continue
            if selector.get("method") not in (None, method):
                continue
            values = np.asarray(metrics[metric_name], dtype=float).reshape(-1)
            if electrode_groups is not None:
                if values.size != electrode_groups.size:
                    raise ValueError(
                        f"{dataset}/{model}/layer-{layer}: target manifest has "
                        f"{electrode_groups.size} lag targets but metrics have {values.size}"
                    )
                fisher_z = np.arctanh(np.clip(values, -1 + 1e-7, 1 - 1e-7))
                electrode_z = np.bincount(
                    electrode_groups, weights=fisher_z, minlength=electrode_count
                ) / electrode_sizes
                values = np.tanh(electrode_z)
            rows.append(
                pd.DataFrame(
                    {
                        "rank": int(selector["rank"]),
                        "fold": selector.get("fold"),
                        "target": np.arange(values.size),
                        "correlation": values,
                    }
                )
            )
        if not rows:
            raise ValueError(
                f"No target-level {method.upper()} correlations for {dataset}/{model}/layer-{layer}"
            )
        return pd.concat(rows, ignore_index=True)


    return (load_rrr_target_performances,)


@app.cell(hide_code=True)
def plot_rank_target_performances(np, plt):
    def plot_rank_target_performances(
        performances,
        dataset_label,
        model,
        layer,
    ):
        """Plot target-level RRR correlation distributions at each rank."""
        ranks = sorted(performances["rank"].unique())
        values = [
            performances.loc[performances["rank"] == rank, "correlation"]
            .dropna()
            .to_numpy()
            for rank in ranks
        ]
        nonempty = [(rank, value) for rank, value in zip(ranks, values) if value.size]
        if not nonempty:
            raise ValueError(f"No non-empty target distributions for {dataset_label}")
        ranks, values = zip(*nonempty)
        positions = np.arange(1, len(ranks) + 1, dtype=float)

        fig, ax = plt.subplots(figsize=(max(6.0, 0.48 * len(ranks)), 4.2))
        violin = ax.violinplot(
            values, positions=positions, widths=0.82,
            showextrema=False, showmeans=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor("#63bc51")
            body.set_edgecolor("#45963a")
            body.set_alpha(0.28)
            body.set_linewidth(0.8)

        boxes = ax.boxplot(
            values, positions=positions, widths=0.14, patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.1},
            whiskerprops={"color": "#4b5563", "linewidth": 0.7},
            capprops={"color": "#4b5563", "linewidth": 0.7},
            boxprops={"color": "#4b5563", "linewidth": 0.7},
        )
        for patch in boxes["boxes"]:
            patch.set_facecolor("white")
            patch.set_alpha(0.8)

        means = [float(np.nanmean(value)) for value in values]
        ax.scatter(positions, means, marker="D", s=22, color="black", zorder=4)
        ax.set(
            xticks=positions,
            xticklabels=[str(rank) for rank in ranks],
            xlabel="RRR rank $k$",
            ylabel="Per-target correlation",
            title=(
                f"{dataset_label}: target performance by RRR rank\n"
                f"{model} / {'layer '+str(layer) if layer is not None else 'layerless'}"
            ),
        )
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        target_label = "voxels" if dataset_label == "fMRI" else "electrodes (Fisher-z mean across lags)"
        ax.text(
            0.99, 0.98,
            f"{target_label}; {len(performances):,} values across ranks/folds",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#4b5563",
        )
        fig.tight_layout()
        return fig, ax


    return (plot_rank_target_performances,)


@app.cell(hide_code=True)
def _(
    Artifact,
    DATASET_ROOTS,
    Path,
    REPO_ROOT,
    fisher_mean,
    lru_cache,
    np,
    pd,
):
    def find_loso_model_layers(
        *,
        results_root=REPO_ROOT / "results",
        dataset="podcast",
    ):
        """Find model/layer pairs with saved leave-one-subject-out results."""
        models_root = (
            Path(results_root) / "run_rrr" / "leave_one_subject_out"
            / DATASET_ROOTS[dataset] / "offset-0"
        )
        pairs = []
        if not models_root.is_dir():
            return pairs
        for model_root in models_root.iterdir():
            if not model_root.is_dir():
                continue
            for layer_root in model_root.glob("layer-*"):
                if not layer_root.is_dir():
                    continue
                suffix = layer_root.name.removeprefix("layer-")
                if suffix == "none":
                    pairs.append((model_root.name, None))
                elif suffix.isdigit():
                    pairs.append((model_root.name, int(suffix)))
        return sorted(set(pairs), key=lambda item: (item[0], item[1] is not None, item[1] or -1))


    @lru_cache(maxsize=None)
    def load_loso_rank_correlations(
        *,
        results_root=REPO_ROOT / "results",
        dataset="podcast",
        model="gpt2",
        layer=None,
    ):
        """Load subject-level LOSO rank correlations for the three saved phases."""
        artifact = Artifact(
            mode="leave_one_subject_out",
            dataset=DATASET_ROOTS[dataset],
            model=model,
            layer=layer,
            results_root=results_root,
        ).load_artifacts(["rank_metrics"])
        is_fmri = dataset == "fmri"
        metric_names = {
            ("rrr", "baseline"): "corr_per_target" if is_fmri else "nr_orth_corr",
            ("rrr", "transfer"): "corr_per_target" if is_fmri else "transfer_orth_corr",
            ("pca", "baseline"): "corr_per_target" if is_fmri else "pca_corr",
        }
        method_labels = {
            ("rrr", "baseline"): "RRR baseline",
            ("rrr", "transfer"): "RRR transfer",
            ("pca", "baseline"): "PCA baseline",
        }
        rows = []
        for selector, metrics in artifact.rank_metrics.items():
            selector = dict(selector)
            phase_key = (selector.get("method"), selector.get("loso_phase"))
            metric_name = metric_names.get(phase_key)
            if metric_name is None or metric_name not in metrics or "rank" not in selector:
                continue
            values = np.asarray(metrics[metric_name], dtype=float)
            correlation = (
                float(np.tanh(np.nanmean(np.arctanh(np.clip(values, -1 + 1e-7, 1 - 1e-7)))))
                if values.ndim
                else float(values)
            )
            rows.append(
                {
                    "subject": str(selector["held_out_subject"]),
                    "method": method_labels[phase_key],
                    "rank": int(selector["rank"]),
                    "fold": selector.get("fold"),
                    "correlation": correlation,
                }
            )
        if not rows:
            raise ValueError(
                f"No LOSO rank metrics for {dataset}/{model}/layer-{layer}"
            )
        frame = pd.DataFrame(rows)
        return (
            frame.groupby(["subject", "method", "rank"], as_index=False)["correlation"]
            .agg(fisher_mean)
        )

    def load_loso_target_correlations(
        *, results_root=REPO_ROOT / "results", dataset="podcast", model="gpt2", layer=None
    ):
        """Load LOSO per-target correlations, averaged across CV folds per target."""
        artifact = Artifact(
            mode="leave_one_subject_out", dataset=DATASET_ROOTS[dataset],
            model=model, layer=layer, results_root=results_root,
        ).load_artifacts(["rank_metrics", "target_manifest"])
        metric_names = {
            ("rrr", "baseline"): "corr_per_target" if dataset == "fmri" else "nr_orth_corr_per_target",
            ("rrr", "transfer"): "corr_per_target" if dataset == "fmri" else "transfer_orth_corr_per_target",
            ("pca", "baseline"): "corr_per_target" if dataset == "fmri" else "pca_corr_per_target",
        }
        method_labels = {
            ("rrr", "baseline"): "RRR baseline",
            ("rrr", "transfer"): "RRR transfer",
            ("pca", "baseline"): "PCA baseline",
        }
        manifests = {
            (str(dict(selector).get("held_out_subject")),
             dict(selector).get("method"), dict(selector).get("loso_phase")): value
            for selector, value in artifact.target_manifest.items()
        }
        rows = []
        for selector, metrics in artifact.rank_metrics.items():
            selector = dict(selector)
            phase_key = (selector.get("method"), selector.get("loso_phase"))
            metric_name = metric_names.get(phase_key)
            if metric_name is None or metric_name not in metrics or "rank" not in selector:
                continue
            held_out = str(selector.get("held_out_subject"))
            manifest = manifests.get((held_out, phase_key[0], phase_key[1]))
            if manifest is None:
                raise ValueError(f"Missing LOSO target manifest for {dataset}/{model}/layer-{layer}/{held_out}/{phase_key}")
            values = np.asarray(metrics[metric_name], dtype=float).reshape(-1)
            if len(values) != len(manifest):
                raise ValueError(
                    f"{dataset}/{model}/layer-{layer}: {metric_name} has {len(values)} targets, "
                    f"but manifest has {len(manifest)} rows for held-out subject {held_out}"
                )
            clipped = np.arctanh(np.clip(values, -1 + 1e-7, 1 - 1e-7))
            if "electrode" in manifest.columns:
                target_keys = manifest.groupby(["subject", "electrode"], sort=False).ngroup().to_numpy()
            elif "voxel_index" in manifest.columns:
                target_keys = manifest.groupby(["subject", "voxel_index"], sort=False).ngroup().to_numpy()
            else:
                target_keys = np.arange(len(values))
            _target_frame = pd.DataFrame({
                "target": target_keys,
                "fisher_z": clipped,
            })
            _target_frame = _target_frame.groupby("target", sort=False, as_index=False)["fisher_z"].mean()
            _target_frame["correlation"] = np.tanh(_target_frame.pop("fisher_z"))
            _target_frame["subject"] = held_out
            _target_frame["method"] = method_labels[phase_key]
            _target_frame["rank"] = int(selector["rank"])
            _target_frame["fold"] = selector.get("fold")
            rows.append(_target_frame)
        if not rows:
            raise ValueError(f"No per-target LOSO rank metrics for {dataset}/{model}/layer-{layer}")
        frame = pd.concat(rows, ignore_index=True)
        frame["fisher_z"] = np.arctanh(np.clip(
            frame.pop("correlation").to_numpy(dtype=float), -1 + 1e-7, 1 - 1e-7,
        ))
        averaged = frame.groupby(
            ["subject", "target", "method", "rank"], as_index=False
        )["fisher_z"].mean()
        averaged["correlation"] = np.tanh(averaged.pop("fisher_z"))
        return averaged



    return find_loso_model_layers, load_loso_rank_correlations


@app.cell(hide_code=True)
def load_loso_rank_correlations(fisher_mean, plt):
    def plot_loso_rank_correlations(frame, dataset_label, model, layer):
        """Plot subject-level LOSO curves and across-subject means."""
        method_order = ["RRR baseline", "RRR transfer", "PCA baseline"]
        method_colors = {
            "RRR baseline": "#6ab2e2",
            "RRR transfer": "#4b98cc",
            "PCA baseline": "#63bc51",
        }
        ranks = sorted(frame["rank"].unique())
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        for method in method_order:
            method_frame = frame[frame["method"] == method]
            for subject, subject_frame in method_frame.groupby("subject", sort=True):
                subject_frame = subject_frame.sort_values("rank")
                ax.plot(
                    subject_frame["rank"],
                    subject_frame["correlation"],
                    color=method_colors[method],
                    linestyle="--" if "transfer" in method.lower() else "-",
                    alpha=0.18,
                    linewidth=0.8,
                    zorder=1,
                )
            mean_frame = (
                method_frame.groupby("rank", as_index=False)["correlation"]
                .agg(fisher_mean)
                .sort_values("rank")
            )
            ax.plot(
                mean_frame["rank"],
                mean_frame["correlation"],
                color=method_colors[method],
                linestyle="--" if "transfer" in method.lower() else "-",
                linewidth=2.2,
                marker="o",
                markersize=3.5,
                label=method,
                zorder=3,
            )

        ax.set(
            xlabel="RRR rank $k$",
            ylabel="Correlation",
            title=(
                f"{dataset_label}: leave-one-subject-out performance\n"
                f"{model} / {'layer '+str(layer) if layer is not None else 'layerless'}"
            ),
            xscale="log",
            xticks=ranks,
            xticklabels=[str(rank) for rank in ranks],
        )
        ax.tick_params(axis="x", labelrotation=90)
        ax.grid(axis="both", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        return fig, ax


    return (plot_loso_rank_correlations,)


@app.cell(hide_code=True)
def paired_distribution_plot(np, plt):
    def plot_paired_distributions(
        distributions,
        *,
        colors=None,
        ax=None,
        figsize=None,
    ):
        """Plot paired distributions in dictionary order.

        Values at the same index are treated as paired observations. All
        distributions must therefore contain the same number of values.
        """
        if not distributions:
            raise ValueError("at least one distribution is required")

        labels = list(distributions)
        paired_values = {
            label: np.asarray(values, dtype=float).reshape(-1)
            for label, values in distributions.items()
        }
        lengths = {len(values) for values in paired_values.values()}
        if len(lengths) != 1:
            raise ValueError("paired distributions must contain the same number of values")
        if not lengths or next(iter(lengths)) == 0:
            raise ValueError("distributions must contain at least one value")

        if colors is None:
            cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            colors = {
                label: cycle[index % len(cycle)]
                for index, label in enumerate(labels)
            }
        else:
            missing_colors = [label for label in labels if label not in colors]
            if missing_colors:
                raise ValueError(f"missing colors for distributions: {missing_colors}")

        positions = np.arange(1, len(labels) + 1, dtype=float)
        finite_values = [values[np.isfinite(values)] for values in paired_values.values()]
        empty_labels = [
            label for label, values in zip(labels, finite_values) if not len(values)
        ]
        if empty_labels:
            raise ValueError(f"distributions contain no finite values: {empty_labels}")

        if ax is None:
            if figsize is None:
                figsize = (max(4.2, 1.6 * len(labels)), 4.4)
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        violin = ax.violinplot(
            finite_values,
            positions=positions,
            widths=0.72,
            showextrema=False,
        )
        for body, label in zip(violin["bodies"], labels):
            body.set_facecolor(colors[label])
            body.set_edgecolor(colors[label])
            body.set_alpha(0.22)
            body.set_linewidth(1.0)

        boxes = ax.boxplot(
            finite_values,
            positions=positions,
            widths=0.20,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"color": "#4b5563", "linewidth": 0.8},
            capprops={"color": "#4b5563", "linewidth": 0.8},
            boxprops={"color": "#4b5563", "linewidth": 0.8},
        )
        for patch, label in zip(boxes["boxes"], labels):
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.18)

        observation_count = next(iter(lengths))
        offsets = (
            np.linspace(-0.12, 0.12, observation_count)
            if observation_count > 1
            else np.zeros(1)
        )
        for observation, offset in enumerate(offsets):
            point_positions = positions + offset
            point_values = np.asarray(
                [paired_values[label][observation] for label in labels]
            )
            for left in range(len(labels) - 1):
                pair = point_values[left : left + 2]
                if np.isfinite(pair).all():
                    ax.plot(
                        point_positions[left : left + 2],
                        pair,
                        color="#9ca3af",
                        linewidth=0.65,
                        alpha=0.5,
                        zorder=2,
                    )
            for position, value, label in zip(point_positions, point_values, labels):
                if np.isfinite(value):
                    ax.scatter(
                        position,
                        value,
                        color=colors[label],
                        s=19,
                        edgecolors="white",
                        linewidths=0.35,
                        zorder=3,
                    )

        ax.set_xticks(positions, labels)
        ax.set_xlim(0.45, len(labels) + 0.55)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        return fig, ax


    return (plot_paired_distributions,)


@app.cell
def _(mo):
    mo.md("""
    ## LOSO
    """)
    return


@app.cell(hide_code=True)
def _(
    REPO_ROOT,
    find_loso_model_layers,
    fisher_mean,
    internal_rrr_artifact,
    load_loso_rank_correlations,
    lru_cache,
    np,
    pd,
):
    @lru_cache(maxsize=None)
    def load_internal_rrr_subject_correlations(*, dataset, model, layer, results_root=REPO_ROOT / "results"):
        """Load full cross-subject RRR correlations averaged over each subject's targets."""
        artifact = internal_rrr_artifact(
            dataset=dataset,
            model=model,
            layer=layer,
            results_root=results_root,
        ).load_artifacts(["rank_metrics", "target_manifest"])
        target_manifest = artifact.target_manifest.get((("method", "rrr"),))
        if target_manifest is None:
            target_manifest = artifact.target_manifest[()]
        target_subjects = target_manifest["subject"].astype(str).to_numpy()
        metric_name = "corr_per_target" if dataset == "fmri" else "nr_orth_corr_per_target"
        rows = []
        for selector, metrics in artifact.rank_metrics.items():
            selector = dict(selector)
            if selector.get("method") != "rrr" or "rank" not in selector:
                continue
            values = np.asarray(metrics.get(metric_name), dtype=float)
            if values.ndim != 1 or len(values) != len(target_subjects):
                continue
            for subject in np.unique(target_subjects):
                subject_values = values[target_subjects == subject]
                fisher_z = np.arctanh(np.clip(subject_values, -1 + 1e-7, 1 - 1e-7))
                rows.append(
                    {
                        "subject": subject,
                        "rank": int(selector["rank"]),
                        "fold": selector.get("fold"),
                        "correlation": float(np.tanh(np.nanmean(fisher_z))),
                    }
                )
        if not rows:
            raise ValueError(f"No full cross-subject RRR metrics for {dataset}/{model}/layer-{layer}")
        return pd.DataFrame(rows).groupby(
            ["subject", "rank"], as_index=False
        )["correlation"].agg(fisher_mean)


    def load_loso_summary_curves(*, dataset, rrr_baseline_source="loso", results_root=REPO_ROOT / "results"):
        """Load paired subject curves for both LOSO summary statistics."""
        method_order = ["RRR baseline", "RRR transfer", "PCA baseline"]
        rows = []
        pairs = find_loso_model_layers(results_root=results_root, dataset=dataset)
        for model, layer in pairs:
            try:
                loso_frame = load_loso_rank_correlations(
                    results_root=results_root,
                    dataset=dataset,
                    model=model,
                    layer=layer,
                )
                baseline_frame = (
                    load_internal_rrr_subject_correlations(
                        dataset=dataset,
                        model=model,
                        layer=layer,
                        results_root=results_root,
                    )
                    if rrr_baseline_source == "full"
                    else loso_frame[loso_frame["method"] == "RRR baseline"]
                )
            except Exception as error:
                print(f"Skipping {dataset}/{model}/layer-{layer}: {error}")
                continue
            method_frames = {
                "RRR baseline": baseline_frame,
                "RRR transfer": loso_frame[loso_frame["method"] == "RRR transfer"],
                "PCA baseline": loso_frame[loso_frame["method"] == "PCA baseline"],
            }
            subjects = sorted(set.intersection(*[
                set(method_frame["subject"].astype(str))
                for method_frame in method_frames.values()
            ]))
            for method, method_frame in method_frames.items():
                selected = method_frame[method_frame["subject"].astype(str).isin(subjects)].copy()
                selected["dataset"] = dataset
                selected["model"] = model
                selected["layer"] = layer
                selected["method"] = method
                rows.append(selected)
        if not rows:
            raise ValueError(f"No usable LOSO curves for {dataset}")
        return pd.concat(rows, ignore_index=True)


    return (load_loso_summary_curves,)


@app.cell(hide_code=True)
def load_loso_auc_values_and_plot(
    REPO_ROOT,
    annotate_pairwise_tests,
    load_loso_summary_curves,
    np,
    paired_permutation_tests,
    plot_paired_distributions,
    summarize_rank_curves,
):
    def calculate_loso_summary_values(curves, *, metric, fraction=1.0):
        values = summarize_rank_curves(
            curves, ["dataset", "model", "layer", "subject", "method"],
            metric=metric, fraction=fraction,
        )
        # Only average subjects with finite summaries for all three methods.
        counts = values.groupby(["dataset", "model", "layer", "subject"], dropna=False)["method"].transform("nunique")
        return values[counts == 3].reset_index(drop=True)


    def load_loso_auc_values(*, dataset, rrr_baseline_source="loso", results_root=REPO_ROOT / "results"):
        return calculate_loso_summary_values(load_loso_summary_curves(
            dataset=dataset, rrr_baseline_source=rrr_baseline_source, results_root=results_root
        ), metric="auc")


    def plot_loso_auc_distributions(auc_values, dataset_label):
        """Plot paired model-layer AUC distributions for the three LOSO conditions."""
        method_order = ["RRR baseline", "RRR transfer", "PCA baseline"]
        method_colors = {
            "RRR baseline": "#6ab2e2",
            "RRR transfer": "#4b98cc",
            "PCA baseline": "#63bc51",
        }
        means = auc_values.groupby(["model", "layer", "method"], as_index=False)["auc"].mean()
        paired = means.pivot_table(
            index=["model", "layer"], columns="method", values="auc", aggfunc="first"
        ).dropna(subset=method_order).reset_index()
        if paired.empty:
            raise ValueError(f"No model/layer pairs with all LOSO conditions for {dataset_label}")
        distributions = {
            method: paired[method].to_numpy(dtype=float) for method in method_order
        }
        fig, ax = plot_paired_distributions(
            distributions,
            colors=method_colors,
            figsize=(7.0, 4.4),
        )
        positions = np.arange(1, len(method_order) + 1, dtype=float)
        for position, method in zip(positions, method_order):
            mean_auc = float(paired[method].mean())
            ax.scatter(position, mean_auc, marker="D", s=28, color="black", zorder=5)
            ax.annotate(f"mean = {mean_auc:.2f}", (position, mean_auc), xytext=(0, 8), textcoords="offset points", ha="center", va="bottom", fontsize=7)
        auc_min = float(paired[method_order].min().min()); auc_max = float(paired[method_order].max().max()); auc_span = max(auc_max - auc_min, 0.05)
        ax.set(ylabel="Normalized curve AUC", title=f"{dataset_label}: leave-one-subject-out AUC ({paired['model'].nunique()} models, {len(paired)} paired layers)", ylim=(auc_min - 0.08 * auc_span, auc_max + 0.15 * auc_span))
        results = paired_permutation_tests(distributions, [
            {"first": "RRR transfer", "second": "PCA baseline", "alternative": "greater"},
            {"first": "RRR baseline", "second": "RRR transfer", "alternative": "greater"},
        ])
        annotate_pairwise_tests(ax, results, method_order)
        fig.tight_layout()
        return fig, ax


    return calculate_loso_summary_values, plot_loso_auc_distributions


@app.cell(hide_code=True)
def plot_average_loso_normalized_curves(normalized_performance_curve, np, plt):
    def plot_average_loso_normalized_curves(
        correlations, dataset_label, aggregation_label="model layers", grid_size=200
    ):
        """Plot subject-averaged model/layer LOSO curves and their equal-weighted means."""
        method_styles = {
            "RRR baseline": ("#6ab2e2", "RRR baseline"),
            "RRR transfer": ("#4b98cc", "RRR transfer"),
            "PCA baseline": ("#63bc51", "PCA baseline"),
        }
        curves = {}
        for (model, layer, method), frame in correlations.groupby(
            ["model", "layer", "method"], dropna=False, sort=True
        ):
            x, y = normalized_performance_curve(
                frame["rank"], frame["correlation"]
            )
            if x.size:
                curves[(model, layer, method)] = (x, y)
        if not curves:
            raise ValueError(f"No usable normalized LOSO curves for {dataset_label}")

        linthresh = 0.01
        positive_grid = np.geomspace(linthresh / 100, 1.0, grid_size - 1)
        grid = np.concatenate(([0.0], positive_grid))
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        contributor_counts = []
        for method, (color, label) in method_styles.items():
            method_curves = [
                curve for key, curve in curves.items() if key[2] == method
            ]
            if not method_curves:
                continue
            interpolated = np.empty((len(method_curves), grid.size))
            for index, (x, y) in enumerate(method_curves):
                ax.plot(x, y, color=color, alpha=0.10, linewidth=0.8, zorder=1,
                            linestyle="--" if "transfer" in method.lower() else "-")
                interpolated[index] = np.interp(grid, x, y)
            ax.plot(
                grid,
                interpolated.mean(axis=0),
                color=color,
                linestyle="--" if "transfer" in method.lower() else "-",
                linewidth=2.6,
                label=label,
                zorder=3,
            )
            contributor_counts.append(f"{label}: {len(method_curves)}")

        ax.set_xscale("symlog", linthresh=linthresh, linscale=1.0)
        ax.set_xticks(
            [0.0, 0.001, 0.01, 0.1, 1.0],
            ["0%", "0.1%", "1%", "10%", "100%"],
        )
        ax.set(
            xlim=(0.0, 1.0),
            ylim=(-0.02, 1.02),
            xlabel="Relative rank",
            ylabel="Normalized monotonic performance",
            title=f"{dataset_label}: mean LOSO convergence across {aggregation_label}",
        )
        ax.grid(axis="both", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        ax.text(
            0.02,
            0.03,
            "Subject-averaged curves contributing throughout: "
            + "; ".join(contributor_counts),
            transform=ax.transAxes,
            fontsize=8,
            color="#4b5563",
        )
        fig.tight_layout()
        return fig, ax

    return (plot_average_loso_normalized_curves,)


@app.cell
def loso_auc_controls(mo):
    loso_auc_rrr_baseline_source = mo.ui.radio(
        options={
            "LOSO RRR baseline": "loso",
            "Full cross-subject RRR baseline": "full",
        },
        value="LOSO RRR baseline",
        label="RRR baseline source",
    )
    loso_auc_rrr_baseline_source
    return (loso_auc_rrr_baseline_source,)


@app.cell
def _(mo):
    loso_saturation_percent = mo.ui.slider(
        start=1, stop=100, step=1, value=99, show_value=True,
        label="LOSO: percent of maximum correlation", debounce=True,
    )
    loso_saturation_percent
    return (loso_saturation_percent,)


@app.cell(hide_code=True)
def _(
    find_loso_model_layers,
    load_loso_rank_correlations,
    pd,
    plot_model_filter,
):
    loso_rank_curves_by_dataset = {}
    for _dataset in ("podcast", "cars-2", "fmri"):
        _frames = []
        for _model, _layer in find_loso_model_layers(dataset=_dataset):
            if _model not in plot_model_filter.value:
                continue
            try:
                _frame = load_loso_rank_correlations(dataset=_dataset, model=_model, layer=_layer).copy()
            except Exception as _error:
                print(f"Skipping {_dataset}/{_model}/layer-{_layer}: {_error}")
                continue
            _frame["model"] = _model
            _frame["layer"] = _layer
            _frames.append(_frame)
        loso_rank_curves_by_dataset[_dataset] = pd.concat(_frames, ignore_index=True)
    return (loso_rank_curves_by_dataset,)


@app.cell(hide_code=True)
def _(load_loso_summary_curves, plot_model_filter):
    loso_summary_curves_by_source = {
        _source: {
            _dataset: load_loso_summary_curves(dataset=_dataset, rrr_baseline_source=_source)
            for _dataset in ("podcast", "cars-2", "fmri")
        }
        for _source in ("loso", "full")
    }

    loso_summary_curves_by_source = {
        _source: {
            _dataset: _frame[_frame["model"].isin(plot_model_filter.value)].copy()
            for _dataset, _frame in _datasets.items()
        }
        for _source, _datasets in loso_summary_curves_by_source.items()
    }
    return (loso_summary_curves_by_source,)


@app.cell(hide_code=True)
def _(loso_auc_rrr_baseline_source, loso_summary_curves_by_source):
    loso_summary_curves_by_dataset = loso_summary_curves_by_source[loso_auc_rrr_baseline_source.value]
    return (loso_summary_curves_by_dataset,)


@app.cell(hide_code=True)
def _(
    QUANTILE_PEAK_FLOOR,
    filter_normalized_curves,
    loso_auc_rrr_baseline_source,
    loso_curve_depth,
    loso_curve_mode,
    loso_curve_model,
    loso_quantile_saturation,
    loso_saturation_percent,
    loso_summary_curves_by_dataset,
    mo,
    plot_rank_summary_distributions,
):
    loso_saturation_by_dataset = {}
    loso_saturation_plots = []
    for _dataset, _label in (("podcast", "Podcast"), ("cars-2", "Cars 2"), ("fmri", "fMRI")):
        _selected, _ = filter_normalized_curves(
            loso_summary_curves_by_dataset[_dataset], loso_curve_mode.value,
            loso_curve_model.value, loso_curve_depth.value,
        )
        _values = loso_quantile_saturation(
            _dataset, _selected, loso_saturation_percent.value / 100,
            baseline_source=loso_auc_rrr_baseline_source.value,
        )
        loso_saturation_by_dataset[_dataset] = _values
        loso_saturation_plots.append(plot_rank_summary_distributions(
            _values, _label + ": LOSO",
            {"RRR baseline": "RRR baseline", "RRR transfer": "RRR transfer", "PCA baseline": "PCA baseline"},
            {"RRR baseline": "#6ab2e2", "RRR transfer": "#4b98cc", "PCA baseline": "#63bc51"},
            metric="rank", percentage=loso_saturation_percent.value,
            tests=[
                {"first": "RRR transfer", "second": "PCA baseline", "alternative": "less"},
                {"first": "RRR baseline", "second": "RRR transfer", "alternative": "less"},
            ],
        ))
    mo.vstack([mo.md(f"LOSO target quantiles peaking above {QUANTILE_PEAK_FLOOR:g} are calculated per subject; saturation ranks are then averaged per model/layer. Holm-adjusted paired permutation tests compare the requested methods."), *[_plot[0] for _plot in loso_saturation_plots]])
    return


@app.cell(hide_code=True)
def loso_auc_plot(
    calculate_loso_summary_values,
    filter_normalized_curves,
    loso_curve_depth,
    loso_curve_mode,
    loso_curve_model,
    loso_summary_curves_by_dataset,
    mo,
    plot_loso_auc_distributions,
):
    loso_auc_by_dataset = {}
    loso_auc_plots = []
    for _dataset, _label in (("podcast", "Podcast"), ("cars-2", "Cars 2"), ("fmri", "fMRI")):
        _auc_values = calculate_loso_summary_values(loso_summary_curves_by_dataset[_dataset], metric="auc")
        _auc_values, _ = filter_normalized_curves(
            _auc_values, loso_curve_mode.value, loso_curve_model.value, loso_curve_depth.value
        )
        loso_auc_by_dataset[_dataset] = _auc_values
        loso_auc_plots.append(plot_loso_auc_distributions(_auc_values, _label))
    mo.vstack([plot[0] for plot in loso_auc_plots])
    return


@app.cell
def loso_normalized_curve_controls(
    find_loso_model_layers,
    mo,
    normalized_curve_filter_controls,
    plot_model_filter,
):
    _datasets = ("podcast", "cars-2", "fmri")
    _loso_sets = [
        {pair for pair in find_loso_model_layers(dataset=dataset) if pair[0] in plot_model_filter.value}
        for dataset in _datasets
    ]
    _loso_pairs = set.intersection(*_loso_sets)
    (
        loso_curve_mode,
        loso_curve_model,
        loso_curve_depth,
        _loso_controls,
    ) = normalized_curve_filter_controls(mo, _loso_pairs, "LOSO")
    _loso_controls
    return loso_curve_depth, loso_curve_mode, loso_curve_model


@app.cell
def _(fisher_mean, loso_rank_curves_by_dataset):
    loso_mean_curves_by_dataset = {
        _dataset: _frame.groupby(
            ["model", "layer", "method", "rank"], as_index=False, dropna=False
        )["correlation"].agg(fisher_mean)
        for _dataset, _frame in loso_rank_curves_by_dataset.items()
    }
    return (loso_mean_curves_by_dataset,)


@app.cell
def loso_gap_controls(mo):
    loso_gap_grouping = mo.ui.radio(
        options={
            "Model depth (one line per depth)": "depth",
            "Model (average across layers)": "model",
        },
        value="Model depth (one line per depth)",
        label="RRR - PCA grouping",
        inline=True,
    )
    loso_gap_grouping
    return (loso_gap_grouping,)


@app.cell(hide_code=True)
def loso_rrr_pca_gap_plot(
    loso_gap_grouping,
    loso_mean_curves_by_dataset,
    mo,
    plot_rrr_pca_gap_groups,
):
    loso_rrr_pca_gap_plots = []
    for _dataset, _label in (
        ("podcast", "Podcast"),
        ("cars-2", "Cars 2"),
        ("fmri", "fMRI"),
    ):
        loso_rrr_pca_gap_plots.append(
            plot_rrr_pca_gap_groups(
                loso_mean_curves_by_dataset[_dataset],
                [
                    ("RRR transfer", "PCA baseline", "RRR transfer - PCA"),
                ],
                loso_gap_grouping.value,
                _label + " - LOSO",
            )
        )
    mo.vstack([
        mo.md(
            "Faint lines are individual model-layer curves; the thick line "
            "is the mean RRR transfer - PCA gap."
        ),
        *[plot[0] for plot in loso_rrr_pca_gap_plots],
    ])
    return


@app.cell(hide_code=True)
def loso_rrr_pca_gap_auc_plot(
    calculate_rrr_pca_gap_auc_values,
    loso_gap_grouping,
    loso_mean_curves_by_dataset,
    mo,
    plot_rrr_pca_gap_auc_distributions,
):
    loso_rrr_pca_gap_auc_plots = []
    for _dataset, _label in (
        ("podcast", "Podcast"),
        ("cars-2", "Cars 2"),
        ("fmri", "fMRI"),
    ):
        _auc_values = calculate_rrr_pca_gap_auc_values(
            loso_mean_curves_by_dataset[_dataset],
            rrr_method="RRR transfer",
            pca_method="PCA baseline",
        )
        loso_rrr_pca_gap_auc_plots.append(
            plot_rrr_pca_gap_auc_distributions(
                _auc_values,
                loso_gap_grouping.value,
                _label + " - LOSO",
            )
        )
    mo.vstack([
        mo.md(
            "Each point is one model-layer signed AUC of the RRR transfer - PCA "
            "gap over normalized log-rank. Lines connect paired model layers."
        ),
        *[plot[0] for plot in loso_rrr_pca_gap_auc_plots],
    ])
    return


@app.cell(hide_code=True)
def loso_normalized_curve_plot(
    filter_normalized_curves,
    loso_curve_depth,
    loso_curve_mode,
    loso_curve_model,
    loso_mean_curves_by_dataset,
    mo,
    plot_average_loso_normalized_curves,
):
    loso_normalized_curve_plots = []
    for _dataset, _label in (
        ("podcast", "Podcast"),
        ("cars-2", "Cars 2"),
        ("fmri", "fMRI"),
    ):
        _loso_curves = loso_mean_curves_by_dataset[_dataset]
        _loso_curves, _aggregation_label = filter_normalized_curves(
            _loso_curves,
            loso_curve_mode.value,
            loso_curve_model.value,
            loso_curve_depth.value,
        )
        loso_normalized_curve_plots.append(
            plot_average_loso_normalized_curves(
                _loso_curves, _label, aggregation_label=_aggregation_label
            )
        )
    mo.vstack([
        mo.md(
            "Subjects are averaged at each rank within a model layer before curve "
            "normalization. Each faint line is one model layer; thick lines are "
            "equal-weighted means across model layers."
        ),
        *[plot[0] for plot in loso_normalized_curve_plots],
    ])
    return


@app.cell(hide_code=True)
def loso_rank_plot(
    loso_model_layer_selector,
    loso_rank_curves_by_dataset,
    mo,
    plot_loso_rank_correlations,
    select_model_layer,
):
    loso_selected_model, loso_selected_layer = loso_model_layer_selector.value
    loso_plots = []
    for _dataset, _label in (("podcast", "Podcast"), ("cars-2", "Cars 2"), ("fmri", "fMRI")):
        _loso_frame = select_model_layer(loso_rank_curves_by_dataset[_dataset], loso_selected_model, loso_selected_layer)
        loso_plots.append(
            plot_loso_rank_correlations(
                _loso_frame,
                _label,
                loso_selected_model,
                loso_selected_layer,
            )
        )
    mo.vstack([plot[0] for plot in loso_plots])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Cross Dataset
    """)
    return


@app.cell(hide_code=True)
def _(
    Artifact,
    DATASET_ROOTS,
    FMRI_INPUT_SCALING_VARIANT,
    Path,
    REPO_ROOT,
    fisher_mean,
    internal_rrr_artifact,
    lru_cache,
    np,
    pd,
):
    def find_cross_model_layers(
        *,
        results_root=REPO_ROOT / "results",
        dataset="podcast",
    ):
        """Find model/layer pairs with saved cross-dataset results."""
        models_root = (
            Path(results_root) / "run_rrr" / "cross_dataset"
            / DATASET_ROOTS[dataset] / "offset-0"
        )
        pairs = []
        if not models_root.is_dir():
            return pairs
        for model_root in models_root.iterdir():
            if not model_root.is_dir():
                continue
            for layer_root in model_root.glob("layer-*"):
                if layer_root.is_dir() and (layer_root / "manifest.json").is_file():
                    suffix = layer_root.name.removeprefix("layer-")
                    if suffix == "none":
                        pairs.append((model_root.name, None))
                    elif suffix.isdigit():
                        pairs.append((model_root.name, int(suffix)))
        return sorted(set(pairs), key=lambda item: (item[0], item[1] is not None, item[1] or -1))


    def _cross_dataset_metric_value(metrics, metric_name):
        values = np.asarray(metrics[metric_name], dtype=float)
        if values.ndim:
            return float(
                np.tanh(np.nanmean(np.arctanh(np.clip(values, -1 + 1e-7, 1 - 1e-7))))
            )
        return float(values)


    @lru_cache(maxsize=None)
    def load_cross_dataset_rank_correlations(
        *,
        results_root=REPO_ROOT / "results",
        dataset="podcast",
        model="gpt2",
        layer=None,
    ):
        """Load within and directed cross-dataset rank correlations by fold."""
        is_fmri = dataset == "fmri"
        dataset_labels = {
            "podcast": "Podcast",
            "cars-2": "Cars 2",
            "brain-treebank-cars-2": "Cars 2",
            "fmri": "fMRI",
            "deep-fmri": "fMRI",
        }
        target_label = dataset_labels[dataset]
        rows = []

        internal = internal_rrr_artifact(
            dataset=dataset, model=model, layer=layer, results_root=results_root,
        ).load_artifacts(["rank_metrics"])
        for selector, metrics in internal.rank_metrics.items():
            selector = dict(selector)
            method = selector.get("method")
            if method not in {"rrr", "pca"} or "rank" not in selector:
                continue
            metric_name = (
                "corr_per_target" if is_fmri
                else ("nr_orth_corr" if method == "rrr" else "pca_corr")
            )
            if metric_name not in metrics:
                continue
            rows.append(
                {
                    "fold": selector.get("fold", 0),
                    "method": "Within RRR" if method == "rrr" else "Within PCA",
                    "rank": int(selector["rank"]),
                    "correlation": _cross_dataset_metric_value(metrics, metric_name),
                }
            )

        cross_base = Artifact(
            mode="cross_dataset", dataset=DATASET_ROOTS[dataset], model=model,
            layer=layer, results_root=results_root,
        )
        index = cross_base._read_json(cross_base._leaf() / "manifest.json", default={})
        variants = sorted(index.get("variants", {}))
        if is_fmri:
            variants = [
                variant
                for variant in variants
                if FMRI_INPUT_SCALING_VARIANT in variant.split("/")
            ]
        if not variants:
            qualifier = (
                f" with {FMRI_INPUT_SCALING_VARIANT}"
                if is_fmri
                else ""
            )
            raise ValueError(
                f"No cross-dataset variants{qualifier} for "
                f"{dataset}/{model}/layer-{layer}"
            )
        for variant in variants:
            variant_parts = variant.split("/")
            try:
                source_mode = "full_fit" if "full_fit" in variant_parts else "internal"
                source_dataset = variant_parts[variant_parts.index(source_mode) + 1]
            except (ValueError, IndexError) as error:
                raise ValueError(f"Cannot identify source dataset from variant {variant!r}") from error
            source_label = dataset_labels.get(source_dataset, source_dataset)
            direction = f"{source_label} → {target_label}"
            transfer_artifact = Artifact(
                mode="cross_dataset", dataset=DATASET_ROOTS[dataset], model=model,
                layer=layer, variant=variant, results_root=results_root,
            )
            variant_manifest = transfer_artifact._manifest_path()
            if not variant_manifest.is_file():
                print(f"Skipping unavailable cross-dataset variant: {variant_manifest}")
                continue
            transfer = transfer_artifact.load_artifacts(["rank_metrics"])
            for selector, metrics in transfer.rank_metrics.items():
                selector = dict(selector)
                method = selector.get("method")
                if method not in {"rrr", "pca"} or "rank" not in selector:
                    continue
                if is_fmri:
                    metric_name = "corr_per_target"
                elif method == "rrr":
                    metric_name = "nr_orth_corr" if "nr_orth_corr" in metrics else "nr_corr"
                else:
                    metric_name = "pca_corr"
                if metric_name not in metrics:
                    continue
                rows.append(
                    {
                        "fold": selector.get("fold", 0),
                        "method": f"Transfer {'RRR' if method == 'rrr' else 'PCA'}: {direction}",
                        "rank": int(selector["rank"]),
                        "correlation": _cross_dataset_metric_value(metrics, metric_name),
                    }
                )
        if not rows:
            raise ValueError(f"No cross-dataset rank metrics for {dataset}/{model}/layer-{layer}")
        return pd.DataFrame(rows).groupby(
            ["fold", "method", "rank"], as_index=False
        )["correlation"].agg(fisher_mean)


    return find_cross_model_layers, load_cross_dataset_rank_correlations


@app.cell(hide_code=True)
def load_cross_dataset_rank_correlations(fisher_mean, plt):
    def plot_cross_dataset_rank_correlations(frame, dataset_label, model, layer):
        """Plot within and directed cross-dataset rank curves by fold."""
        transfer_methods = sorted(
            method for method in frame["method"].unique()
            if method.startswith("Transfer ")
        )
        method_order = ["Within RRR"] + [
            method for method in transfer_methods if method.startswith("Transfer RRR:")
        ] + ["Within PCA"] + [
            method for method in transfer_methods if method.startswith("Transfer PCA:")
        ]
        method_colors = {"Within RRR": "#6ab2e2", "Within PCA": "#63bc51"}
        for method in transfer_methods:
            method_colors[method] = (
                "#4b98cc" if method.startswith("Transfer RRR:") else "#45963a"
            )

        ranks = sorted(frame["rank"].unique())
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for method in method_order:
            method_frame = frame[frame["method"] == method]
            for fold, fold_frame in method_frame.groupby("fold", sort=True):
                fold_frame = fold_frame.sort_values("rank")
                ax.plot(
                    fold_frame["rank"], fold_frame["correlation"],
                    color=method_colors[method], alpha=0.18, linewidth=0.8, zorder=1,
                        linestyle="--" if method.startswith("Transfer ") else "-",
                )
            mean_frame = (
                method_frame.groupby("rank", as_index=False)["correlation"]
                .agg(fisher_mean).sort_values("rank")
            )
            ax.plot(
                mean_frame["rank"], mean_frame["correlation"],
                color=method_colors[method], linewidth=2.2, marker="o", markersize=3.5,
                label=method, zorder=3,
                    linestyle="--" if method.startswith("Transfer ") else "-",
            )
        ax.set(
            xlabel="RRR rank $k$",
            ylabel="Correlation",
            title=(
                f"{dataset_label}: within vs directed cross-dataset performance\n"
                f"{model} / {'layer '+str(layer) if layer is not None else 'layerless'}"
            ),
            xscale="log",
            xticks=ranks,
            xticklabels=[str(rank) for rank in ranks],
        )
        ax.tick_params(axis="x", labelrotation=90)
        ax.grid(axis="both", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        return fig, ax


    def plot_cross_dataset_rank_grid(frames_by_target, model, layer):
        """Plot one source-to-target transfer pair per cell in a 3x2 grid."""
        dataset_labels = {"podcast": "Podcast", "cars-2": "Cars 2", "fmri": "fMRI"}
        dataset_order = ["podcast", "cars-2", "fmri"]
        method_colors = {
            "Within RRR": "#6ab2e2",
            "Transfer RRR": "#4b98cc",
            "Within PCA": "#63bc51",
            "Transfer PCA": "#45963a",
        }
        fig, axes = plt.subplots(3, 2, figsize=(13.0, 12.0), squeeze=False)
        legend_handles = []
        legend_labels = []
        for row_index, target_dataset in enumerate(dataset_order):
            frame = frames_by_target[target_dataset]
            source_datasets = [source for source in dataset_order if source != target_dataset]
            target_label = dataset_labels[target_dataset]
            for column_index, source_dataset in enumerate(source_datasets):
                source_label = dataset_labels[source_dataset]
                direction = f"{source_label} → {target_label}"
                selected_methods = {
                    "Within RRR": "Within RRR",
                    f"Transfer RRR: {direction}": "Transfer RRR",
                    "Within PCA": "Within PCA",
                    f"Transfer PCA: {direction}": "Transfer PCA",
                }
                ax = axes[row_index, column_index]
                ranks = sorted(frame["rank"].unique())
                for source_method, display_method in selected_methods.items():
                    method_frame = frame[frame["method"] == source_method]
                    if method_frame.empty:
                        continue
                    for _, fold_frame in method_frame.groupby("fold", sort=True):
                        fold_frame = fold_frame.sort_values("rank")
                        ax.plot(
                            fold_frame["rank"], fold_frame["correlation"],
                            color=method_colors[display_method], alpha=0.16,
                            linewidth=0.75, zorder=1,
                            linestyle="--" if display_method.startswith("Transfer ") else "-",
                        )
                    mean_frame = (
                        method_frame.groupby("rank", as_index=False)["correlation"]
                        .agg(fisher_mean).sort_values("rank")
                    )
                    mean_line = ax.plot(
                        mean_frame["rank"], mean_frame["correlation"],
                        color=method_colors[display_method], linewidth=2.0,
                        marker="o", markersize=3.0, label=display_method, zorder=3,
                        linestyle="--" if display_method.startswith("Transfer ") else "-",
                    )[0]
                    if display_method not in legend_labels:
                        legend_handles.append(mean_line)
                        legend_labels.append(display_method)
                ax.set(
                    xlabel="RRR rank $k$",
                    ylabel="Correlation",
                    title=direction,
                    xscale="log",
                    xticks=ranks,
                    xticklabels=[str(rank) for rank in ranks],
                )
                ax.tick_params(axis="x", labelrotation=90)
                ax.grid(axis="both", alpha=0.22)
                ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(
            f"Directed cross-dataset performance\n"
            f"{model} / {'layer '+str(layer) if layer is not None else 'layerless'}",
            y=0.995,
        )
        if legend_handles:
            fig.legend(
                legend_handles, legend_labels, loc="lower center",
                ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.005),
            )
        fig.tight_layout(rect=(0, 0.045, 1, 0.97))
        return fig


    return (plot_cross_dataset_rank_grid,)


@app.cell(hide_code=True)
def plot_cross_dataset_normalized_grid(normalized_performance_curve, np, plt):
    def plot_cross_dataset_normalized_grid(
        curves, aggregation_label="model layers", grid_size=200
    ):
        """Plot four-way normalized convergence for every directed dataset pair."""
        dataset_labels = {"podcast": "Podcast", "cars-2": "Cars 2", "fmri": "fMRI"}
        dataset_order = ["podcast", "cars-2", "fmri"]
        method_styles = {
            "Within RRR": ("#6ab2e2", "RRR"),
            "Within PCA": ("#63bc51", "PCA"),
            "Transfer RRR": ("#4b98cc", "RRR transfer"),
            "Transfer PCA": ("#45963a", "PCA transfer"),
        }
        linthresh = 0.01
        positive_grid = np.geomspace(linthresh / 100, 1.0, grid_size - 1)
        grid = np.concatenate(([0.0], positive_grid))
        fig, axes = plt.subplots(3, 2, figsize=(13.0, 12.0), squeeze=False)
        legend_handles = []
        legend_labels = []

        for row_index, target_dataset in enumerate(dataset_order):
            source_datasets = [dataset for dataset in dataset_order if dataset != target_dataset]
            for column_index, source_dataset in enumerate(source_datasets):
                ax = axes[row_index, column_index]
                panel = curves[
                    (curves["target_dataset"] == target_dataset)
                    & (curves["source_dataset"] == source_dataset)
                ]
                counts = []
                for method, (color, label) in method_styles.items():
                    method_curves = []
                    for _, frame in panel[panel["method"] == method].groupby(
                        ["model", "layer"], dropna=False, sort=True
                    ):
                        x, y = normalized_performance_curve(
                            frame["rank"], frame["correlation"]
                        )
                        if x.size:
                            method_curves.append((x, y))
                    if not method_curves:
                        continue
                    interpolated = np.empty((len(method_curves), grid.size))
                    for index, (x, y) in enumerate(method_curves):
                        ax.plot(x, y, color=color, alpha=0.10, linewidth=0.8, zorder=1,
                                linestyle="--" if method.startswith("Transfer ") else "-")
                        interpolated[index] = np.interp(grid, x, y)
                    mean_line = ax.plot(
                        grid,
                        interpolated.mean(axis=0),
                        color=color,
                        linewidth=2.4,
                        label=label,
                        zorder=3,
                        linestyle="--" if method.startswith("Transfer ") else "-",
                    )[0]
                    counts.append(f"{label}: {len(method_curves)}")
                    if label not in legend_labels:
                        legend_handles.append(mean_line)
                        legend_labels.append(label)

                ax.set_xscale("symlog", linthresh=linthresh, linscale=1.0)
                ax.set_xticks(
                    [0.0, 0.001, 0.01, 0.1, 1.0],
                    ["0%", "0.1%", "1%", "10%", "100%"],
                )
                ax.set(
                    xlim=(0.0, 1.0),
                    ylim=(-0.02, 1.02),
                    xlabel="Relative rank",
                    ylabel="Normalized monotonic performance",
                    title=f"{dataset_labels[source_dataset]} → {dataset_labels[target_dataset]}",
                )
                ax.grid(axis="both", alpha=0.22)
                ax.spines[["top", "right"]].set_visible(False)
                ax.text(
                    0.02,
                    0.03,
                    "; ".join(counts),
                    transform=ax.transAxes,
                    fontsize=7,
                    color="#4b5563",
                )

        fig.suptitle(
            f"Mean cross-dataset convergence across {aggregation_label}", y=0.995
        )
        if legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                ncol=4,
                frameon=False,
                bbox_to_anchor=(0.5, 0.005),
            )
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        return fig, axes

    return (plot_cross_dataset_normalized_grid,)


@app.cell(hide_code=True)
def cross_dataset_auc_helpers(
    annotate_pairwise_tests,
    normalized_log_rank_auc,
    np,
    paired_permutation_tests,
    pd,
    plot_paired_distributions,
):
    def calculate_cross_dataset_auc_values(curves):
        """Calculate normalized-curve AUC per model layer and directed method."""
        rows = []
        for keys, frame in curves.groupby(
            ["target_dataset", "source_dataset", "model", "layer", "method"],
            dropna=False,
            sort=True,
        ):
            auc = normalized_log_rank_auc(frame["rank"], frame["correlation"])
            if np.isfinite(auc):
                target_dataset, source_dataset, model, layer, method = keys
                rows.append({
                    "target_dataset": target_dataset,
                    "source_dataset": source_dataset,
                    "model": model,
                    "layer": layer,
                    "method": method,
                    "auc": auc,
                })
        if not rows:
            raise ValueError("No usable cross-dataset curves for AUC")
        return pd.DataFrame(rows)


    def plot_cross_dataset_auc_distribution(
        auc_values,
        source_label,
        target_label,
    ):
        """Plot paired four-method AUC distributions for one directed dataset pair."""
        method_order = ["Within RRR", "Within PCA", "Transfer RRR", "Transfer PCA"]
        display_labels = {
            "Within RRR": "RRR",
            "Within PCA": "PCA",
            "Transfer RRR": "RRR transfer",
            "Transfer PCA": "PCA transfer",
        }
        method_colors = {
            "RRR": "#6ab2e2",
            "PCA": "#63bc51",
            "RRR transfer": "#4b98cc",
            "PCA transfer": "#45963a",
        }
        frame = auc_values.copy()
        frame["layer_key"] = frame["layer"].astype(object).where(
            frame["layer"].notna(), "none"
        )
        paired = (
            frame.pivot_table(
                index=["model", "layer_key"],
                columns="method",
                values="auc",
                aggfunc="first",
            )
            .dropna(subset=method_order)
            .reset_index()
        )
        if paired.empty:
            raise ValueError(f"No complete four-method AUC pairs for {source_label} → {target_label}")
        distributions = {
            display_labels[method]: paired[method].to_numpy(dtype=float)
            for method in method_order
        }
        fig, ax = plot_paired_distributions(
            distributions,
            colors=method_colors,
            figsize=(7.4, 4.6),
        )
        for position, method in enumerate(method_order, start=1):
            mean_auc = float(paired[method].mean())
            ax.scatter(position, mean_auc, marker="D", s=28, color="black", zorder=5)
            ax.annotate(
                f"mean = {mean_auc:.2f}",
                (position, mean_auc),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        auc_min = float(paired[method_order].min().min())
        auc_max = float(paired[method_order].max().max())
        auc_span = max(auc_max - auc_min, 0.05)
        ax.set(
            ylabel="Normalized curve AUC",
            title=(
                f"{source_label} to {target_label}: cross-dataset convergence AUC "
                f"({paired['model'].nunique()} models, {len(paired)} paired layers)"
            ),
            ylim=(auc_min - 0.08 * auc_span, auc_max + 0.15 * auc_span),
        )
        results = paired_permutation_tests(distributions, [
            {"first": "RRR transfer", "second": "PCA transfer", "alternative": "greater"},
            {"first": "RRR", "second": "PCA", "alternative": "greater"},
            {"first": "RRR", "second": "RRR transfer", "alternative": "greater"},
            {"first": "PCA", "second": "RRR transfer", "alternative": "greater"},
        ])
        annotate_pairwise_tests(ax, results, list(distributions))
        fig.tight_layout()
        return fig, ax

    return (
        calculate_cross_dataset_auc_values,
        plot_cross_dataset_auc_distribution,
    )


@app.cell(hide_code=True)
def cross_dataset_rank_plot(
    cross_model_layer_selector,
    cross_rank_curves_by_dataset,
    plot_cross_dataset_rank_grid,
    select_model_layer,
):
    cross_selected_model, cross_selected_layer = cross_model_layer_selector.value
    cross_frames_by_target = {
        _dataset: select_model_layer(cross_rank_curves_by_dataset[_dataset], cross_selected_model, cross_selected_layer)
        for _dataset in ("podcast", "cars-2", "fmri")
    }
    plot_cross_dataset_rank_grid(
        cross_frames_by_target,
        cross_selected_model,
        cross_selected_layer,
    )
    return


@app.cell(hide_code=True)
def cross_dataset_normalized_curve_controls(
    find_cross_model_layers,
    mo,
    normalized_curve_filter_controls,
    plot_model_filter,
):
    _datasets = ("podcast", "cars-2", "fmri")
    _cross_sets = [
        set(find_cross_model_layers(dataset=dataset)) for dataset in _datasets
    ]
    _cross_pairs = {pair for pair in set.union(*_cross_sets) if pair[0] in plot_model_filter.value}
    (
        cross_curve_mode,
        cross_curve_model,
        cross_curve_depth,
        _cross_controls,
    ) = normalized_curve_filter_controls(mo, _cross_pairs, "Cross-dataset")
    _cross_controls
    return cross_curve_depth, cross_curve_mode, cross_curve_model


@app.cell(hide_code=True)
def _(
    find_cross_model_layers,
    load_cross_dataset_rank_correlations,
    pd,
    plot_model_filter,
):
    cross_rank_curves_by_dataset = {}
    for _dataset in ("podcast", "cars-2", "fmri"):
        _frames = []
        for _model, _layer in find_cross_model_layers(dataset=_dataset):
            if _model not in plot_model_filter.value:
                continue
            try:
                _frame = load_cross_dataset_rank_correlations(dataset=_dataset, model=_model, layer=_layer).copy()
            except Exception as _error:
                print(f"Skipping {_dataset}/{_model}/layer-{_layer}: {_error}")
                continue
            _frame["model"] = _model
            _frame["layer"] = _layer
            _frames.append(_frame)
        cross_rank_curves_by_dataset[_dataset] = pd.concat(_frames, ignore_index=True)
    return (cross_rank_curves_by_dataset,)


@app.cell(hide_code=True)
def _(cross_rank_curves_by_dataset, fisher_mean, pd):
    def load_all_cross_dataset_normalized_curves():
        """Load complete four-method curves, averaging folds within each model layer."""
        dataset_labels = {"podcast": "Podcast", "cars-2": "Cars 2", "fmri": "fMRI"}
        rows = []
        for target_dataset in dataset_labels:
            target_label = dataset_labels[target_dataset]
            for (model, layer), frame in cross_rank_curves_by_dataset[target_dataset].groupby(
                ["model", "layer"], dropna=False, sort=True
            ):
                fold_mean = (
                    frame.groupby(["method", "rank"], as_index=False)["correlation"]
                    .agg(fisher_mean)
                )
                directions = sorted({
                    method.split(": ", 1)[1]
                    for method in fold_mean["method"].unique()
                    if method.startswith("Transfer ") and ": " in method
                })
                for direction in directions:
                    source_label, direction_target_label = direction.split(" → ", 1)
                    if direction_target_label != target_label:
                        continue
                    source_dataset = next(
                        (
                            dataset
                            for dataset, label in dataset_labels.items()
                            if label == source_label
                        ),
                        None,
                    )
                    if source_dataset is None:
                        continue
                    method_map = {
                        "Within RRR": "Within RRR",
                        "Within PCA": "Within PCA",
                        f"Transfer RRR: {direction}": "Transfer RRR",
                        f"Transfer PCA: {direction}": "Transfer PCA",
                    }
                    if not set(method_map).issubset(set(fold_mean["method"])):
                        continue
                    for source_method, display_method in method_map.items():
                        method_frame = fold_mean[fold_mean["method"] == source_method]
                        for row in method_frame.itertuples(index=False):
                            rows.append({
                                "target_dataset": target_dataset,
                                "source_dataset": source_dataset,
                                "model": model,
                                "layer": layer,
                                "method": display_method,
                                "rank": row.rank,
                                "correlation": row.correlation,
                            })
        if not rows:
            raise ValueError("No complete four-method cross-dataset curves found")
        return pd.DataFrame(rows)

    cross_dataset_normalized_curves = load_all_cross_dataset_normalized_curves()
    return (cross_dataset_normalized_curves,)


@app.cell(hide_code=True)
def cross_gap_controls(mo):
    cross_gap_grouping = mo.ui.radio(
        options={
            "Model depth (one line per depth)": "depth",
            "Model (average across layers)": "model",
        },
        value="Model depth (one line per depth)",
        label="RRR - PCA grouping",
        inline=True,
    )
    cross_gap_grouping
    return (cross_gap_grouping,)


@app.cell(hide_code=True)
def cross_rrr_pca_gap_plot(
    cross_dataset_normalized_curves,
    cross_gap_grouping,
    mo,
    plot_rrr_pca_gap_groups,
):
    _cross_gap_labels = {
        "podcast": "Podcast",
        "cars-2": "Cars 2",
        "fmri": "fMRI",
    }
    _cross_gap_figures = []
    for _target_dataset in ("podcast", "cars-2", "fmri"):
        for _source_dataset in (
            dataset for dataset in ("podcast", "cars-2", "fmri")
            if dataset != _target_dataset
        ):
            _panel = cross_dataset_normalized_curves[
                (cross_dataset_normalized_curves["target_dataset"] == _target_dataset)
                & (cross_dataset_normalized_curves["source_dataset"] == _source_dataset)
            ]
            _cross_gap_figures.append(
                plot_rrr_pca_gap_groups(
                    _panel,
                    [
                        ("Transfer RRR", "Transfer PCA", "Transfer RRR - PCA"),
                    ],
                    cross_gap_grouping.value,
                    f"{_cross_gap_labels[_source_dataset]} to {_cross_gap_labels[_target_dataset]}",
                )
            )
    mo.vstack([
        mo.md(
            "Faint lines are individual model-layer curves; the thick line "
            "is the mean Transfer RRR - Transfer PCA gap."
        ),
        *[figure[0] for figure in _cross_gap_figures],
    ])
    return


@app.cell(hide_code=True)
def cross_rrr_pca_gap_auc_plot(
    calculate_rrr_pca_gap_auc_values,
    cross_dataset_normalized_curves,
    cross_gap_grouping,
    mo,
    plot_rrr_pca_gap_auc_distributions,
):
    _cross_auc_labels = {
        "podcast": "Podcast",
        "cars-2": "Cars 2",
        "fmri": "fMRI",
    }
    cross_rrr_pca_gap_auc_plots = []
    for _target_dataset in ("podcast", "cars-2", "fmri"):
        for _source_dataset in (
            dataset for dataset in ("podcast", "cars-2", "fmri")
            if dataset != _target_dataset
        ):
            _panel = cross_dataset_normalized_curves[
                (cross_dataset_normalized_curves["target_dataset"] == _target_dataset)
                & (cross_dataset_normalized_curves["source_dataset"] == _source_dataset)
            ]
            _auc_values = calculate_rrr_pca_gap_auc_values(
                _panel,
                rrr_method="Transfer RRR",
                pca_method="Transfer PCA",
            )
            cross_rrr_pca_gap_auc_plots.append(
                plot_rrr_pca_gap_auc_distributions(
                    _auc_values,
                    cross_gap_grouping.value,
                    f"{_cross_auc_labels[_source_dataset]} to {_cross_auc_labels[_target_dataset]}",
                )
            )
    mo.vstack([
        mo.md(
            "Each point is one model-layer signed AUC of the Transfer RRR - "
            "Transfer PCA gap over normalized log-rank. Lines connect paired "
            "model layers."
        ),
        *[plot[0] for plot in cross_rrr_pca_gap_auc_plots],
    ])
    return


@app.cell(hide_code=True)
def cross_dataset_normalized_plot(
    cross_curve_depth,
    cross_curve_mode,
    cross_curve_model,
    cross_dataset_normalized_curves,
    filter_normalized_curves,
    mo,
    plot_cross_dataset_normalized_grid,
):
    _selected_cross_curves, _aggregation_label = filter_normalized_curves(
        cross_dataset_normalized_curves,
        cross_curve_mode.value,
        cross_curve_model.value,
        cross_curve_depth.value,
    )
    cross_dataset_normalized_plot = plot_cross_dataset_normalized_grid(
        _selected_cross_curves, aggregation_label=_aggregation_label
    )
    mo.vstack([
        mo.md(
            "Folds are averaged within each model layer before normalization. Each "
            "faint line is one model layer; thick lines are equal-weighted means. "
            "Panels include only model layers with all four within/transfer methods."
        ),
        cross_dataset_normalized_plot[0],
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    cross_dataset_saturation_percent = mo.ui.slider(
        start=1, stop=100, step=1, value=99, show_value=True,
        label="Cross-dataset: percent of maximum correlation", debounce=True,
    )
    cross_dataset_saturation_percent
    return (cross_dataset_saturation_percent,)


@app.cell(hide_code=True)
def _(
    QUANTILE_PEAK_FLOOR,
    cross_curve_depth,
    cross_curve_mode,
    cross_curve_model,
    cross_dataset_normalized_curves,
    cross_dataset_saturation_percent,
    cross_quantile_saturation,
    filter_normalized_curves,
    mo,
    plot_rank_summary_distributions,
):
    _selected_cross_curves, _ = filter_normalized_curves(
        cross_dataset_normalized_curves,
        cross_curve_mode.value, cross_curve_model.value, cross_curve_depth.value,
    )
    cross_dataset_saturation_values = cross_quantile_saturation(
        _selected_cross_curves, cross_dataset_saturation_percent.value / 100,
    )
    cross_dataset_saturation_plots = []
    _labels = {"podcast": "Podcast", "cars-2": "Cars 2", "fmri": "fMRI"}
    for (_target, _source), _values in cross_dataset_saturation_values.groupby(["target_dataset", "source_dataset"]):
        cross_dataset_saturation_plots.append(plot_rank_summary_distributions(
            _values, f"{_labels[_source]} to {_labels[_target]}",
            {"RRR": "Within RRR", "PCA": "Within PCA", "RRR transfer": "Transfer RRR", "PCA transfer": "Transfer PCA"},
            {"RRR": "#6ab2e2", "PCA": "#63bc51", "RRR transfer": "#4b98cc", "PCA transfer": "#45963a"},
            metric="rank", percentage=cross_dataset_saturation_percent.value,
            tests=[
                {"first": "RRR transfer", "second": "PCA transfer", "alternative": "less"},
                {"first": "RRR", "second": "PCA", "alternative": "less"},
                {"first": "RRR", "second": "RRR transfer", "alternative": "less"},
                {"first": "PCA", "second": "RRR transfer", "alternative": "less"},
            ],
        ))
    mo.vstack([mo.md(f"Cross-dataset saturation uses per-target fold means and requires all quantiles peaking above {QUANTILE_PEAK_FLOOR:g} to cross their thresholds. Each direction pairs model layers across all four methods."), *[_plot[0] for _plot in cross_dataset_saturation_plots]])
    return


@app.cell(hide_code=True)
def cross_dataset_auc_plot(
    calculate_cross_dataset_auc_values,
    cross_curve_depth,
    cross_curve_mode,
    cross_curve_model,
    cross_dataset_normalized_curves,
    filter_normalized_curves,
    mo,
    plot_cross_dataset_auc_distribution,
):
    _selected_cross_curves, _ = filter_normalized_curves(
        cross_dataset_normalized_curves,
        cross_curve_mode.value, cross_curve_model.value, cross_curve_depth.value,
    )
    cross_dataset_auc_values = calculate_cross_dataset_auc_values(
        _selected_cross_curves
    )
    _cross_dataset_labels = {
        "podcast": "Podcast",
        "cars-2": "Cars 2",
        "fmri": "fMRI",
    }
    cross_dataset_auc_plots = []
    for _target_dataset in ("podcast", "cars-2", "fmri"):
        for _source_dataset in (
            dataset for dataset in ("podcast", "cars-2", "fmri")
            if dataset != _target_dataset
        ):
            _pair_auc = cross_dataset_auc_values[
                (cross_dataset_auc_values["target_dataset"] == _target_dataset)
                & (cross_dataset_auc_values["source_dataset"] == _source_dataset)
            ]
            cross_dataset_auc_plots.append(
                plot_cross_dataset_auc_distribution(
                    _pair_auc,
                    _cross_dataset_labels[_source_dataset],
                    _cross_dataset_labels[_target_dataset],
                )
            )
    mo.vstack([
        mo.md(
            "Each plot is one directed source→target comparison. Points are paired "
            "model layers with all four methods; folds were averaged before curve "
            "normalization and AUC calculation."
        ),
        *[plot[0] for plot in cross_dataset_auc_plots],
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Language Decoding
    """)
    return


@app.cell(hide_code=True)
def language_decoding_helpers(
    LanguageDecodingArtifact,
    Path,
    REPO_ROOT,
    json,
    pd,
):
    def load_language_decoding_summaries(
        *, results_root=REPO_ROOT / "results"
    ):
        """Discover every currently saved decoding summary, including partial runs."""
        _root = Path(results_root) / "run_language_decoding"
        _frames = []
        for _mode in ("within_dataset", "cross_dataset"):
            _dataset_glob = "*"
            for _index_path in _root.glob(
                f"{_mode}/{_dataset_glob}/offset-*/*/layer-*/manifest.json"
            ):
                try:
                    _index = json.loads(_index_path.read_text(encoding="utf-8"))
                    _identity = _index["identity"]
                except (OSError, KeyError, json.JSONDecodeError):
                    continue
                for _variant, _state in _index.get("variants", {}).items():
                    _artifact = LanguageDecodingArtifact(
                        mode=_identity["mode"],
                        dataset=_identity["dataset"],
                        offset=_identity["offset"],
                        model=_identity["model"],
                        layer=_identity["layer"],
                        variant=_variant,
                        results_root=results_root,
                    )
                    try:
                        _summary = _artifact.load_field("summary").copy()
                        _manifest = _artifact.manifest()
                    except (FileNotFoundError, KeyError, OSError):
                        continue
                    _upstream = next(
                        (
                            _item for _item in _manifest.get("upstreams", [])
                            if _item.get("script") == "run_rrr"
                        ),
                        {},
                    )
                    _source_dataset = _upstream.get("dataset")
                    _configuration = _manifest.get("configuration", {})
                    _label_offset = int(_configuration.get("label_offset", 0))
                    _summary["target_dataset"] = _identity["dataset"]
                    _summary["model"] = _identity["model"]
                    _summary["layer"] = _identity["layer"]
                    _summary["generation"] = (
                        "cross_dataset"
                        if _mode == "cross_dataset" or _variant.startswith("source-")
                        else "internal"
                    )
                    _summary["source_dataset"] = _source_dataset
                    _summary["label_offset"] = _label_offset
                    _summary["variant"] = _variant
                    _summary["artifact_complete"] = bool(_state.get("complete"))
                    _frames.append(_summary)
        if not _frames:
            return pd.DataFrame()
        return pd.concat(_frames, ignore_index=True)


    def language_decoding_dataset_label(_dataset):
        return {
            "podcast": "Podcast",
            "brain-treebank-cars-2": "Cars 2",
            "deep-fmri": "fMRI",
        }.get(_dataset, str(_dataset))


    def language_decoding_space_label(_generation, _source_dataset):
        if _generation == "internal":
            return "Internal space"
        return f"Cross-dataset space from {language_decoding_dataset_label(_source_dataset)}"


    def language_decoding_metric_label(_metric):
        return {
            "primary_score": "Primary score",
            "correlation": "Correlation",
            "r2": "R²",
            "multivariate_r2": "Multivariate R²",
            "mean_cosine_similarity": "Mean cosine similarity",
            "balanced_accuracy": "Balanced accuracy",
            "macro_f1": "Macro F1",
            "accuracy": "Accuracy",
        }.get(_metric, str(_metric).replace("_", " ").title())


    def select_language_decoding_rows(
        _results, _dataset, _model_layer, _space=None, _label_offset=None
    ):
        _model, _layer = _model_layer
        _selected = _results[
            (_results["target_dataset"] == _dataset)
            & (_results["model"] == _model)
            & (
                _results["layer"].isna()
                if _layer is None
                else _results["layer"] == _layer
            )
        ]
        if _space is not None:
            _generation, _source = _space
            _selected = _selected[
                (_selected["generation"] == _generation)
                & (
                    _selected["source_dataset"].isna()
                    if pd.isna(_source)
                    else _selected["source_dataset"] == _source
                )
            ]
        if _label_offset is not None:
            _selected = _selected[_selected["label_offset"] == _label_offset]
        return _selected

    return (
        language_decoding_dataset_label,
        language_decoding_metric_label,
        language_decoding_space_label,
        load_language_decoding_summaries,
        select_language_decoding_rows,
    )


@app.cell(hide_code=True)
def language_decoding_data(load_language_decoding_summaries):
    language_decoding_results = load_language_decoding_summaries()
    return (language_decoding_results,)


@app.cell(hide_code=True)
def language_decoding_dataset_control(
    language_decoding_dataset_label,
    language_decoding_results,
    mo,
):
    _ld_datasets = sorted(language_decoding_results["target_dataset"].dropna().unique())
    _ld_dataset_options = {
        language_decoding_dataset_label(_dataset): _dataset
        for _dataset in _ld_datasets
    }
    ld_dataset_selector = mo.ui.dropdown(
        options=_ld_dataset_options,
        value=next(iter(_ld_dataset_options)),
        label="Dataset",
    )
    ld_dataset_selector
    return (ld_dataset_selector,)


@app.cell(hide_code=True)
def language_decoding_model_control(
    language_decoding_results,
    ld_dataset_selector,
    mo,
    pd,
):
    _ld_dataset_rows = language_decoding_results[
        language_decoding_results["target_dataset"] == ld_dataset_selector.value
    ]
    _ld_model_layers = sorted(
        {
            (_row.model, None if pd.isna(_row.layer) else int(_row.layer))
            for _row in _ld_dataset_rows[["model", "layer"]].itertuples(index=False)
        },
        key=lambda _item: (_item[0], _item[1] is not None, _item[1] or -1),
    )
    _ld_model_options = {
        f"{_model} / {'layerless' if _layer is None else f'layer {_layer}'}": (_model, _layer)
        for _model, _layer in _ld_model_layers
    }
    ld_model_layer_selector = mo.ui.dropdown(
        options=_ld_model_options,
        value=next(iter(_ld_model_options)),
        label="Model layer",
    )
    ld_model_layer_selector
    return (ld_model_layer_selector,)


@app.cell(hide_code=True)
def language_decoding_space_control(
    language_decoding_results,
    language_decoding_space_label,
    ld_dataset_selector,
    ld_model_layer_selector,
    mo,
    select_language_decoding_rows,
):
    _ld_model_rows = select_language_decoding_rows(
        language_decoding_results,
        ld_dataset_selector.value,
        ld_model_layer_selector.value,
    )
    _ld_spaces = sorted(
        {
            (_row.generation, _row.source_dataset)
            for _row in _ld_model_rows[["generation", "source_dataset"]].itertuples(index=False)
        },
        key=lambda _item: (0 if _item[0] == "internal" else 1, str(_item[1])),
    )
    _ld_space_options = {
        language_decoding_space_label(_generation, _source): (_generation, _source)
        for _generation, _source in _ld_spaces
    }
    ld_space_selector = mo.ui.dropdown(
        options=_ld_space_options,
        value=next(iter(_ld_space_options)),
        label="Space",
    )
    ld_space_selector
    return (ld_space_selector,)


@app.cell(hide_code=True)
def language_decoding_offset_control(
    language_decoding_results,
    ld_dataset_selector,
    ld_model_layer_selector,
    ld_space_selector,
    mo,
    select_language_decoding_rows,
):
    _ld_space_rows = select_language_decoding_rows(
        language_decoding_results,
        ld_dataset_selector.value,
        ld_model_layer_selector.value,
        ld_space_selector.value,
    )
    _ld_offsets = sorted(int(_offset) for _offset in _ld_space_rows["label_offset"].unique())
    _ld_offset_options = {
        f"{_offset:+d}" if _offset else "0": _offset for _offset in _ld_offsets
    }
    ld_offset_selector = mo.ui.dropdown(
        options=_ld_offset_options,
        value=next(iter(_ld_offset_options)),
        label="Label offset",
    )
    ld_offset_selector
    return (ld_offset_selector,)


@app.cell(hide_code=True)
def language_decoding_metric_controls(
    language_decoding_metric_label,
    language_decoding_results,
    ld_dataset_selector,
    ld_model_layer_selector,
    ld_offset_selector,
    ld_space_selector,
    mo,
    select_language_decoding_rows,
):
    _ld_selected_rows = select_language_decoding_rows(
        language_decoding_results,
        ld_dataset_selector.value,
        ld_model_layer_selector.value,
        ld_space_selector.value,
        ld_offset_selector.value,
    )
    _ld_metric_candidates = [
        "primary_score", "correlation", "r2", "multivariate_r2",
        "mean_cosine_similarity", "balanced_accuracy", "macro_f1", "accuracy",
    ]

    def _ld_available_metrics(_kind):
        _kind_rows = _ld_selected_rows[_ld_selected_rows["kind"] == _kind]
        return [
            _metric for _metric in _ld_metric_candidates
            if _metric in _kind_rows and _kind_rows[_metric].notna().any()
        ]

    _ld_continuous_metrics = _ld_available_metrics("continuous")
    _ld_categorical_metrics = _ld_available_metrics("categorical")
    _ld_continuous_options = {
        language_decoding_metric_label(_metric): _metric
        for _metric in _ld_continuous_metrics
    }
    _ld_categorical_options = {
        language_decoding_metric_label(_metric): _metric
        for _metric in _ld_categorical_metrics
    }
    ld_continuous_metric_selector = mo.ui.dropdown(
        options=_ld_continuous_options,
        value=next(iter(_ld_continuous_options), None),
        label="Continuous-target metric",
    )
    ld_categorical_metric_selector = mo.ui.dropdown(
        options=_ld_categorical_options,
        value=next(iter(_ld_categorical_options), None),
        label="Categorical-target metric",
    )
    mo.hstack(
        [ld_continuous_metric_selector, ld_categorical_metric_selector],
        justify="start",
        gap=2,
    )
    return ld_categorical_metric_selector, ld_continuous_metric_selector


@app.cell(hide_code=True)
def language_decoding_relative_difference_config(np):
    language_decoding_relative_epsilon = 0.05

    def language_decoding_relative_curve_auc(_ranks, _values):
        """Integrate one relative curve over normalized log-rank."""
        _ranks = np.asarray(_ranks, dtype=float)
        _values = np.asarray(_values, dtype=float)
        _valid = np.isfinite(_ranks) & np.isfinite(_values) & (_ranks > 0)
        _ranks, _values = _ranks[_valid], _values[_valid]
        if len(_ranks) < 2:
            return np.nan
        _order = np.argsort(_ranks)
        _log_rank = np.log(_ranks[_order])
        _span = _log_rank[-1] - _log_rank[0]
        if _span <= 0:
            return np.nan
        _x = (_log_rank - _log_rank[0]) / _span
        return float(np.trapezoid(_values[_order], _x))

    return (
        language_decoding_relative_curve_auc,
        language_decoding_relative_epsilon,
    )


@app.cell(hide_code=True)
def language_decoding_table_view(
    language_decoding_metric_label,
    language_decoding_relative_curve_auc,
    language_decoding_relative_epsilon,
    language_decoding_results,
    ld_categorical_metric_selector,
    ld_continuous_metric_selector,
    ld_dataset_selector,
    ld_model_layer_selector,
    ld_offset_selector,
    ld_space_selector,
    mo,
    np,
    pd,
    select_language_decoding_rows,
):
    _ld_selected_rows = select_language_decoding_rows(
        language_decoding_results,
        ld_dataset_selector.value,
        ld_model_layer_selector.value,
        ld_space_selector.value,
        ld_offset_selector.value,
    )
    _ld_rank_rows = _ld_selected_rows[
        (_ld_selected_rows["space"] == "rank")
        & (_ld_selected_rows["subspace"].isin(["rrr", "pca"]))
        & (_ld_selected_rows["k"].notna())
    ].copy()
    _ld_table_rows = []
    for _task, _task_rows in _ld_rank_rows.groupby("task", sort=True):
        _kind = str(_task_rows["kind"].iloc[0])
        _metric = (
            ld_continuous_metric_selector.value
            if _kind == "continuous"
            else ld_categorical_metric_selector.value
        )
        _row = {
            "Target label": _task,
            "Target type": _kind,
            "Metric": language_decoding_metric_label(_metric),
        }
        _relative_ranks = []
        _relative_values = []
        for _rank in sorted(int(_rank) for _rank in _task_rows["k"].unique()):
            _rank_rows = _task_rows[_task_rows["k"] == _rank]
            _rrr_values = _rank_rows.loc[_rank_rows["subspace"] == "rrr", _metric]
            _pca_values = _rank_rows.loc[_rank_rows["subspace"] == "pca", _metric]
            _rrr = float(_rrr_values.iloc[0]) if not _rrr_values.empty else np.nan
            _pca = float(_pca_values.iloc[0]) if not _pca_values.empty else np.nan
            _row[f"Rank {_rank} · RRR"] = _rrr
            _row[f"Rank {_rank} · PCA"] = _pca
            _difference = _rrr - _pca
            _relative_rrr = max(_rrr, 0)
            _relative_pca = max(_pca, 0)
            _relative_difference = (_relative_rrr - _relative_pca) / (
                (_relative_rrr + _relative_pca) / 2
                + language_decoding_relative_epsilon
            )
            _row[f"Rank {_rank} · Δ"] = _difference
            _row[f"Rank {_rank} · Relative Δ"] = _relative_difference
            _relative_ranks.append(_rank)
            _relative_values.append(_relative_difference)
        _row["Relative-difference AUC"] = language_decoding_relative_curve_auc(
            _relative_ranks, _relative_values
        )
        _ld_table_rows.append(_row)
    language_decoding_table = (
        pd.DataFrame(_ld_table_rows)
        .sort_values("Relative-difference AUC", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    _ld_complete = bool(_ld_selected_rows["artifact_complete"].all()) if len(_ld_selected_rows) else False
    mo.vstack([
        mo.hstack(
            [ld_dataset_selector, ld_model_layer_selector, ld_space_selector, ld_offset_selector],
            justify="start",
            gap=2,
        ),
        mo.md(
            f"Showing **{len(language_decoding_table)} target labels** from "
            f"{'a complete' if _ld_complete else 'an incomplete'} saved artifact. "
            f"Δ is raw RRR − PCA; Relative Δ first clamps each score to zero, "
            f"then divides their difference by their mean + "
            f"{language_decoding_relative_epsilon:g}. Blank cells are ranks or metrics "
            "not currently available. Rows are ordered by relative-difference AUC "
            "over normalized log-rank."
        ),
        mo.ui.table(
            language_decoding_table,
            selection=None,
            pagination=False,
            show_column_summaries=False,
            freeze_columns_left=["Target label", "Target type", "Metric"],
        ),
    ])
    return


@app.cell(hide_code=True)
def language_decoding_global_difference_heading(mo):
    mo.md("""
    ### Consistent relative RRR advantages across decoding ranks

    Each row summarizes one result across ranks where RRR performance is positive. Results must beat PCA at two or more ranks and are ordered by the AUC of their relative-difference curve.
    """)
    return


@app.cell(hide_code=True)
def language_decoding_global_difference_table(
    language_decoding_dataset_label,
    language_decoding_relative_curve_auc,
    language_decoding_relative_epsilon,
    language_decoding_results,
    language_decoding_space_label,
    ld_categorical_metric_selector,
    ld_continuous_metric_selector,
    mo,
    np,
    pd,
):
    _ld_all_rank_rows = language_decoding_results[
        (language_decoding_results["space"] == "rank")
        & (language_decoding_results["subspace"].isin(["rrr", "pca"]))
        & (language_decoding_results["k"].notna())
    ].copy()
    _ld_continuous_metric = ld_continuous_metric_selector.value
    _ld_categorical_metric = ld_categorical_metric_selector.value
    _ld_all_rank_rows["Selected metric"] = np.where(
        _ld_all_rank_rows["kind"] == "continuous",
        _ld_continuous_metric,
        _ld_categorical_metric,
    )
    _ld_all_rank_rows["Score"] = np.where(
        _ld_all_rank_rows["kind"] == "continuous",
        _ld_all_rank_rows[_ld_continuous_metric],
        _ld_all_rank_rows[_ld_categorical_metric],
    )
    _ld_all_rank_rows["Source space"] = _ld_all_rank_rows.apply(
        lambda _row: language_decoding_space_label(
            _row["generation"], _row["source_dataset"]
        ),
        axis=1,
    )
    _ld_all_rank_rows["Dataset"] = _ld_all_rank_rows["target_dataset"].map(
        language_decoding_dataset_label
    )
    _ld_all_rank_rows["Layer"] = _ld_all_rank_rows["layer"].apply(
        lambda _layer: "layerless" if pd.isna(_layer) else int(_layer)
    )
    _ld_global_keys = [
        "Dataset", "model", "Layer", "Source space", "label_offset",
        "task", "kind", "k", "Selected metric", "variant", "artifact_complete",
    ]
    _ld_rrr_scores = (
        _ld_all_rank_rows.loc[
            _ld_all_rank_rows["subspace"] == "rrr",
            _ld_global_keys + ["Score"],
        ]
        .drop_duplicates(_ld_global_keys)
        .rename(columns={"Score": "RRR"})
    )
    _ld_pca_scores = (
        _ld_all_rank_rows.loc[
            _ld_all_rank_rows["subspace"] == "pca",
            _ld_global_keys + ["Score"],
        ]
        .drop_duplicates(_ld_global_keys)
        .rename(columns={"Score": "PCA"})
    )
    _ld_global_wide = _ld_rrr_scores.merge(
        _ld_pca_scores,
        on=_ld_global_keys,
        how="outer",
        validate="one_to_one",
    )
    _ld_global_wide["Difference"] = (
        _ld_global_wide["RRR"] - _ld_global_wide["PCA"]
    )
    _ld_relative_rrr = _ld_global_wide["RRR"].clip(lower=0)
    _ld_relative_pca = _ld_global_wide["PCA"].clip(lower=0)
    _ld_global_wide["Relative difference"] = (
        _ld_relative_rrr - _ld_relative_pca
    ) / (
        (_ld_relative_rrr + _ld_relative_pca) / 2
        + language_decoding_relative_epsilon
    )
    language_decoding_rank_differences = _ld_global_wide.copy()
    _ld_positive_rrr = _ld_global_wide[_ld_global_wide["RRR"] > 0].copy()
    _ld_positive_rrr["Advantage"] = _ld_positive_rrr["Difference"] > 0
    _ld_summary_keys = [
        "Dataset", "model", "Layer", "Source space", "label_offset",
        "task", "kind", "Selected metric", "variant", "artifact_complete",
    ]
    _ld_relative_auc = (
        _ld_positive_rrr.groupby(_ld_summary_keys, dropna=False)
        .apply(
            lambda _group: language_decoding_relative_curve_auc(
                _group["k"], _group["Relative difference"]
            ),
            include_groups=False,
        )
        .rename("Relative-difference AUC")
        .reset_index()
    )
    all_language_decoding_differences = (
        _ld_positive_rrr.groupby(_ld_summary_keys, as_index=False, dropna=False)
        .agg(
            **{
                "Positive-RRR ranks": ("k", "nunique"),
                "Advantage ranks": ("Advantage", "sum"),
                "Advantage fraction": ("Advantage", "mean"),
                "Ranks": ("k", lambda _values: ", ".join(
                    str(int(_rank)) for _rank in sorted(set(_values))
                )),
                "Mean RRR": ("RRR", "mean"),
                "Mean PCA": ("PCA", "mean"),
                "Mean difference": ("Difference", "mean"),
                "Mean relative difference": ("Relative difference", "mean"),
            }
        )
        .loc[lambda _frame: _frame["Advantage ranks"] >= 2]
        .merge(_ld_relative_auc, on=_ld_summary_keys, how="left", validate="one_to_one")
        .rename(columns={
            "model": "Model",
            "label_offset": "Label offset",
            "task": "Target label",
            "kind": "Target type",
            "artifact_complete": "Complete",
        })
        .sort_values("Relative-difference AUC", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    _ld_global_display_columns = [
        "Dataset", "Model", "Layer", "Source space", "Label offset",
        "Target label", "Target type", "Selected metric", "Positive-RRR ranks",
        "Advantage ranks", "Advantage fraction", "Ranks", "Mean RRR", "Mean PCA",
        "Mean difference", "Mean relative difference", "Relative-difference AUC",
        "Complete",
    ]
    language_decoding_global_table = mo.ui.table(
        all_language_decoding_differences,
        selection="single",
        pagination=True,
        page_size=25,
        show_column_summaries=False,
        visible_columns=_ld_global_display_columns,
    )
    mo.vstack([
        mo.md(
            f"**{len(all_language_decoding_differences):,} results** with RRR > PCA "
            f"at two or more ranks after excluding ranks where RRR ≤ 0. Relative "
            f"relative difference clamps RRR and PCA to zero, then computes "
            f"(RRR − PCA) / (mean(RRR, PCA) + "
            f"{language_decoding_relative_epsilon:g}). Rows are ordered by "
            "relative-difference AUC over normalized log-rank."
        ),
        language_decoding_global_table,
    ])
    return language_decoding_global_table, language_decoding_rank_differences


@app.cell(hide_code=True)
def language_decoding_selected_relative_curve(
    language_decoding_global_table,
    language_decoding_rank_differences,
    mo,
    plt,
):
    _ld_selected_summary = language_decoding_global_table.value
    if _ld_selected_summary is None or len(_ld_selected_summary) == 0:
        _ld_curve_output = mo.callout(
            mo.md("Select a row in the ranked table above to display its rank curve."),
            kind="info",
        )
    else:
        _ld_selected = _ld_selected_summary.iloc[0]
        _ld_curve = language_decoding_rank_differences[
            (language_decoding_rank_differences["Dataset"] == _ld_selected["Dataset"])
            & (language_decoding_rank_differences["model"] == _ld_selected["Model"])
            & (language_decoding_rank_differences["Layer"] == _ld_selected["Layer"])
            & (language_decoding_rank_differences["Source space"] == _ld_selected["Source space"])
            & (language_decoding_rank_differences["label_offset"] == _ld_selected["Label offset"])
            & (language_decoding_rank_differences["task"] == _ld_selected["Target label"])
            & (language_decoding_rank_differences["kind"] == _ld_selected["Target type"])
            & (language_decoding_rank_differences["Selected metric"] == _ld_selected["Selected metric"])
            & (language_decoding_rank_differences["variant"] == _ld_selected["variant"])
            & (language_decoding_rank_differences["RRR"] > 0)
        ].sort_values("k")
        _ld_curve_fig, _ld_curve_ax = plt.subplots(figsize=(9, 4.5))
        _ld_curve_ax.plot(
            _ld_curve["k"], _ld_curve["Relative difference"],
            color="#475569", linewidth=1.8, marker="o",
            markerfacecolor="white", markeredgecolor="#475569",
            label="Relative difference",
        )
        _ld_advantage = _ld_curve["Difference"] > 0
        _ld_curve_ax.scatter(
            _ld_curve.loc[_ld_advantage, "k"],
            _ld_curve.loc[_ld_advantage, "Relative difference"],
            color="#6ab2e2", s=55, zorder=3, label="RRR > PCA",
        )
        _ld_curve_ax.scatter(
            _ld_curve.loc[~_ld_advantage, "k"],
            _ld_curve.loc[~_ld_advantage, "Relative difference"],
            color="#63bc51", s=45, zorder=3, label="RRR ≤ PCA",
        )
        _ld_curve_ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        _ld_curve_ax.set_xscale("log", base=2)
        _ld_curve_ax.set_xticks(_ld_curve["k"].to_numpy())
        _ld_curve_ax.set_xticklabels([str(int(_rank)) for _rank in _ld_curve["k"]])
        _ld_curve_ax.set_xlabel("Rank")
        _ld_curve_ax.set_ylabel("Relative difference after clamping scores to 0")
        _ld_curve_ax.set_title(
            f"{_ld_selected['Dataset']} · {_ld_selected['Model']} / "
            f"{_ld_selected['Layer']} · {_ld_selected['Target label']}\n"
            f"{_ld_selected['Source space']} · offset {_ld_selected['Label offset']}"
        )
        _ld_curve_ax.grid(axis="y", alpha=0.25)
        _ld_curve_ax.spines[["top", "right"]].set_visible(False)
        _ld_curve_ax.legend(frameon=False)
        _ld_curve_fig.tight_layout()
        _ld_curve_output = _ld_curve_fig
    _ld_curve_output
    return


@app.cell(hide_code=True)
def language_decoding_summary_heading(mo):
    mo.md("""
    ## Averaged language-decoding summaries

    Each row is one decoded target within a dataset/source-space condition. Select a row to compare relative RRR-PCA curves split by normalized model depth or model type. Random GPT-2 and layerless GloVe are excluded.
    """)
    return


@app.cell(hide_code=True)
def language_decoding_summary_helpers(
    language_decoding_relative_curve_auc,
    language_decoding_relative_epsilon,
    np,
    pd,
):
    def prepare_language_decoding_summary_curves(
        rank_results, summary_kind, averaging_mode, epsilon
    ):
        """Build member and averaged rank curves for one summary strategy."""
        _data = rank_results[
            ~rank_results["model"].isin(["random-gpt2", "glove-300"])
        ].copy()
        _data = _data[_data["Layer"] != "layerless"].copy()
        _data["Layer"] = _data["Layer"].astype(int)
        _data["RRR clipped"] = _data["RRR"].clip(lower=0)
        _data["PCA clipped"] = _data["PCA"].clip(lower=0)
        _data["Member relative difference"] = (
            _data["RRR clipped"] - _data["PCA clipped"]
        ) / (
            (_data["RRR clipped"] + _data["PCA clipped"]) / 2 + epsilon
        )
        _architecture = {
            "gpt2": "Causal",
            "gpt2-xl-batch": "Causal",
            "gpt-j-6b": "Causal",
            "llama3-8b": "Causal",
            "roberta-large": "Bidirectional",
        }
        if summary_kind == "depth":
            _architectural_max_layers = {
                "gpt2": 12,
                "gpt2-xl-batch": 48,
                "gpt-j-6b": 28,
                "llama3-8b": 32,
                "roberta-large": 24,
            }
            _observed_max_layers = _data.groupby("model")["Layer"].transform("max")
            _max_layers = _data["model"].map(_architectural_max_layers).fillna(
                _observed_max_layers
            )
            _data["Summary group"] = (
                (100 * _data["Layer"] / _max_layers).round().astype(int).astype(str)
                + "% depth"
            )
            _data["Context"] = _data["Dataset"] + " · " + _data["Source space"]
            _data["Member"] = _data["model"]
        elif summary_kind == "model":
            _data["Summary group"] = _data["model"]
            _data["Context"] = _data["Dataset"] + " · " + _data["Source space"]
            _data["Member"] = "Layer " + _data["Layer"].astype(str)
        elif summary_kind == "condition":
            _data["Summary group"] = np.where(
                _data["Source space"] == "Internal space",
                "Internal space",
                "Cross-dataset space",
            )
            _data["Context"] = (
                _data["model"] + " / layer " + _data["Layer"].astype(str)
            )
            _data["Member"] = _data["Dataset"] + " · " + _data["Source space"]
        elif summary_kind == "architecture":
            _data["Summary group"] = _data["model"].map(_architecture)
            _data = _data[_data["Summary group"].notna()].copy()
            _data["Context"] = _data["Dataset"] + " · " + _data["Source space"]
            _data["Member"] = (
                _data["model"] + " / layer " + _data["Layer"].astype(str)
            )
        else:
            raise ValueError(f"Unknown summary kind: {summary_kind}")

        _keys = [
            "Summary group", "Context", "label_offset", "task", "kind",
            "Selected metric", "k",
        ]
        _averaged = (
            _data.groupby(_keys, as_index=False, dropna=False)
            .agg(
                **{
                    "Average RRR": ("RRR clipped", "mean"),
                    "Average PCA": ("PCA clipped", "mean"),
                    "Mean member relative difference": (
                        "Member relative difference", "mean"
                    ),
                    "Members available": ("Member", "nunique"),
                }
            )
        )
        _averaged["Average difference"] = (
            _averaged["Average RRR"] - _averaged["Average PCA"]
        )
        if averaging_mode == "relative_first":
            _averaged["Average relative difference"] = _averaged[
                "Mean member relative difference"
            ]
        elif averaging_mode == "scores_first":
            _averaged["Average relative difference"] = _averaged[
                "Average difference"
            ] / (
                (_averaged["Average RRR"] + _averaged["Average PCA"]) / 2
                + epsilon
            )
        else:
            raise ValueError(f"Unknown averaging mode: {averaging_mode}")

        _summary_keys = [
            "Context", "label_offset", "task", "kind",
            "Selected metric",
        ]
        _ids = _averaged[_summary_keys].drop_duplicates().reset_index(drop=True)
        _ids["Summary ID"] = np.arange(len(_ids))
        _averaged = _averaged.merge(_ids, on=_summary_keys, how="left")
        _data = _data.merge(_ids, on=_summary_keys, how="inner")
        return _data, _averaged


    def summarize_language_decoding_average_curves(averaged_curves, averaging_mode):
        """Summarize one row per target and dataset/source condition."""
        _keys = ["Summary ID", "Context", "label_offset", "task", "kind", "Selected metric", "k"]
        _weighted = averaged_curves.assign(
            _weight=averaged_curves["Members available"],
            _rrr_weighted=averaged_curves["Average RRR"] * averaged_curves["Members available"],
            _pca_weighted=averaged_curves["Average PCA"] * averaged_curves["Members available"],
            _relative_weighted=averaged_curves["Average relative difference"] * averaged_curves["Members available"],
        )
        _overall = _weighted.groupby(_keys, as_index=False, dropna=False).agg(
            _weight=("_weight", "sum"),
            _rrr_weighted=("_rrr_weighted", "sum"),
            _pca_weighted=("_pca_weighted", "sum"),
            _relative_weighted=("_relative_weighted", "sum"),
        )
        _overall["Average RRR"] = _overall["_rrr_weighted"] / _overall["_weight"]
        _overall["Average PCA"] = _overall["_pca_weighted"] / _overall["_weight"]
        _overall["Average difference"] = _overall["Average RRR"] - _overall["Average PCA"]
        if averaging_mode == "relative_first":
            _overall["Average relative difference"] = _overall["_relative_weighted"] / _overall["_weight"]
        elif averaging_mode == "scores_first":
            _overall["Average relative difference"] = _overall["Average difference"] / (
                (_overall["Average RRR"] + _overall["Average PCA"]) / 2
                + language_decoding_relative_epsilon
            )
        else:
            raise ValueError(f"Unknown averaging mode: {averaging_mode}")
        _summary_keys = ["Summary ID", "Context", "label_offset", "task", "kind", "Selected metric"]
        _base = _overall.groupby(_summary_keys, as_index=False, dropna=False).agg(
            **{
                "Ranks": ("k", lambda values: ", ".join(str(int(rank)) for rank in sorted(set(values)))),
                "Mean RRR": ("Average RRR", "mean"),
                "Mean PCA": ("Average PCA", "mean"),
                "Mean difference": ("Average difference", "mean"),
                "Mean relative difference": ("Average relative difference", "mean"),
                "Maximum members": ("_weight", "max"),
            }
        )
        _positive = _overall[_overall["Average RRR"] > 0].copy()
        _positive["Advantage"] = _positive["Average difference"] > 0
        if _positive.empty:
            _positive_stats = pd.DataFrame(columns=_summary_keys + [
                "Positive-RRR ranks", "Advantage ranks", "Advantage fraction",
            ])
        else:
            _positive_stats = _positive.groupby(_summary_keys, as_index=False, dropna=False).agg(
                **{
                    "Positive-RRR ranks": ("k", "nunique"),
                    "Advantage ranks": ("Advantage", "sum"),
                    "Advantage fraction": ("Advantage", "mean"),
                }
            )
        _group_positive = averaged_curves[averaged_curves["Average RRR"] > 0].copy()
        if _group_positive.empty:
            _relative_auc = pd.DataFrame(columns=_summary_keys + [
                "Relative-difference AUC", "Minimum relative-difference AUC",
            ])
        else:
            _relative_auc_by_group = (
                _group_positive.groupby(_summary_keys + ["Summary group"], dropna=False)
                .apply(
                    lambda group: language_decoding_relative_curve_auc(
                        group["k"], group["Average relative difference"]
                    ),
                    include_groups=False,
                )
                .rename("Relative-difference AUC")
                .reset_index()
            )
            _relative_auc = (
                _relative_auc_by_group.groupby(_summary_keys, as_index=False, dropna=False)
                .agg(
                    **{
                        "Relative-difference AUC": ("Relative-difference AUC", "max"),
                        "Minimum relative-difference AUC": ("Relative-difference AUC", "min"),
                    }
                )
            )
        _summary = (
            _base.merge(_positive_stats, on=_summary_keys, how="left", validate="one_to_one")
            .merge(_relative_auc, on=_summary_keys, how="left", validate="one_to_one")
        )
        _summary["AUC gap (max - min)"] = (
            _summary["Relative-difference AUC"]
            - _summary["Minimum relative-difference AUC"]
        )
        for _column in ("Positive-RRR ranks", "Advantage ranks"):
            _summary[_column] = _summary[_column].fillna(0).astype(int)
        _summary["Advantage fraction"] = _summary["Advantage fraction"].fillna(0.0)
        return (
            _summary.rename(columns={"label_offset": "Label offset", "task": "Target label", "kind": "Target type"})
            .sort_values("Relative-difference AUC", ascending=False, na_position="last")
            .reset_index(drop=True)
        )


    def plot_language_decoding_average_curve(selected_row, member_curves, averaged_curves, plt):
        """Plot one selected target with a separate curve for each model group."""
        _summary_id = int(selected_row["Summary ID"])
        _members = member_curves[(member_curves["Summary ID"] == _summary_id) & (member_curves["RRR clipped"] > 0)].copy()
        _average = averaged_curves[(averaged_curves["Summary ID"] == _summary_id) & (averaged_curves["Average RRR"] > 0)].copy()
        _is_depth = _average["Summary group"].astype(str).str.endswith("% depth").any()
        _group_label = "Model depth" if _is_depth else "Model type"
        _groups = _average["Summary group"].dropna().unique().tolist()
        _groups = sorted(_groups, key=lambda value: int(str(value).split("%")[0])) if _is_depth else sorted(_groups)
        _fig, _ax = plt.subplots(figsize=(11, 5.5))
        _colors = plt.get_cmap("tab10")
        _coverage_notes = []
        for _index, _group_name in enumerate(_groups):
            _color = _colors(_index % 10)
            _group_average = _average[_average["Summary group"] == _group_name].sort_values("k")
            _group_members = _members[_members["Summary group"] == _group_name]
            for _, _curve in _group_members.groupby("Member", sort=True):
                _ax.plot(_curve["k"], _curve["Member relative difference"], color=_color, alpha=0.16, linewidth=0.9)
            _ax.plot(_group_average["k"], _group_average["Average relative difference"], color=_color, linewidth=2.5, marker="o", label=str(_group_name))
            _advantage = _group_average["Average difference"] > 0
            _ax.scatter(_group_average.loc[_advantage, "k"], _group_average.loc[_advantage, "Average relative difference"], color="#6ab2e2", edgecolors="white", linewidths=0.6, s=52, zorder=4)
            _ax.scatter(_group_average.loc[~_advantage, "k"], _group_average.loc[~_advantage, "Average relative difference"], color="#63bc51", edgecolors="white", linewidths=0.6, s=48, zorder=4)
            _total = int(_group_members["Member"].nunique())
            _coverage = _group_members.groupby("k")["Member"].nunique()
            _partial = int((_coverage < _total).sum()) if _total else 0
            _coverage_notes.append(f"{_group_name}: {_total} members, {_partial}/{len(_coverage)} ranks with partial coverage")
        _ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        _ax.set_xscale("log", base=2)
        _ranks = sorted(set(_average["k"].astype(int)))
        if _ranks:
            _ax.set_xticks(_ranks)
            _ax.set_xticklabels([str(rank) for rank in _ranks])
        _ax.set_xlabel("Rank")
        _ax.set_ylabel("Relative difference (RRR vs PCA)")
        _ax.set_title(f"{selected_row['Target label']}\n{selected_row['Context']} · offset {selected_row['Label offset']}")
        _ax.grid(axis="y", alpha=0.25)
        _ax.spines[["top", "right"]].set_visible(False)
        if _groups:
            _ax.legend(frameon=False, title=_group_label)
        _coverage_note = "Blue markers: average RRR > PCA; green markers: average RRR <= PCA. " + ("; ".join(_coverage_notes) if _coverage_notes else "No positive-RRR curves to show.")
        _fig.tight_layout()
        return _fig, _coverage_note


    return (
        plot_language_decoding_average_curve,
        prepare_language_decoding_summary_curves,
        summarize_language_decoding_average_curves,
    )


@app.cell(hide_code=True)
def language_decoding_summary_controls(
    language_decoding_rank_differences,
    mo,
    pd,
):
    _summary_data = language_decoding_rank_differences[
        language_decoding_rank_differences["Layer"].notna()
    ]
    _models = sorted(_summary_data["model"].dropna().unique())
    _layers = sorted(
        int(layer)
        for layer in _summary_data["Layer"].dropna().unique()
        if str(layer) != "layerless"
    )
    _layer_options = {str(layer): layer for layer in _layers}
    _layer_options["Layerless"] = "layerless"
    _default_groups = {
        "gpt-j-6b": "Causal",
        "gpt2": "Causal",
        "gpt2-xl-batch": "Causal",
        "llama3-8b": "Causal",
        "roberta-large": "Bidirectional",
        "random-gpt2": "Random",
        "glove-300": "GloVe",
    }
    _group_assignments = pd.DataFrame({
        "Model": _models,
        "Group": [_default_groups.get(model, model) for model in _models],
    })
    language_decoding_summary_group_editor = mo.ui.data_editor(
        _group_assignments,
        label="Assign each model to one group (edit the Group column)",
        editable_columns=["Group"],
        pagination=False,
    )
    language_decoding_summary_models = mo.ui.multiselect(
        options=_models, value=_models, label="Models included"
    )
    language_decoding_summary_layers = mo.ui.multiselect(
        options=_layer_options,
        value=list(_layer_options.keys()),
        label="Layer indices included",
    )
    language_decoding_summary_type = mo.ui.radio(
        options={
            "Model depth (across types)": "depth",
            "Model type (across layers)": "model",
            "Custom model groups": "custom",
        },
        value="Model depth (across types)",
        label="Split curves by",
        inline=True,
    )
    language_decoding_averaging_mode = mo.ui.radio(
        options={
            "Mean of relative curves": "relative_first",
            "Average scores first": "scores_first",
        },
        value="Mean of relative curves",
        label="Averaging",
        inline=True,
    )
    mo.vstack([
        mo.hstack([language_decoding_summary_type, language_decoding_averaging_mode], justify="start", gap=2),
        mo.hstack([language_decoding_summary_models, language_decoding_summary_layers], justify="start", gap=2),
        mo.md("Edit the Group column to define your custom comparison groups; blank groups are omitted."),
        language_decoding_summary_group_editor,
    ])
    return (
        language_decoding_averaging_mode,
        language_decoding_summary_group_editor,
        language_decoding_summary_layers,
        language_decoding_summary_models,
        language_decoding_summary_type,
    )


@app.cell(hide_code=True)
def language_decoding_summary_data(
    language_decoding_averaging_mode,
    language_decoding_rank_differences,
    language_decoding_relative_epsilon,
    language_decoding_summary_group_editor,
    language_decoding_summary_layers,
    language_decoding_summary_models,
    language_decoding_summary_type,
    np,
    prepare_language_decoding_summary_curves,
    summarize_language_decoding_average_curves,
):
    _summary_filtered_rank_differences = language_decoding_rank_differences[
        language_decoding_rank_differences["model"].isin(
            language_decoding_summary_models.value
        )
        & language_decoding_rank_differences["Layer"].astype(str).isin(
            [str(layer) for layer in language_decoding_summary_layers.value]
        )
    ].copy()
    if language_decoding_summary_type.value == "custom":
        _group_assignments = language_decoding_summary_group_editor.value.copy()
        _group_assignments["Group"] = _group_assignments["Group"].fillna("").astype(str).str.strip()
        _model_group_map = dict(zip(_group_assignments["Model"], _group_assignments["Group"]))
        _custom_data = _summary_filtered_rank_differences.copy()
        _custom_data["Summary group"] = _custom_data["model"].map(_model_group_map)
        _custom_data = _custom_data[_custom_data["Summary group"].notna() & _custom_data["Summary group"].ne("")].copy()
        _custom_data["RRR clipped"] = _custom_data["RRR"].clip(lower=0)
        _custom_data["PCA clipped"] = _custom_data["PCA"].clip(lower=0)
        _custom_data["Member relative difference"] = (
            _custom_data["RRR clipped"] - _custom_data["PCA clipped"]
        ) / (
            (_custom_data["RRR clipped"] + _custom_data["PCA clipped"]) / 2
            + language_decoding_relative_epsilon
        )
        _custom_data["Context"] = _custom_data["Dataset"] + " · " + _custom_data["Source space"]
        _custom_data["Member"] = (
            _custom_data["model"] + " / layer " + _custom_data["Layer"].astype(str)
        )
        _custom_keys = [
            "Summary group", "Context", "label_offset", "task", "kind",
            "Selected metric", "k",
        ]
        language_decoding_summary_curves = (
            _custom_data.groupby(_custom_keys, as_index=False, dropna=False)
            .agg(
                **{
                    "Average RRR": ("RRR clipped", "mean"),
                    "Average PCA": ("PCA clipped", "mean"),
                    "Mean member relative difference": ("Member relative difference", "mean"),
                    "Members available": ("Member", "nunique"),
                }
            )
        )
        language_decoding_summary_curves["Average difference"] = (
            language_decoding_summary_curves["Average RRR"]
            - language_decoding_summary_curves["Average PCA"]
        )
        if language_decoding_averaging_mode.value == "relative_first":
            language_decoding_summary_curves["Average relative difference"] = language_decoding_summary_curves["Mean member relative difference"]
        else:
            language_decoding_summary_curves["Average relative difference"] = (
                language_decoding_summary_curves["Average difference"]
                / (
                    (language_decoding_summary_curves["Average RRR"] + language_decoding_summary_curves["Average PCA"]) / 2
                    + language_decoding_relative_epsilon
                )
            )
        _custom_summary_keys = ["Context", "label_offset", "task", "kind", "Selected metric"]
        _custom_ids = language_decoding_summary_curves[_custom_summary_keys].drop_duplicates().reset_index(drop=True)
        _custom_ids["Summary ID"] = np.arange(len(_custom_ids))
        language_decoding_summary_curves = language_decoding_summary_curves.merge(
            _custom_ids, on=_custom_summary_keys, how="left"
        )
        language_decoding_summary_members = _custom_data.merge(
            _custom_ids, on=_custom_summary_keys, how="inner"
        )
    else:
        (
            language_decoding_summary_members,
            language_decoding_summary_curves,
        ) = prepare_language_decoding_summary_curves(
            _summary_filtered_rank_differences,
            language_decoding_summary_type.value,
            language_decoding_averaging_mode.value,
            language_decoding_relative_epsilon,
        )
    language_decoding_summary_results = summarize_language_decoding_average_curves(
        language_decoding_summary_curves, language_decoding_averaging_mode.value
    )
    return (
        language_decoding_summary_curves,
        language_decoding_summary_members,
        language_decoding_summary_results,
    )


@app.cell(hide_code=True)
def language_decoding_summary_table_view(
    language_decoding_summary_results,
    mo,
):
    _ld_summary_columns = [
        "Context", "Label offset", "Target label", "Target type",
        "Selected metric", "Positive-RRR ranks", "Advantage ranks",
        "Advantage fraction", "Ranks", "Mean RRR", "Mean PCA",
        "Mean difference", "Mean relative difference", "Relative-difference AUC",
        "Minimum relative-difference AUC", "AUC gap (max - min)", "Maximum members",
    ]
    language_decoding_summary_table = mo.ui.table(
        language_decoding_summary_results,
        selection="single",
        pagination=True,
        page_size=25,
        show_column_summaries=False,
        visible_columns=_ld_summary_columns,
    )
    mo.vstack([
        mo.md(
            f"**{len(language_decoding_summary_results):,} target-condition rows.** "
            "Each row is one decoded target for a dataset/source-space condition and label offset. "
            "Select a row to compare its relative curves across model depths or model types. "
            "The AUC columns show the maximum, minimum, and max-minus-min gap across those group curves."
        ),
        language_decoding_summary_table,
    ])
    return (language_decoding_summary_table,)


@app.cell(hide_code=True)
def language_decoding_summary_selected_curve(
    language_decoding_summary_curves,
    language_decoding_summary_members,
    language_decoding_summary_table,
    mo,
    plot_language_decoding_average_curve,
    plt,
):
    _ld_summary_selection = language_decoding_summary_table.value
    if _ld_summary_selection is None or len(_ld_summary_selection) == 0:
        _ld_summary_plot_output = mo.callout(
            mo.md("Select a summary row above to display its averaged curve."),
            kind="info",
        )
    else:
        _ld_summary_figure, _ld_summary_coverage = plot_language_decoding_average_curve(
            _ld_summary_selection.iloc[0],
            language_decoding_summary_members,
            language_decoding_summary_curves,
            plt,
        )
        _ld_summary_plot_output = mo.vstack([
            _ld_summary_figure,
            mo.callout(mo.md(_ld_summary_coverage), kind="warn"),
        ])
    _ld_summary_plot_output
    return


@app.cell(hide_code=True)
def language_decoding_transfer_heading(mo):
    mo.md("""
    ## Internal versus cross-dataset relative advantage

    Each observation is a matched model/layer/task/offset. Curve AUC summarizes the clamped relative RRR–PCA advantage over normalized log-rank.
    """)
    return


@app.cell(hide_code=True)
def language_decoding_transfer_helpers(
    language_decoding_relative_curve_auc,
    np,
    pd,
):
    def build_language_decoding_transfer_auc_pairs(rank_differences):
        """Match cross-dataset AUCs to source- and target-internal AUCs."""
        _identity = [
            "Dataset", "model", "Layer", "Source space", "label_offset",
            "task", "kind", "Selected metric",
        ]
        _rank_means = (
            rank_differences.groupby(_identity + ["k"], as_index=False, dropna=False)
            ["Relative difference"].mean()
        )
        _auc = (
            _rank_means.groupby(_identity, as_index=False, dropna=False)
            .apply(
                lambda _group: language_decoding_relative_curve_auc(
                    _group["k"], _group["Relative difference"]
                ),
                include_groups=False,
            )
            .rename(columns={None: "Relative AUC"})
        )
        if "Relative AUC" not in _auc:
            _auc = _auc.rename(columns={_auc.columns[-1]: "Relative AUC"})
        _internal = _auc[_auc["Source space"] == "Internal space"].copy()
        _cross = _auc[_auc["Source space"].str.startswith(
            "Cross-dataset space from ", na=False
        )].copy()
        _cross["Source Dataset"] = _cross["Source space"].str.removeprefix(
            "Cross-dataset space from "
        )
        _cross = _cross.rename(columns={
            "Dataset": "Target Dataset",
            "Relative AUC": "Cross-dataset AUC",
        })
        _match = ["model", "Layer", "label_offset", "task", "kind", "Selected metric"]
        _internal_lookup = _internal.rename(columns={
            "Dataset": "Internal Dataset",
            "Relative AUC": "Internal AUC",
        })
        _source = _cross.merge(
            _internal_lookup[["Internal Dataset", *_match, "Internal AUC"]],
            left_on=["Source Dataset", *_match],
            right_on=["Internal Dataset", *_match],
            how="inner",
        )
        _source["Internal baseline"] = "Source internal"
        _target = _cross.merge(
            _internal_lookup[["Internal Dataset", *_match, "Internal AUC"]],
            left_on=["Target Dataset", *_match],
            right_on=["Internal Dataset", *_match],
            how="inner",
        )
        _target["Internal baseline"] = "Target internal"
        _pairs = pd.concat([_source, _target], ignore_index=True)
        _pairs["Observation"] = (
            _pairs["model"] + " / "
            + _pairs["Layer"].apply(
                lambda _layer: "layerless" if pd.isna(_layer) or type(_layer) is str else f"layer {int(_layer)}"
            )
            + " / " + _pairs["task"]
            + " / offset " + _pairs["label_offset"].astype(int).astype(str)
        )
        return _pairs


    def summarize_language_decoding_transfer_correlations(pairs):
        """Compute one Pearson correlation for every directed dataset pair."""
        _rows = []
        for (_source, _target), _group in pairs.groupby(
            ["Source Dataset", "Target Dataset"], sort=True
        ):
            _valid = _group[["Internal AUC", "Cross-dataset AUC"]].dropna()
            _correlation = (
                float(_valid["Internal AUC"].corr(_valid["Cross-dataset AUC"]))
                if len(_valid) >= 2
                else np.nan
            )
            _rows.append({
                "Source Dataset": _source,
                "Target Dataset": _target,
                "Correlation": _correlation,
                "Observations": int(len(_valid)),
                "Tasks": int(_group["task"].nunique()),
                "Model layers": int(
                    _group[["model", "Layer"]].drop_duplicates().shape[0]
                ),
            })
        return pd.DataFrame(_rows).sort_values(
            "Correlation", ascending=False, na_position="last"
        ).reset_index(drop=True)


    def plot_language_decoding_transfer_correlation(selected_row, pairs, plt):
        """Plot one selected AUC relationship with feature-aware hover labels."""
        import plotly.graph_objects as go

        _points = pairs[
            (pairs["Source Dataset"] == selected_row["Source Dataset"])
            & (pairs["Target Dataset"] == selected_row["Target Dataset"])
        ].dropna(subset=["Internal AUC", "Cross-dataset AUC"])
        _fig = go.Figure()
        _offset_colors = {-1: "#f59e0b", 0: "#2563eb", 1: "#16a34a"}
        for _offset, _offset_points in _points.groupby("label_offset", sort=True):
            _custom = np.column_stack([
                _offset_points["task"].astype(str),
                _offset_points["model"].astype(str),
                _offset_points["Layer"].astype(str),
                _offset_points["label_offset"].astype(int).astype(str),
            ])
            _fig.add_trace(go.Scatter(
                x=_offset_points["Internal AUC"],
                y=_offset_points["Cross-dataset AUC"],
                mode="markers",
                name=f"Offset {int(_offset):+d}",
                marker={"color": _offset_colors.get(int(_offset), "#64748b"), "size": 8, "opacity": 0.72, "line": {"color": "white", "width": 0.5}},
                customdata=_custom,
                hovertemplate=(
                    "Feature: %{customdata[0]}<br>"
                    "Model: %{customdata[1]}<br>"
                    "Layer: %{customdata[2]}<br>"
                    "Label offset: %{customdata[3]}<br>"
                    "Internal AUC: %{x:.3f}<br>"
                    "Cross-dataset AUC: %{y:.3f}<extra></extra>"
                ),
            ))
        if len(_points) >= 2 and _points["Internal AUC"].nunique() >= 2:
            _slope, _intercept = np.polyfit(
                _points["Internal AUC"], _points["Cross-dataset AUC"], 1
            )
            _line_x = np.linspace(
                _points["Internal AUC"].min(), _points["Internal AUC"].max(), 100
            )
            _fig.add_trace(go.Scatter(
                x=_line_x, y=_slope * _line_x + _intercept,
                mode="lines", name="Linear fit", line={"color": "#111827", "width": 2.5},
                hoverinfo="skip",
            ))
        _limits = [
            min(_points["Internal AUC"].min(), _points["Cross-dataset AUC"].min()),
            max(_points["Internal AUC"].max(), _points["Cross-dataset AUC"].max()),
        ]
        _fig.add_trace(go.Scatter(
            x=_limits, y=_limits, mode="lines", name="Identity",
            line={"color": "#94a3b8", "dash": "dash", "width": 1},
            hoverinfo="skip",
        ))
        _correlation = _points["Internal AUC"].corr(_points["Cross-dataset AUC"])
        _fig.update_layout(
            title=f"{selected_row['Source Dataset']} -> {selected_row['Target Dataset']}<br><sup>Pearson r = {_correlation:.3f}; n = {len(_points):,}</sup>",
            xaxis_title=f"{_points['Internal baseline'].iloc[0]} relative-curve AUC",
            yaxis_title="Cross-dataset relative-curve AUC",
            template="plotly_white",
            hovermode="closest",
            legend_title="Label offset",
            width=720,
            height=620,
            margin={"l": 70, "r": 30, "t": 90, "b": 70},
        )
        _fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return _fig


    return (
        build_language_decoding_transfer_auc_pairs,
        plot_language_decoding_transfer_correlation,
        summarize_language_decoding_transfer_correlations,
    )


@app.cell(hide_code=True)
def language_decoding_transfer_controls(
    language_decoding_rank_differences,
    mo,
):
    _ld_transfer_offsets = sorted(
        int(_offset) for _offset in language_decoding_rank_differences["label_offset"].unique()
    )
    _ld_transfer_offset_options = {
        f"{_offset:+d}" if _offset else "0": _offset
        for _offset in _ld_transfer_offsets
    }
    language_decoding_internal_baseline = mo.ui.radio(
        options={
            "Source internal": "Source internal",
            "Target internal": "Target internal",
        },
        value="Source internal",
        label="Internal X-axis",
        inline=True,
    )
    language_decoding_transfer_offsets = mo.ui.multiselect(
        options=_ld_transfer_offset_options,
        value=list(_ld_transfer_offset_options),
        label="Label offsets",
    )
    mo.hstack(
        [language_decoding_internal_baseline, language_decoding_transfer_offsets],
        justify="start", gap=2,
    )
    return (
        language_decoding_internal_baseline,
        language_decoding_transfer_offsets,
    )


@app.cell(hide_code=True)
def language_decoding_transfer_auc_data(
    build_language_decoding_transfer_auc_pairs,
    language_decoding_rank_differences,
):
    language_decoding_transfer_auc_pairs = (
        build_language_decoding_transfer_auc_pairs(language_decoding_rank_differences)
    )
    return (language_decoding_transfer_auc_pairs,)


@app.cell(hide_code=True)
def language_decoding_transfer_correlation_data(
    language_decoding_internal_baseline,
    language_decoding_transfer_auc_pairs,
    language_decoding_transfer_offsets,
    summarize_language_decoding_transfer_correlations,
):
    language_decoding_transfer_selected_pairs = language_decoding_transfer_auc_pairs[
        (language_decoding_transfer_auc_pairs["Internal baseline"]
         == language_decoding_internal_baseline.value)
        & (language_decoding_transfer_auc_pairs["label_offset"].isin(
            language_decoding_transfer_offsets.value
        ))
    ]
    language_decoding_transfer_correlations = (
        summarize_language_decoding_transfer_correlations(language_decoding_transfer_selected_pairs)
    )
    return (
        language_decoding_transfer_correlations,
        language_decoding_transfer_selected_pairs,
    )


@app.cell(hide_code=True)
def language_decoding_transfer_correlation_table_view(
    language_decoding_transfer_correlations,
    language_decoding_transfer_selected_pairs,
    mo,
):
    language_decoding_transfer_correlation_table = mo.ui.table(
        language_decoding_transfer_correlations,
        selection="single",
        pagination=False,
        show_column_summaries=False,
    )
    mo.vstack([
        mo.md(
            f"**{len(language_decoding_transfer_correlations)} directed dataset pairs** "
            f"using **{len(language_decoding_transfer_selected_pairs):,} matched observations**. "
            "Select a row to inspect its scatter plot."
        ),
        language_decoding_transfer_correlation_table,
    ])
    return (language_decoding_transfer_correlation_table,)


@app.cell(hide_code=True)
def language_decoding_transfer_selected_scatter(
    language_decoding_transfer_correlation_table,
    language_decoding_transfer_selected_pairs,
    mo,
    plot_language_decoding_transfer_correlation,
    plt,
):
    _ld_transfer_selection = language_decoding_transfer_correlation_table.value
    if _ld_transfer_selection is None or len(_ld_transfer_selection) == 0:
        _ld_transfer_plot_output = mo.callout(
            mo.md("Select a directed dataset pair above to display its AUC scatter plot."),
            kind="info",
        )
    else:
        _ld_transfer_plot_output = plot_language_decoding_transfer_correlation(
            _ld_transfer_selection.iloc[0], language_decoding_transfer_selected_pairs, plt
        )
    _ld_transfer_plot_output
    return


@app.cell
def language_decoding_internal_consensus_heading(mo):
    mo.md("""
    ## Within-dataset RRR–PCA feature agreement

    For a selected model/layer and label offset, each tile shows the existing relative-difference AUC across decoding ranks. Blue means RRR decodes the feature better than PCA on average over log-rank; green means PCA does better. A check mark marks a feature with positive AUC in all three datasets. Only features available in all three are shown.
    """)
    return


@app.cell(hide_code=True)
def language_decoding_internal_consensus_data(
    language_decoding_rank_differences,
    language_decoding_relative_curve_auc,
):
    _consensus_internal = language_decoding_rank_differences[
        language_decoding_rank_differences["Source space"] == "Internal space"
    ].copy()
    _consensus_keys = [
        "Dataset", "model", "Layer", "label_offset", "task", "kind", "Selected metric"
    ]
    _consensus_rank_means = (
        _consensus_internal.groupby(_consensus_keys + ["k"], as_index=False, dropna=False)
        ["Relative difference"].mean()
    )
    _consensus_auc = (
        _consensus_rank_means.groupby(_consensus_keys, dropna=False)
        .apply(
            lambda _group: language_decoding_relative_curve_auc(
                _group["k"], _group["Relative difference"]
            ),
            include_groups=False,
        )
        .rename("Relative AUC")
        .reset_index()
    )
    language_decoding_internal_consensus_auc = (
        _consensus_auc.pivot_table(
            index=["model", "Layer", "label_offset", "task"],
            columns="Dataset",
            values="Relative AUC",
        )
        .dropna(subset=["Podcast", "Cars 2", "fMRI"])
        .reset_index()
    )
    return (language_decoding_internal_consensus_auc,)


@app.cell(hide_code=True)
def language_decoding_internal_consensus_controls(
    language_decoding_internal_consensus_auc,
    mo,
):
    _consensus_model_layers = sorted(
        {
            (_row.model, _row.Layer)
            for _row in language_decoding_internal_consensus_auc[["model", "Layer"]].itertuples(index=False)
        },
        key=lambda _item: (_item[0], _item[1] != "layerless", _item[1] if _item[1] != "layerless" else -1),
    )
    _consensus_model_options = {
        f"{_model} / {'layerless' if _layer == 'layerless' else f'layer {_layer}'}": (_model, _layer)
        for _model, _layer in _consensus_model_layers
    }
    _consensus_default = next(
        (_label for _label, _value in _consensus_model_options.items() if _value == ("gpt2", 0)),
        next(iter(_consensus_model_options)),
    )
    language_decoding_consensus_model_layer = mo.ui.dropdown(
        options=_consensus_model_options,
        value=_consensus_default,
        label="Model / layer",
    )
    _consensus_offsets = sorted(language_decoding_internal_consensus_auc["label_offset"].unique())
    _consensus_offset_options = {
        f"{int(_offset):+d}" if _offset else "0": int(_offset)
        for _offset in _consensus_offsets
    }
    language_decoding_consensus_offset = mo.ui.dropdown(
        options=_consensus_offset_options,
        value="0" if "0" in _consensus_offset_options else next(iter(_consensus_offset_options)),
        label="Label offset",
    )
    mo.hstack([language_decoding_consensus_model_layer, language_decoding_consensus_offset], justify="start", gap=2)
    return (
        language_decoding_consensus_model_layer,
        language_decoding_consensus_offset,
    )


@app.cell(hide_code=True)
def language_decoding_internal_consensus_heatmap(
    language_decoding_consensus_model_layer,
    language_decoding_consensus_offset,
    language_decoding_internal_consensus_auc,
    mo,
    np,
):
    import plotly.graph_objects as go

    _consensus_model, _consensus_layer = language_decoding_consensus_model_layer.value
    _consensus_selected = language_decoding_internal_consensus_auc[
        (language_decoding_internal_consensus_auc["model"] == _consensus_model)
        & (language_decoding_internal_consensus_auc["Layer"] == _consensus_layer)
        & (language_decoding_internal_consensus_auc["label_offset"] == language_decoding_consensus_offset.value)
    ].copy()
    _consensus_datasets = ["Podcast", "Cars 2", "fMRI"]
    _consensus_selected["All three positive"] = (_consensus_selected[_consensus_datasets] > 0).all(axis=1)
    _consensus_selected["Mean AUC"] = _consensus_selected[_consensus_datasets].mean(axis=1)
    _consensus_selected = _consensus_selected.sort_values(
        ["All three positive", "Mean AUC"], ascending=[False, False]
    )
    _consensus_values = _consensus_selected[_consensus_datasets].to_numpy()
    _consensus_limit = max(float(np.nanmax(np.abs(_consensus_values))), 0.01)
    _consensus_labels = [
        f"{'✓ ' if _positive else ''}{_task}"
        for _task, _positive in zip(
            _consensus_selected["task"], _consensus_selected["All three positive"]
        )
    ]
    _consensus_fig = go.Figure(go.Heatmap(
        z=_consensus_values,
        x=_consensus_datasets,
        y=_consensus_labels,
        zmin=-_consensus_limit,
        zmax=_consensus_limit,
        colorscale=[[0, "#63bc51"], [0.5, "#f8fafc"], [1, "#6ab2e2"]],
        colorbar={"title": "Relative AUC"},
        hovertemplate="Feature: %{y}<br>Dataset: %{x}<br>Relative AUC: %{z:.3f}<extra></extra>",
    ))
    _consensus_fig.update_layout(
        template="plotly_white",
        height=680,
        margin={"l": 240, "r": 40, "t": 20, "b": 55},
        yaxis={"autorange": "reversed", "automargin": True},
        xaxis={"side": "top"},
    )
    mo.vstack([
        mo.md(
            f"**{int(_consensus_selected['All three positive'].sum())} of {len(_consensus_selected)} shared features** "
            "have positive RRR–PCA AUC in all three datasets."
        ),
        _consensus_fig,
    ])
    return (go,)


@app.cell(hide_code=True)
def language_decoding_within_pair_controls(
    language_decoding_internal_consensus_auc,
    mo,
):
    _within_pair_offset_values = sorted(
        int(_offset) for _offset in language_decoding_internal_consensus_auc["label_offset"].unique()
    )
    _within_pair_offset_options = {
        f"{_offset:+d}" if _offset else "0": _offset
        for _offset in _within_pair_offset_values
    }
    language_decoding_within_pair_offsets = mo.ui.multiselect(
        options=_within_pair_offset_options,
        value=list(_within_pair_offset_options),
        label="Offsets in scatter plots",
    )
    mo.vstack([
        mo.md("""
    ### Within-dataset feature correlations

    Each point is one model/layer × shared feature × selected offset, using its within-dataset RRR–PCA relative AUC. Pearson correlations pool the plotted points across model/layers; the dashed line marks equal AUC in both datasets.
    """),
        language_decoding_within_pair_offsets,
    ])
    return (language_decoding_within_pair_offsets,)


@app.cell(hide_code=True)
def language_decoding_within_pair_scatter(
    go,
    language_decoding_internal_consensus_auc,
    language_decoding_within_pair_offsets,
    mo,
    np,
):
    from plotly.subplots import make_subplots as _make_subplots

    _within_pair_rows = language_decoding_internal_consensus_auc[
        language_decoding_internal_consensus_auc["label_offset"].isin(
            language_decoding_within_pair_offsets.value
        )
    ].copy()
    if _within_pair_rows.empty:
        _within_pair_output = mo.callout(
            mo.md("Select at least one label offset to show the within-dataset correlations."),
            kind="info",
        )
    else:
        _within_pair_specs = [
            ("fMRI", "Cars 2"),
            ("Podcast", "Cars 2"),
            ("fMRI", "Podcast"),
        ]
        _within_pair_titles = []
        for _x_name, _y_name in _within_pair_specs:
            _r = _within_pair_rows[_x_name].corr(_within_pair_rows[_y_name])
            _within_pair_titles.append(
                f"{_x_name} vs {_y_name}<br><sup>Pearson r = {_r:.3f}; n = {len(_within_pair_rows)}</sup>"
            )
        _within_pair_fig = _make_subplots(
            rows=1, cols=3, subplot_titles=_within_pair_titles,
            horizontal_spacing=0.08,
        )
        _within_pair_colors = {-1: "#f59e0b", 0: "#2563eb", 1: "#16a34a"}
        _within_pair_bound = max(
            float(np.nanmax(np.abs(_within_pair_rows[["Podcast", "Cars 2", "fMRI"]].to_numpy()))),
            0.05,
        )
        _within_pair_limits = [-1.05 * _within_pair_bound, 1.05 * _within_pair_bound]
        for _col, (_x_name, _y_name) in enumerate(_within_pair_specs, start=1):
            for _offset, _points in _within_pair_rows.groupby("label_offset", sort=True):
                _within_pair_fig.add_trace(go.Scatter(
                    x=_points[_x_name],
                    y=_points[_y_name],
                    mode="markers",
                    name=f"Offset {int(_offset):+d}",
                    legendgroup=str(int(_offset)),
                    showlegend=_col == 1,
                    marker={
                        "color": _within_pair_colors.get(int(_offset), "#64748b"),
                        "size": 6,
                        "opacity": 0.45,
                        "line": {"color": "white", "width": 0.5},
                    },
                    customdata=np.column_stack([
                        _points["task"].astype(str),
                        _points["model"].astype(str),
                        _points["Layer"].astype(str),
                        _points["label_offset"].astype(int).astype(str),
                    ]),
                    hovertemplate=(
                        "Feature: %{customdata[0]}<br>"
                        "Model: %{customdata[1]}<br>"
                        "Layer: %{customdata[2]}<br>"
                        "Offset: %{customdata[3]}<br>"
                        + _x_name + " AUC: %{x:.3f}<br>"
                        + _y_name + " AUC: %{y:.3f}<extra></extra>"
                    ),
                ), row=1, col=_col)
            _within_pair_fig.add_trace(go.Scatter(
                x=_within_pair_limits,
                y=_within_pair_limits,
                mode="lines",
                name="Identity",
                showlegend=_col == 1,
                line={"color": "#94a3b8", "dash": "dash", "width": 1},
                hoverinfo="skip",
            ), row=1, col=_col)
            if len(_within_pair_rows) >= 2 and _within_pair_rows[_x_name].nunique() >= 2:
                _slope, _intercept = np.polyfit(
                    _within_pair_rows[_x_name], _within_pair_rows[_y_name], 1
                )
                _line_x = np.linspace(
                    _within_pair_rows[_x_name].min(), _within_pair_rows[_x_name].max(), 100
                )
                _within_pair_fig.add_trace(go.Scatter(
                    x=_line_x,
                    y=_slope * _line_x + _intercept,
                    mode="lines",
                    name="Linear fit",
                    showlegend=_col == 1,
                    line={"color": "#111827", "width": 2},
                    hoverinfo="skip",
                ), row=1, col=_col)
            _within_pair_fig.update_xaxes(
                title_text=f"{_x_name} relative AUC",
                range=_within_pair_limits,
                row=1, col=_col,
            )
            _within_pair_fig.update_yaxes(
                title_text=f"{_y_name} relative AUC",
                range=_within_pair_limits,
                scaleanchor=f"x{_col}" if _col > 1 else "x",
                scaleratio=1,
                row=1, col=_col,
            )
        _within_pair_fig.update_layout(
            template="plotly_white",
            width=1450,
            height=520,
            hovermode="closest",
            legend_title="Label offset",
            margin={"l": 75, "r": 30, "t": 85, "b": 80},
        )
        _within_pair_output = _within_pair_fig
    _within_pair_output
    return


@app.cell(hide_code=True)
def language_decoding_pooled_curve_controls(
    language_decoding_rank_differences,
    mo,
):
    _pooled_models = sorted(language_decoding_rank_differences["model"].dropna().unique())
    language_decoding_pooled_model_filter = mo.ui.multiselect(
        options=_pooled_models, value=_pooled_models, label="Models"
    )
    language_decoding_pooled_layer_depth = {}
    for _model, _group in language_decoding_rank_differences[["model", "Layer"]].drop_duplicates().groupby("model"):
        _numeric_layers = sorted(int(_layer) for _layer in _group["Layer"] if _layer != "layerless")
        _max_layer = max(_numeric_layers) if _numeric_layers else None
        for _layer in _group["Layer"]:
            language_decoding_pooled_layer_depth[(_model, _layer)] = (
                "layerless" if _layer == "layerless" else
                int(round(100 * int(_layer) / _max_layer)) if _max_layer else 0
            )
    _pooled_depths = sorted(
        set(language_decoding_pooled_layer_depth.values()),
        key=lambda _depth: -1 if _depth == "layerless" else _depth,
    )
    _pooled_depth_options = {
        "Layerless" if _depth == "layerless" else f"{_depth}% depth": _depth
        for _depth in _pooled_depths
    }
    language_decoding_pooled_depth_filter = mo.ui.multiselect(
        options=_pooled_depth_options, value=list(_pooled_depth_options), label="Layer depths"
    )
    _pooled_offsets = sorted(int(_offset) for _offset in language_decoding_rank_differences["label_offset"].unique())
    _pooled_offset_options = {
        f"{_offset:+d}" if _offset else "0": _offset for _offset in _pooled_offsets
    }
    language_decoding_pooled_offset_filter = mo.ui.dropdown(
        options=_pooled_offset_options,
        value="0" if "0" in _pooled_offset_options else next(iter(_pooled_offset_options)),
        label="Label offset",
    )
    mo.vstack([
        mo.md("""
    ## Average RRR–PCA decoding curves across selected model layers

    Select one decoded target to compare its internal RRR–PCA curves across datasets at the chosen label offset. At each rank, each curve is the equal-weighted mean of selected model/layers' existing relative differences, including ranks with nonpositive RRR scores. Depth is relative to each model's configured maximum layer; GloVe is layerless.
    """),
        mo.hstack([language_decoding_pooled_model_filter, language_decoding_pooled_depth_filter, language_decoding_pooled_offset_filter], justify="start", gap=2),
    ])
    return (
        language_decoding_pooled_depth_filter,
        language_decoding_pooled_layer_depth,
        language_decoding_pooled_model_filter,
        language_decoding_pooled_offset_filter,
    )


@app.cell(hide_code=True)
def language_decoding_pooled_curve_table(
    language_decoding_pooled_depth_filter,
    language_decoding_pooled_layer_depth,
    language_decoding_pooled_model_filter,
    language_decoding_rank_differences,
    language_decoding_relative_curve_auc,
    mo,
    np,
    pd,
):
    language_decoding_pooled_members = language_decoding_rank_differences[
        language_decoding_rank_differences["model"].isin(language_decoding_pooled_model_filter.value)
        & (language_decoding_rank_differences["Source space"] == "Internal space")
    ].copy()
    language_decoding_pooled_members["Depth"] = [
        language_decoding_pooled_layer_depth[(_model, _layer)]
        for _model, _layer in zip(
            language_decoding_pooled_members["model"], language_decoding_pooled_members["Layer"]
        )
    ]
    language_decoding_pooled_members = language_decoding_pooled_members[
        language_decoding_pooled_members["Depth"].isin(language_decoding_pooled_depth_filter.value)
        & np.isfinite(language_decoding_pooled_members["Relative difference"])
    ].copy()
    language_decoding_pooled_members["Model layer"] = (
        language_decoding_pooled_members["model"].astype(str)
        + " / " + language_decoding_pooled_members["Layer"].astype(str)
    )
    _pooled_keys = ["Dataset", "Source space", "label_offset", "task", "kind", "Selected metric"]
    if language_decoding_pooled_members.empty:
        language_decoding_pooled_curves = pd.DataFrame(columns=[
            *_pooled_keys, "k", "Average relative difference", "Members available"
        ])
        language_decoding_pooled_summary = pd.DataFrame(columns=[
            "Dataset", "Source space", "Label offset", "Target label", "Target type",
            "Selected metric", "Relative-difference AUC", "RRR-favored ranks",
            "Ranks", "Min model layers", "Max model layers",
        ])
    else:
        language_decoding_pooled_curves = (
            language_decoding_pooled_members.groupby(_pooled_keys + ["k"], as_index=False, dropna=False)
            .agg(**{
                "Average relative difference": ("Relative difference", "mean"),
                "Members available": ("Model layer", "nunique"),
            })
        )
        _pooled_auc = (
            language_decoding_pooled_curves.groupby(_pooled_keys, dropna=False)
            .apply(
                lambda _group: language_decoding_relative_curve_auc(
                    _group["k"], _group["Average relative difference"]
                ), include_groups=False,
            )
            .rename("Relative-difference AUC").reset_index()
        )
        _pooled_stats = (
            language_decoding_pooled_curves.assign(
                _rrr_favored=lambda _frame: _frame["Average relative difference"] > 0
            )
            .groupby(_pooled_keys, as_index=False, dropna=False)
            .agg(**{
                "RRR-favored ranks": ("_rrr_favored", "sum"),
                "Ranks": ("k", "nunique"),
                "Min model layers": ("Members available", "min"),
                "Max model layers": ("Members available", "max"),
            })
        )
        language_decoding_pooled_summary = (
            _pooled_stats.merge(_pooled_auc, on=_pooled_keys, validate="one_to_one")
            .rename(columns={
                "label_offset": "Label offset", "task": "Target label", "kind": "Target type"
            })
            .sort_values("Relative-difference AUC", ascending=False, na_position="last")
            .reset_index(drop=True)
        )
    _pooled_target_keys = ["Target label", "Target type", "Selected metric"]
    if language_decoding_pooled_summary.empty:
        language_decoding_pooled_target_summary = pd.DataFrame(columns=[
            *_pooled_target_keys, "Datasets", "Offsets"
        ])
    else:
        language_decoding_pooled_target_summary = (
            language_decoding_pooled_summary.groupby(
                _pooled_target_keys, as_index=False, dropna=False
            )
            .agg(**{
                "Datasets": ("Dataset", "nunique"),
                "Offsets": ("Label offset", "nunique"),
            })
            .sort_values("Target label")
            .reset_index(drop=True)
        )
    language_decoding_pooled_table = mo.ui.table(
        language_decoding_pooled_target_summary,
        selection="single", pagination=True, page_size=25,
        show_column_summaries=False,
    )
    mo.vstack([
        mo.md(
            f"**{len(language_decoding_pooled_target_summary)} decoded targets** from selected models and depths. "
            "Select a target to compare its averaged internal curves across datasets."
        ),
        language_decoding_pooled_table,
    ])
    return language_decoding_pooled_curves, language_decoding_pooled_table


@app.cell(hide_code=True)
def language_decoding_pooled_curve_plot(
    language_decoding_pooled_curves,
    language_decoding_pooled_offset_filter,
    language_decoding_pooled_table,
    language_decoding_relative_curve_auc,
    mo,
    plt,
):
    _pooled_selection = language_decoding_pooled_table.value
    if _pooled_selection is None or len(_pooled_selection) == 0:
        _pooled_plot_output = mo.callout(
            mo.md("Select a decoded target above to compare its within-dataset curves."),
            kind="info",
        )
    else:
        _row = _pooled_selection.iloc[0]
        _curves = language_decoding_pooled_curves[
            (language_decoding_pooled_curves["task"] == _row["Target label"])
            & (language_decoding_pooled_curves["kind"] == _row["Target type"])
            & (language_decoding_pooled_curves["Selected metric"] == _row["Selected metric"])
            & (language_decoding_pooled_curves["label_offset"] == language_decoding_pooled_offset_filter.value)
        ].copy()
        if _curves.empty:
            _pooled_plot_output = mo.callout(
                mo.md("No internal curves are available for this target at the selected offset."),
                kind="info",
            )
        else:
            _dataset_colors = {"Podcast": "#2563eb", "Cars 2": "#f59e0b", "fMRI": "#16a34a"}
            _fig, _ax = plt.subplots(figsize=(10, 5.5))
            for _dataset in ("Podcast", "Cars 2", "fMRI"):
                _curve = _curves[_curves["Dataset"] == _dataset].sort_values("k")
                if _curve.empty:
                    continue
                _auc = language_decoding_relative_curve_auc(
                    _curve["k"], _curve["Average relative difference"]
                )
                _ax.plot(
                    _curve["k"], _curve["Average relative difference"],
                    color=_dataset_colors[_dataset], linewidth=2.3,
                    marker="o", markersize=5,
                    label=f"{_dataset} (AUC {_auc:+.2f})",
                )
            _ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            _ax.set_xscale("log", base=2)
            _ranks = sorted(_curves["k"].astype(int).unique())
            _ax.set_xticks(_ranks, [str(_rank) for _rank in _ranks])
            _ax.set_xlabel("Rank")
            _ax.set_ylabel("Mean relative difference (RRR vs PCA)")
            _ax.set_title(
                f"{_row['Target label']} · offset {language_decoding_pooled_offset_filter.value}"
            )
            _ax.grid(axis="y", alpha=0.25)
            _ax.spines[["top", "right"]].set_visible(False)
            _ax.legend(frameon=False)
            _fig.tight_layout()
            _coverage_min = int(_curves["Members available"].min())
            _coverage_max = int(_curves["Members available"].max())
            _pooled_plot_output = mo.vstack([
                _fig,
                mo.md(
                    f"**{_curves['Dataset'].nunique()} internal dataset curves** are shown. "
                    f"Each rank averages {_coverage_min}–{_coverage_max} selected model layers, "
                    "depending on the dataset and rank."
                ),
            ])
    _pooled_plot_output
    return


@app.cell(hide_code=True)
def paired_permutation_tests(np, pd):
    def paired_permutation_tests(
        distributions,
        tests,
        *,
        n_permutations=10_000,
        seed=0,
    ):
        """Run one-sided paired sign-flip permutation tests with Holm correction.

        Each test is a dictionary containing ``first``, ``second``, and
        ``alternative``. The observed statistic is always first minus second.
        """
        if not distributions:
            raise ValueError("at least one distribution is required")
        if n_permutations < 1:
            raise ValueError("n_permutations must be positive")

        values = {
            label: np.asarray(distribution, dtype=float).reshape(-1)
            for label, distribution in distributions.items()
        }
        rng = np.random.default_rng(seed)
        rows = []
        for test in tests:
            first = test["first"]
            second = test["second"]
            alternative = test["alternative"]
            if first not in values or second not in values:
                raise ValueError(f"unknown distribution in test: {first!r}, {second!r}")
            if alternative not in {"greater", "less"}:
                raise ValueError("alternative must be 'greater' or 'less'")
            if len(values[first]) != len(values[second]):
                raise ValueError(f"paired distributions must have equal lengths: {first!r}, {second!r}")

            differences = values[first] - values[second]
            differences = differences[np.isfinite(differences)]
            if not len(differences):
                raise ValueError(f"test has no finite paired values: {first!r}, {second!r}")
            observed = float(np.mean(differences))
            shuffled_statistics = np.empty(n_permutations, dtype=float)
            for permutation in range(n_permutations):
                signs = rng.choice((-1.0, 1.0), size=len(differences))
                shuffled_statistics[permutation] = np.mean(differences * signs)
            if alternative == "greater":
                exceedances = np.count_nonzero(shuffled_statistics >= observed)
            else:
                exceedances = np.count_nonzero(shuffled_statistics <= observed)
            rows.append({
                "first": first,
                "second": second,
                "alternative": alternative,
                "n_pairs": len(differences),
                "observed_difference": observed,
                "p_value": (exceedances + 1) / (n_permutations + 1),
            })

        results = pd.DataFrame(rows)
        if results.empty:
            results["p_value_holm"] = pd.Series(dtype=float)
            results["significant"] = pd.Series(dtype=bool)
            return results
        order = np.argsort(results["p_value"].to_numpy())
        adjusted = np.empty(len(results), dtype=float)
        sorted_p = results["p_value"].to_numpy()[order]
        sorted_adjusted = np.maximum.accumulate(
            np.minimum(1.0, (len(results) - np.arange(len(results))) * sorted_p)
        )
        adjusted[order] = sorted_adjusted
        results["p_value_holm"] = adjusted
        results["significant"] = results["p_value_holm"] < 0.05
        return results


    def annotate_pairwise_tests(ax, results, labels, *, compact=False):
        """Add brackets and adjusted p-values for requested pairwise tests."""
        if results is None or results.empty:
            return
        positions = {label: index for index, label in enumerate(labels, start=1)}
        if compact:
            transform = ax.get_xaxis_transform()
            top = 0.90
            tick = 0.025
            for offset, result in enumerate(results.to_dict("records")):
                first, second = result["first"], result["second"]
                if first not in positions or second not in positions:
                    raise ValueError(f"test label is not plotted: {first!r}, {second!r}")
                left, right = sorted((positions[first], positions[second]))
                y = top - offset * 0.12
                ax.plot(
                    [left, left, right, right],
                    [y, y + tick, y + tick, y],
                    transform=transform,
                    color="#374151",
                    linewidth=1.0,
                    clip_on=False,
                    zorder=6,
                )
                p_value = result["p_value_holm"]
                p_text = "p<.001" if p_value < 0.001 else f"p={p_value:.3f}"
                mark = "*" if result["significant"] else "n.s."
                ax.text(
                    (left + right) / 2,
                    y + tick,
                    f"{mark} {p_text}",
                    transform=transform,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="#374151",
                    clip_on=False,
                    zorder=7,
                )
            return

        lower, upper = ax.get_ylim()
        span = max(upper - lower, 1e-6)
        base = upper + 0.05 * span
        step = 0.12 * span
        for offset, result in enumerate(results.to_dict("records")):
            first, second = result["first"], result["second"]
            if first not in positions or second not in positions:
                raise ValueError(f"test label is not plotted: {first!r}, {second!r}")
            left, right = sorted((positions[first], positions[second]))
            y = base + offset * step
            tick = 0.025 * span
            ax.plot(
                [left, left, right, right],
                [y, y + tick, y + tick, y],
                color="#374151",
                linewidth=0.8,
                clip_on=False,
                zorder=6,
            )
            p_value = result["p_value_holm"]
            p_text = "p<.001" if p_value < 0.001 else f"p={p_value:.3f}"
            mark = "*" if result["significant"] else "n.s."
            ax.text(
                (left + right) / 2,
                y + tick,
                f"{mark} {p_text}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#374151",
                clip_on=False,
                zorder=7,
            )
        ax.set_ylim(lower, base + len(results) * step)


    return annotate_pairwise_tests, paired_permutation_tests


@app.cell(hide_code=True)
def language_decoding_feature_correlations(
    REPO_ROOT,
    language_decoding_dataset_label,
    language_decoding_results,
    mo,
    np,
    pd,
    plt,
):
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    def _feature_table(_dataset):
        _paths = {
            "podcast": [REPO_ROOT / "processed_data/language_features/podcast/podcast/tokens.tsv"],
            "brain-treebank-cars-2": [REPO_ROOT / "processed_data/language_features/brain_treebank/cars-2/tokens.tsv"],
            "deep-fmri": sorted((REPO_ROOT / "processed_data/language_features/fmri").glob("*/tokens.tsv")),
        }[_dataset]
        _tables = [pd.read_csv(_path, sep="\t") for _path in _paths if _path.is_file()]
        return pd.concat(_tables, ignore_index=True) if _tables else pd.DataFrame()

    def _association_matrix(_table, _features, _kinds):
        """Pairwise association: Pearson, correlation ratio, or Cramer's V."""
        _features = [name for name in _features if name in _table]
        _matrix = pd.DataFrame(np.eye(len(_features)), index=_features, columns=_features)
        for _i, _left_name in enumerate(_features):
            for _right_name in _features[_i + 1:]:
                _left, _right = _table[_left_name], _table[_right_name]
                _categorical_left = _kinds.get(_left_name) == "categorical"
                _categorical_right = _kinds.get(_right_name) == "categorical"
                _paired = pd.DataFrame({"left": _left, "right": _right}).dropna()
                if len(_paired) < 2:
                    _value = np.nan
                elif not _categorical_left and not _categorical_right:
                    _value = _paired["left"].corr(_paired["right"])
                elif _categorical_left != _categorical_right:
                    _cat_key, _num_key = ("left", "right") if _categorical_left else ("right", "left")
                    _groups = _paired.groupby(_cat_key, observed=True)[_num_key]
                    _overall = _paired[_num_key].mean()
                    _denominator = ((_paired[_num_key] - _overall) ** 2).sum()
                    _value = np.sqrt(
                        sum(len(_group) * (_group.mean() - _overall) ** 2 for _, _group in _groups)
                        / _denominator
                    ) if _denominator > 0 else np.nan
                else:
                    _counts = pd.crosstab(_paired["left"], _paired["right"]).to_numpy(dtype=float)
                    _n = _counts.sum()
                    _expected = np.outer(_counts.sum(axis=1), _counts.sum(axis=0)) / _n
                    _chi2 = np.sum((_counts - _expected) ** 2 / _expected)
                    _denom = min(_counts.shape[0] - 1, _counts.shape[1] - 1)
                    _value = np.sqrt(_chi2 / (_n * _denom)) if _denom > 0 else np.nan
                _matrix.loc[_left_name, _right_name] = _matrix.loc[_right_name, _left_name] = _value
        return _matrix

    def _plot_matrix(_matrix, _title, *, _signed):
        _matrix = _matrix.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if len(_matrix) < 2:
            return None
        _values = _matrix.to_numpy(dtype=float)
        _distance = 1.0 - np.nan_to_num(np.abs(_values), nan=0.0)
        np.fill_diagonal(_distance, 0.0)
        _order = leaves_list(linkage(squareform(_distance, checks=False), method="average"))
        _matrix = _matrix.iloc[_order, _order]
        _fig, _ax = plt.subplots(figsize=(max(7, 0.34 * len(_matrix)), max(6, 0.32 * len(_matrix))))
        _image = _ax.imshow(_matrix, vmin=-1 if _signed else 0, vmax=1, cmap="coolwarm" if _signed else "viridis", interpolation="nearest", aspect="auto")
        _ax.set_xticks(range(len(_matrix)), _matrix.columns, rotation=90, fontsize=7)
        _ax.set_yticks(range(len(_matrix)), _matrix.index, fontsize=7)
        _ax.set_title(_title)
        _fig.colorbar(_image, ax=_ax, label="Pearson r / association strength" if _signed else "Pearson r")
        _fig.tight_layout()
        return _fig

    _outputs = [mo.md("## Language feature correlations by dataset")]
    for _dataset in ("podcast", "brain-treebank-cars-2", "deep-fmri"):
        _dataset_rows = language_decoding_results[
            language_decoding_results["target_dataset"] == _dataset
        ]
        if _dataset_rows.empty:
            continue
        _label = language_decoding_dataset_label(_dataset)
        _outputs.append(mo.md(f"### {_label}"))

        # Raw feature relationships use all available word rows. Categorical features
        # use association strength, while continuous pairs use signed Pearson r.
        _feature_kinds = (
            _dataset_rows[["task", "kind"]].drop_duplicates("task")
            .set_index("task")["kind"].to_dict()
        )
        _raw = _feature_table(_dataset)
        _raw_features = sorted(set(_feature_kinds) & set(_raw.columns))
        if len(_raw_features) >= 2:
            _raw_numeric = _raw.copy()
            for _feature in _raw_features:
                if _feature_kinds.get(_feature) == "continuous":
                    _raw_numeric[_feature] = pd.to_numeric(_raw_numeric[_feature], errors="coerce")
                else:
                    _raw_numeric[_feature] = _raw_numeric[_feature].astype("string")
            _raw_matrix = _association_matrix(_raw_numeric, _raw_features, _feature_kinds)
            _raw_fig = _plot_matrix(_raw_matrix, "Raw feature associations across words", _signed=True)
            if _raw_fig is not None:
                _outputs.append(_raw_fig)
                _outputs.append(mo.md("Continuous pairs show signed Pearson correlation; pairs involving categorical features show unsigned correlation-ratio or Cramer's V association."))
        else:
            _outputs.append(mo.md("Raw feature tables or matching feature columns are unavailable."))

        # Compare tasks over matched internal model/layer runs at label offset zero.
        _scores = _dataset_rows[
            (_dataset_rows["generation"] == "internal")
            & (_dataset_rows["label_offset"] == 0)
            & (_dataset_rows["space"] == "rank")
            & (_dataset_rows["subspace"] == "rrr")
        ].copy()
        if not _scores.empty:
            _run_keys = ["model", "layer", "task"]
            _score_wide = _scores.groupby(_run_keys, dropna=False)["primary_score"].mean().unstack("task")
            _score_matrix = _score_wide.corr(min_periods=3)
            _score_fig = _plot_matrix(_score_matrix, "Decoding-score correlations across model/layer runs (internal, offset 0; ranks averaged)", _signed=True)
            if _score_fig is not None:
                _outputs.append(_score_fig)
            _run_centered_scores = _score_wide.sub(_score_wide.mean(axis=1), axis=0)
            _centered_matrix = _run_centered_scores.corr(min_periods=3)
            _centered_fig = _plot_matrix(
                _centered_matrix,
                'Run centered decoding score correlations (each run minus its mean across tasks)',
                _signed=True,
            )
            if _centered_fig is not None:
                _outputs.append(_centered_fig)
                _outputs.append(mo.md('For each model/layer run, this subtracts that run’s average score across available tasks before correlating features. Remaining correlations reflect task specific score deviations beyond overall run quality.'))
        else:
            _outputs.append(mo.md("No internal, offset-zero RRR decoding scores are available for this dataset."))
    mo.vstack(_outputs)
    return


@app.cell
def _(REPO_ROOT, mo, np, pd, plt):
    def load_significant_target_counts(results_root=REPO_ROOT / "results"):
        """Count significant neural targets from completed significance artifacts."""
        import csv
        import glob
        import json

        _datasets = {
            "Podcast": ("run_significance/sig_lags/podcast", "podcast"),
            "Cars 2": ("run_significance/sig_lags/brain-treebank-cars-2", "brain-treebank-cars-2"),
            "fMRI": ("run_significance/held_out_session/deep-fmri", "deep-fmri"),
        }
        _rows = []
        for _label, (_relative_root, _dataset_identity) in _datasets.items():
            _root = results_root / _relative_root
            for _manifest_path in glob.glob(str(_root / "**/manifest.json"), recursive=True):
                with open(_manifest_path) as _handle:
                    _manifest = json.load(_handle)
                _identity = _manifest.get("identity", {})
                _entries = _manifest.get("entries", {})
                if (
                    not _manifest.get("complete")
                    or _identity.get("dataset") != _dataset_identity
                    or not _entries.get("target_significance")
                ):
                    continue
                _artifact_root = __import__("pathlib").Path(_manifest_path).parent
                _significant = 0
                _total = 0
                for _entry in _entries["target_significance"].values():
                    _csv_path = _artifact_root / _entry["file"]
                    if not _csv_path.is_file():
                        continue
                    with open(_csv_path, newline="") as _handle:
                        for _row in csv.DictReader(_handle):
                            _total += 1
                            _significant += _row["significant"].strip().lower() == "true"
                if _total:
                    _rows.append({
                        "dataset": _label,
                        "model": _identity["model"],
                        "layer": _identity.get("layer"),
                        "significant_targets": _significant,
                        "total_targets": _total,
                    })
        _frame = pd.DataFrame(_rows)
        if _frame.empty:
            raise ValueError("No completed significance target artifacts were found")
        if _frame.duplicated(["dataset", "model", "layer"]).any():
            _duplicates = _frame[_frame.duplicated(["dataset", "model", "layer"], keep=False)]
            raise ValueError(f"Multiple completed significance artifacts found for a model/layer: {_duplicates}")
        return _frame


    def plot_significant_target_model_layers(counts, dataset_label, model_colors):
        """Plot significant target counts as model-grouped, layer-ordered bars."""
        _data = counts[counts["dataset"] == dataset_label].copy()
        _fig, _ax = plt.subplots(figsize=(13, 5.2))
        _positions = []
        _tick_labels = []
        _next_position = 0.0
        for _model in sorted(_data["model"].unique()):
            _model_rows = _data[_data["model"] == _model].copy()
            _model_rows["_layer_order"] = _model_rows["layer"].map(
                lambda _layer: (pd.isna(_layer), -1 if pd.isna(_layer) else int(_layer))
            )
            _model_rows = _model_rows.sort_values("_layer_order")
            _model_positions = np.arange(len(_model_rows), dtype=float) + _next_position
            _ax.bar(
                _model_positions,
                _model_rows["significant_targets"],
                width=0.82,
                color=model_colors[_model],
            )
            _positions.extend(_model_positions)
            _tick_labels.extend(
                "layerless" if pd.isna(_layer) else str(int(_layer))
                for _layer in _model_rows["layer"]
            )
            _ax.text(
                float(_model_positions.mean()), -0.16, _model,
                transform=_ax.get_xaxis_transform(), ha="center", va="top", fontsize=8,
            )
            _next_position = float(_model_positions[-1]) + 1.8
        _ax.set_xticks(_positions, _tick_labels, rotation=45, ha="right")
        _ax.set_ylabel("Significant targets")
        _ax.set_title(dataset_label)
        _ax.grid(axis="y", alpha=0.25)
        _ax.spines[["top", "right"]].set_visible(False)
        _fig.tight_layout()
        return _fig, _ax


    significant_target_counts = load_significant_target_counts()
    _significance_models = sorted(significant_target_counts["model"].unique())
    _significance_cmap = plt.get_cmap("tab20", max(len(_significance_models), 1))
    significant_target_model_colors = {
        _model: _significance_cmap(_index)
        for _index, _model in enumerate(_significance_models)
    }
    significant_target_plots = [
        plot_significant_target_model_layers(significant_target_counts, _dataset, significant_target_model_colors)[0]
        for _dataset in ("Podcast", "Cars 2", "fMRI")
        if _dataset in significant_target_counts["dataset"].values
    ]
    mo.vstack([
        mo.md(
            "## Significant targets by model layer\n"
            "Each bar counts targets marked significant in the completed significance artifact. "
            "Cars 2 uses Laplacian rereferencing; fMRI counts voxels across held-out sessions."
        ),
        *significant_target_plots,
    ])
    return


if __name__ == "__main__":
    app.run()
