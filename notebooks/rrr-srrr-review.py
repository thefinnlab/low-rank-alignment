import marimo

__generated_with = "0.20.0"
app = marimo.App(width="medium")


# ---------------------------------------------------------------------------
# Foundation
# ---------------------------------------------------------------------------


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import kurtosis as sp_kurtosis, entropy as sp_entropy, skew as sp_skew
    from pathlib import Path
    return Path, alt, mo, np, pd, plt, sp_entropy, sp_kurtosis, sp_skew


@app.cell
def _():
    import sys
    import os
    _cwd = os.path.abspath(".")
    _root = _cwd
    while _root != os.path.dirname(_root):
        if os.path.isdir(os.path.join(_root, "utils")):
            break
        _root = os.path.dirname(_root)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from utils.embeddings import load_embeddings
    from utils.examples import get_mo_display_activations, get_example_display_controls
    from utils.encoding import load_encoding_results
    from utils.data_loaders import load_sig_electrodes, load_raws
    from utils.atlas import get_coord_atlas_labels, build_electrode_region_map
    return (
        build_electrode_region_map,
        get_coord_atlas_labels,
        get_example_display_controls,
        get_mo_display_activations,
        load_embeddings,
        load_encoding_results,
        load_raws,
        load_sig_electrodes,
    )


@app.cell
def _(mo):
    mo.md("# RRR Interpretability Explorer")
    return


@app.cell
def _(mo):
    results_root_input = mo.ui.text(
        value="results/gpt2xl_layer24_offset0",
        label="Results root",
        full_width=True,
    )
    data_root_input = mo.ui.text(
        value="data/",
        label="Data root",
        full_width=True,
    )
    layer_input = mo.ui.number(value=24, label="Layer")
    mo.hstack([results_root_input, data_root_input, layer_input])
    return data_root_input, layer_input, results_root_input


@app.cell
def _(Path, mo, results_root_input):
    _rr = Path(results_root_input.value)
    mo.stop(
        not _rr.exists(),
        mo.callout(mo.md(f"Results root `{_rr}` does not exist."), kind="warn"),
    )

    rrr_options = []

    # cross_subject
    if (_rr / "rrr" / "grid_results.csv").exists():
        rrr_options.append("cross_subject")

    # per_subject
    _ps_rrr = _rr / "per_subject_rrr" / "subjects"
    if _ps_rrr.exists():
        for _p in sorted(_ps_rrr.glob("sub_*/grid_results.csv")):
            rrr_options.append(f"per_subject/{_p.parent.name}")

    # per_region
    _pr_rrr = _rr / "per_region_rrr" / "regions"
    if _pr_rrr.exists():
        for _p in sorted(_pr_rrr.glob("*/grid_results.csv")):
            rrr_options.append(f"per_region/{_p.parent.name}")

    return (rrr_options,)


@app.cell
def _(data_root_input, layer_input, load_embeddings, np):
    aligned_embeddings, df_word = load_embeddings(
        data_root_input.value, int(layer_input.value)
    )

    class _AllWords:
        selection = np.arange(len(df_word))

    all_words_epochs = _AllWords()
    return aligned_embeddings, all_words_epochs, df_word


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@app.cell
def _(Path, results_root_input):
    def get_rrr_dir(source: str) -> Path:
        rr = Path(results_root_input.value)
        if source == "cross_subject":
            return rr / "rrr"
        elif source.startswith("per_subject/"):
            return rr / "per_subject_rrr" / "subjects" / source.split("/", 1)[1]
        else:
            return rr / "per_region_rrr" / "regions" / source.split("/", 1)[1]

    return (get_rrr_dir,)


@app.cell
def _(np):
    def load_rrr_AB(rrr_dir, fold_str, k):
        """Load A and B matrices from the correct fold directory, truncated to rank k."""
        fold_dir = rrr_dir / fold_str
        A_full = np.load(fold_dir / "A.npy")  # (n_embed, max_k)
        B_full = np.load(fold_dir / "B.npy")  # (n_targets, max_k)
        k_clamp = min(k, A_full.shape[1], B_full.shape[1])
        return A_full[:, :k_clamp], B_full[:, :k_clamp]

    return (load_rrr_AB,)


@app.cell
def _(
    Path, load_encoding_results, load_raws, load_sig_electrodes, np,
    results_root_input, data_root_input,
):
    def load_electrode_metadata(source: str):
        """Load electrode metadata (names, lags, coords, y-column mapping) for a source.

        Returns dict with keys:
            subjects: list of {sub_id, elec_names, lags, y_col_start, y_col_end, coords}
            lags: shared lag array (seconds)
            all_elec_names: flat list of electrode names in Y column order
            all_coords: (n_total_elecs, 3) MNI coordinates in mm
        """
        import os
        rr = Path(results_root_input.value)
        data_root = data_root_input.value

        # Determine subject IDs and per-subject electrode lists
        sig_electrodes, _ = load_sig_electrodes(
            str(rr), list(range(1, 10))
        )

        if source == "cross_subject":
            subs_with_sig = [s for s in sorted(sig_electrodes.keys()) if sig_electrodes[s]]
            per_sub_elecs = {s: sig_electrodes[s] for s in subs_with_sig}
        elif source.startswith("per_subject/"):
            sub_name = source.split("/", 1)[1]  # e.g. "sub_3"
            sub_id = int(sub_name.replace("sub_", ""))
            per_sub_elecs = {sub_id: sig_electrodes.get(sub_id, [])}
        elif source.startswith("per_region/"):
            region_name = source.split("/", 1)[1]
            # Re-derive which electrodes are in this region
            all_sub_ids = [s for s in sorted(sig_electrodes.keys()) if sig_electrodes[s]]
            try:
                elec_region_map = build_electrode_region_map(
                    all_sub_ids, data_root, None, None,
                )
                region_sub_elecs = elec_region_map.get(region_name, {})
                per_sub_elecs = {}
                for sid, elecs in region_sub_elecs.items():
                    sig_set = set(sig_electrodes.get(sid, []))
                    filtered = [e for e in elecs if e in sig_set]
                    if filtered:
                        per_sub_elecs[sid] = filtered
            except Exception:
                per_sub_elecs = {}
        else:
            per_sub_elecs = {}

        # Load metadata per subject
        subjects_info = []
        y_cursor = 0
        lags = None
        all_elec_names = []
        all_coords = []

        for sub_id in sorted(per_sub_elecs.keys()):
            elec_names = per_sub_elecs[sub_id]
            enc_path = os.path.join(
                str(rr), "subjects", f"sub_{sub_id}", "encoding_results.joblib"
            )
            if not os.path.exists(enc_path):
                continue
            enc_results = load_encoding_results(enc_path)
            sub_lags = enc_results["lags"]
            if lags is None:
                lags = sub_lags
            n_lags = len(sub_lags)
            n_target_cols = len(elec_names) * n_lags

            # Load MNE raw header (no data) for MNI coordinates
            try:
                raws = load_raws(
                    [sub_id], data_root,
                    use_high_gamma=True,
                    per_subject_electrodes={sub_id: elec_names},
                    no_preload=True,
                )
                raw = raws[0]
                coords = np.array([
                    raw.info["chs"][raw.ch_names.index(ch)]["loc"][:3] * 1000
                    for ch in elec_names if ch in raw.ch_names
                ])
            except Exception:
                coords = np.zeros((len(elec_names), 3))

            subjects_info.append({
                "sub_id": sub_id,
                "elec_names": elec_names,
                "lags": sub_lags,
                "y_col_start": y_cursor,
                "y_col_end": y_cursor + n_target_cols,
                "coords": coords,
            })
            all_elec_names.extend(elec_names)
            all_coords.append(coords)
            y_cursor += n_target_cols

        return {
            "subjects": subjects_info,
            "lags": lags if lags is not None else np.array([]),
            "all_elec_names": all_elec_names,
            "all_coords": np.vstack(all_coords) if all_coords else np.zeros((0, 3)),
        }

    return (load_electrode_metadata,)


# ---------------------------------------------------------------------------
# Section 1: RRR — Rank vs Explained Variance
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md("---\n## Section 1: RRR — Rank vs Explained Variance")
    return


@app.cell
def _(mo, rrr_options):
    mo.stop(
        len(rrr_options) == 0,
        mo.callout(mo.md("No RRR results found. Run `scripts/run_rrr.py` first."), kind="warn"),
    )
    return


@app.cell
def _(mo, rrr_options):
    s1_source = mo.ui.dropdown(
        options=rrr_options,
        value=rrr_options[0] if rrr_options else None,
        label="RRR source",
    )
    s1_show_folds = mo.ui.checkbox(value=False, label="Show individual folds")
    s1_threshold = mo.ui.slider(
        start=0.0, stop=0.10, value=0.01, step=0.005,
        label="Sufficiency threshold (full_r2 − minimal_r2 ≤)",
    )
    mo.hstack([s1_source, s1_show_folds, s1_threshold])
    return s1_show_folds, s1_source, s1_threshold


@app.cell
def _(get_rrr_dir, mo, pd, s1_source):
    _unit_dir = get_rrr_dir(s1_source.value)
    _summary_path = _unit_dir / "summary.csv"
    _grid_path = _unit_dir / "grid_results.csv"
    mo.stop(
        not _summary_path.exists(),
        mo.callout(mo.md(f"summary.csv not found at `{_summary_path}`."), kind="warn"),
    )
    s1_summary = pd.read_csv(_summary_path)
    s1_grid = pd.read_csv(_grid_path) if _grid_path.exists() else None
    return s1_grid, s1_summary


@app.cell
def _(s1_grid, s1_source, s1_summary):
    from utils.plotting import plot_rrr_rank_results
    plot_rrr_rank_results(
        s1_summary,
        grid_df=s1_grid,
        title=f"RRR: R² vs rank ({s1_source.value})",
    )
    return


@app.cell
def _(alt, mo, s1_summary):
    # CCA / Procrustes (optional)
    _opt_cols = [c for c in ["cca_similarity", "procrustes_similarity"] if c in s1_summary.columns]
    mo.stop(len(_opt_cols) == 0, mo.md("*No CCA or Procrustes similarity columns found.*"))
    _long_opt = s1_summary.melt(
        id_vars="k", value_vars=_opt_cols, var_name="metric", value_name="value"
    )
    _opt_chart = (
        alt.Chart(_long_opt)
        .mark_line(point=True)
        .encode(
            x=alt.X("k:Q", scale=alt.Scale(type="log"), title="Rank k"),
            y=alt.Y("value:Q", title="Similarity", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("metric:N"),
            tooltip=["k:Q", "metric:N", alt.Tooltip("value:Q", format=".4f")],
        )
        .properties(width="container", height=200, title="CCA / Procrustes similarity vs rank")
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    mo.ui.altair_chart(_opt_chart)
    return


@app.cell
def _(mo, np, s1_summary, s1_threshold):
    # Sufficiency summary
    if "full_r2" in s1_summary.columns and "minimal_r2" in s1_summary.columns:
        _gap = s1_summary["full_r2"] - s1_summary["minimal_r2"]
        _ratio = s1_summary["minimal_r2"] / s1_summary["full_r2"].replace(0, float("nan"))
        _suf = s1_summary[_gap <= s1_threshold.value]
        if len(_suf) > 0:
            _min_k = int(_suf["k"].min())
            _gap_at_k = float(_gap[s1_summary["k"] == _min_k].values[0])
            _ratio_at_k = float(_ratio[s1_summary["k"] == _min_k].values[0])
            mo.callout(
                mo.md(f"**Minimum sufficient rank (RRR):** k = **{_min_k}**  \n"
                      f"First rank where avg. `full_r2 − minimal_r2 ≤ {s1_threshold.value:.3f}` "
                      f"(gap = {_gap_at_k:.4f}, ratio = {_ratio_at_k:.3f})"),
                kind="success",
            )
        else:
            mo.callout(
                mo.md(f"No rank achieves `full_r2 − minimal_r2 ≤ {s1_threshold.value:.3f}`. "
                      "Try a larger threshold or a larger rank grid."),
                kind="warn",
            )
    else:
        mo.callout(mo.md("*`full_r2` or `minimal_r2` not found in summary.csv.*"), kind="warn")
    return


# ---------------------------------------------------------------------------
# Section 2: Direction Interpretability
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md("---\n## Section 2: Direction Interpretability")
    return


@app.cell
def _(mo, rrr_options):
    mo.stop(
        len(rrr_options) == 0,
        mo.callout(mo.md("No RRR results found."), kind="warn"),
    )
    s2_source = mo.ui.dropdown(
        options=rrr_options,
        value=rrr_options[0] if rrr_options else None,
        label="Source",
    )
    s2_fold = mo.ui.dropdown(
        options=["fold_0", "fold_1"],
        value="fold_0",
        label="Fold",
    )
    s2_k = mo.ui.number(value=10, start=1, stop=50, label="Rank k")
    s2_dir_idx = mo.ui.number(value=0, start=0, stop=49, label="Direction index")
    mo.hstack([s2_source, s2_fold, s2_k, s2_dir_idx])
    return s2_dir_idx, s2_fold, s2_k, s2_source


@app.cell
def _(
    aligned_embeddings, get_rrr_dir, load_rrr_AB,
    mo, np, s2_dir_idx, s2_fold, s2_k, s2_source,
):
    _unit_dir = get_rrr_dir(s2_source.value)
    _fold_dir = _unit_dir / s2_fold.value
    mo.stop(
        not (_fold_dir / "A.npy").exists(),
        mo.callout(mo.md(f"A.npy not found at `{_fold_dir}`."), kind="warn"),
    )
    _A, _B = load_rrr_AB(_unit_dir, s2_fold.value, int(s2_k.value))
    _dir_idx = min(int(s2_dir_idx.value), _A.shape[1] - 1)
    s2_direction = _A[:, _dir_idx]
    s2_activations = (aligned_embeddings @ s2_direction).astype(np.float64)
    return (s2_activations,)


@app.cell
def _(alt, mo, np, pd, s2_activations, s2_dir_idx, s2_fold, s2_k):
    _counts, _edges = np.histogram(s2_activations, bins=100)
    _hist_df = pd.DataFrame({"x1": _edges[:-1], "x2": _edges[1:], "count": _counts})
    _bars = (
        alt.Chart(_hist_df)
        .mark_bar(color="#93c5fd", stroke="#bfdbfe", strokeWidth=0.5)
        .encode(
            x=alt.X("x1:Q", title="Activation",
                    scale=alt.Scale(domain=[float(_edges[0]), float(_edges[-1])])),
            x2="x2:Q",
            y=alt.Y("count:Q", title="Count"),
            tooltip=[
                alt.Tooltip("x1:Q", title="From", format=".3f"),
                alt.Tooltip("x2:Q", title="To", format=".3f"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
    )
    _chart = _bars.properties(
        width="container", height=220,
        title=f"RRR direction {s2_dir_idx.value} — k={s2_k.value}, {s2_fold.value}",
    ).configure_view(strokeWidth=0).configure_axis(grid=False)
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(get_example_display_controls, mo):
    s2_num_bins, s2_samp_per_bin, s2_ctx, s2_seed = get_example_display_controls()
    mo.hstack([s2_num_bins, s2_samp_per_bin, s2_ctx, s2_seed])
    return s2_ctx, s2_num_bins, s2_samp_per_bin, s2_seed


@app.cell
def _(
    all_words_epochs, df_word, get_mo_display_activations,
    s2_activations, s2_ctx, s2_num_bins, s2_samp_per_bin, s2_seed,
):
    get_mo_display_activations(
        s2_activations,
        df_word,
        all_words_epochs,
        s2_num_bins,
        s2_samp_per_bin,
        s2_ctx,
        s2_seed,
    )
    return


# ---------------------------------------------------------------------------
# Section 3: B Matrix — Temporal Profile
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md("---\n## Section 3: B Matrix — Temporal Profile")
    return


@app.cell
def _(mo, rrr_options):
    mo.stop(
        len(rrr_options) == 0,
        mo.callout(mo.md("No RRR results found."), kind="warn"),
    )
    s3_source = mo.ui.dropdown(
        options=rrr_options,
        value=rrr_options[0] if rrr_options else None,
        label="Source",
    )
    s3_fold = mo.ui.dropdown(
        options=["fold_0", "fold_1"],
        value="fold_0",
        label="Fold",
    )
    s3_k = mo.ui.number(value=10, start=1, stop=50, label="Rank k")
    s3_dir_idx = mo.ui.number(value=0, start=0, stop=49, label="Direction index")
    mo.hstack([s3_source, s3_fold, s3_k, s3_dir_idx])
    return s3_dir_idx, s3_fold, s3_k, s3_source


@app.cell
def _(
    get_rrr_dir, load_electrode_metadata, load_rrr_AB,
    mo, np, pd, s3_dir_idx, s3_fold, s3_k, s3_source,
):
    _unit_dir = get_rrr_dir(s3_source.value)
    _fold_dir = _unit_dir / s3_fold.value
    mo.stop(
        not (_fold_dir / "B.npy").exists(),
        mo.callout(mo.md(f"B.npy not found at `{_fold_dir}`."), kind="warn"),
    )
    _A, _B = load_rrr_AB(_unit_dir, s3_fold.value, int(s3_k.value))
    _dir_idx = min(int(s3_dir_idx.value), _B.shape[1] - 1)
    _b_col = _B[:, _dir_idx]  # (n_targets,)

    _meta = load_electrode_metadata(s3_source.value)
    mo.stop(
        len(_meta["subjects"]) == 0,
        mo.callout(mo.md("Could not load electrode metadata for this source."), kind="warn"),
    )
    _lags = _meta["lags"]
    _n_lags = len(_lags)

    # Build tidy DataFrame: one row per (electrode, lag)
    _rows = []
    for _sub in _meta["subjects"]:
        _start = _sub["y_col_start"]
        _end = _sub["y_col_end"]
        _n_elecs = len(_sub["elec_names"])
        if _end - _start != _n_elecs * _n_lags:
            continue
        _b_sub = _b_col[_start:_end].reshape(_n_elecs, _n_lags)
        for _ei, _ename in enumerate(_sub["elec_names"]):
            for _li in range(_n_lags):
                _rows.append({
                    "electrode": _ename,
                    "subject": f"sub_{_sub['sub_id']}",
                    "lag_sec": float(_lags[_li]),
                    "weight": float(_b_sub[_ei, _li]),
                })

    s3_b_df = pd.DataFrame(_rows)
    s3_meta = _meta
    return s3_b_df, s3_meta


@app.cell
def _(mo, s3_b_df, s3_meta):
    _subs = sorted(s3_b_df["subject"].unique().tolist()) if len(s3_b_df) > 0 else []
    _elecs = sorted(s3_b_df["electrode"].unique().tolist()) if len(s3_b_df) > 0 else []
    s3_subject_filter = mo.ui.multiselect(
        options=_subs,
        value=_subs,
        label="Subjects",
    )
    s3_electrode_filter = mo.ui.multiselect(
        options=_elecs,
        value=_elecs[:20] if len(_elecs) > 20 else _elecs,
        label="Electrodes (select subset for clarity)",
    )
    mo.hstack([s3_subject_filter, s3_electrode_filter])
    return s3_electrode_filter, s3_subject_filter


@app.cell
def _(alt, mo, np, pd, s3_b_df, s3_dir_idx, s3_electrode_filter, s3_subject_filter):
    mo.stop(len(s3_b_df) == 0, mo.md("*No B-matrix data available.*"))

    _filtered = s3_b_df[
        s3_b_df["subject"].isin(s3_subject_filter.value) &
        s3_b_df["electrode"].isin(s3_electrode_filter.value)
    ]

    # Mean +/- std across filtered electrodes
    _agg = _filtered.groupby("lag_sec")["weight"].agg(["mean", "std"]).reset_index()
    _agg["upper"] = _agg["mean"] + _agg["std"]
    _agg["lower"] = _agg["mean"] - _agg["std"]

    _band = (
        alt.Chart(_agg)
        .mark_area(opacity=0.2, color="#3b82f6")
        .encode(
            x=alt.X("lag_sec:Q", title="Lag (seconds)"),
            y=alt.Y("lower:Q", title="B weight"),
            y2="upper:Q",
        )
    )
    _mean_line = (
        alt.Chart(_agg)
        .mark_line(color="#3b82f6", strokeWidth=2)
        .encode(
            x="lag_sec:Q",
            y=alt.Y("mean:Q", title="B weight"),
            tooltip=[
                alt.Tooltip("lag_sec:Q", title="Lag (s)", format=".3f"),
                alt.Tooltip("mean:Q", title="Mean weight", format=".4f"),
                alt.Tooltip("std:Q", title="Std", format=".4f"),
            ],
        )
    )

    # Individual electrode lines (thin, transparent)
    _n_elecs_shown = _filtered["electrode"].nunique()
    _indiv = (
        alt.Chart(_filtered)
        .mark_line(opacity=0.15, strokeWidth=0.8)
        .encode(
            x="lag_sec:Q",
            y="weight:Q",
            color=alt.Color("electrode:N", legend=None),
        )
    ) if _n_elecs_shown <= 50 else alt.Chart().mark_point()

    _chart = (
        alt.layer(_indiv, _band, _mean_line)
        .properties(
            width="container", height=300,
            title=f"B weights over time — direction {s3_dir_idx.value} ({_n_elecs_shown} electrodes)",
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(alt, mo, s3_b_df, s3_dir_idx, s3_electrode_filter, s3_subject_filter):
    # Heatmap: electrodes x lags
    mo.stop(len(s3_b_df) == 0, mo.md("*No B-matrix data available.*"))

    _filtered = s3_b_df[
        s3_b_df["subject"].isin(s3_subject_filter.value) &
        s3_b_df["electrode"].isin(s3_electrode_filter.value)
    ]

    _hm = (
        alt.Chart(_filtered)
        .mark_rect()
        .encode(
            x=alt.X("lag_sec:Q", title="Lag (seconds)", bin=alt.Bin(maxbins=64)),
            y=alt.Y("electrode:N", title="Electrode", sort=alt.SortField("subject")),
            color=alt.Color(
                "weight:Q", title="B weight",
                scale=alt.Scale(scheme="redblue", domainMid=0),
            ),
            tooltip=[
                alt.Tooltip("electrode:N"),
                alt.Tooltip("subject:N"),
                alt.Tooltip("lag_sec:Q", title="Lag (s)", format=".3f"),
                alt.Tooltip("weight:Q", format=".4f"),
            ],
        )
        .properties(
            width="container", height=max(200, len(_filtered["electrode"].unique()) * 12),
            title=f"B weight heatmap — direction {s3_dir_idx.value}",
        )
        .configure_view(strokeWidth=0)
    )
    mo.ui.altair_chart(_hm)
    return


@app.cell
def _(mo, np, pd, s3_b_df, s3_electrode_filter, s3_subject_filter):
    # Summary table: per-electrode peak lag, peak weight, mean |weight|
    mo.stop(len(s3_b_df) == 0, mo.md("*No B-matrix data available.*"))

    _filtered = s3_b_df[
        s3_b_df["subject"].isin(s3_subject_filter.value) &
        s3_b_df["electrode"].isin(s3_electrode_filter.value)
    ]

    _summary_rows = []
    for (_elec, _sub), _grp in _filtered.groupby(["electrode", "subject"]):
        _abs_max_idx = _grp["weight"].abs().idxmax()
        _peak_row = _grp.loc[_abs_max_idx]
        _summary_rows.append({
            "electrode": _elec,
            "subject": _sub,
            "peak_lag_sec": round(float(_peak_row["lag_sec"]), 3),
            "peak_weight": round(float(_peak_row["weight"]), 5),
            "mean_abs_weight": round(float(_grp["weight"].abs().mean()), 5),
        })

    _summary_df = pd.DataFrame(_summary_rows).sort_values("mean_abs_weight", ascending=False)
    mo.vstack([
        mo.md("### Electrode Summary"),
        mo.ui.table(_summary_df),
    ])
    return


# ---------------------------------------------------------------------------
# Section 4: B Matrix — Brain Map
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md("---\n## Section 4: B Matrix — Brain Map")
    return


@app.cell
def _(mo, rrr_options):
    mo.stop(
        len(rrr_options) == 0,
        mo.callout(mo.md("No RRR results found."), kind="warn"),
    )
    s4_source = mo.ui.dropdown(
        options=rrr_options,
        value=rrr_options[0] if rrr_options else None,
        label="Source",
    )
    s4_fold = mo.ui.dropdown(
        options=["fold_0", "fold_1"],
        value="fold_0",
        label="Fold",
    )
    s4_k = mo.ui.number(value=10, start=1, stop=50, label="Rank k")
    s4_dir_idx = mo.ui.number(value=0, start=0, stop=49, label="Direction index")
    s4_time_mode = mo.ui.dropdown(
        options=["single timepoint", "time window average"],
        value="single timepoint",
        label="Time mode",
    )
    s4_timepoint = mo.ui.slider(
        start=-2.0, stop=2.0, value=0.0, step=0.05,
        label="Timepoint (seconds)",
    )
    s4_time_start = mo.ui.number(value=-0.5, start=-2.0, stop=2.0, step=0.05, label="Window start (s)")
    s4_time_end = mo.ui.number(value=0.5, start=-2.0, stop=2.0, step=0.05, label="Window end (s)")
    mo.vstack([
        mo.hstack([s4_source, s4_fold, s4_k, s4_dir_idx]),
        mo.hstack([s4_time_mode, s4_timepoint, s4_time_start, s4_time_end]),
    ])
    return (
        s4_dir_idx, s4_fold, s4_k, s4_source,
        s4_time_end, s4_time_mode, s4_time_start, s4_timepoint,
    )


@app.cell
def _(
    get_rrr_dir, load_electrode_metadata, load_rrr_AB,
    mo, np, pd, plt,
    s4_dir_idx, s4_fold, s4_k, s4_source,
    s4_time_end, s4_time_mode, s4_time_start, s4_timepoint,
):
    _unit_dir = get_rrr_dir(s4_source.value)
    _fold_dir = _unit_dir / s4_fold.value
    mo.stop(
        not (_fold_dir / "B.npy").exists(),
        mo.callout(mo.md(f"B.npy not found at `{_fold_dir}`."), kind="warn"),
    )
    _A, _B = load_rrr_AB(_unit_dir, s4_fold.value, int(s4_k.value))
    _dir_idx = min(int(s4_dir_idx.value), _B.shape[1] - 1)
    _b_col = _B[:, _dir_idx]

    _meta = load_electrode_metadata(s4_source.value)
    mo.stop(
        len(_meta["subjects"]) == 0 or len(_meta["lags"]) == 0,
        mo.callout(mo.md("Could not load electrode metadata."), kind="warn"),
    )
    _lags = _meta["lags"]
    _n_lags = len(_lags)

    # Compute one scalar per electrode
    if s4_time_mode.value == "single timepoint":
        _t = float(s4_timepoint.value)
        _lag_idx = int(np.argmin(np.abs(_lags - _t)))
        _time_label = f"t = {_lags[_lag_idx]:.3f}s"
    else:
        _t_start = float(s4_time_start.value)
        _t_end = float(s4_time_end.value)
        _lag_mask = (_lags >= _t_start) & (_lags <= _t_end)
        mo.stop(
            not _lag_mask.any(),
            mo.callout(mo.md("No lags in the selected time window."), kind="warn"),
        )
        _time_label = f"t = [{_t_start:.2f}, {_t_end:.2f}]s"

    _values = []
    _coords = []
    _elec_names = []
    _sub_labels = []

    for _sub in _meta["subjects"]:
        _start = _sub["y_col_start"]
        _end = _sub["y_col_end"]
        _n_elecs = len(_sub["elec_names"])
        if _end - _start != _n_elecs * _n_lags:
            continue
        _b_sub = _b_col[_start:_end].reshape(_n_elecs, _n_lags)
        if s4_time_mode.value == "single timepoint":
            _v = _b_sub[:, _lag_idx]
        else:
            _v = _b_sub[:, _lag_mask].mean(axis=1)
        _values.append(_v)
        _coords.append(_sub["coords"][:_n_elecs])
        _elec_names.extend(_sub["elec_names"])
        _sub_labels.extend([f"sub_{_sub['sub_id']}"] * _n_elecs)

    mo.stop(
        len(_values) == 0,
        mo.callout(mo.md("No electrode data available for brain map."), kind="warn"),
    )

    _values = np.concatenate(_values)
    _coords = np.vstack(_coords)

    # Check coordinates are valid (non-zero)
    _valid = np.any(_coords != 0, axis=1)
    mo.stop(
        not _valid.any(),
        mo.callout(mo.md("No valid MNI coordinates found for electrodes."), kind="warn"),
    )

    _values_valid = _values[_valid]
    _coords_valid = _coords[_valid]
    _names_valid = [n for n, v in zip(_elec_names, _valid) if v]
    _subs_valid = [s for s, v in zip(_sub_labels, _valid) if v]

    # Sort by absolute value so largest are plotted on top
    _order = np.argsort(np.abs(_values_valid))

    try:
        from nilearn.plotting import plot_markers
        _fig = plt.figure(figsize=(14, 5))
        _vmax = max(abs(_values_valid.min()), abs(_values_valid.max()))
        plot_markers(
            _values_valid[_order],
            _coords_valid[_order],
            node_size=40,
            display_mode="lzr",
            node_cmap="RdBu_r",
            node_vmin=-_vmax,
            node_vmax=_vmax,
            colorbar=True,
        )
        plt.suptitle(f"RRR direction {_dir_idx}, {_time_label}", fontsize=12, y=1.02)
        mo.mpl(_fig)
    except ImportError:
        mo.callout(mo.md("nilearn not installed — cannot render brain map."), kind="warn")

    # Electrode table
    _tbl_df = pd.DataFrame({
        "electrode": _elec_names,
        "subject": _sub_labels,
        "weight": np.round(_values, 5),
        "mni_x": np.round(_coords[:, 0], 1),
        "mni_y": np.round(_coords[:, 1], 1),
        "mni_z": np.round(_coords[:, 2], 1),
    }).sort_values("weight", key=abs, ascending=False)

    mo.vstack([
        mo.md(f"### Electrode Weights — direction {_dir_idx}, {_time_label}"),
        mo.ui.table(_tbl_df),
    ])
    return


# ---------------------------------------------------------------------------
# Section 5: Cross-Fold Consistency
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md("---\n## Section 5: Cross-Fold Consistency")
    return


@app.cell
def _(mo, rrr_options):
    mo.stop(
        len(rrr_options) == 0,
        mo.callout(mo.md("No RRR results found."), kind="warn"),
    )
    s5_source = mo.ui.dropdown(
        options=rrr_options,
        value=rrr_options[0] if rrr_options else None,
        label="Source",
    )
    s5_k = mo.ui.number(value=10, start=1, stop=50, label="Rank k")
    s5_ndirs = mo.ui.slider(start=2, stop=50, value=10, step=1, label="Directions to compare")
    mo.hstack([s5_source, s5_k, s5_ndirs])
    return s5_k, s5_ndirs, s5_source


@app.cell
def _(alt, get_rrr_dir, mo, np, pd, s5_k, s5_ndirs, s5_source):
    _unit_dir = get_rrr_dir(s5_source.value)
    _f0 = _unit_dir / "fold_0" / "A.npy"
    _f1 = _unit_dir / "fold_1" / "A.npy"
    mo.stop(
        not _f0.exists() or not _f1.exists(),
        mo.callout(mo.md("Need both fold_0 and fold_1 A.npy for consistency analysis."), kind="warn"),
    )

    _k = int(s5_k.value)
    _A0 = np.load(_f0)[:, :_k]
    _A1 = np.load(_f1)[:, :_k]
    _n = min(int(s5_ndirs.value), _A0.shape[1], _A1.shape[1])

    # Normalize columns
    _A0_norm = _A0[:, :_n] / np.linalg.norm(_A0[:, :_n], axis=0, keepdims=True)
    _A1_norm = _A1[:, :_n] / np.linalg.norm(_A1[:, :_n], axis=0, keepdims=True)
    _cos_sim = _A0_norm.T @ _A1_norm  # (n, n)
    _diag_cos = np.abs(np.diag(_cos_sim))

    # --- Tab A: Cosine heatmap ---
    _hm_rows = []
    for _i in range(_n):
        for _j in range(_n):
            _hm_rows.append({"fold_0_dir": _i, "fold_1_dir": _j, "cos_sim": float(_cos_sim[_i, _j])})
    _hm_df = pd.DataFrame(_hm_rows)
    _hm_chart = (
        alt.Chart(_hm_df)
        .mark_rect()
        .encode(
            x=alt.X("fold_0_dir:O", title="Fold 0 direction"),
            y=alt.Y("fold_1_dir:O", title="Fold 1 direction"),
            color=alt.Color(
                "cos_sim:Q", title="Cosine sim",
                scale=alt.Scale(scheme="rdbu", domain=[-1, 1]),
            ),
            tooltip=[
                alt.Tooltip("fold_0_dir:O", title="Fold 0"),
                alt.Tooltip("fold_1_dir:O", title="Fold 1"),
                alt.Tooltip("cos_sim:Q", format=".3f"),
            ],
        )
        .properties(width="container", height=400, title="Cosine similarity: fold 0 vs fold 1")
        .configure_view(strokeWidth=0)
    )

    # --- Tab B: Diagonal consistency bar chart ---
    _diag_df = pd.DataFrame({"direction": np.arange(_n), "abs_cos_sim": _diag_cos})
    _diag_chart = (
        alt.Chart(_diag_df)
        .mark_bar(color="#3b82f6")
        .encode(
            x=alt.X("direction:O", title="Direction"),
            y=alt.Y("abs_cos_sim:Q", title="|Cosine similarity|", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("direction:O"),
                alt.Tooltip("abs_cos_sim:Q", format=".4f"),
            ],
        )
        .properties(width="container", height=250, title="Per-direction fold consistency")
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    # --- Tab C: Consistency table ---
    _tbl_df = pd.DataFrame({
        "direction": np.arange(_n),
        "abs_cos_sim": np.round(_diag_cos, 4),
    })

    mo.tabs({
        "Cosine heatmap": mo.ui.altair_chart(_hm_chart),
        "Diagonal consistency": mo.ui.altair_chart(_diag_chart),
        "Table": mo.ui.table(_tbl_df),
    })
    return


# ---------------------------------------------------------------------------
# Section 6: Cross-Source Comparison
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md("---\n## Section 6: Cross-Source Comparison")
    return


@app.cell
def _(mo, rrr_options):
    mo.stop(
        len(rrr_options) < 2,
        mo.callout(mo.md("Need ≥ 2 RRR sources for cross-source comparison."), kind="warn"),
    )
    s6_source_a = mo.ui.dropdown(options=rrr_options, value=rrr_options[0], label="Source A")
    s6_fold_a = mo.ui.dropdown(options=["fold_0", "fold_1"], value="fold_0", label="Fold A")
    s6_source_b = mo.ui.dropdown(
        options=rrr_options,
        value=rrr_options[1] if len(rrr_options) > 1 else rrr_options[0],
        label="Source B",
    )
    s6_fold_b = mo.ui.dropdown(options=["fold_0", "fold_1"], value="fold_0", label="Fold B")
    s6_k = mo.ui.number(value=10, start=1, stop=50, label="Rank k")
    s6_topk = mo.ui.slider(start=2, stop=50, value=10, step=1, label="Top-k directions")
    mo.vstack([
        mo.hstack([s6_source_a, s6_fold_a, s6_source_b, s6_fold_b]),
        mo.hstack([s6_k, s6_topk]),
    ])
    return s6_fold_a, s6_fold_b, s6_k, s6_source_a, s6_source_b, s6_topk


@app.cell
def _(
    aligned_embeddings, alt, get_rrr_dir, load_rrr_AB,
    mo, np, pd,
    s6_fold_a, s6_fold_b, s6_k, s6_source_a, s6_source_b, s6_topk,
):
    _dir_a = get_rrr_dir(s6_source_a.value)
    _dir_b = get_rrr_dir(s6_source_b.value)
    mo.stop(
        not (_dir_a / s6_fold_a.value / "A.npy").exists() or
        not (_dir_b / s6_fold_b.value / "A.npy").exists(),
        mo.callout(mo.md("A.npy not found for one or both sources."), kind="warn"),
    )

    _k = int(s6_k.value)
    _A_a, _ = load_rrr_AB(_dir_a, s6_fold_a.value, _k)
    _A_b, _ = load_rrr_AB(_dir_b, s6_fold_b.value, _k)
    _n = min(int(s6_topk.value), _A_a.shape[1], _A_b.shape[1])

    _A_a_k = _A_a[:, :_n]
    _A_b_k = _A_b[:, :_n]

    # Normalize columns
    _A_a_norm = _A_a_k / np.linalg.norm(_A_a_k, axis=0, keepdims=True)
    _A_b_norm = _A_b_k / np.linalg.norm(_A_b_k, axis=0, keepdims=True)
    _cos_sim = _A_a_norm.T @ _A_b_norm  # (n, n)

    # Best matching pair
    _abs_cs = np.abs(_cos_sim)
    _best_flat = int(np.argmax(_abs_cs))
    _best_i = _best_flat // _n
    _best_j = _best_flat % _n
    _best_cos = float(_cos_sim[_best_i, _best_j])

    # Activation correlation
    _acts_a = (aligned_embeddings @ _A_a_k).astype(np.float64)
    _acts_b = (aligned_embeddings @ _A_b_k).astype(np.float64)
    _a_z = (_acts_a - _acts_a.mean(0)) / np.where(_acts_a.std(0) > 0, _acts_a.std(0), 1.0)
    _b_z = (_acts_b - _acts_b.mean(0)) / np.where(_acts_b.std(0) > 0, _acts_b.std(0), 1.0)
    _act_corr = (_a_z.T @ _b_z) / len(_a_z)

    # --- Tab A: Cosine heatmap ---
    _cs_rows = []
    for _i in range(_n):
        for _j in range(_n):
            _cs_rows.append({"dir_A": _i, "dir_B": _j, "cos_sim": float(_cos_sim[_i, _j])})
    _cs_df = pd.DataFrame(_cs_rows)
    _cs_chart = (
        alt.Chart(_cs_df)
        .mark_rect()
        .encode(
            x=alt.X("dir_A:O", title="Source A direction"),
            y=alt.Y("dir_B:O", title="Source B direction"),
            color=alt.Color("cos_sim:Q", title="Cosine sim",
                           scale=alt.Scale(scheme="rdbu", domain=[-1, 1])),
            tooltip=[
                alt.Tooltip("dir_A:O"), alt.Tooltip("dir_B:O"),
                alt.Tooltip("cos_sim:Q", format=".3f"),
            ],
        )
        .properties(
            width="container", height=400,
            title=f"Cosine similarity: {s6_source_a.value} vs {s6_source_b.value}",
        )
        .configure_view(strokeWidth=0)
    )

    # --- Tab B: Activation scatter (best pair) ---
    _scatter_df = pd.DataFrame({
        "activation_A": _acts_a[:, _best_i],
        "activation_B": _acts_b[:, _best_j],
    })
    _scatter_base = alt.Chart(_scatter_df).encode(
        x=alt.X("activation_A:Q", title=f"Source A dir {_best_i}"),
        y=alt.Y("activation_B:Q", title=f"Source B dir {_best_j}"),
    )
    _scatter_pts = _scatter_base.mark_point(opacity=0.3, size=20, color="#3b82f6")
    _scatter_reg = _scatter_base.transform_regression(
        "activation_A", "activation_B"
    ).mark_line(color="#ef4444", strokeWidth=2)
    _scatter_chart = (
        (_scatter_pts + _scatter_reg)
        .properties(
            width="container", height=350,
            title=f"Best pair: A[{_best_i}] vs B[{_best_j}] (cos={_best_cos:.3f})",
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    # --- Tab C: Top pairs table ---
    _tp_rows = []
    for _i in range(_n):
        _bj = int(np.argmax(np.abs(_cos_sim[_i])))
        _tp_rows.append({
            "dir_A": _i, "best_dir_B": _bj,
            "cos_sim": round(float(_cos_sim[_i, _bj]), 4),
            "abs_cos_sim": round(float(np.abs(_cos_sim[_i, _bj])), 4),
        })
    _tp_df = pd.DataFrame(_tp_rows).sort_values("abs_cos_sim", ascending=False)

    # --- Tab D: Activation correlation heatmap ---
    _ac_rows = []
    for _i in range(_n):
        for _j in range(_n):
            _ac_rows.append({"dir_A": _i, "dir_B": _j, "pearson_r": float(_act_corr[_i, _j])})
    _ac_df = pd.DataFrame(_ac_rows)
    _ac_chart = (
        alt.Chart(_ac_df)
        .mark_rect()
        .encode(
            x=alt.X("dir_A:O", title="Source A direction"),
            y=alt.Y("dir_B:O", title="Source B direction"),
            color=alt.Color("pearson_r:Q", title="Pearson r",
                           scale=alt.Scale(scheme="rdbu", domain=[-1, 1])),
            tooltip=[
                alt.Tooltip("dir_A:O"), alt.Tooltip("dir_B:O"),
                alt.Tooltip("pearson_r:Q", format=".3f"),
            ],
        )
        .properties(width="container", height=400, title="Activation Pearson correlations")
        .configure_view(strokeWidth=0)
    )

    mo.tabs({
        "A. Cosine heatmap": mo.ui.altair_chart(_cs_chart),
        "B. Activation scatter": mo.ui.altair_chart(_scatter_chart),
        "C. Top pairs": mo.ui.table(_tp_df),
        "D. Activation correlations": mo.ui.altair_chart(_ac_chart),
    })
    return


# ---------------------------------------------------------------------------
# Section 7: Cross-Region Electrode Similarity from B
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md("---\n## Section 7: Cross-Region Electrode Similarity from B")
    return


@app.cell
def _(mo, rrr_options):
    mo.stop(
        len(rrr_options) == 0,
        mo.callout(mo.md("No RRR results found."), kind="warn"),
    )
    s7_source = mo.ui.dropdown(
        options=rrr_options,
        value=rrr_options[0] if rrr_options else None,
        label="Source",
    )
    s7_fold = mo.ui.dropdown(
        options=["fold_0", "fold_1"],
        value="fold_0",
        label="Fold",
    )
    s7_k = mo.ui.number(value=10, start=1, stop=50, label="Rank k")
    s7_profile_mode = mo.ui.dropdown(
        options=["all lags (full profile)", "lag-averaged"],
        value="lag-averaged",
        label="Electrode profile",
    )
    mo.hstack([s7_source, s7_fold, s7_k, s7_profile_mode])
    return s7_fold, s7_k, s7_profile_mode, s7_source


@app.cell
def _(
    alt, get_coord_atlas_labels, get_rrr_dir, load_electrode_metadata, load_rrr_AB,
    mo, np, pd, s7_fold, s7_k, s7_profile_mode, s7_source,
):
    _unit_dir = get_rrr_dir(s7_source.value)
    _fold_dir = _unit_dir / s7_fold.value
    mo.stop(
        not (_fold_dir / "B.npy").exists(),
        mo.callout(mo.md(f"B.npy not found at `{_fold_dir}`."), kind="warn"),
    )
    _k = int(s7_k.value)
    _A, _B = load_rrr_AB(_unit_dir, s7_fold.value, _k)

    _meta = load_electrode_metadata(s7_source.value)
    mo.stop(
        len(_meta["subjects"]) == 0 or len(_meta["lags"]) == 0,
        mo.callout(mo.md("Could not load electrode metadata."), kind="warn"),
    )
    _lags = _meta["lags"]
    _n_lags = len(_lags)
    _n_dirs = _B.shape[1]

    # Build electrode profile matrix
    # For each electrode, extract its B weights across all directions
    _profiles = []  # list of (n_feature,) per electrode
    _elec_names = []
    _sub_labels = []
    _coords_list = []

    for _sub in _meta["subjects"]:
        _start = _sub["y_col_start"]
        _end = _sub["y_col_end"]
        _n_elecs = len(_sub["elec_names"])
        if _end - _start != _n_elecs * _n_lags:
            continue
        # B_sub: (n_elecs, n_lags, n_dirs) from B: (n_targets, n_dirs)
        _B_sub = _B[_start:_end, :].reshape(_n_elecs, _n_lags, _n_dirs)
        for _ei, _ename in enumerate(_sub["elec_names"]):
            if s7_profile_mode.value == "lag-averaged":
                # Average over lags: (n_dirs,)
                _prof = _B_sub[_ei, :, :].mean(axis=0)
            else:
                # Full profile: (n_lags * n_dirs,)
                _prof = _B_sub[_ei, :, :].ravel()
            _profiles.append(_prof)
            _elec_names.append(_ename)
            _sub_labels.append(f"sub_{_sub['sub_id']}")
            if _ei < len(_sub["coords"]):
                _coords_list.append(_sub["coords"][_ei])
            else:
                _coords_list.append(np.zeros(3))

    mo.stop(
        len(_profiles) < 2,
        mo.callout(mo.md("Need ≥ 2 electrodes for similarity analysis."), kind="warn"),
    )

    _profile_mat = np.array(_profiles)  # (n_electrodes, n_features)
    _coords_arr = np.array(_coords_list)  # (n_electrodes, 3)

    # Assign regions via atlas
    _valid_coords = np.any(_coords_arr != 0, axis=1)
    _regions = ["unknown"] * len(_elec_names)
    if _valid_coords.any():
        try:
            _atlas_labels = get_coord_atlas_labels(_coords_arr[_valid_coords])
            _vi = 0
            for _idx in range(len(_elec_names)):
                if _valid_coords[_idx]:
                    _regions[_idx] = _atlas_labels[_vi]
                    _vi += 1
        except Exception:
            pass

    # Cosine similarity matrix between all electrodes
    _norms = np.linalg.norm(_profile_mat, axis=1, keepdims=True)
    _norms = np.where(_norms > 0, _norms, 1.0)
    _normed = _profile_mat / _norms
    _sim_mat = _normed @ _normed.T  # (n_elecs, n_elecs)

    # Group by region
    _unique_regions = sorted(set(_regions))
    _region_indices = {r: [i for i, reg in enumerate(_regions) if reg == r] for r in _unique_regions}

    # Region-by-region mean similarity
    _rr_rows = []
    for _ra in _unique_regions:
        for _rb in _unique_regions:
            _idxs_a = _region_indices[_ra]
            _idxs_b = _region_indices[_rb]
            if len(_idxs_a) == 0 or len(_idxs_b) == 0:
                continue
            _block = _sim_mat[np.ix_(_idxs_a, _idxs_b)]
            if _ra == _rb:
                # Exclude diagonal (self-similarity)
                _mask = ~np.eye(len(_idxs_a), len(_idxs_b), dtype=bool)
                _vals = _block[_mask] if _mask.any() else _block.ravel()
            else:
                _vals = _block.ravel()
            _rr_rows.append({
                "region_A": _ra, "region_B": _rb,
                "mean_sim": round(float(_vals.mean()), 4) if len(_vals) > 0 else 0.0,
                "n_elecs_A": len(_idxs_a), "n_elecs_B": len(_idxs_b),
            })

    _rr_df = pd.DataFrame(_rr_rows)

    # Within vs across region summary
    _within = []
    _across = []
    for _row in _rr_rows:
        if _row["region_A"] == _row["region_B"]:
            _within.append(_row["mean_sim"])
        else:
            _across.append(_row["mean_sim"])

    _within_mean = float(np.mean(_within)) if _within else float("nan")
    _across_mean = float(np.mean(_across)) if _across else float("nan")

    # Heatmap
    _hm = (
        alt.Chart(_rr_df)
        .mark_rect()
        .encode(
            x=alt.X("region_A:N", title="Region A"),
            y=alt.Y("region_B:N", title="Region B"),
            color=alt.Color(
                "mean_sim:Q", title="Mean cosine sim",
                scale=alt.Scale(scheme="viridis"),
            ),
            tooltip=[
                alt.Tooltip("region_A:N"), alt.Tooltip("region_B:N"),
                alt.Tooltip("mean_sim:Q", format=".4f"),
                alt.Tooltip("n_elecs_A:Q"), alt.Tooltip("n_elecs_B:Q"),
            ],
        )
        .properties(width="container", height=400, title="Region-by-region electrode similarity (B profiles)")
        .configure_view(strokeWidth=0)
    )

    # Within vs across bar
    _wa_df = pd.DataFrame([
        {"category": "within-region", "mean_sim": _within_mean},
        {"category": "across-region", "mean_sim": _across_mean},
    ])
    _wa_chart = (
        alt.Chart(_wa_df)
        .mark_bar()
        .encode(
            x=alt.X("category:N", title=""),
            y=alt.Y("mean_sim:Q", title="Mean cosine similarity"),
            color=alt.Color("category:N", scale=alt.Scale(
                domain=["within-region", "across-region"],
                range=["#3b82f6", "#f59e0b"],
            ), legend=None),
            tooltip=[alt.Tooltip("mean_sim:Q", format=".4f")],
        )
        .properties(width=200, height=250, title="Within vs across region similarity")
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    mo.tabs({
        "Region similarity heatmap": mo.ui.altair_chart(_hm),
        "Within vs across": mo.vstack([
            mo.ui.altair_chart(_wa_chart),
            mo.callout(
                mo.md(f"**Within-region mean:** {_within_mean:.4f}  \n"
                      f"**Across-region mean:** {_across_mean:.4f}"),
                kind="info",
            ),
        ]),
        "Region table": mo.ui.table(_rr_df),
    })
    return


# ---------------------------------------------------------------------------
# Section 8: Activation Distribution Statistics
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md("---\n## Section 8: Activation Distribution Statistics")
    return


@app.cell
def _(mo, rrr_options):
    mo.stop(
        len(rrr_options) == 0,
        mo.callout(mo.md("No RRR results found."), kind="warn"),
    )
    s8_source = mo.ui.dropdown(
        options=rrr_options,
        value=rrr_options[0] if rrr_options else None,
        label="Source",
    )
    s8_fold = mo.ui.dropdown(
        options=["fold_0", "fold_1"],
        value="fold_0",
        label="Fold",
    )
    s8_k = mo.ui.number(value=10, start=1, stop=50, label="Rank k")
    mo.hstack([s8_source, s8_fold, s8_k])
    return s8_fold, s8_k, s8_source


@app.cell
def _(
    aligned_embeddings, alt, get_rrr_dir, load_rrr_AB,
    mo, np, pd, s8_fold, s8_k, s8_source,
    sp_entropy, sp_kurtosis, sp_skew,
):
    _unit_dir = get_rrr_dir(s8_source.value)
    _fold_dir = _unit_dir / s8_fold.value
    mo.stop(
        not (_fold_dir / "A.npy").exists(),
        mo.callout(mo.md(f"A.npy not found at `{_fold_dir}`."), kind="warn"),
    )
    _k = int(s8_k.value)
    _A, _ = load_rrr_AB(_unit_dir, s8_fold.value, _k)
    _n = _A.shape[1]

    _acts = (aligned_embeddings @ _A).astype(np.float64)  # (n_words, k)

    _stats_rows = []
    for _i in range(_n):
        _a = _acts[:, _i]
        _counts, _ = np.histogram(_a, bins=50)
        _probs = _counts / _counts.sum()
        _ent = float(sp_entropy(_probs + 1e-12))
        _stats_rows.append({
            "direction": _i,
            "mean": round(float(_a.mean()), 4),
            "std": round(float(_a.std()), 4),
            "kurtosis": round(float(sp_kurtosis(_a)), 4),
            "skewness": round(float(sp_skew(_a)), 4),
            "entropy": round(_ent, 4),
        })

    _stats_df = pd.DataFrame(_stats_rows)

    _kurt_chart = (
        alt.Chart(_stats_df)
        .mark_bar(color="#a78bfa")
        .encode(
            x=alt.X("direction:O", title="Direction"),
            y=alt.Y("kurtosis:Q", title="Kurtosis (excess)"),
            tooltip=[
                alt.Tooltip("direction:O"),
                alt.Tooltip("kurtosis:Q", format=".3f"),
                alt.Tooltip("skewness:Q", format=".3f"),
                alt.Tooltip("entropy:Q", format=".3f"),
            ],
        )
        .properties(
            width="container", height=250,
            title="Kurtosis per direction (higher = more semantically selective)",
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    mo.vstack([
        mo.ui.altair_chart(_kurt_chart),
        mo.ui.table(_stats_df),
    ])
    return


if __name__ == "__main__":
    app.run()
