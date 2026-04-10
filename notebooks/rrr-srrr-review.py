import marimo

__generated_with = "0.20.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import (
        kurtosis as sp_kurtosis,
        entropy as sp_entropy,
        skew as sp_skew,
    )
    from pathlib import Path
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, inconsistent

    return (
        Path,
        alt,
        dendrogram,
        fcluster,
        inconsistent,
        linkage,
        mo,
        np,
        pd,
        plt,
        sp_entropy,
        sp_kurtosis,
        sp_skew,
    )


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
    from src.utils.embeddings import load_embeddings
    from src.utils.examples import (
        get_mo_display_activations,
        get_example_display_controls,
    )
    from src.utils.encoding import load_encoding_results
    from src.utils.data_loaders import load_sig_electrodes, load_raws
    from src.utils.atlas import (
        get_coord_atlas_labels,
        build_electrode_region_map,
        _lookup_atlas_labels,
    )

    return (
        build_electrode_region_map,
        get_coord_atlas_labels,
        get_example_display_controls,
        get_mo_display_activations,
        load_embeddings,
        load_encoding_results,
        load_raws,
        load_sig_electrodes,
        os,
    )


@app.cell
def _(mo):
    mo.md("""
    # RRR Interpretability Explorer
    """)
    return


@app.cell(hide_code=True)
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
    sig_dir_input = mo.ui.text(
        value="",
        label="Significance dir (blank = results root)",
        full_width=True,
    )
    model_type_input = mo.ui.text(
        value="gpt2xl",
        label="Model"
    )
    model_configs_input = mo.ui.text(
        value="layer:24",
        label="Model Config"
    )
    layer_input = mo.ui.number(value=24, label="Layer")
    mo.hstack([results_root_input, data_root_input, model_type_input, model_configs_input, sig_dir_input, layer_input])
    return (
        data_root_input,
        model_configs_input,
        model_type_input,
        results_root_input,
        sig_dir_input,
    )


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(
    data_root_input,
    load_embeddings,
    model_configs_input,
    model_type_input,
    np,
):
    model_config = {}
    for _config in model_configs_input.value.split(","):
        _k, _v = _config.split(":")
        model_config[_k] = _v

    aligned_embeddings, df_word = load_embeddings(
        data_root_input.value, model_type_input.value, **model_config
    )

    class _AllWords:
        selection = np.arange(len(df_word))

    all_words_epochs = _AllWords()
    return aligned_embeddings, all_words_epochs, df_word


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(np):
    def load_rrr_AB(rrr_dir, fold_str, k):
        """Load A and B matrices from the correct fold directory, truncated to rank k."""
        fold_dir = rrr_dir / fold_str
        A_full = np.load(fold_dir / "A.npy")  # (n_embed, max_k)
        B_full = np.load(fold_dir / "B.npy")  # (n_targets, max_k)
        k_clamp = min(k, A_full.shape[1], B_full.shape[1])
        return A_full[:, :k_clamp], B_full[:, :k_clamp]

    return (load_rrr_AB,)


@app.cell(hide_code=True)
def _(
    Path,
    build_electrode_region_map,
    data_root_input,
    load_encoding_results,
    load_raws,
    load_sig_electrodes,
    np,
    results_root_input,
    sig_dir_input,
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
        _sig_dir = sig_dir_input.value.strip() or None

        # Determine subject IDs and per-subject electrode lists
        sig_electrodes, _ = load_sig_electrodes(
            str(rr),
            list(range(1, 10)),
            sig_dir=_sig_dir,
        )

        if source == "cross_subject":
            subs_with_sig = [
                s for s in sorted(sig_electrodes.keys()) if sig_electrodes[s]
            ]
            per_sub_elecs = {s: sig_electrodes[s] for s in subs_with_sig}
        elif source.startswith("per_subject/"):
            sub_name = source.split("/", 1)[1]  # e.g. "sub_3"
            sub_id = int(sub_name.replace("sub_", ""))
            per_sub_elecs = {sub_id: sig_electrodes.get(sub_id, [])}
        elif source.startswith("per_region/"):
            region_name = source.split("/", 1)[1]
            # Re-derive which electrodes are in this region
            all_sub_ids = [
                s for s in sorted(sig_electrodes.keys()) if sig_electrodes[s]
            ]
            try:
                elec_region_map = build_electrode_region_map(
                    all_sub_ids,
                    data_root,
                    None,
                    None,
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
            enc_results = load_encoding_results(enc_path, location="cpu")
            sub_lags = enc_results["lags"]
            if lags is None:
                lags = sub_lags
            n_lags = len(sub_lags)
            n_target_cols = len(elec_names) * n_lags

            # Load MNE raw header (no data) for MNI coordinates
            try:
                raws = load_raws(
                    [sub_id],
                    data_root,
                    use_high_gamma=True,
                    per_subject_electrodes={sub_id: elec_names},
                    no_preload=True,
                )
                raw = raws[0]
                coords = np.array(
                    [
                        raw.info["chs"][raw.ch_names.index(ch)]["loc"][:3] * 1000
                        for ch in elec_names
                        if ch in raw.ch_names
                    ]
                )
            except Exception:
                coords = np.zeros((len(elec_names), 3))

            subjects_info.append(
                {
                    "sub_id": sub_id,
                    "elec_names": elec_names,
                    "lags": sub_lags,
                    "y_col_start": y_cursor,
                    "y_col_end": y_cursor + n_target_cols,
                    "coords": coords,
                }
            )
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


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Section 1: RRR — Rank vs Explained Variance
    """)
    return


@app.cell(hide_code=True)
def _(mo, rrr_options):
    mo.stop(
        len(rrr_options) == 0,
        mo.callout(
            mo.md("No RRR results found. Run `scripts/run_rrr.py` first."), kind="warn"
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo, rrr_options):
    s1_source = mo.ui.dropdown(
        options=rrr_options,
        value=rrr_options[0] if rrr_options else None,
        label="RRR source",
    )
    s1_show_folds = mo.ui.checkbox(value=False, label="Show individual folds")
    s1_threshold = mo.ui.slider(
        start=0.0,
        stop=0.10,
        value=0.01,
        step=0.005,
        label="Sufficiency threshold (full_r2 − rrr_r2 ≤)",
    )
    mo.hstack([s1_source, s1_show_folds, s1_threshold])
    return s1_source, s1_threshold


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(s1_grid, s1_source, s1_summary):
    from src.utils.plotting import plot_rrr_rank_results

    plot_rrr_rank_results(
        s1_summary,
        grid_df=s1_grid,
        title=f"RRR: R² vs rank ({s1_source.value})",
    )
    return


@app.cell(hide_code=True)
def _(alt, mo, s1_summary):
    # CCA / Procrustes (optional)
    _opt_cols = [
        c
        for c in ["cca_similarity", "procrustes_similarity"]
        if c in s1_summary.columns
    ]
    mo.stop(
        len(_opt_cols) == 0, mo.md("*No CCA or Procrustes similarity columns found.*")
    )
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
        .properties(
            width="container", height=200, title="CCA / Procrustes similarity vs rank"
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    mo.ui.altair_chart(_opt_chart)
    return


@app.cell(hide_code=True)
def _(mo, s1_summary, s1_threshold):
    # Sufficiency summary: find smallest k where RRR explains nearly as much as full model
    if "full_r2" in s1_summary.columns and "rrr_r2" in s1_summary.columns:
        _gap = s1_summary["full_r2"] - s1_summary["rrr_r2"]
        _ratio = s1_summary["rrr_r2"] / s1_summary["full_r2"].replace(
            0, float("nan")
        )
        _suf = s1_summary[_gap <= s1_threshold.value]
        if len(_suf) > 0:
            _min_k = int(_suf["k"].min())
            _gap_at_k = float(_gap[s1_summary["k"] == _min_k].values[0])
            _ratio_at_k = float(_ratio[s1_summary["k"] == _min_k].values[0])
            mo.callout(
                mo.md(
                    f"**Minimum sufficient rank (RRR):** k = **{_min_k}**  \n"
                    f"First rank where avg. `full_r2 − rrr_r2 ≤ {s1_threshold.value:.3f}` "
                    f"(gap = {_gap_at_k:.4f}, ratio = {_ratio_at_k:.3f})"
                ),
                kind="success",
            )
        else:
            mo.callout(
                mo.md(
                    f"No rank achieves `full_r2 − rrr_r2 ≤ {s1_threshold.value:.3f}`. "
                    "Try a larger threshold or a larger rank grid."
                ),
                kind="warn",
            )
    else:
        mo.callout(
            mo.md("*`full_r2` or `rrr_r2` not found in summary.csv.*"), kind="warn"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Section 2: Direction Interpretability
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(
    aligned_embeddings,
    get_rrr_dir,
    load_rrr_AB,
    mo,
    np,
    s2_dir_idx,
    s2_fold,
    s2_k,
    s2_source,
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


@app.cell(hide_code=True)
def _(alt, mo, np, pd, s2_activations, s2_dir_idx, s2_fold, s2_k):
    _counts, _edges = np.histogram(s2_activations, bins=100)
    _hist_df = pd.DataFrame({"x1": _edges[:-1], "x2": _edges[1:], "count": _counts})
    _bars = (
        alt.Chart(_hist_df)
        .mark_bar(color="#93c5fd", stroke="#bfdbfe", strokeWidth=0.5)
        .encode(
            x=alt.X(
                "x1:Q",
                title="Activation",
                scale=alt.Scale(domain=[float(_edges[0]), float(_edges[-1])]),
            ),
            x2="x2:Q",
            y=alt.Y("count:Q", title="Count"),
            tooltip=[
                alt.Tooltip("x1:Q", title="From", format=".3f"),
                alt.Tooltip("x2:Q", title="To", format=".3f"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
    )
    _chart = (
        _bars.properties(
            width="container",
            height=220,
            title=f"RRR direction {s2_dir_idx.value} — k={s2_k.value}, {s2_fold.value}",
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell(hide_code=True)
def _(get_example_display_controls, mo):
    s2_num_bins, s2_samp_per_bin, s2_ctx, s2_seed = get_example_display_controls()
    mo.hstack([s2_num_bins, s2_samp_per_bin, s2_ctx, s2_seed])
    return s2_ctx, s2_num_bins, s2_samp_per_bin, s2_seed


@app.cell(hide_code=True)
def _(
    all_words_epochs,
    df_word,
    get_mo_display_activations,
    s2_activations,
    s2_ctx,
    s2_num_bins,
    s2_samp_per_bin,
    s2_seed,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Section 3: B Matrix — Temporal Profile
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(
    get_rrr_dir,
    load_electrode_metadata,
    load_rrr_AB,
    mo,
    s3_fold,
    s3_k,
    s3_source,
):
    _unit_dir = get_rrr_dir(s3_source.value)
    _fold_dir = _unit_dir / s3_fold.value
    mo.stop(
        not (_fold_dir / "B.npy").exists(),
        mo.callout(mo.md(f"B.npy not found at `{_fold_dir}`."), kind="warn"),
    )
    _A, B_3 = load_rrr_AB(_unit_dir, s3_fold.value, int(s3_k.value))

    meta_3 = load_electrode_metadata(s3_source.value)
    return B_3, meta_3


@app.cell(hide_code=True)
def _(B_3, meta, meta_3, mo, pd, s3_dir_idx):
    mo.stop(
        len(meta["subjects"]) == 0,
        mo.callout(
            mo.md("Could not load electrode metadata for this source."), kind="warn"
        ),
    )
    _dir_idx = min(int(s3_dir_idx.value), B_3.shape[1] - 1)
    _b_col = B_3[:, _dir_idx]  # (n_targets,)
    _lags = meta_3["lags"]
    _n_lags = len(_lags)

    # Build tidy DataFrame: one row per (electrode, lag)
    _rows = []
    for _sub in meta_3["subjects"]:
        _start = _sub["y_col_start"]
        _end = _sub["y_col_end"]
        _n_elecs = len(_sub["elec_names"])
        if _end - _start != _n_elecs * _n_lags:
            continue
        _b_sub = _b_col[_start:_end].reshape(_n_elecs, _n_lags)
        for _ei, _ename in enumerate(_sub["elec_names"]):
            for _li in range(_n_lags):
                _rows.append(
                    {
                        "electrode": _ename,
                        "subject": f"sub_{_sub['sub_id']}",
                        "lag_sec": float(_lags[_li]),
                        "weight": float(_b_sub[_ei, _li]),
                    }
                )

    def convert_to_electrode_id(subject, electrode):
        return f"{subject}-{electrode}"

    s3_b_df = pd.DataFrame(_rows)
    s3_meta = meta_3

    all_elec_list = []
    for i, row in s3_b_df.iterrows():
        all_elec_list.append(convert_to_electrode_id(row.subject, row.electrode))
    s3_b_df["elec_name"] = all_elec_list
    # Drop duplicates
    elec_list = list(set(all_elec_list))
    return convert_to_electrode_id, elec_list, s3_b_df


@app.cell(hide_code=True)
def _(elec_list, mo):
    _elecs = sorted(elec_list)
    get_s3_selected_elecs, set_s3_selected_elecs = mo.state(
        elec_list[:20] if len(elec_list) > 20 else elec_list
    )
    return get_s3_selected_elecs, set_s3_selected_elecs


@app.cell(hide_code=True)
def _(elec_list, get_s3_selected_elecs, mo, s3_b_df):
    _subs = sorted(s3_b_df["subject"].unique().tolist()) if len(s3_b_df) > 0 else []
    _elecs = sorted(s3_b_df["electrode"].unique().tolist()) if len(s3_b_df) > 0 else []
    s3_subject_filter = mo.ui.multiselect(
        options=_subs,
        value=_subs,
        label="Subjects",
    )

    s3_electrode_filter = mo.ui.multiselect(
        options=elec_list,
        value=get_s3_selected_elecs(),
        label="Electrodes (select subset for clarity)",
    )
    mo.hstack([s3_subject_filter, s3_electrode_filter])
    return (s3_electrode_filter,)


@app.cell(hide_code=True)
def _(alt, mo, s3_b_df, s3_dir_idx, s3_electrode_filter):
    mo.stop(len(s3_b_df) == 0, mo.md("*No B-matrix data available.*"))

    _filtered = s3_b_df[s3_b_df["elec_name"].isin(s3_electrode_filter.value)]

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
        (
            alt.Chart(_filtered)
            .mark_line(opacity=0.15, strokeWidth=0.8)
            .encode(
                x="lag_sec:Q",
                y="weight:Q",
                color=alt.Color("electrode:N", legend=None),
            )
        )
        if _n_elecs_shown <= 50
        else alt.Chart().mark_point()
    )

    _chart = (
        alt.layer(_indiv, _band, _mean_line)
        .properties(
            width="container",
            height=300,
            title=f"B weights over time — direction {s3_dir_idx.value} ({_n_elecs_shown} electrodes)",
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell(hide_code=True)
def _(alt, mo, s3_b_df, s3_dir_idx, s3_electrode_filter):
    # Heatmap: electrodes x lags
    mo.stop(len(s3_b_df) == 0, mo.md("*No B-matrix data available.*"))

    _filtered = s3_b_df[s3_b_df["elec_name"].isin(s3_electrode_filter.value)]

    _hm = (
        alt.Chart(_filtered)
        .mark_rect()
        .encode(
            x=alt.X("lag_sec:Q", title="Lag (seconds)", bin=alt.Bin(maxbins=64)),
            y=alt.Y("electrode:N", title="Electrode", sort=alt.SortField("subject")),
            color=alt.Color(
                "weight:Q",
                title="B weight",
                scale=alt.Scale(scheme="redblue", domainMid=0, reverse=True),
            ),
            tooltip=[
                alt.Tooltip("electrode:N"),
                alt.Tooltip("subject:N"),
                alt.Tooltip("lag_sec:Q", title="Lag (s)", format=".3f"),
                alt.Tooltip("weight:Q", format=".4f"),
            ],
        )
        .properties(
            width="container",
            height=max(200, len(_filtered["electrode"].unique()) * 12),
            title=f"B weight heatmap — direction {s3_dir_idx.value}",
        )
        .configure_view(strokeWidth=0)
    )
    mo.ui.altair_chart(_hm)
    return


@app.cell(hide_code=True)
def _(mo, pd, s3_b_df, s3_electrode_filter):
    # Summary table: per-electrode peak lag, peak weight, mean |weight|
    mo.stop(len(s3_b_df) == 0, mo.md("*No B-matrix data available.*"))

    _filtered = s3_b_df[s3_b_df["elec_name"].isin(s3_electrode_filter.value)]

    _summary_rows = []
    for (_elec, _sub), _grp in _filtered.groupby(["electrode", "subject"]):
        _abs_max_idx = _grp["weight"].abs().idxmax()
        _peak_row = _grp.loc[_abs_max_idx]
        _summary_rows.append(
            {
                "electrode": _elec,
                "subject": _sub,
                "peak_lag_sec": round(float(_peak_row["lag_sec"]), 3),
                "peak_weight": round(float(_peak_row["weight"]), 5),
                "mean_abs_weight": round(float(_grp["weight"].abs().mean()), 5),
            }
        )

    _summary_df = pd.DataFrame(_summary_rows).sort_values(
        "mean_abs_weight", ascending=False
    )
    mo.vstack(
        [
            mo.md("### Electrode Summary"),
            mo.ui.table(_summary_df),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Section 4: B Matrix — Brain Map
    """)
    return


@app.cell(hide_code=True)
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
        start=-2.0,
        stop=2.0,
        value=0.0,
        step=0.05,
        label="Timepoint (seconds)",
    )
    s4_time_start = mo.ui.number(
        value=-0.5, start=-2.0, stop=2.0, step=0.05, label="Window start (s)"
    )
    s4_time_end = mo.ui.number(
        value=0.5, start=-2.0, stop=2.0, step=0.05, label="Window end (s)"
    )
    mo.vstack(
        [
            mo.hstack([s4_source, s4_fold, s4_k, s4_dir_idx]),
            mo.hstack([s4_time_mode, s4_timepoint, s4_time_start, s4_time_end]),
        ]
    )
    return (
        s4_dir_idx,
        s4_fold,
        s4_k,
        s4_source,
        s4_time_end,
        s4_time_mode,
        s4_time_start,
        s4_timepoint,
    )


@app.cell(hide_code=True)
def _(
    get_rrr_dir,
    load_electrode_metadata,
    load_rrr_AB,
    mo,
    s4_fold,
    s4_k,
    s4_source,
):
    _unit_dir = get_rrr_dir(s4_source.value)
    _fold_dir = _unit_dir / s4_fold.value
    mo.stop(
        not (_fold_dir / "B.npy").exists(),
        mo.callout(mo.md(f"B.npy not found at `{_fold_dir}`."), kind="warn"),
    )
    _A, B = load_rrr_AB(_unit_dir, s4_fold.value, int(s4_k.value))

    meta = load_electrode_metadata(s4_source.value)
    return B, meta


@app.cell(hide_code=True)
def _(
    B,
    meta,
    mo,
    np,
    plt,
    s4_dir_idx,
    s4_time_end,
    s4_time_mode,
    s4_time_start,
    s4_timepoint,
):
    mo.stop(
        len(meta["subjects"]) == 0 or len(meta["lags"]) == 0,
        mo.callout(mo.md("Could not load electrode metadata."), kind="warn"),
    )
    dir_idx = int(s4_dir_idx.value)
    _b_col = B[:, dir_idx]
    _lags = meta["lags"]
    _n_lags = len(_lags)

    # Compute one scalar per electrode
    if s4_time_mode.value == "single timepoint":
        _t = float(s4_timepoint.value)
        _lag_idx = int(np.argmin(np.abs(_lags - _t)))
        time_label = f"t = {_lags[_lag_idx]:.3f}s"
    else:
        _t_start = float(s4_time_start.value)
        _t_end = float(s4_time_end.value)
        _lag_mask = (_lags >= _t_start) & (_lags <= _t_end)
        mo.stop(
            not _lag_mask.any(),
            mo.callout(mo.md("No lags in the selected time window."), kind="warn"),
        )
        time_label = f"t = [{_t_start:.2f}, {_t_end:.2f}]s"

    values = []
    coords = []
    elec_names = []
    sub_labels = []

    for _sub in meta["subjects"]:
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
        values.append(_v)
        coords.append(_sub["coords"][:_n_elecs])
        elec_names.extend(_sub["elec_names"])
        sub_labels.extend([f"sub_{_sub['sub_id']}"] * _n_elecs)

    mo.stop(
        len(values) == 0,
        mo.callout(mo.md("No electrode data available for brain map."), kind="warn"),
    )

    values = np.concatenate(values)
    coords = np.vstack(coords)

    # Check coordinates are valid (non-zero)
    _valid = np.any(coords != 0, axis=1)
    mo.stop(
        not _valid.any(),
        mo.callout(
            mo.md("No valid MNI coordinates found for electrodes."), kind="warn"
        ),
    )

    _values_valid = values[_valid]
    _coords_valid = coords[_valid]
    _names_valid = [n for n, v in zip(elec_names, _valid) if v]
    _subs_valid = [s for s, v in zip(sub_labels, _valid) if v]

    # Sort by absolute value so largest are plotted on top
    _order = np.argsort(np.abs(_values_valid))

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
    plt.suptitle(f"RRR direction {dir_idx}, {time_label}", fontsize=12, y=1.02)
    plt.show()
    return coords, dir_idx, elec_names, sub_labels, time_label, values


@app.cell(hide_code=True)
def _(coords, dir_idx, elec_names, mo, np, pd, sub_labels, time_label, values):
    # Electrode table
    _tbl_df = pd.DataFrame(
        {
            "electrode": elec_names,
            "subject": sub_labels,
            "weight": np.round(values, 5),
            "mni_x": np.round(coords[:, 0], 1),
            "mni_y": np.round(coords[:, 1], 1),
            "mni_z": np.round(coords[:, 2], 1),
        }
    ).sort_values("weight", key=abs, ascending=False)

    mo_electrode_table = mo.ui.table(_tbl_df)

    mo.vstack(
        [
            mo.md(f"### Electrode Weights — direction {dir_idx}, {time_label}"),
            mo_electrode_table,
        ]
    )
    return (mo_electrode_table,)


@app.cell(hide_code=True)
def _(mo):
    update_elecs_run = mo.ui.run_button(
        label="Click to Update Section 3 Electrodes with Table Selections"
    )
    update_elecs_run
    return (update_elecs_run,)


@app.cell(hide_code=True)
def _(
    convert_to_electrode_id,
    mo,
    mo_electrode_table,
    set_s3_selected_elecs,
    update_elecs_run,
):
    mo.stop(not update_elecs_run.value, "Click button to update section 3 selections")

    _subs = mo_electrode_table.value["subject"].tolist()
    _elecs = mo_electrode_table.value["electrode"].tolist()
    set_s3_selected_elecs(
        [convert_to_electrode_id(sub, elec) for sub, elec in zip(_subs, _elecs)]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Section 5: Cross-Fold Consistency
    """)
    return


@app.cell(hide_code=True)
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
    s5_ndirs = mo.ui.slider(
        start=2, stop=50, value=10, step=1, label="Directions to compare"
    )
    mo.hstack([s5_source, s5_k, s5_ndirs])
    return s5_k, s5_ndirs, s5_source


@app.cell(hide_code=True)
def _(alt, get_rrr_dir, mo, np, pd, s5_k, s5_ndirs, s5_source):
    _unit_dir = get_rrr_dir(s5_source.value)
    _f0 = _unit_dir / "fold_0" / "A.npy"
    _f1 = _unit_dir / "fold_1" / "A.npy"
    mo.stop(
        not _f0.exists() or not _f1.exists(),
        mo.callout(
            mo.md("Need both fold_0 and fold_1 A.npy for consistency analysis."),
            kind="warn",
        ),
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
            _hm_rows.append(
                {"fold_0_dir": _i, "fold_1_dir": _j, "cos_sim": float(_cos_sim[_i, _j])}
            )
    _hm_df = pd.DataFrame(_hm_rows)
    _hm_chart = (
        alt.Chart(_hm_df)
        .mark_rect()
        .encode(
            x=alt.X("fold_0_dir:O", title="Fold 0 direction"),
            y=alt.Y("fold_1_dir:O", title="Fold 1 direction"),
            color=alt.Color(
                "cos_sim:Q",
                title="Cosine sim",
                scale=alt.Scale(scheme="redblue", domain=[1, -1]),
            ),
            tooltip=[
                alt.Tooltip("fold_0_dir:O", title="Fold 0"),
                alt.Tooltip("fold_1_dir:O", title="Fold 1"),
                alt.Tooltip("cos_sim:Q", format=".3f"),
            ],
        )
        .properties(
            width="container", height=400, title="Cosine similarity: fold 0 vs fold 1"
        )
        .configure_view(strokeWidth=0)
    )

    # --- Tab B: Diagonal consistency bar chart ---
    _diag_df = pd.DataFrame({"direction": np.arange(_n), "abs_cos_sim": _diag_cos})
    _diag_chart = (
        alt.Chart(_diag_df)
        .mark_bar(color="#3b82f6")
        .encode(
            x=alt.X("direction:O", title="Direction"),
            y=alt.Y(
                "abs_cos_sim:Q",
                title="|Cosine similarity|",
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=[
                alt.Tooltip("direction:O"),
                alt.Tooltip("abs_cos_sim:Q", format=".4f"),
            ],
        )
        .properties(
            width="container", height=250, title="Per-direction fold consistency"
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    # --- Tab C: Consistency table ---
    _tbl_df = pd.DataFrame(
        {
            "direction": np.arange(_n),
            "abs_cos_sim": np.round(_diag_cos, 4),
        }
    )

    mo.tabs(
        {
            "Cosine heatmap": mo.ui.altair_chart(_hm_chart),
            "Diagonal consistency": mo.ui.altair_chart(_diag_chart),
            "Table": mo.ui.table(_tbl_df),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Section 6: Cross-Source Comparison
    """)
    return


@app.cell(hide_code=True)
def _(mo, rrr_options):
    mo.stop(
        len(rrr_options) < 2,
        mo.callout(
            mo.md("Need ≥ 2 RRR sources for cross-source comparison."), kind="warn"
        ),
    )
    s6_source_a = mo.ui.dropdown(
        options=rrr_options, value=rrr_options[0], label="Source A"
    )
    s6_fold_a = mo.ui.dropdown(
        options=["fold_0", "fold_1"], value="fold_0", label="Fold A"
    )
    s6_source_b = mo.ui.dropdown(
        options=rrr_options,
        value=rrr_options[1] if len(rrr_options) > 1 else rrr_options[0],
        label="Source B",
    )
    s6_fold_b = mo.ui.dropdown(
        options=["fold_0", "fold_1"], value="fold_0", label="Fold B"
    )
    s6_k = mo.ui.number(value=10, start=1, stop=50, label="Rank k")
    s6_topk = mo.ui.slider(start=2, stop=50, value=10, step=1, label="Top-k directions")
    mo.vstack(
        [
            mo.hstack([s6_source_a, s6_fold_a, s6_source_b, s6_fold_b]),
            mo.hstack([s6_k, s6_topk]),
        ]
    )
    return s6_fold_a, s6_fold_b, s6_k, s6_source_a, s6_source_b, s6_topk


@app.cell(hide_code=True)
def _(
    aligned_embeddings,
    alt,
    get_rrr_dir,
    load_rrr_AB,
    mo,
    np,
    pd,
    s6_fold_a,
    s6_fold_b,
    s6_k,
    s6_source_a,
    s6_source_b,
    s6_topk,
):
    _dir_a = get_rrr_dir(s6_source_a.value)
    _dir_b = get_rrr_dir(s6_source_b.value)
    mo.stop(
        not (_dir_a / s6_fold_a.value / "A.npy").exists()
        or not (_dir_b / s6_fold_b.value / "A.npy").exists(),
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
    _a_z = (_acts_a - _acts_a.mean(0)) / np.where(
        _acts_a.std(0) > 0, _acts_a.std(0), 1.0
    )
    _b_z = (_acts_b - _acts_b.mean(0)) / np.where(
        _acts_b.std(0) > 0, _acts_b.std(0), 1.0
    )
    _act_corr = (_a_z.T @ _b_z) / len(_a_z)

    # --- Tab A: Cosine heatmap ---
    _cs_rows = []
    for _i in range(_n):
        for _j in range(_n):
            _cs_rows.append(
                {"dir_A": _i, "dir_B": _j, "cos_sim": float(_cos_sim[_i, _j])}
            )
    _cs_df = pd.DataFrame(_cs_rows)
    _cs_chart = (
        alt.Chart(_cs_df)
        .mark_rect()
        .encode(
            x=alt.X("dir_A:O", title="Source A direction"),
            y=alt.Y("dir_B:O", title="Source B direction"),
            color=alt.Color(
                "cos_sim:Q",
                title="Cosine sim",
                scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True),
            ),
            tooltip=[
                alt.Tooltip("dir_A:O"),
                alt.Tooltip("dir_B:O"),
                alt.Tooltip("cos_sim:Q", format=".3f"),
            ],
        )
        .properties(
            width="container",
            height=400,
            title=f"Cosine similarity: {s6_source_a.value} vs {s6_source_b.value}",
        )
        .configure_view(strokeWidth=0)
    )

    # --- Tab B: Activation scatter (best pair) ---
    _scatter_df = pd.DataFrame(
        {
            "activation_A": _acts_a[:, _best_i],
            "activation_B": _acts_b[:, _best_j],
        }
    )
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
            width="container",
            height=350,
            title=f"Best pair: A[{_best_i}] vs B[{_best_j}] (cos={_best_cos:.3f})",
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    # --- Tab C: Top pairs table ---
    _tp_rows = []
    for _i in range(_n):
        _bj = int(np.argmax(np.abs(_cos_sim[_i])))
        _tp_rows.append(
            {
                "dir_A": _i,
                "best_dir_B": _bj,
                "cos_sim": round(float(_cos_sim[_i, _bj]), 4),
                "abs_cos_sim": round(float(np.abs(_cos_sim[_i, _bj])), 4),
            }
        )
    _tp_df = pd.DataFrame(_tp_rows).sort_values("abs_cos_sim", ascending=False)

    # --- Tab D: Activation correlation heatmap ---
    _ac_rows = []
    for _i in range(_n):
        for _j in range(_n):
            _ac_rows.append(
                {"dir_A": _i, "dir_B": _j, "pearson_r": float(_act_corr[_i, _j])}
            )
    _ac_df = pd.DataFrame(_ac_rows)
    _ac_chart = (
        alt.Chart(_ac_df)
        .mark_rect()
        .encode(
            x=alt.X("dir_A:O", title="Source A direction"),
            y=alt.Y("dir_B:O", title="Source B direction"),
            color=alt.Color(
                "pearson_r:Q",
                title="Pearson r",
                scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True),
            ),
            tooltip=[
                alt.Tooltip("dir_A:O"),
                alt.Tooltip("dir_B:O"),
                alt.Tooltip("pearson_r:Q", format=".3f"),
            ],
        )
        .properties(
            width="container", height=400, title="Activation Pearson correlations"
        )
        .configure_view(strokeWidth=0)
    )

    mo.tabs(
        {
            "A. Cosine heatmap": mo.ui.altair_chart(_cs_chart),
            "B. Activation scatter": mo.ui.altair_chart(_scatter_chart),
            "C. Top pairs": mo.ui.table(_tp_df),
            "D. Activation correlations": mo.ui.altair_chart(_ac_chart),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Section 7: Cross-Region Electrode Similarity from B
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _():
    REGION_GROUPS = {
        "EAC": [
            "L G_temp_sup-G_T_transv",
            "R G_temp_sup-G_T_transv",
            "L S_temporal_transverse",
            "L G_temp_sup-Plan_tempo",
        ],
        "STG": [
            "L G_temp_sup-Lateral",
            "R G_temp_sup-Lateral",
            "L S_temporal_sup",
            "R S_temporal_sup",
        ],
        "IFG": [
            "L G_front_inf-Opercular",
            "R G_front_inf-Opercular",
            "L G_front_inf-Triangul",
            "R G_front_inf-Triangul",
            "L G_front_inf-Orbital",
            "R G_front_inf-Orbital",
        ],
        "PRC": [
            "L G_precentral",
            "R G_precentral",
            "L S_precentral-inf-part",
            "L S_precentral-sup-part",
            "L G_and_S_subcentral",
        ],
        "MFG": ["L G_front_middle", "R G_front_middle", "L S_front_inf"],
        "TMP": ["L Pole_temporal", "R Pole_temporal", "L G_temp_sup-Plan_polar"],
    }
    return (REGION_GROUPS,)


@app.cell(hide_code=True)
def _(
    REGION_GROUPS,
    alt,
    get_coord_atlas_labels,
    get_rrr_dir,
    load_electrode_metadata,
    load_rrr_AB,
    mo,
    np,
    pd,
    s7_fold,
    s7_k,
    s7_profile_mode,
    s7_source,
):
    _unit_dir = get_rrr_dir(s7_source.value)
    _fold_dir = _unit_dir / s7_fold.value
    mo.stop(
        not (_fold_dir / "B.npy").exists(),
        mo.callout(mo.md(f"B.npy not found at `{_fold_dir}`."), kind="warn"),
    )
    _destrieux_to_region = {}
    for _area, _regions in REGION_GROUPS.items():
        for _region in _regions:
            _destrieux_to_region[_region] = _area

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
        from src.utils.atlas import DESTRIEUX_2009_LABELS, group_electrodes_by_region

        _atlas_labels = get_coord_atlas_labels(
            _coords_arr[_valid_coords],
            "atlases/destrieux_2009/destrieux2009_rois_lateralized.nii.gz",
            DESTRIEUX_2009_LABELS,
        )
        _region_labels = [
            _destrieux_to_region[x] if x in _destrieux_to_region else None
            for x in _atlas_labels
        ]
        _vi = 0
        for _idx in range(len(_elec_names)):
            if _valid_coords[_idx] and _region_labels[_vi]:
                _regions[_idx] = _region_labels[_vi]
                _vi += 1
            elif not _region_labels[_vi]:
                _vi += 1

    # Cosine similarity matrix between all electrodes
    _norms = np.linalg.norm(_profile_mat, axis=1, keepdims=True)
    _norms = np.where(_norms > 0, _norms, 1.0)
    _normed = _profile_mat / _norms
    _sim_mat = _normed @ _normed.T  # (n_elecs, n_elecs)

    # Group by region
    _unique_regions = sorted(set(_regions))
    _region_indices = {
        r: [i for i, reg in enumerate(_regions) if reg == r] for r in _unique_regions
    }

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
            _rr_rows.append(
                {
                    "region_A": _ra,
                    "region_B": _rb,
                    "mean_sim": (
                        round(float(_vals.mean()), 4) if len(_vals) > 0 else 0.0
                    ),
                    "n_elecs_A": len(_idxs_a),
                    "n_elecs_B": len(_idxs_b),
                }
            )

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
                "mean_sim:Q",
                title="Mean cosine sim",
                scale=alt.Scale(scheme="viridis"),
            ),
            tooltip=[
                alt.Tooltip("region_A:N"),
                alt.Tooltip("region_B:N"),
                alt.Tooltip("mean_sim:Q", format=".4f"),
                alt.Tooltip("n_elecs_A:Q"),
                alt.Tooltip("n_elecs_B:Q"),
            ],
        )
        .properties(
            width="container",
            height=400,
            title="Region-by-region electrode similarity (B profiles)",
        )
        .configure_view(strokeWidth=0)
    )

    # Within vs across bar
    _wa_df = pd.DataFrame(
        [
            {"category": "within-region", "mean_sim": _within_mean},
            {"category": "across-region", "mean_sim": _across_mean},
        ]
    )
    _wa_chart = (
        alt.Chart(_wa_df)
        .mark_bar()
        .encode(
            x=alt.X("category:N", title=""),
            y=alt.Y("mean_sim:Q", title="Mean cosine similarity"),
            color=alt.Color(
                "category:N",
                scale=alt.Scale(
                    domain=["within-region", "across-region"],
                    range=["#3b82f6", "#f59e0b"],
                ),
                legend=None,
            ),
            tooltip=[alt.Tooltip("mean_sim:Q", format=".4f")],
        )
        .properties(width=200, height=250, title="Within vs across region similarity")
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    mo.tabs(
        {
            "Region similarity heatmap": mo.ui.altair_chart(_hm),
            "Within vs across": mo.vstack(
                [
                    mo.ui.altair_chart(_wa_chart),
                    mo.callout(
                        mo.md(
                            f"**Within-region mean:** {_within_mean:.4f}  \n"
                            f"**Across-region mean:** {_across_mean:.4f}"
                        ),
                        kind="info",
                    ),
                ]
            ),
            "Region table": mo.ui.table(_rr_df),
        }
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Section 8: Activation Distribution Statistics
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(
    aligned_embeddings,
    alt,
    get_rrr_dir,
    load_rrr_AB,
    mo,
    np,
    pd,
    s8_fold,
    s8_k,
    s8_source,
    sp_entropy,
    sp_kurtosis,
    sp_skew,
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
        _stats_rows.append(
            {
                "direction": _i,
                "mean": round(float(_a.mean()), 4),
                "std": round(float(_a.std()), 4),
                "kurtosis": round(float(sp_kurtosis(_a)), 4),
                "skewness": round(float(sp_skew(_a)), 4),
                "entropy": round(_ent, 4),
            }
        )

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
            width="container",
            height=250,
            title="Kurtosis per direction (higher = more semantically selective)",
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    mo.vstack(
        [
            mo.ui.altair_chart(_kurt_chart),
            mo.ui.table(_stats_df),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 9: Cluster Analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo, rrr_options):
    mo.stop(
        len(rrr_options) == 0,
        mo.callout(mo.md("No RRR results found."), kind="warn"),
    )
    s9_source = mo.ui.dropdown(
        options=rrr_options,
        value=rrr_options[0] if rrr_options else None,
        label="Source",
    )
    s9_fold = mo.ui.dropdown(
        options=["fold_0", "fold_1"],
        value="fold_0",
        label="Fold",
    )
    s9_k = mo.ui.number(value=20, start=1, stop=50, label="Rank k")
    mo.hstack([s9_source, s9_fold, s9_k])
    return s9_k, s9_source


@app.cell(hide_code=True)
def _(
    aligned_embeddings,
    get_rrr_dir,
    load_encoding_results,
    load_rrr_AB,
    np,
    os,
    pd,
    s9_k,
    s9_source,
):
    _unit_dir = get_rrr_dir(s9_source.value)

    _enc_path = os.path.join(
                get_rrr_dir(s9_source.value).parent, "subjects", f"sub_1", "encoding_results.joblib"
            )
    _test_idxs = load_encoding_results(_enc_path, location="cpu")["test_idxs"]

    cluster_preds = []
    for _fold in range(len(_test_idxs)):
        _A, _B = load_rrr_AB(_unit_dir, "fold_" + str(_fold), int(s9_k.value))
        _curr_embeds = aligned_embeddings[_test_idxs[_fold]]
        cluster_preds.append(_curr_embeds @ _A @ _B.T)

    cluster_preds = np.concatenate(cluster_preds, axis=0)

    _detailed_sigs = pd.read_csv(_unit_dir.parent / "significance_detailed.csv")
    _sig_elecs = pd.read_csv(_unit_dir.parent / "all_subject_sig.csv")

    # 1. Repeat each row 128 times
    _b_labels = _sig_elecs.loc[_sig_elecs.index.repeat(128)].reset_index(drop=True)

    # 2. Add the 'sample_index' column (0 to 127 for each block)
    _b_labels['lag_idx'] = np.tile(np.arange(128), len(_sig_elecs))
    _b_labels["electrode"] = _b_labels["elec"]
    _detailed_df_filtered = _detailed_sigs.merge(
        _b_labels[['subject', 'electrode', 'lag_idx']], 
        on=['subject', 'electrode', 'lag_idx'], 
        how='inner'
    )
    _b_labels = _b_labels.merge(_detailed_df_filtered, on=['subject', 'electrode', 'lag_idx'], how='left')

    # 2. Index into the rows where the boolean is True
    final_b_labels = _b_labels[_b_labels['sig']].copy()
    sig_cluster_preds = cluster_preds[:, final_b_labels.index]
    return final_b_labels, sig_cluster_preds


@app.cell(hide_code=True)
def _():
    import torch

    def get_correlation_gpu(array_a, array_b):
        a = torch.tensor(array_a, device='cuda', dtype=torch.float32)
        b = torch.tensor(array_b, device='cuda', dtype=torch.float32)

        # Standardize on GPU
        a = (a - a.mean(dim=0)) / a.std(dim=0)
        b = (b - b.mean(dim=0)) / b.std(dim=0)

        corr_matrix = (a.T @ b) / array_a.shape[0]
        return corr_matrix.cpu().numpy()

    def get_correlation(array_a, array_B):
        a = (array_a - array_a.mean(axis=0)) / array_a.std(axis=0)
        b = (array_B - array_B.mean(axis=0)) / array_B.std(axis=0)

        corr_matrix = (a.T @ b) / array_a.shape[0]
        return corr_matrix.cpu().numpy()

    return (get_correlation_gpu,)


@app.cell
def _(get_correlation_gpu, np, sig_cluster_preds):
    # rng = np.random.default_rng(42)
    # sub_idx = rng.choice(cluster_preds.shape[1], size=1000, replace=False)
    # pairwise_corrs = get_correlation_gpu(cluster_preds[:, sub_idx], cluster_preds[:, sub_idx])
    # pairwise_D = np.arccos(pairwise_corrs)
    # np.fill_diagonal(pairwise_D, 0)


    pairwise_corrs = get_correlation_gpu(sig_cluster_preds, sig_cluster_preds)
    pairwise_D = np.arccos(np.abs(pairwise_corrs))
    np.fill_diagonal(pairwise_D, 0)
    return (pairwise_D,)


@app.cell(hide_code=True)
def _(dendrogram, fcluster, inconsistent, np, plt):
    def plot_cluster_diagnostics(Z, labels=None, figsize=(18, 20), tail_fraction=0.05):
        """
        Diagnostic plots for choosing a dendrogram distance cutoff.

        Parameters
        ----------
        Z : ndarray
            Linkage matrix from scipy.cluster.hierarchy.linkage.
        labels : list, optional
            Labels for leaf nodes in the dendrogram.
        figsize : tuple
            Figure size.
        tail_fraction : float
            Fraction of top merges to focus on for detailed plots (default 5%).
        """
        merge_dists = Z[:, 2]
        n_merges = len(merge_dists)
        n_samples = n_merges + 1
        n_tail = max(int(n_merges * tail_fraction), 50)

        fig, axes = plt.subplots(3, 2, figsize=figsize)

        # 1. Dendrogram (truncated for large datasets)
        ax = axes[0, 0]
        truncate = n_samples > 80
        dendrogram(
            Z,
            ax=ax,
            truncate_mode='lastp' if truncate else None,
            p=40 if truncate else n_samples,
            leaf_rotation=90,
            leaf_font_size=8,
            labels=labels if not truncate else None,
            color_threshold=0,
        )
        ax.set_title(f'Dendrogram{" (truncated)" if truncate else ""}')
        ax.set_ylabel('Merge distance')

        # 2. Sorted merge distances — zoomed to tail
        ax = axes[0, 1]
        sorted_dists = np.sort(merge_dists)
        tail = sorted_dists[-n_tail:]
        ax.plot(range(n_tail), tail, marker='.', ms=4)
        gaps = np.diff(sorted_dists)
        top_gap_idxs = np.argsort(gaps)[-min(5, len(gaps)):]
        for idx in top_gap_idxs:
            cut_val = (sorted_dists[idx] + sorted_dists[idx + 1]) / 2
            if cut_val >= tail[0]:
                ax.axhline(cut_val, color='red', alpha=0.5, ls='--', lw=1)
        ax.set_title(f'Last {n_tail} merge distances (red = largest gaps)')
        ax.set_xlabel(f'Merge index (of top {n_tail})')
        ax.set_ylabel('Distance')

        # 3. Number of clusters vs. cut threshold — focused on interesting range
        ax = axes[1, 0]
        t_lo = np.percentile(merge_dists, max(80, 100 - tail_fraction * 200))
        t_hi = merge_dists.max() * 1.05
        t_range = np.linspace(t_lo, t_hi, 300)
        n_clusters = np.array([
            len(np.unique(fcluster(Z, t=t, criterion='distance'))) for t in t_range
        ])
        ax.plot(t_range, n_clusters)
        ax.set_title('Number of clusters vs. distance threshold')
        ax.set_xlabel('Distance threshold t')
        ax.set_ylabel('Number of clusters')
        ax.set_yscale('log')
        for nc in np.unique(n_clusters):
            mask = n_clusters == nc
            t_vals = t_range[mask]
            t_width = t_vals.max() - t_vals.min()
            if t_width / (t_range[-1] - t_range[0]) > 0.03:
                ax.axhspan(nc - 0.3, nc + 0.3, alpha=0.08, color='green')

        # 4. Top merge distance gaps
        ax = axes[1, 1]
        n_gaps_show = min(40, len(gaps))
        top_gaps = np.sort(gaps)[::-1][:n_gaps_show]
        ax.bar(range(n_gaps_show), top_gaps, width=0.8)
        ax.set_title(f'Top {n_gaps_show} merge distance gaps')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Gap size')

        # 5. Inconsistency coefficients
        ax = axes[2, 0]
        incons = inconsistent(Z, d=2)
        incons_vals = incons[:, 3]
        ax.hist(incons_vals, bins=80, edgecolor='black', linewidth=0.3)
        # zoom into the right tail via a twin axis
        threshold_99 = np.percentile(incons_vals, 99)
        ax.axvline(threshold_99, color='red', ls='--', lw=1,
                   label=f'99th percentile ({threshold_99:.2f})')
        ax.legend()
        ax.set_title('Inconsistency coefficient distribution (depth=2)')
        ax.set_xlabel('Inconsistency coefficient')
        ax.set_ylabel('Count')

        # 6. Cluster size distribution at candidate thresholds
        ax = axes[2, 1]
        candidates = []
        if len(gaps) > 0:
            for idx in np.argsort(gaps)[-min(4, len(gaps)):]:
                candidates.append((sorted_dists[idx] + sorted_dists[idx + 1]) / 2)
        candidates = sorted(set(candidates))

        if candidates:
            bar_width = 0.8 / len(candidates)
            for i, t in enumerate(candidates):
                cl = fcluster(Z, t=t, criterion='distance')
                sizes = np.sort(np.bincount(cl)[1:])[::-1]
                # only show top clusters to keep readable
                n_show = min(30, len(sizes))
                ax.bar(
                    np.arange(n_show) + i * bar_width,
                    sizes[:n_show],
                    width=bar_width,
                    label=f't={t:.3f} ({len(sizes)} cl)',
                    alpha=0.7,
                )
            ax.legend(fontsize=8)
        ax.set_title('Cluster size distribution at candidate thresholds')
        ax.set_xlabel('Cluster rank (largest first)')
        ax.set_ylabel('Cluster size')

        plt.tight_layout()
        plt.savefig('cluster_diagnostics.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Summary table
        print(f"\n=== Candidate cut thresholds ({n_samples} samples, {n_merges} merges) ===")
        if candidates:
            for t in candidates:
                cl = fcluster(Z, t=t, criterion='distance')
                sizes = np.sort(np.bincount(cl)[1:])[::-1]
                print(f"  t={t:.4f} → {len(sizes):4d} clusters | "
                      f"largest: {sizes[0]:5d} | median: {np.median(sizes):5.0f} | "
                      f"singletons: {np.sum(sizes == 1):4d}")

    return (plot_cluster_diagnostics,)


@app.cell(hide_code=True)
def _(np, plt):
    from matplotlib.gridspec import GridSpec
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize
    from sklearn.manifold import MDS

    ALL_DIAGNOSTICS = {"hopkins", "silhouette", "gap", "elbow", "pca", "tsne", "umap", "hdbscan"}


    def analyze_clusterability(
        X: np.ndarray,
        k_range: tuple = (2, 11),
        k_step: int = 10,
        random_state: int = 42,
        metric: str = "euclidean",
        subsample: int | None = None,
        n_gap_refs: int = 10,
        diagnostics: set[str] | None = None,
        color_by: str = "kmeans",
    ):
        """
        Analyze how clusterable a dataset is using multiple diagnostics.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, dim) or (n_samples, n_samples)
            The input data, or a precomputed distance matrix if metric="precomputed".
        k_range : tuple
            Range of K values to test (start, stop). Default (2, 11) tests K=2..10.
        k_step : int
            Step between K values.
        random_state : int
            Random seed for reproducibility.
        metric : str
            Distance metric: "euclidean", "cosine", or "precomputed".
        subsample : int or None
            If set, subsample to this many points for expensive operations.
        n_gap_refs : int
            Number of reference datasets for the gap statistic. Default 10.
        diagnostics : set of str, or None
            Which analyses to run. Any subset of:
                {"hopkins", "silhouette", "gap", "elbow", "pca", "tsne", "umap", "hdbscan"}
            None (default) runs all of them.
        color_by : str
            How to color embedding scatter plots: "kmeans" (default) or "hdbscan".
            If "hdbscan", the hdbscan diagnostic is implicitly enabled.

        Returns
        -------
        results : dict
            Dictionary containing all computed metrics and embeddings.
        fig : matplotlib.figure.Figure or None
            The summary figure, or None if no plottable diagnostics were requested.
        """
        # ── Resolve diagnostics set ───────────────────────────────────────────
        if diagnostics is None:
            run = ALL_DIAGNOSTICS.copy()
        else:
            unknown = diagnostics - ALL_DIAGNOSTICS
            if unknown:
                raise ValueError(
                    f"Unknown diagnostics: {unknown}. Choose from {sorted(ALL_DIAGNOSTICS)}"
                )
            run = diagnostics.copy()

        if color_by not in ("kmeans", "hdbscan"):
            raise ValueError(f"color_by must be 'kmeans' or 'hdbscan', got '{color_by}'")

        # If coloring by hdbscan, make sure it runs
        if color_by == "hdbscan":
            run.add("hdbscan")

        has_embeddings = bool(run & {"pca", "tsne", "umap"})
        needs_kmeans = bool(run & {"silhouette", "gap", "elbow"}) or (
            color_by == "kmeans" and has_embeddings
        )

        rng = np.random.default_rng(random_state)
        ks = list(range(k_range[0], k_range[1], k_step))
        precomputed = metric == "precomputed"
        use_cosine = metric == "cosine"

        # ── 0. Prepare data ───────────────────────────────────────────────────
        if precomputed:
            assert X.shape[0] == X.shape[1], "Precomputed metric requires a square distance matrix"
            D = X
            n_samples = D.shape[0]
            dim = None

            n_mds_components = min(50, n_samples - 1)
            mds = MDS(
                n_components=n_mds_components, dissimilarity="precomputed",
                random_state=random_state, normalized_stress="auto",
            )
            X_embedded = mds.fit_transform(D)
            X_clust = X_embedded
            dim_label = f"MDS-{n_mds_components}"
        else:
            n_samples = X.shape[0]
            dim = X.shape[1]
            dim_label = str(dim)
            D = None
            X_clust = normalize(X, norm="l2") if use_cosine else X.copy()

        # Subsample
        do_subsample = subsample is not None and subsample < n_samples
        if do_subsample:
            sub_idx = rng.choice(n_samples, size=subsample, replace=False)
            X_sub = X_clust[sub_idx]
            if precomputed:
                D_sub = D[np.ix_(sub_idx, sub_idx)]
                X_sub_raw = X_embedded[sub_idx]
            else:
                X_sub_raw = X[sub_idx]
            print(f"Subsampling: {subsample}/{n_samples} points for t-SNE, UMAP, silhouette")
        else:
            sub_idx = np.arange(n_samples)
            X_sub = X_clust
            if precomputed:
                D_sub = D
                X_sub_raw = X_embedded
            else:
                X_sub_raw = X

        use_minibatch = n_samples > 20_000
        if use_minibatch and needs_kmeans:
            print(f"Using MiniBatchKMeans (n={n_samples})")

        def make_kmeans(k, n_init=10):
            if use_minibatch:
                return MiniBatchKMeans(
                    n_clusters=k, n_init=n_init, random_state=random_state, batch_size=2048
                )
            return KMeans(n_clusters=k, n_init=n_init, random_state=random_state)

        results = {"metric": metric}

        # ── 1. Hopkins Statistic ──────────────────────────────────────────────
        if "hopkins" in run:
            m = min(n_samples // 10, 200)
            idx = rng.choice(n_samples, size=m, replace=False)
            X_sample = X_clust[idx]

            mins, maxs = X_clust.min(axis=0), X_clust.max(axis=0)
            X_uniform = rng.uniform(mins, maxs, size=(m, X_clust.shape[1]))
            if use_cosine:
                X_uniform = normalize(X_uniform, norm="l2")

            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(X_clust)

            u_dists, _ = nn.kneighbors(X_uniform, n_neighbors=1)
            w_dists, _ = nn.kneighbors(X_sample, n_neighbors=2)
            w_dists = w_dists[:, 1]

            hopkins = u_dists.sum() / (u_dists.sum() + w_dists.sum())
            results["hopkins"] = hopkins
            results["hopkins_note"] = (
                "computed on MDS embedding" if precomputed else "computed on feature space"
            )

        # ── 2. KMeans ─────────────────────────────────────────────────────────
        all_km_models = {}
        labels_full_by_k = {}
        inertias = []
        silhouettes = []

        if needs_kmeans:
            for k in ks:
                km = make_kmeans(k)
                labels_full = km.fit_predict(X_clust)
                all_km_models[k] = km
                labels_full_by_k[k] = labels_full

                if "elbow" in run:
                    inertias.append(km.inertia_)

                if "silhouette" in run:
                    if do_subsample:
                        labels_sub = labels_full[sub_idx]
                        if len(np.unique(labels_sub)) >= 2:
                            if precomputed:
                                silhouettes.append(
                                    silhouette_score(D_sub, labels_sub, metric="precomputed")
                                )
                            else:
                                silhouettes.append(silhouette_score(X_sub, labels_sub))
                        else:
                            silhouettes.append(0.0)
                    else:
                        if precomputed:
                            silhouettes.append(
                                silhouette_score(D, labels_full, metric="precomputed")
                            )
                        else:
                            silhouettes.append(silhouette_score(X_clust, labels_full))

            if "silhouette" in run:
                results["silhouettes"] = dict(zip(ks, silhouettes))
                best_k_idx = int(np.argmax(silhouettes))
                best_k_sil = ks[best_k_idx]
                results["best_k_silhouette"] = best_k_sil
            if "elbow" in run:
                results["inertias"] = dict(zip(ks, inertias))
            results["kmeans_models"] = all_km_models

        # ── 3. Gap Statistic ─────────────────────────────────────────────────
        if "gap" in run:
            gap_n = min(n_samples, 20_000) if do_subsample else n_samples
            X_gap = X_clust[:gap_n]
            mins_gap, maxs_gap = X_gap.min(axis=0), X_gap.max(axis=0)

            gaps = []
            gap_stds = []
            for k in ks:
                km = make_kmeans(k, n_init=5)
                km.fit(X_gap)
                log_wk = np.log(km.inertia_)

                ref_log_wks = []
                for _ in range(n_gap_refs):
                    X_ref = rng.uniform(mins_gap, maxs_gap, size=(gap_n, X_clust.shape[1]))
                    if use_cosine:
                        X_ref = normalize(X_ref, norm="l2")
                    km_ref = make_kmeans(k, n_init=3)
                    km_ref.fit(X_ref)
                    ref_log_wks.append(np.log(km_ref.inertia_))

                gap = np.mean(ref_log_wks) - log_wk
                gap_std = np.std(ref_log_wks) * np.sqrt(1 + 1 / n_gap_refs)
                gaps.append(gap)
                gap_stds.append(gap_std)

            results["gaps"] = dict(zip(ks, gaps))
            results["gap_stds"] = dict(zip(ks, gap_stds))

        # ── 4. HDBSCAN ───────────────────────────────────────────────────────
        hdbscan_labels_full = None
        hdbscan_labels_sub = None
        has_hdbscan = False

        if "hdbscan" in run:
            try:
                import hdbscan as _hdbscan

                if precomputed:
                    hdb = _hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=15)
                    hdbscan_labels_full = hdb.fit_predict(D)
                else:
                    hdb_metric = "euclidean"  # hdbscan handles cosine poorly; use normalized data
                    hdb = _hdbscan.HDBSCAN(metric=hdb_metric, min_cluster_size=15)
                    hdbscan_labels_full = hdb.fit_predict(X_clust)

                hdbscan_labels_sub = hdbscan_labels_full[sub_idx]
                n_hdb_clusters = len(set(hdbscan_labels_full)) - (1 if -1 in hdbscan_labels_full else 0)
                n_noise = int((hdbscan_labels_full == -1).sum())
                has_hdbscan = True

                results["hdbscan_labels_full"] = hdbscan_labels_full
                results["hdbscan_labels_sub"] = hdbscan_labels_sub
                results["hdbscan_n_clusters"] = n_hdb_clusters
                results["hdbscan_n_noise"] = n_noise
                results["hdbscan_probabilities"] = hdb.probabilities_
                results["hdbscan_model"] = hdb

                # Silhouette for HDBSCAN (excluding noise) on subsample
                non_noise_mask = hdbscan_labels_sub != -1
                if non_noise_mask.sum() > 1 and len(np.unique(hdbscan_labels_sub[non_noise_mask])) >= 2:
                    if precomputed:
                        hdb_sil = silhouette_score(
                            D_sub[np.ix_(non_noise_mask, non_noise_mask)],
                            hdbscan_labels_sub[non_noise_mask],
                            metric="precomputed",
                        )
                    else:
                        hdb_sil = silhouette_score(
                            X_sub[non_noise_mask], hdbscan_labels_sub[non_noise_mask]
                        )
                    results["hdbscan_silhouette"] = hdb_sil

            except ImportError:
                print("hdbscan not installed, skipping HDBSCAN diagnostic")

        # ── 5. Dimensionality Reduction Embeddings ────────────────────────────
        X_pca = X_tsne = X_umap = None
        pca_var = (None, None)
        has_umap = False

        if "pca" in run:
            if precomputed:
                mds_2d = MDS(
                    n_components=2, dissimilarity="precomputed",
                    random_state=random_state, normalized_stress="auto",
                )
                X_pca_full = mds_2d.fit_transform(D)
                pca_var = (None, None)
            else:
                pca = PCA(n_components=2, random_state=random_state)
                X_pca_full = pca.fit_transform(X)
                pca_var = pca.explained_variance_ratio_
            X_pca = X_pca_full[sub_idx]
            results["pca_embedding"] = X_pca
            results["pca_explained_var"] = pca_var

        if "tsne" in run:
            perplexity = min(30, len(X_sub) - 1)
            if precomputed:
                tsne = TSNE(
                    n_components=2, perplexity=perplexity,
                    metric="precomputed", init="random", random_state=random_state,
                )
                X_tsne = tsne.fit_transform(D_sub if do_subsample else D)
            else:
                tsne_metric = "cosine" if use_cosine else "euclidean"
                tsne = TSNE(
                    n_components=2, perplexity=perplexity,
                    metric=tsne_metric, random_state=random_state,
                )
                X_tsne = tsne.fit_transform(X_sub_raw if use_cosine else X_sub)
            results["tsne_embedding"] = X_tsne

        if "umap" in run:
            try:
                import umap
                if precomputed:
                    reducer = umap.UMAP(
                        n_components=2, metric="precomputed", random_state=random_state,
                    )
                    X_umap = reducer.fit_transform(D_sub if do_subsample else D)
                else:
                    umap_metric = "cosine" if use_cosine else "euclidean"
                    reducer = umap.UMAP(
                        n_components=2, metric=umap_metric, random_state=random_state,
                    )
                    X_umap = reducer.fit_transform(X_sub_raw if use_cosine else X_sub)
                has_umap = True
                results["umap_embedding"] = X_umap
            except ImportError:
                has_umap = False

        # ── 6. Resolve coloring labels for embeddings ─────────────────────────
        color_labels_sub = None
        color_k_label = None  # for the suptitle

        if color_by == "hdbscan" and has_hdbscan:
            color_labels_sub = hdbscan_labels_sub
            color_k_label = f"HDBSCAN ({results['hdbscan_n_clusters']} clusters, {results['hdbscan_n_noise']} noise)"
        elif color_by == "hdbscan" and not has_hdbscan:
            print("HDBSCAN unavailable, falling back to KMeans coloring")
            color_by = "kmeans"  # fall through

        if color_by == "kmeans":
            best_k = results.get("best_k_silhouette")
            if best_k is not None:
                labels_best_full = labels_full_by_k[best_k]
                color_labels_sub = labels_best_full[sub_idx]
                color_k_label = f"KMeans K={best_k}"
            elif all_km_models:
                best_k = ks[len(ks) // 2]
                labels_best_full = labels_full_by_k.get(best_k)
                if labels_best_full is not None:
                    color_labels_sub = labels_best_full[sub_idx]
                    color_k_label = f"KMeans K={best_k}"

        if color_labels_sub is not None:
            results["color_labels_sub"] = color_labels_sub
            results["color_by"] = color_by
        results["subsample_idx"] = sub_idx

        # ── 7. Plot ───────────────────────────────────────────────────────────
        top_panels = [d for d in ["silhouette", "gap", "elbow"] if d in run]
        bot_panels = [d for d in ["pca", "tsne", "umap"] if d in run and
                      (d != "umap" or has_umap)]

        # HDBSCAN gets its own summary panel in the top row
        if "hdbscan" in run and has_hdbscan:
            top_panels.append("hdbscan")

        if not top_panels and not bot_panels:
            return results, None

        n_rows = (1 if top_panels else 0) + (1 if bot_panels else 0)
        n_cols = max(len(top_panels), len(bot_panels), 1)
        fig = plt.figure(figsize=(6 * n_cols, 6 * n_rows + 1))
        fig.patch.set_facecolor("#fafafa")
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.35)

        # Colormap: need to handle noise label (-1) for HDBSCAN
        if color_labels_sub is not None:
            n_unique = len(set(color_labels_sub) - {-1})
        else:
            n_unique = 2
        cmap = "tab10" if n_unique <= 10 else "tab20"

        row = 0

        # -- Hopkins banner --
        if "hopkins" in run and "hopkins" in results:
            ax_title = fig.add_axes([0.0, 0.95, 1.0, 0.05])
            ax_title.set_axis_off()
            h = results["hopkins"]
            hop_color = "#2ecc71" if h > 0.7 else ("#f39c12" if h > 0.55 else "#e74c3c")
            hop_label = (
                "strong clustering" if h > 0.7
                else ("moderate" if h > 0.55 else "weak / random")
            )
            hop_note = " (via MDS embedding)" if precomputed else ""
            ax_title.text(
                0.5, 0.5,
                f"Hopkins Statistic = {h:.3f}  →  {hop_label}{hop_note}  [metric: {metric}]",
                ha="center", va="center", fontsize=16, fontweight="bold", color=hop_color,
                transform=ax_title.transAxes,
            )

        # -- Top row: metric plots --
        if top_panels:
            for col, panel in enumerate(top_panels):
                ax = fig.add_subplot(gs[row, col])
                if panel == "silhouette":
                    ax.plot(ks, silhouettes, "o-", color="#2c3e50", linewidth=2, markersize=6)
                    ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
                    ax.set_xlabel("K"); ax.set_ylabel("Silhouette Score")
                    sil_note = (
                        " (precomputed dist)" if precomputed
                        else (" (subsampled)" if do_subsample else "")
                    )
                    ax.set_title(f"Silhouette Score vs K{sil_note}")
                    ax.set_xticks(ks)
                elif panel == "gap":
                    g = list(results["gaps"].values())
                    gs_ = list(results["gap_stds"].values())
                    ax.plot(ks, g, "o-", color="#8e44ad", linewidth=2, markersize=6)
                    ax.fill_between(
                        ks, np.array(g) - np.array(gs_), np.array(g) + np.array(gs_),
                        alpha=0.2, color="#8e44ad",
                    )
                    ax.set_xlabel("K"); ax.set_ylabel("Gap")
                    ax.set_title(f"Gap Statistic vs K{' (via MDS)' if precomputed else ''}")
                    ax.set_xticks(ks)
                elif panel == "elbow":
                    ax.plot(ks, inertias, "o-", color="#e67e22", linewidth=2, markersize=6)
                    ax.set_xlabel("K"); ax.set_ylabel("Inertia")
                    ax.set_title(f"Elbow Plot{' (via MDS)' if precomputed else ''}")
                    ax.set_xticks(ks)
                elif panel == "hdbscan":
                    # Bar chart: cluster sizes + noise
                    hdb_labels = results["hdbscan_labels_full"]
                    unique_labels = sorted(set(hdb_labels))
                    counts = [int((hdb_labels == l).sum()) for l in unique_labels]
                    bar_labels = [f"Noise" if l == -1 else f"C{l}" for l in unique_labels]
                    bar_colors = ["#bdc3c7" if l == -1 else plt.cm.tab10(l % 10) for l in unique_labels]
                    ax.bar(bar_labels, counts, color=bar_colors, edgecolor="white")
                    ax.set_ylabel("Count")
                    hdb_sil_str = ""
                    if "hdbscan_silhouette" in results:
                        hdb_sil_str = f"  sil={results['hdbscan_silhouette']:.3f}"
                    ax.set_title(
                        f"HDBSCAN: {results['hdbscan_n_clusters']} clusters, "
                        f"{results['hdbscan_n_noise']} noise{hdb_sil_str}"
                    )
                    if len(bar_labels) > 15:
                        ax.tick_params(axis="x", rotation=90, labelsize=7)
            row += 1

        # -- Bottom row: embeddings --
        if bot_panels:
            scatter_kw = dict(s=12, alpha=0.7, edgecolors="none")

            # Build color array: noise points get grey
            if color_labels_sub is not None:
                c_array = color_labels_sub.astype(float)
                noise_mask = color_labels_sub == -1
            else:
                c_array = "#2c3e50"
                noise_mask = None

            for col, panel in enumerate(bot_panels):
                ax = fig.add_subplot(gs[row, col])
                if panel == "pca":
                    emb = X_pca
                elif panel == "tsne":
                    emb = X_tsne
                elif panel == "umap":
                    emb = X_umap

                if noise_mask is not None and noise_mask.any():
                    # Plot noise in grey first, then clusters on top
                    ax.scatter(
                        emb[noise_mask, 0], emb[noise_mask, 1],
                        c="#bdc3c7", s=8, alpha=0.3, edgecolors="none", label="noise",
                    )
                    cluster_mask = ~noise_mask
                    ax.scatter(
                        emb[cluster_mask, 0], emb[cluster_mask, 1],
                        c=c_array[cluster_mask], cmap=cmap, **scatter_kw,
                    )
                else:
                    ax.scatter(emb[:, 0], emb[:, 1], c=c_array, cmap=cmap, **scatter_kw)

                if panel == "pca":
                    if precomputed:
                        ax.set_title("MDS (2D from distance matrix)")
                        ax.set_xlabel("MDS1"); ax.set_ylabel("MDS2")
                    else:
                        ax.set_title(f"PCA  (var: {pca_var[0]:.1%} + {pca_var[1]:.1%})")
                        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
                elif panel == "tsne":
                    ax.set_title("t-SNE")
                elif panel == "umap":
                    ax.set_title("UMAP")

        n_plot = len(X_sub)
        color_str = color_k_label or "uncolored"
        fig.suptitle(
            f"Clusterability Analysis  (n={n_samples}, dim={dim_label}, "
            f"colored by {color_str}, plotting {n_plot} pts)",
            fontsize=14, y=1.0,
        )

        if precomputed:
            results["mds_embedding_full"] = X_embedded

        return results, fig

    return


@app.cell(hide_code=True)
def _():
    # analyze_clusterability(pairwise_D, metric="precomputed", diagnostics=set(["hopkins"]), k_range = (2, 30), k_step = 5)
    # plt.show()
    return


@app.cell
def _():
    from scipy.spatial.distance import squareform

    return (squareform,)


@app.cell
def _(linkage, pairwise_D, plot_cluster_diagnostics, squareform):
    condensed = squareform(pairwise_D, checks=False)
    Z_corr = linkage(condensed, method='ward')
    plot_cluster_diagnostics(Z_corr)
    return (Z_corr,)


@app.cell
def _(Z_corr, fcluster, final_b_labels, np, plt):
    num_clusters = 7
    cluster_labels = fcluster(Z_corr, t=num_clusters, criterion='maxclust')

    final_b_labels["cluster"] = cluster_labels
    cluster_counts = final_b_labels[["subject", "lag_sec", "cluster"]].groupby("cluster").value_counts().reset_index()

    from matplotlib.colors import ListedColormap

    _all_clusters = np.sort(final_b_labels["cluster"].unique())
    _base_cmap = plt.cm.get_cmap("tab10", max(len(_all_clusters), 1))
    cluster_colors = {c: _base_cmap(i) for i, c in enumerate(_all_clusters)}
    return cluster_colors, cluster_counts


@app.cell(hide_code=True)
def _(cluster_colors, cluster_counts, plt):
    for cluster in sorted(cluster_counts["cluster"].unique()):
        lag_counts = (
            cluster_counts[cluster_counts["cluster"] == cluster]
            .groupby("lag_sec")["count"]
            .sum()
            .reset_index()
        )
        plt.plot(
            lag_counts["lag_sec"],
            lag_counts["count"],
            label=f"cluster {cluster}",
            color=cluster_colors[cluster],
        )
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(cluster_colors, final_b_labels, meta, mo, np):
    def _():
        mo.stop(
            len(meta["subjects"]) == 0 or len(meta["lags"]) == 0,
            mo.callout(mo.md("Could not load electrode metadata."), kind="warn"),
        )
        from collections import defaultdict
        elec_cluster_counts = defaultdict(lambda: defaultdict(int))
        elec_total_counts = defaultdict(int)

        for _, row in final_b_labels.iterrows():
            key = (row["subject"], row["elec"])
            elec_cluster_counts[key][row["cluster"]] += 1
            elec_total_counts[key] += 1

        cluster_map = {}
        for _, row in final_b_labels.iterrows():
            cluster_map[(row["subject"], row["elec"])] = row["cluster"]

        all_coords = []
        all_cluster_ids = []
        all_names = []
        for _sub in meta["subjects"]:
            sub_id = _sub["sub_id"]
            for i, name in enumerate(_sub["elec_names"]):
                coord = _sub["coords"][i]
                if np.all(coord == 0):
                    continue
                all_coords.append(coord)
                all_names.append(f"sub{sub_id}_{name}")
                all_cluster_ids.append(cluster_map.get((sub_id, name), None))

        mo.stop(
            len(all_coords) == 0,
            mo.callout(mo.md("No valid electrode coordinates found."), kind="warn"),
        )
        all_coords = np.vstack(all_coords)
        unique_clusters = np.sort([c for c in set(all_cluster_ids) if c is not None])

        from nilearn.plotting import plot_glass_brain
        import matplotlib.pyplot as plt

        for clust in unique_clusters:
            mask = np.array([c == clust for c in all_cluster_ids])
            masked_names = [all_names[i] for i in range(len(all_names)) if mask[i]]
            masked_coords = all_coords[mask]

            base_color = cluster_colors[clust]
            alpha_values = []
            for name in masked_names:
                parts = name.split("_", 1)
                sub_id = int(parts[0].replace("sub", ""))
                elec = parts[1]
                key = (sub_id, elec)
                frac = elec_cluster_counts[key][clust] / elec_total_counts[key]
                alpha_values.append(max(frac, 0.05))

            alpha_values = np.array(alpha_values)

            # Render empty glass brain as a background
            display = plot_glass_brain(
                None,
                display_mode="lzr",
                figure=plt.figure(figsize=(14, 5)),
            )

            # Scatter electrodes directly onto each axis
            for key, nilearn_ax in display.axes.items():
                ax = nilearn_ax.ax
                direction = key  # 'l', 'r', or 'z'

                for i, coord_3d in enumerate(masked_coords):
                    x, y, z = coord_3d

                    # Project 3D MNI -> 2D depending on the view
                    if direction in ("l", "r"):  # sagittal
                        px, py = y, z
                    elif direction == "z":       # axial
                        px, py = x, y
                    else:
                        continue

                    ax.scatter(
                        px, py,
                        s=50,
                        color=(*base_color[:3], alpha_values[i]),
                        edgecolors="none",
                        zorder=10,
                    )

            plt.suptitle(f"Cluster {clust}", fontsize=13, y=1.02)
            plt.show()

    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
