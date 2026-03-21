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
    from pathlib import Path
    return Path, alt, mo, np, pd


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
    return get_example_display_controls, get_mo_display_activations, load_embeddings


@app.cell
def _(mo):
    mo.md("# RRR / SRRR Results Reviewer")
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
    srrr_options = []

    # cross_subject
    if (_rr / "rrr" / "grid_results.csv").exists():
        rrr_options.append("cross_subject")
    if (_rr / "srrr" / "grid_results.csv").exists():
        srrr_options.append("cross_subject")

    # per_subject
    _ps_rrr = _rr / "per_subject_rrr" / "subjects"
    if _ps_rrr.exists():
        for _p in sorted(_ps_rrr.glob("sub_*/grid_results.csv")):
            rrr_options.append(f"per_subject/{_p.parent.name}")
    _ps_srrr = _rr / "per_subject_srrr" / "subjects"
    if _ps_srrr.exists():
        for _p in sorted(_ps_srrr.glob("sub_*/grid_results.csv")):
            srrr_options.append(f"per_subject/{_p.parent.name}")

    # per_region
    _pr_rrr = _rr / "per_region_rrr" / "regions"
    if _pr_rrr.exists():
        for _p in sorted(_pr_rrr.glob("*/grid_results.csv")):
            rrr_options.append(f"per_region/{_p.parent.name}")
    _pr_srrr = _rr / "per_region_srrr" / "regions"
    if _pr_srrr.exists():
        for _p in sorted(_pr_srrr.glob("*/grid_results.csv")):
            srrr_options.append(f"per_region/{_p.parent.name}")

    return rrr_options, srrr_options


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
    def _rrr_unit_dir(source: str) -> Path:
        rr = Path(results_root_input.value)
        if source == "cross_subject":
            return rr / "rrr"
        elif source.startswith("per_subject/"):
            return rr / "per_subject_rrr" / "subjects" / source.split("/", 1)[1]
        else:
            return rr / "per_region_rrr" / "regions" / source.split("/", 1)[1]

    def _srrr_unit_dir(source: str) -> Path:
        rr = Path(results_root_input.value)
        if source == "cross_subject":
            return rr / "srrr"
        elif source.startswith("per_subject/"):
            return rr / "per_subject_srrr" / "subjects" / source.split("/", 1)[1]
        else:
            return rr / "per_region_srrr" / "regions" / source.split("/", 1)[1]

    def _svd_dir_for_srrr(source: str) -> Path:
        rr = Path(results_root_input.value)
        if source == "cross_subject":
            return rr / "svd"
        elif source.startswith("per_subject/"):
            return rr / "per_subject_svd" / "subjects" / source.split("/", 1)[1]
        else:
            return rr / "per_region_svd" / "regions" / source.split("/", 1)[1]

    get_rrr_dir = _rrr_unit_dir
    get_srrr_dir = _srrr_unit_dir
    get_svd_dir = _svd_dir_for_srrr
    return get_rrr_dir, get_srrr_dir, get_svd_dir


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
        start=0.01, stop=0.10, value=0.05, step=0.01,
        label="Sufficiency threshold (recovery_r2 <)",
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
def _(alt, mo, pd, s1_grid, s1_show_folds, s1_summary):
    # Build long-form summary for 3-series line chart
    _cols = ["full_r2", "minimal_r2", "recovery_r2"]
    _long = s1_summary.melt(
        id_vars="k", value_vars=[c for c in _cols if c in s1_summary.columns],
        var_name="metric", value_name="r2",
    )
    _color_map = {
        "full_r2": "#9ca3af",
        "minimal_r2": "#3b82f6",
        "recovery_r2": "#ef4444",
    }
    _dash_map = {
        "full_r2": [4, 2],
        "minimal_r2": [1, 0],
        "recovery_r2": [1, 0],
    }

    _summary_lines = (
        alt.Chart(_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("k:Q", scale=alt.Scale(type="log"), title="Rank k"),
            y=alt.Y("r2:Q", title="R²"),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(
                    domain=list(_color_map.keys()),
                    range=list(_color_map.values()),
                ),
                legend=alt.Legend(title="Metric"),
            ),
            strokeDash=alt.StrokeDash(
                "metric:N",
                scale=alt.Scale(
                    domain=list(_dash_map.keys()),
                    range=list(_dash_map.values()),
                ),
                legend=None,
            ),
            tooltip=["k:Q", "metric:N", alt.Tooltip("r2:Q", format=".4f")],
        )
    )

    _layers = [_summary_lines]

    if s1_show_folds and s1_grid is not None:
        _fold_long = s1_grid.melt(
            id_vars=["k", "fold"], value_vars=[c for c in _cols if c in s1_grid.columns],
            var_name="metric", value_name="r2",
        )
        _fold_lines = (
            alt.Chart(_fold_long)
            .mark_line(opacity=0.25, strokeWidth=1)
            .encode(
                x=alt.X("k:Q", scale=alt.Scale(type="log")),
                y="r2:Q",
                color=alt.Color("metric:N", scale=alt.Scale(
                    domain=list(_color_map.keys()), range=list(_color_map.values())
                ), legend=None),
                detail="fold:N",
            )
        )
        _layers = [_fold_lines, _summary_lines]

    _chart = (
        alt.layer(*_layers)
        .properties(width="container", height=280, title="RRR: R² metrics vs rank")
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(alt, mo, pd, s1_grid):
    # Fold heatmap: rows=fold, cols=k, color=recovery_r2
    mo.stop(
        s1_grid is None,
        mo.callout(mo.md("grid_results.csv not found; fold heatmap unavailable."), kind="warn"),
    )
    _hm_df = s1_grid[["fold", "k", "recovery_r2"]].copy()
    _hm_df["fold"] = _hm_df["fold"].astype(str)
    _hm = (
        alt.Chart(_hm_df)
        .mark_rect()
        .encode(
            x=alt.X("k:O", title="Rank k"),
            y=alt.Y("fold:O", title="Fold"),
            color=alt.Color(
                "recovery_r2:Q",
                scale=alt.Scale(scheme="reds", reverse=True),
                legend=alt.Legend(title="recovery_r2"),
            ),
            tooltip=["fold:O", "k:O", alt.Tooltip("recovery_r2:Q", format=".4f")],
        )
        .properties(width="container", height=100, title="Recovery R² per fold × rank (lower = more sufficient)")
        .configure_view(strokeWidth=0)
    )
    mo.ui.altair_chart(_hm)
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
def _(mo, s1_summary, s1_threshold):
    # Sufficiency summary
    _suf = s1_summary[s1_summary["recovery_r2"] < s1_threshold.value]
    if len(_suf) > 0:
        _min_k = int(_suf["k"].min())
        mo.callout(
            mo.md(f"**Minimum sufficient rank (RRR):** k = **{_min_k}**  \n"
                  f"First rank where avg. `recovery_r2 < {s1_threshold.value:.2f}`"),
            kind="success",
        )
    else:
        mo.callout(
            mo.md(f"No rank achieves `recovery_r2 < {s1_threshold.value:.2f}`. "
                  "Try a higher threshold or a larger rank grid."),
            kind="warn",
        )
    return


# ---------------------------------------------------------------------------
# Section 2: SRRR — Rank, Lambda, and Direction Selection
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md("---\n## Section 2: SRRR — Rank, Lambda, and Direction Selection")
    return


@app.cell
def _(mo, srrr_options):
    mo.stop(
        len(srrr_options) == 0,
        mo.callout(mo.md("No SRRR results found. Run `scripts/run_srrr.py` first."), kind="warn"),
    )
    return


@app.cell
def _(mo, srrr_options):
    s2_source = mo.ui.dropdown(
        options=srrr_options,
        value=srrr_options[0] if srrr_options else None,
        label="SRRR source",
    )
    s2_lambda_mode = mo.ui.dropdown(
        options=["knee (auto)", "manual"],
        value="knee (auto)",
        label="Lambda selection",
    )
    s2_show_folds = mo.ui.checkbox(value=False, label="Show individual folds")
    mo.hstack([s2_source, s2_lambda_mode, s2_show_folds])
    return s2_lambda_mode, s2_show_folds, s2_source


@app.cell
def _(get_srrr_dir, mo, pd, s2_source):
    _unit_dir = get_srrr_dir(s2_source.value)
    _summary_path = _unit_dir / "summary.csv"
    _grid_path = _unit_dir / "grid_results.csv"
    _knee_path = _unit_dir / "knee_selection.csv"
    mo.stop(
        not _summary_path.exists(),
        mo.callout(mo.md(f"summary.csv not found at `{_summary_path}`."), kind="warn"),
    )
    s2_summary = pd.read_csv(_summary_path)
    s2_grid = pd.read_csv(_grid_path) if _grid_path.exists() else None
    s2_knee = pd.read_csv(_knee_path) if _knee_path.exists() else None
    return s2_grid, s2_knee, s2_summary


@app.cell
def _(mo, s2_summary):
    _lams = sorted(s2_summary["lam"].unique().tolist())
    s2_manual_lam = mo.ui.dropdown(
        options=[str(l) for l in _lams],
        value=str(_lams[0]) if _lams else None,
        label="Manual lambda (used when mode = manual)",
    )
    s2_manual_lam
    return (s2_manual_lam,)


@app.cell
def _(s2_knee, s2_lambda_mode, s2_manual_lam, s2_summary):
    # Resolve which lambda to use per k
    if s2_lambda_mode.value == "knee (auto)" and s2_knee is not None:
        _lam_per_k = dict(zip(s2_knee["k"], s2_knee["knee_lam"]))
        s2_active = s2_summary[
            s2_summary.apply(lambda r: abs(r["lam"] - _lam_per_k.get(r["k"], -1)) < 1e-12, axis=1)
        ].copy()
    else:
        _lam_val = float(s2_manual_lam.value) if s2_manual_lam.value else s2_summary["lam"].min()
        s2_active = s2_summary[s2_summary["lam"].apply(lambda l: abs(l - _lam_val) < 1e-12)].copy()
    return (s2_active,)


@app.cell
def _(alt, mo, s2_active):
    # Plot 1: sufficient_frac vs rank
    mo.stop(
        "sufficient_frac" not in s2_active.columns,
        mo.callout(mo.md("`sufficient_frac` column not found in SRRR summary."), kind="warn"),
    )
    _first_suf_k = s2_active[s2_active["sufficient_frac"] >= 1.0]["k"].min() if (s2_active["sufficient_frac"] >= 1.0).any() else None
    _bars = (
        alt.Chart(s2_active)
        .mark_bar(color="#60a5fa")
        .encode(
            x=alt.X("k:O", title="Rank k"),
            y=alt.Y("sufficient_frac:Q", title="Sufficient fraction", scale=alt.Scale(domain=[0, 1])),
            tooltip=["k:O", alt.Tooltip("sufficient_frac:Q", format=".3f"),
                     alt.Tooltip("lam:Q", format=".2e")],
        )
    )
    _ref = alt.Chart({"values": [{"y": 1.0}]}).mark_rule(color="#dc2626", strokeDash=[4, 2]).encode(y="y:Q")
    _chart = (
        alt.layer(_bars, _ref)
        .properties(width="container", height=200, title="SRRR: Sufficient fraction vs rank")
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    _suf_note = f"  \n**First fully sufficient rank:** k = **{int(_first_suf_k)}**" if _first_suf_k is not None else "  \n*No rank achieves full sufficiency in this lambda selection.*"
    mo.vstack([mo.ui.altair_chart(_chart), mo.md(_suf_note)])
    return


@app.cell
def _(alt, mo, pd, s2_active, s2_grid, s2_lambda_mode, s2_manual_lam, s2_show_folds):
    # Plot 2: R² metrics vs rank at selected lambda
    _cols = ["full_r2", "minimal_r2", "recovery_r2"]
    _long = s2_active.melt(
        id_vars="k", value_vars=[c for c in _cols if c in s2_active.columns],
        var_name="metric", value_name="r2",
    )
    _color_map = {
        "full_r2": "#9ca3af",
        "minimal_r2": "#3b82f6",
        "recovery_r2": "#ef4444",
    }
    _summary_lines = (
        alt.Chart(_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("k:Q", scale=alt.Scale(type="log"), title="Rank k"),
            y=alt.Y("r2:Q", title="R²"),
            color=alt.Color("metric:N", scale=alt.Scale(
                domain=list(_color_map.keys()), range=list(_color_map.values())
            )),
            tooltip=["k:Q", "metric:N", alt.Tooltip("r2:Q", format=".4f")],
        )
    )
    _layers = [_summary_lines]

    if s2_show_folds and s2_grid is not None:
        if s2_lambda_mode.value == "knee (auto)":
            # Filter fold data to match active lam per k
            _active_kl = set(zip(s2_active["k"], s2_active["lam"]))
            _fold_sub = s2_grid[s2_grid.apply(lambda r: (r["k"], r["lam"]) in _active_kl, axis=1)]
        else:
            _lv = float(s2_manual_lam.value)
            _fold_sub = s2_grid[s2_grid["lam"].apply(lambda l: abs(l - _lv) < 1e-12)]
        _fold_long = _fold_sub.melt(
            id_vars=["k", "fold"], value_vars=[c for c in _cols if c in _fold_sub.columns],
            var_name="metric", value_name="r2",
        )
        _fold_lines = (
            alt.Chart(_fold_long)
            .mark_line(opacity=0.2, strokeWidth=1)
            .encode(
                x=alt.X("k:Q", scale=alt.Scale(type="log")),
                y="r2:Q",
                color=alt.Color("metric:N", scale=alt.Scale(
                    domain=list(_color_map.keys()), range=list(_color_map.values())
                ), legend=None),
                detail="fold:N",
            )
        )
        _layers = [_fold_lines, _summary_lines]

    _chart = (
        alt.layer(*_layers)
        .properties(width="container", height=250, title="SRRR: R² metrics vs rank (at selected lambda)")
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(alt, mo, s2_active):
    # Plot 3: n_selected vs rank
    mo.stop(
        "n_selected" not in s2_active.columns,
        mo.callout(mo.md("`n_selected` column not found."), kind="warn"),
    )
    _chart = (
        alt.Chart(s2_active)
        .mark_line(point=True, color="#8b5cf6")
        .encode(
            x=alt.X("k:Q", scale=alt.Scale(type="log"), title="Rank k"),
            y=alt.Y("n_selected:Q", title="Avg. selected SVD directions"),
            tooltip=["k:Q", alt.Tooltip("n_selected:Q", format=".1f"),
                     alt.Tooltip("lam:Q", format=".2e")],
        )
        .properties(width="container", height=200, title="SRRR: Selected directions vs rank")
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(alt, mo, s2_lambda_mode, s2_summary):
    # Plot 4: Lambda sensitivity heatmap
    _hm_df = s2_summary[["k", "lam", "recovery_r2", "sufficient_frac"]].copy()
    _hm_df["lam_str"] = _hm_df["lam"].apply(lambda v: f"{v:.2e}")
    _hm_df["k_str"] = _hm_df["k"].astype(str)

    _s2_lam_heatmap_col = mo.ui.dropdown(
        options=["recovery_r2", "sufficient_frac"],
        value="sufficient_frac" if s2_lambda_mode.value == "knee (auto)" else "recovery_r2",
        label="Color by",
    )

    _hm = (
        alt.Chart(_hm_df)
        .mark_rect()
        .encode(
            x=alt.X("lam_str:O", sort=sorted(_hm_df["lam_str"].unique()), title="Lambda"),
            y=alt.Y("k_str:O", sort=[str(k) for k in sorted(_hm_df["k"].unique())], title="Rank k"),
            color=alt.Color(
                f"{'sufficient_frac' if s2_lambda_mode.value == 'knee (auto)' else 'recovery_r2'}:Q",
                scale=alt.Scale(scheme="blues"),
                legend=alt.Legend(title="Value"),
            ),
            tooltip=["k:Q", alt.Tooltip("lam:Q", format=".2e"),
                     alt.Tooltip("recovery_r2:Q", format=".4f"),
                     alt.Tooltip("sufficient_frac:Q", format=".3f")],
        )
        .properties(width="container", height=200, title="SRRR: Lambda sensitivity heatmap")
        .configure_view(strokeWidth=0)
    )
    mo.ui.altair_chart(_hm)
    return


@app.cell
def _(mo, s2_knee):
    # Knee selection table
    mo.stop(
        s2_knee is None,
        mo.callout(mo.md("knee_selection.csv not found."), kind="warn"),
    )
    _display_cols = [c for c in ["k", "knee_lam", "minimal_r2", "full_r2", "recovery_r2", "sufficient_frac", "n_selected"] if c in s2_knee.columns]
    mo.vstack([
        mo.md("### Knee Lambda Selection (per rank)"),
        mo.ui.table(s2_knee[_display_cols].round(4)),
    ])
    return


# ---------------------------------------------------------------------------
# Section 3: Direction Interpretability (Word Activations)
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md("---\n## Section 3: Direction Interpretability")
    return


@app.cell
def _(mo, rrr_options, srrr_options):
    _all_methods = (["RRR"] if rrr_options else []) + (["SRRR"] if srrr_options else [])
    mo.stop(
        len(_all_methods) == 0,
        mo.callout(mo.md("No RRR or SRRR results found."), kind="warn"),
    )
    s3_method = mo.ui.dropdown(
        options=_all_methods,
        value=_all_methods[0],
        label="Method",
    )
    s3_method
    return (s3_method,)


@app.cell
def _(mo, rrr_options, s3_method, srrr_options):
    _opts = rrr_options if s3_method.value == "RRR" else srrr_options
    s3_source = mo.ui.dropdown(
        options=_opts,
        value=_opts[0] if _opts else None,
        label="Source",
    )
    s3_source
    return (s3_source,)


@app.cell
def _(Path, get_rrr_dir, get_srrr_dir, mo, np, s3_method, s3_source):
    # Discover available (k, lam) pairs from directory structure
    _unit_dir = get_rrr_dir(s3_source.value) if s3_method.value == "RRR" else get_srrr_dir(s3_source.value)

    _k_lam_options = []
    if s3_method.value == "RRR":
        for _d in sorted(_unit_dir.glob("k*/fold_0")):
            _k_str = _d.parent.name
            if _k_str.startswith("k") and (_d.parent / "fold_0" / "A.npy").exists():
                try:
                    _k_lam_options.append(int(_k_str[1:]))
                except ValueError:
                    pass
        _k_lam_options = sorted(set(_k_lam_options))
        s3_k_options = [str(k) for k in _k_lam_options]
        s3_lam_options = ["N/A"]
    else:
        _seen_k = set()
        _seen_lam = set()
        for _d in sorted(_unit_dir.glob("k*_lam*/fold_0")):
            _parts = _d.parent.name.split("_lam")
            if len(_parts) == 2:
                try:
                    _k_val = int(_parts[0][1:])
                    _lam_val = float(_parts[1])
                    _seen_k.add(_k_val)
                    _seen_lam.add(_lam_val)
                except ValueError:
                    pass
        s3_k_options = [str(k) for k in sorted(_seen_k)]
        s3_lam_options = [f"{l:.2e}" for l in sorted(_seen_lam)]

    mo.stop(
        len(s3_k_options) == 0,
        mo.callout(mo.md(f"No saved A.npy files found under `{_unit_dir}`."), kind="warn"),
    )
    return s3_k_options, s3_lam_options


@app.cell
def _(mo, s3_k_options, s3_lam_options, s3_method):
    s3_k = mo.ui.dropdown(options=s3_k_options, value=s3_k_options[0], label="Rank k")
    s3_lam = mo.ui.dropdown(
        options=s3_lam_options,
        value=s3_lam_options[0],
        label="Lambda (SRRR only)",
    )
    s3_fold = mo.ui.dropdown(
        options=["fold_0", "fold_1"],
        value="fold_0",
        label="Fold",
    )
    s3_dir_idx = mo.ui.slider(start=0, stop=int(s3_k_options[-1]) - 1, value=0, step=1, label="Direction index")
    _row = [s3_k, s3_fold, s3_dir_idx]
    if s3_method.value == "SRRR":
        _row.insert(1, s3_lam)
    mo.hstack(_row)
    return s3_dir_idx, s3_fold, s3_k, s3_lam


@app.cell
def _(
    Path, aligned_embeddings, get_rrr_dir, get_srrr_dir, get_svd_dir,
    mo, np, s3_dir_idx, s3_fold, s3_k, s3_lam, s3_method, s3_source,
):
    _k_val = int(s3_k.value)
    _fold_name = s3_fold.value
    _dir_idx = min(s3_dir_idx.value, _k_val - 1)

    if s3_method.value == "RRR":
        _unit_dir = get_rrr_dir(s3_source.value)
        _fold_dir = _unit_dir / f"k{_k_val}" / _fold_name
        mo.stop(
            not (_fold_dir / "A.npy").exists(),
            mo.callout(mo.md(f"A.npy not found at `{_fold_dir}`."), kind="warn"),
        )
        _A = np.load(_fold_dir / "A.npy")  # (n_embed, k)
        s3_direction = _A[:, _dir_idx]  # (n_embed,)
    else:
        _lam_str = s3_lam.value
        _unit_dir = get_srrr_dir(s3_source.value)
        _fold_dir = _unit_dir / f"k{_k_val}_lam{_lam_str}" / _fold_name
        mo.stop(
            not (_fold_dir / "A.npy").exists(),
            mo.callout(mo.md(f"A.npy not found at `{_fold_dir}`."), kind="warn"),
        )
        _A = np.load(_fold_dir / "A.npy")           # (n_selected, k)
        _sel_idx = np.load(_fold_dir / "selected_idx.npy")  # (n_selected,)
        # Load corresponding fold's Vt from SVD decomposition
        _svd_path = get_svd_dir(s3_source.value) / "svd_decomposition.npz"
        mo.stop(
            not _svd_path.exists(),
            mo.callout(mo.md(f"SVD decomposition not found at `{_svd_path}`. Needed to map SRRR directions back to embedding space."), kind="warn"),
        )
        _svd_data = np.load(_svd_path)
        _fold_num = int(_fold_name.split("_")[1])
        _Vt = _svd_data[f"Vt_fold{_fold_num}"]  # (n_dirs, n_embed)
        s3_direction = _Vt[_sel_idx, :].T @ _A[:, _dir_idx]  # (n_embed,)

    s3_activations = (aligned_embeddings @ s3_direction).astype(np.float64)
    return (s3_activations,)


@app.cell
def _(alt, mo, np, pd, s3_activations, s3_dir_idx, s3_fold, s3_k, s3_method):
    _counts, _edges = np.histogram(s3_activations, bins=100)
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
        title=f"{s3_method.value} direction {s3_dir_idx.value} — k={s3_k.value}, {s3_fold.value}",
    ).configure_view(strokeWidth=0).configure_axis(grid=False)
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(get_example_display_controls, mo):
    s3_num_bins, s3_samp_per_bin, s3_ctx, s3_seed = get_example_display_controls()
    mo.hstack([s3_num_bins, s3_samp_per_bin, s3_ctx, s3_seed])
    return s3_ctx, s3_num_bins, s3_samp_per_bin, s3_seed


@app.cell
def _(
    all_words_epochs, df_word, get_mo_display_activations,
    s3_activations, s3_ctx, s3_num_bins, s3_samp_per_bin, s3_seed,
):
    get_mo_display_activations(
        s3_activations,
        df_word,
        all_words_epochs,
        s3_num_bins,
        s3_samp_per_bin,
        s3_ctx,
        s3_seed,
    )
    return


# ---------------------------------------------------------------------------
# Section 4: RRR vs SRRR Comparison
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md("---\n## Section 4: RRR vs SRRR Comparison")
    return


@app.cell
def _(mo, rrr_options, srrr_options):
    mo.stop(
        len(rrr_options) == 0 or len(srrr_options) == 0,
        mo.callout(
            mo.md("Both RRR and SRRR results are required for comparison."),
            kind="warn",
        ),
    )
    return


@app.cell
def _(mo, rrr_options, srrr_options):
    s4_rrr_source = mo.ui.dropdown(
        options=rrr_options, value=rrr_options[0], label="RRR source"
    )
    s4_srrr_source = mo.ui.dropdown(
        options=srrr_options, value=srrr_options[0], label="SRRR source"
    )
    mo.hstack([s4_rrr_source, s4_srrr_source])
    return s4_rrr_source, s4_srrr_source


@app.cell
def _(get_rrr_dir, get_srrr_dir, mo, pd, s4_rrr_source, s4_srrr_source):
    _rrr_path = get_rrr_dir(s4_rrr_source.value) / "summary.csv"
    _srrr_path = get_srrr_dir(s4_srrr_source.value) / "summary.csv"
    _knee_path = get_srrr_dir(s4_srrr_source.value) / "knee_selection.csv"
    mo.stop(
        not _rrr_path.exists() or not _srrr_path.exists(),
        mo.callout(mo.md("Could not load summary CSVs for selected sources."), kind="warn"),
    )
    s4_rrr_summary = pd.read_csv(_rrr_path)
    s4_srrr_summary = pd.read_csv(_srrr_path)
    s4_srrr_knee = pd.read_csv(_knee_path) if _knee_path.exists() else None
    return s4_rrr_summary, s4_srrr_knee, s4_srrr_summary


@app.cell
def _(mo, s4_srrr_summary):
    _lams = sorted(s4_srrr_summary["lam"].unique().tolist())
    s4_lam_for_cmp = mo.ui.dropdown(
        options=[f"{l:.2e}" for l in _lams],
        value=f"{_lams[0]:.2e}" if _lams else None,
        label="Lambda for SRRR comparison (overrides knee)",
    )
    s4_use_knee = mo.ui.checkbox(value=True, label="Use knee lambda per rank")
    mo.hstack([s4_use_knee, s4_lam_for_cmp])
    return s4_lam_for_cmp, s4_use_knee


@app.cell
def _(
    alt, mo, np, pd,
    s4_lam_for_cmp, s4_rrr_summary, s4_srrr_knee, s4_srrr_summary, s4_use_knee,
):
    # Build RRR series
    _rrr_df = s4_rrr_summary[["k", "recovery_r2"]].copy()
    _rrr_df["method"] = "RRR"

    # Build SRRR series at selected lambda
    if s4_use_knee.value and s4_srrr_knee is not None:
        _lam_per_k = dict(zip(s4_srrr_knee["k"], s4_srrr_knee["knee_lam"]))
        _srrr_filtered = s4_srrr_summary[
            s4_srrr_summary.apply(lambda r: abs(r["lam"] - _lam_per_k.get(r["k"], -1)) < 1e-12, axis=1)
        ].copy()
    else:
        _lv = float(s4_lam_for_cmp.value)
        _srrr_filtered = s4_srrr_summary[s4_srrr_summary["lam"].apply(lambda l: abs(l - _lv) < 1e-12)].copy()

    _srrr_df = _srrr_filtered[["k", "recovery_r2"]].copy()
    _srrr_df["method"] = "SRRR"

    _cmp_df = pd.concat([_rrr_df, _srrr_df], ignore_index=True)

    # Determine sufficient ranks
    _threshold = 0.05
    _rrr_suf = _rrr_df[_rrr_df["recovery_r2"] < _threshold]["k"].min() if (_rrr_df["recovery_r2"] < _threshold).any() else None
    _srrr_suf = _srrr_df[_srrr_df["recovery_r2"] < _threshold]["k"].min() if (_srrr_df["recovery_r2"] < _threshold).any() else None

    _lines = (
        alt.Chart(_cmp_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("k:Q", scale=alt.Scale(type="log"), title="Rank k"),
            y=alt.Y("recovery_r2:Q", title="Recovery R²"),
            color=alt.Color("method:N", scale=alt.Scale(
                domain=["RRR", "SRRR"], range=["#3b82f6", "#f59e0b"]
            )),
            tooltip=["method:N", "k:Q", alt.Tooltip("recovery_r2:Q", format=".4f")],
        )
    )

    _vlines_data = []
    if _rrr_suf is not None:
        _vlines_data.append({"k": float(_rrr_suf), "method": "RRR"})
    if _srrr_suf is not None:
        _vlines_data.append({"k": float(_srrr_suf), "method": "SRRR"})

    _layers = [_lines]
    if _vlines_data:
        _vlines = (
            alt.Chart(pd.DataFrame(_vlines_data))
            .mark_rule(strokeDash=[4, 2], strokeWidth=1.5)
            .encode(
                x="k:Q",
                color=alt.Color("method:N", scale=alt.Scale(
                    domain=["RRR", "SRRR"], range=["#3b82f6", "#f59e0b"]
                ), legend=None),
            )
        )
        _layers.append(_vlines)

    _chart = (
        alt.layer(*_layers)
        .properties(width="container", height=280, title="RRR vs SRRR: Recovery R² vs rank (lower = more sufficient)")
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    # Summary callout
    _lines_md = []
    if _rrr_suf is not None:
        _lines_md.append(f"- **RRR sufficient rank:** k = {int(_rrr_suf)}")
    else:
        _lines_md.append("- **RRR sufficient rank:** not reached (recovery_r2 < 0.05)")
    if _srrr_suf is not None:
        _n_sel = _srrr_filtered[_srrr_filtered["k"] == _srrr_suf]["n_selected"].values
        _n_str = f", avg. {_n_sel[0]:.0f} SVD directions selected" if len(_n_sel) > 0 else ""
        _lines_md.append(f"- **SRRR sufficient rank:** k = {int(_srrr_suf)}{_n_str}")
    else:
        _lines_md.append("- **SRRR sufficient rank:** not reached (recovery_r2 < 0.05)")

    mo.vstack([
        mo.ui.altair_chart(_chart),
        mo.callout(mo.md("\n".join(_lines_md)), kind="info"),
    ])
    return


if __name__ == "__main__":
    app.run()
