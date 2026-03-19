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
    from scipy.stats import kurtosis as sp_kurtosis, entropy as sp_entropy, skew as sp_skew
    from pathlib import Path
    return Path, alt, mo, np, pd, sp_entropy, sp_kurtosis, sp_skew


@app.cell
def _():
    import sys
    import os
    # Find repo root: walk up from CWD until we find a directory containing utils/
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
    mo.md("# SVD Interpretability Explorer")
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

    source_options = []

    _cross = _rr / "svd" / "svd_decomposition.npz"
    if _cross.exists():
        source_options.append("cross_subject")

    _ps_dir = _rr / "per_subject_svd" / "subjects"
    if _ps_dir.exists():
        for _p in sorted(_ps_dir.glob("sub_*/svd_decomposition.npz")):
            source_options.append(f"per_subject/{_p.parent.name}")

    _pr_dir = _rr / "per_region_svd" / "regions"
    if _pr_dir.exists():
        for _p in sorted(_pr_dir.glob("*/svd_decomposition.npz")):
            source_options.append(f"per_region/{_p.parent.name}")

    mo.stop(
        len(source_options) == 0,
        mo.callout(mo.md(f"No SVD results found under `{_rr}`."), kind="warn"),
    )
    return (source_options,)


@app.cell
def _(data_root_input, layer_input, load_embeddings, np):
    aligned_embeddings, df_word = load_embeddings(
        data_root_input.value, int(layer_input.value)
    )

    class AllWords:
        selection = np.arange(len(df_word))

    all_words_epochs = AllWords()
    return aligned_embeddings, all_words_epochs, df_word


@app.cell
def _(Path, np, results_root_input):
    def load_svd(source):
        """Load SVD decomposition .npz for a given source string."""
        rr = Path(results_root_input.value)
        if source == "cross_subject":
            npz_path = rr / "svd" / "svd_decomposition.npz"
        elif source.startswith("per_subject/"):
            sub_name = source.split("/", 1)[1]
            npz_path = (
                rr / "per_subject_svd" / "subjects" / sub_name / "svd_decomposition.npz"
            )
        elif source.startswith("per_region/"):
            region_name = source.split("/", 1)[1]
            npz_path = (
                rr / "per_region_svd" / "regions" / region_name / "svd_decomposition.npz"
            )
        else:
            raise ValueError(f"Unknown source: {source}")

        data = np.load(npz_path)
        n_folds = sum(1 for k in data.keys() if k.startswith("Vt_fold"))
        result = {"n_folds": n_folds}
        for _f in range(n_folds):
            for _prefix in ("Vt", "S", "U", "alignments"):
                _key = f"{_prefix}_fold{_f}"
                if _key in data:
                    result[_key] = data[_key]
        return result

    def get_vt(svd_data, fold_selection):
        """Resolve fold selection string to (Vt, S) arrays."""
        if fold_selection == "fold 0":
            return svd_data["Vt_fold0"], svd_data["S_fold0"]
        elif fold_selection == "fold 1":
            return svd_data["Vt_fold1"], svd_data["S_fold1"]
        elif fold_selection == "average":
            Vt0, S0 = svd_data["Vt_fold0"], svd_data["S_fold0"]
            Vt1, S1 = svd_data["Vt_fold1"], svd_data["S_fold1"]
            n = min(len(Vt0), len(Vt1))
            Vt0, Vt1 = Vt0[:n], Vt1[:n]
            S0, S1 = S0[:n], S1[:n]
            # Sign-flip fold1 rows where dot product with fold0 row < 0
            dots = np.einsum("ij,ij->i", Vt0, Vt1)
            signs = np.where(dots >= 0, 1.0, -1.0)
            Vt1_aligned = Vt1 * signs[:, np.newaxis]

            def _norm(V):
                norms = np.linalg.norm(V, axis=1, keepdims=True)
                return V / np.where(norms > 0, norms, 1.0)

            Vt_avg = _norm((_norm(Vt0) + _norm(Vt1_aligned)) / 2.0)
            S_avg = (S0 + S1) / 2.0
            return Vt_avg, S_avg
        else:
            raise ValueError(f"Unknown fold_selection: {fold_selection}")

    return get_vt, load_svd


# ---------------------------------------------------------------------------
# Section 1: Direction Interpretability
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md("---\n## Section 1: Direction Interpretability")
    return


@app.cell
def _(mo, source_options):
    s1_source = mo.ui.dropdown(
        options=source_options,
        value=source_options[0],
        label="Source",
    )
    s1_fold = mo.ui.dropdown(
        options=["fold 0", "fold 1", "average"],
        value="fold 0",
        label="Fold",
    )
    s1_dir = mo.ui.slider(start=0, stop=99, value=0, step=1, label="Direction index")
    mo.hstack([s1_source, s1_fold, s1_dir])
    return s1_dir, s1_fold, s1_source


@app.cell
def _(load_svd, s1_source):
    s1_svd = load_svd(s1_source.value)
    return (s1_svd,)


@app.cell
def _(aligned_embeddings, get_vt, np, s1_dir, s1_fold, s1_svd):
    _s1_Vt, _s1_S = get_vt(s1_svd, s1_fold.value)
    _s1_idx = min(s1_dir.value, len(_s1_Vt) - 1)
    s1_activations = (aligned_embeddings @ _s1_Vt[_s1_idx]).astype(np.float64)
    s1_singular_value = float(_s1_S[_s1_idx])
    return s1_activations, s1_singular_value


@app.cell
def _(alt, mo, np, pd, s1_activations, s1_dir, s1_singular_value):
    _fine_counts, _fine_edges = np.histogram(s1_activations, bins=100)
    _hist_df = pd.DataFrame({
        "x1": _fine_edges[:-1],
        "x2": _fine_edges[1:],
        "count": _fine_counts,
    })
    _bars = (
        alt.Chart(_hist_df)
        .mark_bar(color="#93c5fd", stroke="#bfdbfe", strokeWidth=0.5)
        .encode(
            x=alt.X(
                "x1:Q",
                title="Activation",
                scale=alt.Scale(domain=[float(_fine_edges[0]), float(_fine_edges[-1])]),
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
    _chart = _bars.properties(
        width="container",
        height=220,
        title=f"Direction {s1_dir.value} — singular value {s1_singular_value:.4f}",
    ).configure_view(strokeWidth=0).configure_axis(grid=False)
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(get_example_display_controls, mo):
    s1_num_bins, s1_samp_per_bin, s1_ctx, s1_seed = get_example_display_controls()
    mo.hstack([s1_num_bins, s1_samp_per_bin, s1_ctx, s1_seed])
    return s1_ctx, s1_num_bins, s1_samp_per_bin, s1_seed


@app.cell
def _(
    all_words_epochs,
    df_word,
    get_mo_display_activations,
    s1_activations,
    s1_ctx,
    s1_num_bins,
    s1_samp_per_bin,
    s1_seed,
):
    get_mo_display_activations(
        s1_activations,
        df_word,
        all_words_epochs,
        s1_num_bins,
        s1_samp_per_bin,
        s1_ctx,
        s1_seed,
    )
    return


# ---------------------------------------------------------------------------
# Section 2: Fold Consistency (single source)
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md("---\n## Section 2: Fold Consistency")
    return


@app.cell
def _(mo, source_options):
    s2_source = mo.ui.dropdown(
        options=source_options,
        value=source_options[0],
        label="Source",
    )
    s2_ndirs = mo.ui.slider(start=5, stop=50, value=20, step=5, label="Top-k directions")
    mo.hstack([s2_source, s2_ndirs])
    return s2_ndirs, s2_source


@app.cell
def _(load_svd, mo, s2_source):
    s2_svd = load_svd(s2_source.value)
    mo.stop(
        s2_svd["n_folds"] < 2,
        mo.callout(
            mo.md(f"Source `{s2_source.value}` has only {s2_svd['n_folds']} fold(s); need ≥ 2."),
            kind="warn",
        ),
    )
    return (s2_svd,)


@app.cell
def _(np, s2_ndirs, s2_svd):
    _Vt0 = s2_svd["Vt_fold0"]
    _Vt1 = s2_svd["Vt_fold1"]
    _S0 = s2_svd["S_fold0"]
    _S1 = s2_svd["S_fold1"]

    _n = min(s2_ndirs.value, len(_Vt0), len(_Vt1))
    _Vt0_k = _Vt0[:_n]
    _Vt1_k = _Vt1[:_n]

    s2_cos_sim_matrix = _Vt0_k @ _Vt1_k.T  # (k, k) — cosine sims (unit rows)
    s2_diag_cos = np.abs(np.diag(s2_cos_sim_matrix))
    s2_S0 = _S0[:_n]
    s2_S1 = _S1[:_n]
    s2_n = _n
    return s2_S0, s2_S1, s2_cos_sim_matrix, s2_diag_cos, s2_n


@app.cell
def _(alt, mo, np, pd, s2_S0, s2_S1, s2_cos_sim_matrix, s2_diag_cos, s2_n):
    # --- Tab A: cosine heatmap ---
    _rows = []
    for _i in range(s2_n):
        for _j in range(s2_n):
            _rows.append({"fold0_dir": _i, "fold1_dir": _j, "cos_sim": float(s2_cos_sim_matrix[_i, _j])})
    _hm_df = pd.DataFrame(_rows)
    _hm_chart = (
        alt.Chart(_hm_df)
        .mark_rect()
        .encode(
            x=alt.X("fold0_dir:O", title="Fold 0 direction"),
            y=alt.Y("fold1_dir:O", title="Fold 1 direction"),
            color=alt.Color(
                "cos_sim:Q",
                title="Cosine sim",
                scale=alt.Scale(scheme="rdbu", domain=[-1, 1]),
            ),
            tooltip=[
                alt.Tooltip("fold0_dir:O", title="Fold 0"),
                alt.Tooltip("fold1_dir:O", title="Fold 1"),
                alt.Tooltip("cos_sim:Q", title="Cosine sim", format=".3f"),
            ],
        )
        .properties(
            width="container", height=400, title="Cosine similarity: fold 0 vs fold 1"
        )
        .configure_view(strokeWidth=0)
    )

    # --- Tab B: SV spectrum ---
    _sv_rows = []
    for _i in range(s2_n):
        _sv_rows.append({"direction": _i, "value": float(s2_S0[_i]), "fold": "fold 0"})
        _sv_rows.append({"direction": _i, "value": float(s2_S1[_i]), "fold": "fold 1"})
    _sv_df = pd.DataFrame(_sv_rows)
    _sv_chart = (
        alt.Chart(_sv_df)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("direction:O", title="Direction"),
            y=alt.Y("value:Q", title="Singular value"),
            color=alt.Color("fold:N", title="Fold"),
            xOffset="fold:N",
            tooltip=[
                alt.Tooltip("direction:O"),
                alt.Tooltip("fold:N"),
                alt.Tooltip("value:Q", title="Singular value", format=".4f"),
            ],
        )
        .properties(width="container", height=300, title="Singular value spectrum")
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    # --- Tab C: consistency table ---
    _tbl_df = pd.DataFrame({
        "direction": np.arange(s2_n),
        "abs_cos_sim": s2_diag_cos,
        "S_fold0": s2_S0,
        "S_fold1": s2_S1,
    })
    _tbl_df["abs_cos_sim"] = _tbl_df["abs_cos_sim"].round(4)
    _tbl_df["S_fold0"] = _tbl_df["S_fold0"].round(4)
    _tbl_df["S_fold1"] = _tbl_df["S_fold1"].round(4)
    _consistency_tbl = mo.ui.table(_tbl_df)

    mo.tabs({
        "Cosine heatmap": mo.ui.altair_chart(_hm_chart),
        "SV spectrum": mo.ui.altair_chart(_sv_chart),
        "Consistency table": _consistency_tbl,
    })
    return


# ---------------------------------------------------------------------------
# Section 3: Cross-Source Comparison
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md("---\n## Section 3: Cross-Source Comparison")
    return


@app.cell
def _(mo, source_options):
    _fold_opts = ["fold 0", "fold 1", "average"]
    s3_source_a = mo.ui.dropdown(options=source_options, value=source_options[0], label="Source A")
    s3_fold_a = mo.ui.dropdown(options=_fold_opts, value="fold 0", label="Fold A")
    s3_source_b = mo.ui.dropdown(options=source_options, value=source_options[0], label="Source B")
    s3_fold_b = mo.ui.dropdown(options=_fold_opts, value="fold 0", label="Fold B")
    s3_topk = mo.ui.slider(start=5, stop=50, value=20, step=5, label="Top-k")
    mo.vstack([
        mo.hstack([s3_source_a, s3_fold_a, s3_source_b, s3_fold_b]),
        s3_topk,
    ])
    return s3_fold_a, s3_fold_b, s3_source_a, s3_source_b, s3_topk


@app.cell
def _(
    aligned_embeddings,
    get_vt,
    load_svd,
    mo,
    np,
    s3_fold_a,
    s3_fold_b,
    s3_source_a,
    s3_source_b,
    s3_topk,
):
    _svd_a = load_svd(s3_source_a.value)
    _svd_b = load_svd(s3_source_b.value)
    _Vt_a_full, _S_a_full = get_vt(_svd_a, s3_fold_a.value)
    _Vt_b_full, _S_b_full = get_vt(_svd_b, s3_fold_b.value)

    _k_req = s3_topk.value
    _k = min(_k_req, len(_Vt_a_full), len(_Vt_b_full))
    s3_clamped = _k < _k_req

    s3_Vt_A = _Vt_a_full[:_k]
    s3_Vt_B = _Vt_b_full[:_k]
    s3_S_A = _S_a_full[:_k]
    s3_S_B = _S_b_full[:_k]
    s3_k = _k

    _clamp_msg = (
        mo.callout(
            mo.md(f"Top-k clamped to {_k} (sources have {len(_Vt_a_full)} and {len(_Vt_b_full)} directions)."),
            kind="info",
        )
        if s3_clamped
        else mo.md("")
    )
    _clamp_msg

    # Cosine similarity matrix (k x k)
    s3_cos_sim_AB = s3_Vt_A @ s3_Vt_B.T

    # Best matching pair (max abs cosine sim)
    _abs_cs = np.abs(s3_cos_sim_AB)
    _best_flat = int(np.argmax(_abs_cs))
    s3_best_i = _best_flat // _k
    s3_best_j = _best_flat % _k
    s3_acts_best_A = (aligned_embeddings @ s3_Vt_A[s3_best_i]).astype(np.float64)
    s3_acts_best_B = (aligned_embeddings @ s3_Vt_B[s3_best_j]).astype(np.float64)
    s3_best_cos = float(s3_cos_sim_AB[s3_best_i, s3_best_j])

    # Activation correlation matrix (k x k Pearson)
    _acts_A = (aligned_embeddings @ s3_Vt_A.T).astype(np.float64)  # (n_words, k)
    _acts_B = (aligned_embeddings @ s3_Vt_B.T).astype(np.float64)
    _A_z = (_acts_A - _acts_A.mean(0)) / np.where(_acts_A.std(0) > 0, _acts_A.std(0), 1.0)
    _B_z = (_acts_B - _acts_B.mean(0)) / np.where(_acts_B.std(0) > 0, _acts_B.std(0), 1.0)
    s3_act_corr_AB = (_A_z.T @ _B_z) / len(_A_z)  # (k, k) Pearson
    return (
        s3_acts_best_A,
        s3_acts_best_B,
        s3_act_corr_AB,
        s3_best_cos,
        s3_best_i,
        s3_best_j,
        s3_cos_sim_AB,
        s3_k,
    )


@app.cell
def _(
    alt,
    mo,
    np,
    pd,
    s3_act_corr_AB,
    s3_acts_best_A,
    s3_acts_best_B,
    s3_best_cos,
    s3_best_i,
    s3_best_j,
    s3_cos_sim_AB,
    s3_fold_a,
    s3_fold_b,
    s3_k,
    s3_source_a,
    s3_source_b,
):
    # --- Tab A: cosine heatmap ---
    _rows_cs = []
    for _i in range(s3_k):
        for _j in range(s3_k):
            _rows_cs.append({
                "dir_A": _i, "dir_B": _j,
                "cos_sim": float(s3_cos_sim_AB[_i, _j]),
            })
    _cs_df = pd.DataFrame(_rows_cs)
    _cs_chart = (
        alt.Chart(_cs_df)
        .mark_rect()
        .encode(
            x=alt.X("dir_A:O", title="Source A direction"),
            y=alt.Y("dir_B:O", title="Source B direction"),
            color=alt.Color(
                "cos_sim:Q",
                title="Cosine sim",
                scale=alt.Scale(scheme="rdbu", domain=[-1, 1]),
            ),
            tooltip=[
                alt.Tooltip("dir_A:O", title="Dir A"),
                alt.Tooltip("dir_B:O", title="Dir B"),
                alt.Tooltip("cos_sim:Q", title="Cosine sim", format=".3f"),
            ],
        )
        .properties(
            width="container",
            height=400,
            title=f"Cosine similarity: {s3_source_a.value} ({s3_fold_a.value}) vs {s3_source_b.value} ({s3_fold_b.value})",
        )
        .configure_view(strokeWidth=0)
    )

    # --- Tab B: activation scatter (best-matching pair) ---
    _scatter_df = pd.DataFrame({
        "activation_A": s3_acts_best_A,
        "activation_B": s3_acts_best_B,
    })
    _scatter_base = alt.Chart(_scatter_df).encode(
        x=alt.X("activation_A:Q", title=f"Source A dir {s3_best_i}"),
        y=alt.Y("activation_B:Q", title=f"Source B dir {s3_best_j}"),
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
            title=f"Best-matching pair: A[{s3_best_i}] vs B[{s3_best_j}] (cosine sim={s3_best_cos:.3f})",
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    # --- Tab C: top pairs table ---
    _tp_rows = []
    for _i in range(s3_k):
        _best_j_for_i = int(np.argmax(np.abs(s3_cos_sim_AB[_i])))
        _tp_rows.append({
            "dir_A": _i,
            "best_dir_B": _best_j_for_i,
            "cos_sim": round(float(s3_cos_sim_AB[_i, _best_j_for_i]), 4),
            "abs_cos_sim": round(float(np.abs(s3_cos_sim_AB[_i, _best_j_for_i])), 4),
        })
    _tp_df = pd.DataFrame(_tp_rows).sort_values("abs_cos_sim", ascending=False)
    _tp_tbl = mo.ui.table(_tp_df)

    # --- Tab D: activation correlation heatmap ---
    _rows_ac = []
    for _i in range(s3_k):
        for _j in range(s3_k):
            _rows_ac.append({
                "dir_A": _i, "dir_B": _j,
                "pearson_r": float(s3_act_corr_AB[_i, _j]),
            })
    _ac_df = pd.DataFrame(_rows_ac)
    _ac_chart = (
        alt.Chart(_ac_df)
        .mark_rect()
        .encode(
            x=alt.X("dir_A:O", title="Source A direction"),
            y=alt.Y("dir_B:O", title="Source B direction"),
            color=alt.Color(
                "pearson_r:Q",
                title="Pearson r",
                scale=alt.Scale(scheme="rdbu", domain=[-1, 1]),
            ),
            tooltip=[
                alt.Tooltip("dir_A:O", title="Dir A"),
                alt.Tooltip("dir_B:O", title="Dir B"),
                alt.Tooltip("pearson_r:Q", title="Pearson r", format=".3f"),
            ],
        )
        .properties(
            width="container",
            height=400,
            title="Activation Pearson correlations",
        )
        .configure_view(strokeWidth=0)
    )

    mo.tabs({
        "A. Cosine heatmap": mo.ui.altair_chart(_cs_chart),
        "B. Activation scatter": mo.ui.altair_chart(_scatter_chart),
        "C. Top pairs": _tp_tbl,
        "D. Activation correlations": mo.ui.altair_chart(_ac_chart),
    })
    return


# ---------------------------------------------------------------------------
# Section 4: Additional Analyses
# ---------------------------------------------------------------------------


@app.cell
def _(mo):
    mo.md("---\n## Section 4: Additional Analyses")
    return


# --- 4A: Activation Distribution Statistics ---


@app.cell
def _(mo):
    mo.md("### 4A: Activation Distribution Statistics")
    return


@app.cell
def _(mo, source_options):
    s4a_source = mo.ui.dropdown(options=source_options, value=source_options[0], label="Source")
    s4a_fold = mo.ui.dropdown(
        options=["fold 0", "fold 1", "average"], value="fold 0", label="Fold"
    )
    s4a_topk = mo.ui.slider(start=5, stop=50, value=20, step=5, label="Top-k directions")
    mo.hstack([s4a_source, s4a_fold, s4a_topk])
    return s4a_fold, s4a_source, s4a_topk


@app.cell
def _(
    aligned_embeddings,
    get_vt,
    load_svd,
    np,
    pd,
    s4a_fold,
    s4a_source,
    s4a_topk,
    sp_entropy,
    sp_kurtosis,
    sp_skew,
):
    _svd = load_svd(s4a_source.value)
    _Vt, _S = get_vt(_svd, s4a_fold.value)
    _k = min(s4a_topk.value, len(_Vt))
    _Vt_k = _Vt[:_k]
    _S_k = _S[:_k]

    # Compute stats per direction
    _acts_all = (aligned_embeddings @ _Vt_k.T).astype(np.float64)  # (n_words, k)

    _stats_rows = []
    for _i in range(_k):
        _a = _acts_all[:, _i]
        # Entropy of binned distribution (50 bins)
        _counts, _ = np.histogram(_a, bins=50)
        _probs = _counts / _counts.sum()
        _ent = float(sp_entropy(_probs + 1e-12))
        _stats_rows.append({
            "direction": _i,
            "singular_value": round(float(_S_k[_i]), 4),
            "mean": round(float(_a.mean()), 4),
            "std": round(float(_a.std()), 4),
            "kurtosis": round(float(sp_kurtosis(_a)), 4),
            "skewness": round(float(sp_skew(_a)), 4),
            "entropy": round(_ent, 4),
        })
    s4a_stats_df = pd.DataFrame(_stats_rows)
    return (s4a_stats_df,)


@app.cell
def _(alt, mo, pd, s4a_stats_df):
    _kurt_chart = (
        alt.Chart(s4a_stats_df)
        .mark_bar(color="#a78bfa")
        .encode(
            x=alt.X("direction:O", title="Direction"),
            y=alt.Y("kurtosis:Q", title="Kurtosis (excess)"),
            tooltip=[
                alt.Tooltip("direction:O"),
                alt.Tooltip("kurtosis:Q", format=".3f"),
                alt.Tooltip("singular_value:Q", title="Singular value", format=".4f"),
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

    _display_df = s4a_stats_df.copy()
    _tbl = mo.ui.table(_display_df)

    mo.vstack([
        mo.ui.altair_chart(_kurt_chart),
        _tbl,
    ])
    return


# --- 4B: Cross-Region Subspace Overlap ---


@app.cell
def _(mo):
    mo.md("### 4B: Cross-Region Subspace Overlap")
    return


@app.cell
def _(get_vt, load_svd, mo, np, pd, source_options):
    _region_sources = [s for s in source_options if s.startswith("per_region/")]

    if len(_region_sources) < 2:
        mo.callout(
            mo.md(f"Need ≥ 2 per-region sources; found {len(_region_sources)}."),
            kind="warn",
        )
    else:
        _k_sub = 20
        _regions = [s.split("/", 1)[1] for s in _region_sources]
        _Vts = {}
        for _src in _region_sources:
            try:
                _svd = load_svd(_src)
                _Vt, _ = get_vt(_svd, "fold 0")
                _Vts[_src.split("/", 1)[1]] = _Vt[:_k_sub]
            except Exception:
                pass

        _valid_regions = list(_Vts.keys())
        _overlap_rows = []
        for _ra in _valid_regions:
            for _rb in _valid_regions:
                _Vta = _Vts[_ra]
                _Vtb = _Vts[_rb]
                _k_use = min(len(_Vta), len(_Vtb))
                _, _sigma, _ = np.linalg.svd(_Vta[:_k_use] @ _Vtb[:_k_use].T)
                _sigma = np.clip(_sigma, -1, 1)
                _angles = np.arccos(_sigma) * 180 / np.pi
                _mean_angle = float(_angles.mean())
                _overlap_rows.append({"region_A": _ra, "region_B": _rb, "mean_angle_deg": round(_mean_angle, 2)})

        _overlap_df = pd.DataFrame(_overlap_rows)
        _hm = (
            alt.Chart(_overlap_df)
            .mark_rect()
            .encode(
                x=alt.X("region_A:N", title="Region A"),
                y=alt.Y("region_B:N", title="Region B"),
                color=alt.Color(
                    "mean_angle_deg:Q",
                    title="Mean principal angle (°)",
                    scale=alt.Scale(scheme="viridis"),
                ),
                tooltip=[
                    alt.Tooltip("region_A:N"),
                    alt.Tooltip("region_B:N"),
                    alt.Tooltip("mean_angle_deg:Q", title="Mean angle (°)", format=".1f"),
                ],
            )
            .properties(
                width="container",
                height=400,
                title=f"Cross-region subspace overlap (top-{_k_sub} directions, fold 0)",
            )
            .configure_view(strokeWidth=0)
        )
        mo.ui.altair_chart(_hm)
    return


# --- 4C: Permutation-Based Significance ---


@app.cell
def _(mo):
    mo.md("### 4C: Permutation-Based Significance of Cosine Similarities")
    return


@app.cell
def _(mo, source_options):
    _fold_opts_4c = ["fold 0", "fold 1", "average"]
    s4c_source_a = mo.ui.dropdown(options=source_options, value=source_options[0], label="Source A")
    s4c_fold_a = mo.ui.dropdown(options=_fold_opts_4c, value="fold 0", label="Fold A")
    s4c_source_b = mo.ui.dropdown(options=source_options, value=source_options[0], label="Source B")
    s4c_fold_b = mo.ui.dropdown(options=_fold_opts_4c, value="fold 0", label="Fold B")
    s4c_topk = mo.ui.slider(start=5, stop=30, value=10, step=5, label="Top-k")
    s4c_n_perm = mo.ui.slider(start=100, stop=1000, value=500, step=100, label="N permutations")
    mo.vstack([
        mo.hstack([s4c_source_a, s4c_fold_a, s4c_source_b, s4c_fold_b]),
        mo.hstack([s4c_topk, s4c_n_perm]),
    ])
    return s4c_fold_a, s4c_fold_b, s4c_n_perm, s4c_source_a, s4c_source_b, s4c_topk


@app.cell
def _(
    alt,
    get_vt,
    load_svd,
    mo,
    np,
    pd,
    s4c_fold_a,
    s4c_fold_b,
    s4c_n_perm,
    s4c_source_a,
    s4c_source_b,
    s4c_topk,
):
    _svd_a = load_svd(s4c_source_a.value)
    _svd_b = load_svd(s4c_source_b.value)
    _Vt_a, _ = get_vt(_svd_a, s4c_fold_a.value)
    _Vt_b, _ = get_vt(_svd_b, s4c_fold_b.value)
    _k = min(s4c_topk.value, len(_Vt_a), len(_Vt_b))
    _Vt_a_k = _Vt_a[:_k]
    _Vt_b_k = _Vt_b[:_k]

    _obs_diag = np.abs(np.diag(_Vt_a_k @ _Vt_b_k.T))  # (k,) observed diagonal cosine sims

    # Permutation null distribution
    _rng = np.random.default_rng(42)
    _n_perm = int(s4c_n_perm.value)
    _null = np.zeros((_n_perm, _k))
    for _perm_i in range(_n_perm):
        _perm_idx = _rng.permutation(_k)
        _null[_perm_i] = np.abs(np.diag(_Vt_a_k @ _Vt_b_k[_perm_idx].T))

    # One-sided p-values: proportion of null >= observed
    _p_vals = ((_null >= _obs_diag[np.newaxis, :]).sum(0) + 1) / (_n_perm + 1)

    # BH FDR correction
    _sorted_idx = np.argsort(_p_vals)
    _ranks = np.empty_like(_sorted_idx)
    _ranks[_sorted_idx] = np.arange(1, _k + 1)
    _p_fdr = np.minimum(1.0, _p_vals * _k / _ranks)
    # Make monotone
    _p_fdr_mono = np.minimum.accumulate(_p_fdr[_sorted_idx][::-1])[::-1]
    _p_fdr_final = np.empty(_k)
    _p_fdr_final[_sorted_idx] = _p_fdr_mono

    _sig_df = pd.DataFrame({
        "direction": np.arange(_k),
        "obs_cos_sim": np.round(_obs_diag, 4),
        "p_val": np.round(_p_vals, 5),
        "neg_log10_p_fdr": np.round(-np.log10(_p_fdr_final + 1e-10), 3),
    })

    _threshold = -np.log10(0.05)
    _scatter = (
        alt.Chart(_sig_df)
        .mark_point(size=80, filled=True)
        .encode(
            x=alt.X("direction:O", title="Direction"),
            y=alt.Y("neg_log10_p_fdr:Q", title="-log10(p_FDR)"),
            color=alt.Color(
                "neg_log10_p_fdr:Q",
                scale=alt.Scale(scheme="reds"),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("direction:O"),
                alt.Tooltip("obs_cos_sim:Q", title="Observed |cosine|", format=".3f"),
                alt.Tooltip("neg_log10_p_fdr:Q", title="-log10(p_FDR)", format=".2f"),
            ],
        )
    )
    _threshold_rule = (
        alt.Chart(pd.DataFrame({"y": [_threshold]}))
        .mark_rule(color="#ef4444", strokeDash=[4, 4])
        .encode(y="y:Q")
    )
    _perm_chart = (
        (_scatter + _threshold_rule)
        .properties(
            width="container",
            height=300,
            title=f"Permutation significance (N={_n_perm}, dashed = p_FDR=0.05)",
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    mo.vstack([
        mo.ui.altair_chart(_perm_chart),
        mo.ui.table(_sig_df),
    ])
    return


# --- 4D: Subspace Projection (Principal Angles) ---


@app.cell
def _(mo):
    mo.md("### 4D: Subspace Projection (Principal Angles)")
    return


@app.cell
def _(mo, source_options):
    _fold_opts_4d = ["fold 0", "fold 1", "average"]
    s4d_source_a = mo.ui.dropdown(options=source_options, value=source_options[0], label="Source A")
    s4d_fold_a = mo.ui.dropdown(options=_fold_opts_4d, value="fold 0", label="Fold A")
    s4d_source_b = mo.ui.dropdown(options=source_options, value=source_options[0], label="Source B")
    s4d_fold_b = mo.ui.dropdown(options=_fold_opts_4d, value="fold 0", label="Fold B")
    s4d_topk = mo.ui.slider(start=5, stop=50, value=20, step=5, label="Top-k")
    mo.vstack([
        mo.hstack([s4d_source_a, s4d_fold_a, s4d_source_b, s4d_fold_b]),
        s4d_topk,
    ])
    return s4d_fold_a, s4d_fold_b, s4d_source_a, s4d_source_b, s4d_topk


@app.cell
def _(
    alt,
    get_vt,
    load_svd,
    mo,
    np,
    pd,
    s4d_fold_a,
    s4d_fold_b,
    s4d_source_a,
    s4d_source_b,
    s4d_topk,
):
    _svd_a = load_svd(s4d_source_a.value)
    _svd_b = load_svd(s4d_source_b.value)
    _Vt_a, _ = get_vt(_svd_a, s4d_fold_a.value)
    _Vt_b, _ = get_vt(_svd_b, s4d_fold_b.value)
    _k = min(s4d_topk.value, len(_Vt_a), len(_Vt_b))
    _Vt_a_k = _Vt_a[:_k]
    _Vt_b_k = _Vt_b[:_k]

    _, _sigma, _ = np.linalg.svd(_Vt_a_k @ _Vt_b_k.T)
    _sigma = np.clip(_sigma, -1, 1)
    _principal_angles = np.arccos(_sigma) * 180 / np.pi
    _overlap_score = float(np.mean(_sigma ** 2))

    _pa_df = pd.DataFrame({
        "index": np.arange(len(_principal_angles)),
        "principal_angle_deg": np.round(_principal_angles, 2),
        "cos_sigma": np.round(_sigma, 4),
    })
    _pa_chart = (
        alt.Chart(_pa_df)
        .mark_bar(color="#6ee7b7")
        .encode(
            x=alt.X("index:O", title="Principal angle index"),
            y=alt.Y(
                "principal_angle_deg:Q",
                title="Principal angle (°)",
                scale=alt.Scale(domain=[0, 90]),
            ),
            tooltip=[
                alt.Tooltip("index:O"),
                alt.Tooltip("principal_angle_deg:Q", title="Angle (°)", format=".1f"),
                alt.Tooltip("cos_sigma:Q", title="cos(angle)", format=".4f"),
            ],
        )
        .properties(
            width="container",
            height=280,
            title=f"Principal angles (0° = perfect overlap) — subspace overlap score: {_overlap_score:.4f}",
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    mo.vstack([
        mo.callout(
            mo.md(f"**Subspace overlap score** (mean σ²): **{_overlap_score:.4f}** — 1.0 = identical subspaces, 0.0 = orthogonal"),
            kind="info",
        ),
        mo.ui.altair_chart(_pa_chart),
    ])
    return


if __name__ == "__main__":
    app.run()
