import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    _cwd = Path.cwd().resolve()
    _root = _cwd
    while _root != _root.parent:
        if (_root / "src").is_dir():
            break
        _root = _root.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from src.utils.subspace_compare import (
        discover_subspace_sources,
        orthonormal_basis,
        principal_angles_from_bases,
        sample_random_principal_angles,
        subspace_distance_metrics,
    )

    return (
        discover_subspace_sources,
        mo,
        np,
        orthonormal_basis,
        pd,
        plt,
        principal_angles_from_bases,
        sample_random_principal_angles,
        subspace_distance_metrics,
    )


@app.cell
def _(mo):
    mo.md("""
    # Compare RRR/PCA Subspaces
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    roots_input = mo.ui.text(
        value="results,.",
        label="Scan roots",
        full_width=True,
    )
    include_null_input = mo.ui.checkbox(value=False, label="Random null")
    null_draws_input = mo.ui.number(
        value=250, start=1, stop=10000, step=1, label="Null draws"
    )
    null_seed_input = mo.ui.number(value=0, start=0, step=1, label="Null seed")
    null_band_input = mo.ui.slider(
        start=50,
        stop=99,
        step=1,
        value=95,
        label="Null band %",
    )
    mo.vstack(
        [
            roots_input,
            mo.hstack(
                [
                    include_null_input,
                    null_draws_input,
                    null_seed_input,
                    null_band_input,
                ]
            ),
        ]
    )
    return (
        include_null_input,
        null_band_input,
        null_draws_input,
        null_seed_input,
        roots_input,
    )


@app.cell(hide_code=True)
def _(discover_subspace_sources, mo, roots_input):
    _roots = [r.strip() for r in roots_input.value.split(",") if r.strip()]
    sources = discover_subspace_sources(_roots)
    mo.stop(
        not sources,
        mo.callout(
            mo.md("No `fold_*/A.npy` subspace factors found under the scan roots."),
            kind="warn",
        ),
    )
    source_options = {
        f"{s.label}  |  kind={s.kind}, fold={s.fold}, shape=({s.ambient_dim}, {s.max_rank})": i
        for i, s in enumerate(sources)
    }
    return source_options, sources


@app.cell(hide_code=True)
def _(mo, source_options, sources):
    _default_b = 1 if len(sources) > 1 else 0
    _source_labels = list(source_options.keys())
    source_a_input = mo.ui.dropdown(
        options=source_options,
        value=_source_labels[0],
        label="Subspace A",
        full_width=True,
    )
    source_b_input = mo.ui.dropdown(
        options=source_options,
        value=_source_labels[_default_b],
        label="Subspace B",
        full_width=True,
    )
    pca_overlay_input = mo.ui.dropdown(
        options={
            "Off": "off",
            "PCA vs A": "a",
            "PCA vs B": "b",
            "PCA vs both": "both",
        },
        value="Off",
        label="PCA overlay",
    )
    mo.vstack([source_a_input, source_b_input, pca_overlay_input])
    return pca_overlay_input, source_a_input, source_b_input


@app.cell(hide_code=True)
def _(mo, source_a_input, source_b_input, sources):
    source_a = sources[source_a_input.value]
    source_b = sources[source_b_input.value]
    rank_a_input = mo.ui.number(
        start=1,
        stop=source_a.max_rank,
        step=1,
        value=min(10, source_a.max_rank),
        label="Rank A",
    )
    rank_b_input = mo.ui.number(
        start=1,
        stop=source_b.max_rank,
        step=1,
        value=min(10, source_b.max_rank),
        label="Rank B",
    )
    mo.hstack([rank_a_input, rank_b_input])
    return rank_a_input, rank_b_input, source_a, source_b


@app.cell(hide_code=True)
def _(mo, pd, sources):
    _rows = [
        {
            "label": s.label,
            "kind": s.kind,
            "fold": s.fold,
            "ambient_dim": s.ambient_dim,
            "max_rank": s.max_rank,
            "path": str(s.path),
        }
        for s in sources
    ]
    mo.ui.table(pd.DataFrame(_rows), selection=None)
    return


@app.cell(hide_code=True)
def _(np, orthonormal_basis, rank_a_input, rank_b_input, source_a, source_b):
    matrix_a = np.load(source_a.path, mmap_mode="r")
    matrix_b = np.load(source_b.path, mmap_mode="r")
    basis_a = orthonormal_basis(matrix_a, int(rank_a_input.value))
    basis_b = orthonormal_basis(matrix_b, int(rank_b_input.value))
    return basis_a, basis_b


@app.cell(hide_code=True)
def _(basis_a, basis_b, mo, principal_angles_from_bases):
    mo.stop(
        basis_a.shape[0] != basis_b.shape[0],
        mo.callout(
            mo.md(
                "Selected subspaces have different ambient dimensions: "
                f"{basis_a.shape[0]} vs {basis_b.shape[0]}."
            ),
            kind="warn",
        ),
    )
    angles = principal_angles_from_bases(basis_a, basis_b)
    return (angles,)


@app.cell(hide_code=True)
def _(np, source_a, source_b, sources):
    def _pca_sibling(source):
        parts = list(source.unit_dir.parts)
        replacements = {
            "rrr": "pca",
            "per_subject_rrr": "per_subject_pca",
            "per_region_rrr": "per_region_pca",
        }
        candidates = []
        for idx, part in enumerate(parts):
            if part in replacements:
                new_parts = parts.copy()
                new_parts[idx] = replacements[part]
                candidates.append(source.unit_dir.__class__(*new_parts))
        for candidate in candidates:
            for other in sources:
                if (
                    other.unit_dir == candidate
                    and other.fold == source.fold
                    and other.ambient_dim == source.ambient_dim
                ):
                    return other
        return None

    pca_a = _pca_sibling(source_a) if not source_a.kind.endswith("pca") else source_a
    pca_b = _pca_sibling(source_b) if not source_b.kind.endswith("pca") else source_b
    _ = np
    return pca_a, pca_b


@app.cell(hide_code=True)
def _(
    np,
    orthonormal_basis,
    pca_a,
    pca_b,
    pca_overlay_input,
    principal_angles_from_bases,
    rank_a_input,
    rank_b_input,
    source_a,
    source_b,
):
    overlay_angles = {}
    _mode = pca_overlay_input.value
    if _mode in {"a", "both"} and pca_a is not None and pca_a.path != source_a.path:
        _basis = orthonormal_basis(np.load(pca_a.path, mmap_mode="r"), rank_a_input.value)
        if _basis.shape[0] == source_a.ambient_dim:
            overlay_angles["PCA sibling vs A"] = principal_angles_from_bases(
                _basis,
                orthonormal_basis(np.load(source_a.path, mmap_mode="r"), rank_a_input.value),
            )
    if _mode in {"b", "both"} and pca_b is not None and pca_b.path != source_b.path:
        _basis = orthonormal_basis(np.load(pca_b.path, mmap_mode="r"), rank_b_input.value)
        if _basis.shape[0] == source_b.ambient_dim:
            overlay_angles["PCA sibling vs B"] = principal_angles_from_bases(
                _basis,
                orthonormal_basis(np.load(source_b.path, mmap_mode="r"), rank_b_input.value),
            )
    return (overlay_angles,)


@app.cell(hide_code=True)
def _(
    angles,
    include_null_input,
    np,
    null_band_input,
    null_draws_input,
    null_seed_input,
    rank_a_input,
    rank_b_input,
    sample_random_principal_angles,
    source_a,
    source_b,
):
    null_angles = None
    null_band = None
    if include_null_input.value:
        null_angles = sample_random_principal_angles(
            source_a.ambient_dim,
            int(rank_a_input.value),
            int(rank_b_input.value),
            int(null_draws_input.value),
            int(null_seed_input.value),
        )
        _alpha = (100 - int(null_band_input.value)) / 2
        null_band = {
            "low": np.percentile(null_angles, _alpha, axis=0),
            "median": np.percentile(null_angles, 50, axis=0),
            "high": np.percentile(null_angles, 100 - _alpha, axis=0),
        }
    _ = angles, source_b
    return null_angles, null_band


@app.cell(hide_code=True)
def _(
    angles,
    mo,
    np,
    null_angles,
    null_band,
    overlay_angles,
    plt,
    source_a,
    source_b,
):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(1, len(angles) + 1)
    ax.plot(x, np.degrees(angles), marker="o", linewidth=2, label="A vs B")
    if null_band is not None:
        ax.fill_between(
            x,
            np.degrees(null_band["low"]),
            np.degrees(null_band["high"]),
            alpha=0.18,
            color="0.4",
            label="Random null band",
        )
        ax.plot(
            x,
            np.degrees(null_band["median"]),
            color="0.35",
            linestyle="--",
            linewidth=1.5,
            label="Random null median",
        )
    for label, values in overlay_angles.items():
        _x = np.arange(1, len(values) + 1)
        ax.plot(_x, np.degrees(values), marker=".", linewidth=1.6, label=label)
    ax.set_xlabel("Principal angle index")
    ax.set_ylabel("Angle (degrees)")
    ax.set_ylim(bottom=0)
    ax.set_title("Principal angles between selected subspaces")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    _notes = [
        f"A: `{source_a.label}`",
        f"B: `{source_b.label}`",
    ]
    if null_angles is not None:
        _notes.append(f"Null draws: {null_angles.shape[0]}")
    mo.vstack([mo.md("\n\n".join(_notes)), fig])
    return


@app.cell(hide_code=True)
def _(angles, mo, np, overlay_angles, pd, subspace_distance_metrics):
    _rows = [{"comparison": "A vs B", **subspace_distance_metrics(angles)}]
    for _label, _angles in overlay_angles.items():
        _rows.append({"comparison": _label, **subspace_distance_metrics(_angles)})
    _df = pd.DataFrame(_rows)
    for _col in ["grassmann", "fubini_study"]:
        _df[f"{_col}_degrees"] = np.degrees(_df[_col])
    mo.ui.table(_df, selection=None)
    return


if __name__ == "__main__":
    app.run()
