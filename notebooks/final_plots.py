"""Curated paper figures from preloaded result summaries."""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import sys
    import marimo as mo

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.utils.plotting.final_data import load_final_data, primary_analysis_data
    from src.utils.plotting.final_figures import (
        average_quantile_grid, cross_aggregate, cross_gap_scatter,
        control_saturation_grid, decoding_aggregate, encoding_grid, model_order, quantile_grid,
        rrr_aggregate, sensitivity_grid, significance_grid,
        significance_latex,
    )

    results_root = repo_root / "results"
    return (
        average_quantile_grid,
        control_saturation_grid,
        cross_aggregate,
        cross_gap_scatter,
        decoding_aggregate,
        encoding_grid,
        load_final_data,
        mo,
        model_order,
        primary_analysis_data,
        quantile_grid,
        results_root,
        rrr_aggregate,
        sensitivity_grid,
        significance_grid,
        significance_latex,
    )


@app.cell
def _(load_final_data, results_root):
    # The only large artifact read in this notebook. Restart to refresh results.
    (internal, loso, cross, quantiles, loso_quantiles, full_rows,
     significant, decoding, gaps) = load_final_data(results_root)
    return (
        cross,
        decoding,
        full_rows,
        gaps,
        internal,
        loso,
        loso_quantiles,
        quantiles,
        significant,
    )


@app.cell(hide_code=True)
def _(
    cross,
    decoding,
    internal,
    loso,
    loso_quantiles,
    primary_analysis_data,
    quantiles,
):
    # Keep all loaded model results available for appendix controls.
    (primary_internal, primary_loso, primary_cross, primary_quantiles,
     primary_loso_quantiles, primary_decoding) = primary_analysis_data(
        internal, loso, cross, quantiles, loso_quantiles, decoding
    )

    return (
        primary_cross,
        primary_decoding,
        primary_internal,
        primary_loso,
        primary_loso_quantiles,
        primary_quantiles,
    )


@app.cell
def _(gaps, mo):
    directed_gaps = [gap for gap in gaps if gap.startswith("directed ")]
    directed_details = "\n".join(f"- {gap}" for gap in directed_gaps)
    other_gaps = [gap for gap in gaps if not gap.startswith("directed ")]
    other_details = "\n".join(f"- {gap}" for gap in other_gaps)
    mo.md(
        "# Final plots\n\n"
        "All saved correlations and target summaries were loaded at startup. "
        "Restart the notebook to pick up new artifacts.\n\n"
        f"**Missing or incomplete results:** {len(gaps)} total, including "
        f"{len(directed_gaps)} directed results. Available completed results are plotted.\n\n"
        f"<details><summary>Show directed result gaps</summary>\n\n{directed_details}\n\n</details>\n\n"
        f"<details><summary>Show other result gaps</summary>\n\n{other_details}\n\n</details>"
    )
    return


@app.cell
def _(mo, primary_quantiles):
    pairs = sorted({(model, layer) for dataset, model, layer, method in primary_quantiles
                    if all((required_dataset, model, layer, required_method) in primary_quantiles
                           for required_dataset in ("podcast", "cars-2", "fmri")
                           for required_method in ("rrr", "pca"))},
                   key=lambda item: (item[0], item[1] is not None, item[1] or -1))
    options = {f"{model} / {'layerless' if layer is None else f'layer {layer}'}": (model, layer)
               for model, layer in pairs}
    selected_pair = mo.ui.dropdown(
        options=options,
        value=next((label for label, pair in options.items() if pair == ("gpt2", 0)),
                   next(iter(options), None)),
        label="Sample model layer",
    )
    selected_pair
    return (selected_pair,)


@app.cell
def _(mo, primary_loso_quantiles):
    _loso_pairs = sorted({(model, layer) for dataset, model, layer, subject, method in primary_loso_quantiles
                    if all(any(key[:3] == (required_dataset, model, layer)
                               and key[-1] == required_method for key in primary_loso_quantiles)
                           for required_dataset in ("podcast", "cars-2", "fmri")
                           for required_method in ("RRR baseline", "RRR transfer", "PCA baseline"))},
                   key=lambda item: (item[0], item[1] is not None, item[1] or -1))
    _loso_options = {f"{model} / {'layerless' if layer is None else f'layer {layer}'}": (model, layer)
               for model, layer in _loso_pairs}
    loso_pair = mo.ui.dropdown(
        options=_loso_options,
        value=next((label for label, pair in _loso_options.items() if pair == ("gpt2", 0)),
                   next(iter(_loso_options), None)),
        label="LOSO sample model layer",
    )
    loso_pair
    return (loso_pair,)


@app.cell
def _(
    full_rows,
    loso_pair,
    mo,
    primary_internal,
    primary_loso,
    primary_loso_quantiles,
    primary_quantiles,
    rrr_aggregate,
    selected_pair,
):
    if selected_pair.value is None:
        aggregate_output = mo.md("No internal RRR quantile curves are available.")
    else:
        pair = selected_pair.value
        figures = [
            mo.md("## Internal RRR aggregate"),
            rrr_aggregate(primary_internal, primary_loso, primary_quantiles, primary_loso_quantiles,
                          full_rows, pair, mode="internal"),
        ]
        if loso_pair.value is not None:
            figures.extend([
                mo.md("## Leave-one-subject-out RRR aggregate"),
                rrr_aggregate(primary_internal, primary_loso, primary_quantiles, primary_loso_quantiles,
                              full_rows, loso_pair.value, mode="loso"),
            ])
        aggregate_output = mo.vstack(figures)
    aggregate_output
    return


@app.cell
def _(cross_aggregate, cross_gap_scatter, mo, primary_cross):
    mo.vstack([
        mo.md("## Directed cross-dataset RRR aggregate"),
        cross_aggregate(primary_cross),
        mo.md("## Internal minus transfer AUC gap correlation"),
        cross_gap_scatter(primary_cross),
    ])
    return


@app.cell
def _(decoding_aggregate, mo, primary_decoding):
    mo.vstack([mo.md("## Language decoding aggregate"),
               decoding_aggregate(primary_decoding)])
    return


@app.cell
def _(encoding_grid, full_rows, mo, model_order):
    order = model_order(full_rows)
    mo.vstack([mo.md("## Full encoding-model performance"),
               encoding_grid(full_rows, order)])
    return (order,)


@app.cell
def _(mo, order, significance_grid, significance_latex, significant):
    latex = significance_latex(significant, order)
    mo.vstack([
        mo.md("## Significant targets by model layer"),
        significance_grid(significant, order),
        mo.md("### Significant-target counts and dataset unions (LaTeX)"),
        mo.md("```latex\n" + latex + "\n```"),
    ])
    return


@app.cell
def _(mo, primary_internal, primary_quantiles, quantile_grid, selected_pair):
    if selected_pair.value is None:
        quantile_output = mo.md("No selected model layer.")
    else:
        quantile_output = mo.vstack([
            mo.md("## RRR and PCA target quantiles by rank"),
            quantile_grid(primary_quantiles, selected_pair.value, primary_internal),
        ])
    quantile_output
    return


@app.cell
def _(average_quantile_grid, mo, primary_quantiles):
    mo.vstack([
        mo.md("## Mean normalized quantile curves across model layers"),
        average_quantile_grid(primary_quantiles),
    ])
    return


@app.cell
def _(mo, primary_internal, sensitivity_grid):
    mo.vstack([
        mo.md("## Saturation sensitivity to required peak performance"),
        sensitivity_grid(primary_internal, (.5, .75, .9, .95, .975, .99, 1.0)),
    ])
    return


@app.cell(hide_code=True)
def _(mo, primary_loso, sensitivity_grid):
    mo.vstack([
        mo.md("## LOSO saturation sensitivity to required peak performance"),
        sensitivity_grid(primary_loso, (.5, .75, .9, .95, .975, .99, 1.0), mode="loso"),
    ])
    return


@app.cell
def _(control_saturation_grid, internal, mo):
    mo.vstack([
        mo.md("## Appendix: random GPT-2 internal saturation ranks"),
        control_saturation_grid(internal, "random-gpt2"),
        mo.md("## Appendix: GloVe internal saturation ranks"),
        control_saturation_grid(internal, "glove-300"),
    ])
    return


if __name__ == "__main__":
    app.run()
