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

    from src.utils.plotting.final_data import load_final_data
    from src.utils.plotting.final_figures import (
        average_quantile_grid, cross_aggregate, cross_gap_scatter,
        decoding_aggregate, encoding_grid, model_order, quantile_grid,
        rrr_aggregate, sensitivity_grid, significance_grid,
        significance_latex,
    )

    results_root = repo_root / "results"
    return (
        average_quantile_grid,
        cross_aggregate,
        cross_gap_scatter,
        decoding_aggregate,
        encoding_grid,
        load_final_data,
        mo,
        model_order,
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
def _(mo, quantiles):
    pairs = sorted({(model, layer) for dataset, model, layer, method in quantiles
                    if all((required_dataset, model, layer, required_method) in quantiles
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
def _(loso_quantiles, mo):
    _loso_pairs = sorted({(model, layer) for dataset, model, layer, subject, method in loso_quantiles
                    if all(any(key[:3] == (required_dataset, model, layer)
                               and key[-1] == required_method for key in loso_quantiles)
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
    internal,
    loso,
    loso_pair,
    loso_quantiles,
    mo,
    quantiles,
    rrr_aggregate,
    selected_pair,
):
    if selected_pair.value is None:
        aggregate_output = mo.md("No internal RRR quantile curves are available.")
    else:
        pair = selected_pair.value
        figures = [
            mo.md("## Internal RRR aggregate"),
            rrr_aggregate(internal, loso, quantiles, loso_quantiles,
                          full_rows, pair, mode="internal"),
        ]
        if loso_pair.value is not None:
            figures.extend([
                mo.md("## Leave-one-subject-out RRR aggregate"),
                rrr_aggregate(internal, loso, quantiles, loso_quantiles,
                              full_rows, loso_pair.value, mode="loso"),
            ])
        aggregate_output = mo.vstack(figures)
    aggregate_output
    return


@app.cell
def _(cross, cross_aggregate, cross_gap_scatter, mo):
    mo.vstack([
        mo.md("## Directed cross-dataset RRR aggregate"),
        cross_aggregate(cross),
        mo.md("## Internal minus transfer AUC gap correlation"),
        cross_gap_scatter(cross),
    ])
    return


@app.cell
def _(decoding, decoding_aggregate, mo):
    mo.vstack([mo.md("## Language decoding aggregate"),
               decoding_aggregate(decoding)])
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
def _(mo, quantile_grid, quantiles, selected_pair):
    if selected_pair.value is None:
        quantile_output = mo.md("No selected model layer.")
    else:
        quantile_output = mo.vstack([
            mo.md("## RRR and PCA target quantiles by rank"),
            quantile_grid(quantiles, selected_pair.value),
        ])
    quantile_output
    return


@app.cell
def _(average_quantile_grid, mo, quantiles):
    mo.vstack([
        mo.md("## Mean normalized quantile curves across model layers"),
        average_quantile_grid(quantiles),
    ])
    return


@app.cell
def _(mo, quantiles, sensitivity_grid):
    mo.vstack([
        mo.md("## Saturation sensitivity to minimum quantile correlation"),
        sensitivity_grid(quantiles, (0, .005, .01), variable="floor"),
        mo.md("## Saturation sensitivity to required peak performance"),
        sensitivity_grid(quantiles, (.5, .75, .9, .95, .99, 1.0), variable="fraction"),
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.utils.plotting.final_export import main

        main()
    else:
        app.run()
