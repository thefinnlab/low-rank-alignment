import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import sys
    sys.path.append(".")

    import matplotlib.pyplot as plt
    import altair as alt
    import pandas as pd

    from scripts.run_significance import Artifact

    return Artifact, alt, pd


@app.cell
def _():
    datasets = {
        # "podcast": "sig_lags", 
        "brain-treebank-cars-2": "sig_lags", 
        # "fmri": "held_out_session",
    }

    model_information = {
        "gpt-j-6b": [0, 7, 14, 21, 28],
        "gpt2-xl-batch": [0, 12, 24, 36, 48],
        "llama3-8b": [0, 8, 16, 24, 32],
        "roberta-large": [0, 6, 12, 18, 24],
        "random-gpt2-xl": [0, 12, 24, 36, 48],
        "gpt2": [0, 3, 6, 9, 12],
    }
    return datasets, model_information


@app.cell
def _():
    model_layer_results = {"gpt-j-6b": [0] * 5,
                           "gpt2-xl-batch": [0] * 5, 
                           "llama3-8b": [0] * 5, 
                           "roberta-large": [0] * 5, 
                           "random-gpt2-xl": [0] * 5, 
                           "gpt2": [0] * 5
                          }
    return (model_layer_results,)


@app.cell
def _(Artifact, datasets, model_information, model_layer_results):
    for _dataset, _mode in datasets.items():
        for _model, _layers in model_information.items():
            for i, _layer in enumerate(_layers):
                artifact_results = Artifact(dataset=_dataset, model=_model, mode=_mode, layer=_layer).load_artifacts(["target_significance"]).target_significance
                for _, sig in artifact_results.items():
                    # if model == "gpt2":
                    #     print(sig.significant.sum())
                    # print(sig.significant.count())
                    model_layer_results[_model][i] += sig.significant.sum()
    return


@app.cell
def _(model_layer_results):
    model_layer_results
    return


@app.cell
def _(model_information, model_layer_results, pd):
    rows = []
    for model, values in model_layer_results.items():
        layers = model_information[model]
        for idx, (layer, val) in enumerate(zip(layers, values)):
            rows.append(
                {
                    "model": model,
                    "step_idx": f"Point {idx + 1}",
                    "layer": layer,
                    "layer_label": f"L{layer}",
                    "value": int(val),
                }
            )

    df = pd.DataFrame(rows)
    return (df,)


@app.cell
def _(alt, df):
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("layer:O", title="Layer", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("value:Q", title="Value"),
            color=alt.Color("model:N", legend=None),
            tooltip=["model", "layer", "value"],
        )
        .properties(width=120, height=200)
        .facet(facet=alt.Facet("model:N", title="Model"), columns=3)
        .resolve_scale(x="independent")
    )

    chart.show()
    return


@app.cell
def _(alt, df):
    per_layer_value_df = df.groupby("step_idx").agg({"step_idx": "first", "value": "sum"})
    _chart = (
        alt.Chart(per_layer_value_df)
        .mark_bar()
        .encode(
            x=alt.X("step_idx:O", title="step_idx", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("value:Q", title="Value"),
            # color=alt.Color("model:N", legend=None),
            tooltip=["step_idx", "value"],
        )
        .properties(width=800, height=200)
        .resolve_scale(x="independent")
    )

    _chart.show()
    return


@app.cell
def _(alt, df):
    per_model_value_df = df.groupby("model").agg({"model": "first", "value": "sum"})
    _chart = (
        alt.Chart(per_model_value_df)
        .mark_bar()
        .encode(
            x=alt.X("model:O", title="model", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("value:Q", title="Value"),
            # color=alt.Color("model:N", legend=None),
            tooltip=["model", "value"],
        )
        .properties(width=1000, height=200)
        .resolve_scale(x="independent")
    )

    _chart.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
